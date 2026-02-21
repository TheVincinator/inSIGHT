"""
camera_client.py

Real-time CV feature extractor + WebSocket streamer (no ROS).
Features:
- Blink detection via EAR (Eye Aspect Ratio)
- Blink rate over rolling window (blinks/min)
- PERCLOS (% frames eyes closed over window)
- Head motion variance (NOW NORMALIZED by face scale)
- Pupil "dilation" proxy (computed only when eyes are open + median filtered):
    - pupil_ratio: blended dark-pixel/contour ratio inside eye ROI (median filtered)
    - pupil_baseline: median pupil_ratio over first BASELINE_SEC seconds
    - pupil_delta: pupil_ratio - pupil_baseline

Requires (inside your venv311):
  pip install opencv-python mediapipe websockets numpy

Run:
  # Terminal 1
  uvicorn fusion_server:app --host 127.0.0.1 --port 8000 --reload

  # Terminal 2
  python camera_client.py

Quit: press 'q' in the OpenCV window.
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import websockets

# ----------------------------
# Tunables
# ----------------------------
WS_URL = "ws://127.0.0.1:8000/ws"  # use 127.0.0.1 to avoid IPv6/DNS weirdness
SEND_HZ = 1.0                      # send metrics once per second
WINDOW_SEC = 15.0                  # rolling window for blink rate / perclos / head motion
MIN_FACE_CONF = 0.5
MIN_TRACK_CONF = 0.5

CAM_WIDTH = 640
CAM_HEIGHT = 480

# Pick camera index based on your ffmpeg device list:
# 0 = MacBook Air Camera
# 1 = iPhone 11 Camera (Continuity)
# 3 = MacBook Air Desk View Camera
CAMERA_INDEX = 0

# On macOS, AVFoundation backend is typically most stable
CAP_BACKEND = cv2.CAP_AVFOUNDATION

# Pupil baseline collection
BASELINE_SEC = 15.0  # first N seconds after pupil proxy becomes available

# ---- Adaptive EAR baseline ----
EAR_BASELINE_SEC = 2.0       # collect "open-eye" EAR for first ~2 seconds
EAR_THRESH_FRAC = 0.75       # threshold = EAR_THRESH_FRAC * median_open_EAR
EAR_FALLBACK_THRESH = 0.21   # used until baseline is ready

# ---- Pupil stability ----
PUPIL_OPEN_MARGIN = 0.02     # only compute pupil when ear > (ear_thresh + margin)
PUPIL_MEDIAN_N = 5           # median filter window for pupil_ratio

# ----------------------------
# MediaPipe FaceMesh landmark indices
# EAR uses 6 points per eye
# ----------------------------
L_EYE = {
    "p1": 33,    # outer corner
    "p4": 133,   # inner corner
    "p2": 160,   # upper inner
    "p6": 144,   # lower inner
    "p3": 158,   # upper outer
    "p5": 153,   # lower outer
}
R_EYE = {
    "p1": 362,   # outer corner
    "p4": 263,   # inner corner
    "p2": 387,   # upper inner
    "p6": 373,   # lower inner
    "p3": 385,   # upper outer
    "p5": 380,   # lower outer
}

# Nose tip for head motion proxy
NOSE_TIP_IDX = 1

# Face scale: inter-ocular distance between outer eye corners
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 362

# Eye ROI indices (small set around each eye, good enough for ROI crop)
LEFT_EYE_ROI_IDXS = [33, 133, 160, 158, 144, 153]
RIGHT_EYE_ROI_IDXS = [362, 263, 387, 385, 373, 380]


def _pt(landmarks, idx: int, w: int, h: int) -> np.ndarray:
    """Return pixel coordinates (x, y) as float array."""
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def ear_from_eye(landmarks, eye_idx_map, w: int, h: int) -> float:
    """
    Eye Aspect Ratio:
      EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    """
    p1 = _pt(landmarks, eye_idx_map["p1"], w, h)
    p2 = _pt(landmarks, eye_idx_map["p2"], w, h)
    p3 = _pt(landmarks, eye_idx_map["p3"], w, h)
    p4 = _pt(landmarks, eye_idx_map["p4"], w, h)
    p5 = _pt(landmarks, eye_idx_map["p5"], w, h)
    p6 = _pt(landmarks, eye_idx_map["p6"], w, h)

    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    hdist = np.linalg.norm(p1 - p4) + 1e-6
    return float((v1 + v2) / (2.0 * hdist))


def eye_roi_from_landmarks(frame_bgr, landmarks, idx_list, pad: int = 8):
    """
    Crop a bounding box around given landmark indices.
    idx_list: list of landmark indices around an eye.
    """
    h, w = frame_bgr.shape[:2]
    pts = np.array([_pt(landmarks, idx, w, h) for idx in idx_list], dtype=np.float32)

    x_min = int(max(0, np.min(pts[:, 0]) - pad))
    x_max = int(min(w - 1, np.max(pts[:, 0]) + pad))
    y_min = int(max(0, np.min(pts[:, 1]) - pad))
    y_max = int(min(h - 1, np.max(pts[:, 1]) + pad))

    if x_max <= x_min or y_max <= y_min:
        return None

    return frame_bgr[y_min:y_max, x_min:x_max]


def pupil_ratio_proxy(eye_bgr) -> Optional[float]:
    """
    Hackathon-friendly proxy for pupil "size" in an eye ROI.

    Returns a ratio ~[0..1] where larger means "more dark blob area".
    Uses:
      - CLAHE to normalize lighting
      - Gaussian blur
      - adaptive threshold (binary_inv) to isolate dark pixels
      - blend largest contour ratio + dark pixel ratio

    This is NOT medically accurate and is sensitive to lighting/glare.
    Use pupil_delta vs baseline for more stability.
    """
    if eye_bgr is None or eye_bgr.size == 0:
        return None

    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)

    # Normalize contrast (helps a lot with uneven lighting)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Smooth noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold: pupil is dark. Invert so pupil becomes white in the mask.
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 7
    )

    # Morphological open to remove specks
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    dark_ratio = float(np.sum(mask > 0) / mask.size)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_area = float(max(areas))
        contour_ratio = max_area / float(mask.size)
        return 0.6 * contour_ratio + 0.4 * dark_ratio

    return dark_ratio


@dataclass
class WindowStats:
    # Each entry: (timestamp, is_closed, blink_event, motion_norm)
    samples: Deque[Tuple[float, bool, int, float]]

    def __init__(self):
        self.samples = deque()

    def prune(self, now: float, window_sec: float):
        while self.samples and (now - self.samples[0][0] > window_sec):
            self.samples.popleft()

    def add(self, ts: float, is_closed: bool, blink_event: int, motion_norm: float):
        self.samples.append((ts, is_closed, blink_event, motion_norm))

    def blink_rate_per_min(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        t0 = self.samples[0][0]
        t1 = self.samples[-1][0]
        dt = max(1e-6, t1 - t0)
        blinks = sum(s[2] for s in self.samples)
        return float(blinks * 60.0 / dt)

    def perclos(self) -> float:
        if not self.samples:
            return 0.0
        closed = sum(1 for s in self.samples if s[1])
        return float(closed / len(self.samples))

    def head_motion_var(self) -> float:
        """Variance of normalized motion across the window."""
        if len(self.samples) < 3:
            return 0.0
        m = np.array([s[3] for s in self.samples], dtype=np.float32)
        return float(np.var(m))


async def camera_loop():
    # Setup camera
    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_BACKEND)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {CAMERA_INDEX}). "
            "Try a different CAMERA_INDEX, and ensure macOS Camera permission for Terminal/VSCode."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    # Setup FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=MIN_FACE_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    win = WindowStats()
    last_send = 0.0

    # Blink state machine: count blink when open->closed->open
    was_closed = False
    blink_armed = False

    # ---- EAR baseline tracking (adaptive threshold) ----
    ear_baseline_samples = []
    ear_baseline_start: Optional[float] = None
    ear_thresh = EAR_FALLBACK_THRESH

    # ---- Head motion normalization ----
    prev_nose_xy: Optional[np.ndarray] = None

    # ---- Pupil median filter ----
    pupil_hist = deque(maxlen=PUPIL_MEDIAN_N)

    # Pupil baseline tracking
    pupil_baseline: Optional[float] = None
    pupil_baseline_samples = []
    baseline_start: Optional[float] = None

    # Connect to server (if server isn't running, this will error)
    async with websockets.connect(WS_URL) as ws:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                await asyncio.sleep(0.01)
                continue

            # Mirror view (more natural)
            frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            res = mesh.process(rgb)
            now = time.time()

            blink_event = 0
            ear = None
            is_closed = False

            pupil_ratio = None
            pupil_delta = None

            motion_norm = 0.0  # normalized per-frame motion

            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                lms = face.landmark

                # EAR
                ear_l = ear_from_eye(lms, L_EYE, w, h)
                ear_r = ear_from_eye(lms, R_EYE, w, h)
                ear = (ear_l + ear_r) / 2.0

                # ---- Adaptive EAR threshold (open-eye baseline) ----
                if ear_baseline_start is None:
                    ear_baseline_start = now

                if (now - ear_baseline_start) <= EAR_BASELINE_SEC:
                    ear_baseline_samples.append(ear)

                if (now - ear_baseline_start) > EAR_BASELINE_SEC and len(ear_baseline_samples) >= 10:
                    open_med = float(np.median(ear_baseline_samples))
                    ear_thresh = max(0.10, EAR_THRESH_FRAC * open_med)

                is_closed = ear < ear_thresh

                # Nose tip
                nose_xy = _pt(lms, NOSE_TIP_IDX, w, h)

                # Face scale = inter-ocular distance (outer eye corners)
                l_outer = _pt(lms, LEFT_EYE_OUTER, w, h)
                r_outer = _pt(lms, RIGHT_EYE_OUTER, w, h)
                face_scale = float(np.linalg.norm(l_outer - r_outer) + 1e-6)

                # Normalized motion (frame-to-frame nose displacement / face scale)
                if prev_nose_xy is not None and np.isfinite(prev_nose_xy).all():
                    motion_norm = float(np.linalg.norm(nose_xy - prev_nose_xy) / face_scale)
                prev_nose_xy = nose_xy

                # Blink logic
                if not was_closed and is_closed:
                    blink_armed = True
                elif was_closed and not is_closed and blink_armed:
                    blink_event = 1
                    blink_armed = False
                was_closed = is_closed

                # ---- Pupil: only compute when eyes open enough ----
                eyes_open_for_pupil = ear > (ear_thresh + PUPIL_OPEN_MARGIN)

                if eyes_open_for_pupil:
                    left_eye_roi = eye_roi_from_landmarks(frame, lms, LEFT_EYE_ROI_IDXS, pad=10)
                    right_eye_roi = eye_roi_from_landmarks(frame, lms, RIGHT_EYE_ROI_IDXS, pad=10)

                    pupil_l = pupil_ratio_proxy(left_eye_roi)
                    pupil_r = pupil_ratio_proxy(right_eye_roi)

                    raw = None
                    if pupil_l is not None and pupil_r is not None:
                        raw = (pupil_l + pupil_r) / 2.0
                    elif pupil_l is not None:
                        raw = pupil_l
                    elif pupil_r is not None:
                        raw = pupil_r

                    if raw is not None:
                        pupil_hist.append(raw)
                        pupil_ratio = float(np.median(np.array(pupil_hist, dtype=np.float32)))

                # Baseline init + collection (only when pupil_ratio is valid)
                if pupil_ratio is not None:
                    if baseline_start is None:
                        baseline_start = now

                    if pupil_baseline is None and (now - baseline_start) <= BASELINE_SEC:
                        pupil_baseline_samples.append(pupil_ratio)

                    if (
                        pupil_baseline is None
                        and (now - baseline_start) > BASELINE_SEC
                        and len(pupil_baseline_samples) > 10
                    ):
                        pupil_baseline = float(np.median(pupil_baseline_samples))

                if pupil_ratio is not None and pupil_baseline is not None:
                    pupil_delta = pupil_ratio - pupil_baseline

                # Add to rolling window (motion_norm is already safe)
                win.add(now, is_closed, blink_event, motion_norm)

                # Overlay
                cv2.putText(
                    frame,
                    f"EAR: {ear:.3f}  thr:{ear_thresh:.3f}  closed={int(is_closed)}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.circle(frame, (int(nose_xy[0]), int(nose_xy[1])), 3, (0, 255, 0), -1)

            else:
                # If face not detected, avoid counting blinks incorrectly
                blink_armed = False
                was_closed = False
                prev_nose_xy = None  # reset motion reference
                cv2.putText(
                    frame,
                    "Face not detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # Prune window
            win.prune(now, WINDOW_SEC)

            blink_rate = win.blink_rate_per_min()
            perclos = win.perclos()
            head_var = win.head_motion_var()

            # Overlays (always show)
            cv2.putText(
                frame,
                f"Blink/min({int(WINDOW_SEC)}s): {blink_rate:.1f}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"PERCLOS({int(WINDOW_SEC)}s): {perclos:.2f}",
                (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"HeadVarNorm({int(WINDOW_SEC)}s): {head_var:.4f}",
                (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Pupil overlays (if available)
            if pupil_ratio is not None:
                base_str = "None" if pupil_baseline is None else f"{pupil_baseline:.4f}"
                cv2.putText(
                    frame,
                    f"PupilRatio: {pupil_ratio:.4f}  Base: {base_str}",
                    (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
            if pupil_delta is not None:
                cv2.putText(
                    frame,
                    f"PupilDelta: {pupil_delta:+.4f}",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("camera_client (CV features)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Send once per second
            if now - last_send >= (1.0 / SEND_HZ):
                payload = {
                    "type": "eye_metrics",
                    "value": {
                        "blink_rate_per_min": blink_rate,
                        "perclos": perclos,
                        "head_motion_var": head_var,  # now normalized-var
                        "ear": ear,
                        "ear_thresh": ear_thresh,
                        "pupil_ratio": pupil_ratio,
                        "pupil_delta": pupil_delta,
                        "pupil_baseline": pupil_baseline,
                        "face_detected": bool(res.multi_face_landmarks),
                        "window_sec": WINDOW_SEC,
                        "baseline_sec": BASELINE_SEC,
                        "camera_index": CAMERA_INDEX,
                    },
                    "timestamp": now,
                }
                await ws.send(json.dumps(payload))
                last_send = now

            await asyncio.sleep(0)  # yield to event loop

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(camera_loop())

#still need to add:
# logic to turn camera to keep face centered (camera turns using servos)