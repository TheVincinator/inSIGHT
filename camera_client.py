import asyncio
import json
import os
import random
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pygame
import requests
import websockets

from config import (
    CAM_HEIGHT, CAM_WIDTH, CAMERA_INDEX,
    EAR_BASELINE_SEC, EAR_FALLBACK_THRESH, EAR_THRESH_FRAC,
    ESP32_SERVO_URL, EYE_SEND_HZ, EYE_WINDOW_SEC,
    LOW_BLINK_THRESH, RETRY_DELAY,
    RPPG_ASSUMED_FPS, RPPG_ENABLED, RPPG_HR_MAX_BPM, RPPG_HR_MIN_BPM,
    RPPG_MIN_WINDOW_SEC, RPPG_RESET_ABSENT_SEC, RPPG_WINDOW_SEC,
    SERVO_ALPHA, SERVO_DEAD_ZONE, SERVO_GAIN, SERVO_SEND_HZ,
    WS_INGEST_URL, WS_SUBSCRIBE_URL,
)
from rppg import RPPGEstimator

# ---- Audio alert setup ----
# On each stress trigger: plays stress_alert.wav first, waits for it to finish,
# then plays a randomly chosen voice clip from the stress_alerts/ folder.
pygame.mixer.init()
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ALERTS_DIR  = os.path.join(_SCRIPT_DIR, "stress_alerts")

# Load the alert tone (stress_alert.wav sits inside stress_alerts/)
_alert_tone_path = os.path.join(_ALERTS_DIR, "stress_alert.wav")
try:
    _alert_tone = pygame.mixer.Sound(_alert_tone_path)
    print(f"[camera_client] loaded alert tone: stress_alert.wav")
except Exception as e:
    _alert_tone = None
    print(f"[camera_client] could not load stress_alert.wav: {e}")

# Load voice clips from stress_alerts/ — exclude stress_alert.wav itself
_stress_alerts = []
if not os.path.isdir(_ALERTS_DIR):
    print(f"[camera_client] alerts folder not found: {_ALERTS_DIR} — voice alerts disabled")
else:
    for filename in sorted(f for f in os.listdir(_ALERTS_DIR)
                           if f.lower().endswith((".wav", ".mp3"))
                           and f != "stress_alert.wav"):
        path = os.path.join(_ALERTS_DIR, filename)
        _stress_alerts.append(path)
        print(f"[camera_client] loaded voice alert: {filename}")

    if not _stress_alerts:
        print(f"[camera_client] no .wav or .mp3 files found in {_ALERTS_DIR} — voice alerts disabled")


def _play_random_alert():
    """
    Play the alert tone, wait for it to finish, then play a random voice clip.
    Runs in a background thread so the async camera loop is never blocked.
    """
    def _run():
        # Step 1: play the alert tone through the Sound channel (non-music)
        if _alert_tone is not None:
            channel = _alert_tone.play()
            if channel is not None:
                while channel.get_busy():
                    pygame.time.wait(50)

        # Step 2: play a random voice clip through the music channel
        if _stress_alerts:
            pygame.mixer.music.load(random.choice(_stress_alerts))
            pygame.mixer.music.play()

    threading.Thread(target=_run, daemon=True).start()

stress_score       = 0.0
stress_state       = "normal"
_prev_stress_state = "normal"  # tracks last state to detect transitions into "stressed"

# macOS capture backend; use cv2.CAP_ANY on Linux/Windows
CAP_BACKEND = cv2.CAP_AVFOUNDATION

MIN_FACE_CONF  = 0.5
MIN_TRACK_CONF = 0.5

servo_filtered  = 0.0
last_servo_send = 0.0


def send_servo_command(speed):
    if not ESP32_SERVO_URL:
        return
    try:
        requests.get(ESP32_SERVO_URL, params={"pan": float(speed)}, timeout=0.02)
    except Exception:
        pass


def stress_color(state: str):
    if state == "calm":
        return (0, 255, 120)
    elif state == "stressed":
        return (0, 80, 255)
    else:
        return (0, 200, 255)


L_EYE = {"p1": 33,  "p4": 133, "p2": 160, "p6": 144, "p3": 158, "p5": 153}
R_EYE = {"p1": 362, "p4": 263, "p2": 387, "p6": 373, "p3": 385, "p5": 380}

NOSE_TIP_IDX    = 1
LEFT_EYE_OUTER  = 33
RIGHT_EYE_OUTER = 362
LEFT_EYE_INNER  = 133   # medial canthus, inner corner of left eye
RIGHT_EYE_INNER = 263   # medial canthus, inner corner of right eye

# Iris landmark indices (available with refine_landmarks=True)
# Left iris centre = 468, right iris centre = 473
LEFT_IRIS_IDX  = 468
RIGHT_IRIS_IDX = 473

last_valid_eye_metrics = None


def _pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def ear_from_eye(landmarks, eye_idx_map, w, h):
    p1 = _pt(landmarks, eye_idx_map["p1"], w, h)
    p2 = _pt(landmarks, eye_idx_map["p2"], w, h)
    p3 = _pt(landmarks, eye_idx_map["p3"], w, h)
    p4 = _pt(landmarks, eye_idx_map["p4"], w, h)
    p5 = _pt(landmarks, eye_idx_map["p5"], w, h)
    p6 = _pt(landmarks, eye_idx_map["p6"], w, h)
    v1    = np.linalg.norm(p2 - p6)
    v2    = np.linalg.norm(p3 - p5)
    hdist = np.linalg.norm(p1 - p4) + 1e-6
    return float((v1 + v2) / (2.0 * hdist))


def iris_diameter_px(landmarks, iris_idx, eye_outer_idx, eye_inner_idx, w, h):
    """
    Estimate iris diameter in pixels, normalised by eye width to remove
    distance-from-camera effects. Returns a scale-invariant ratio.
    Eye width is measured from outer corner to inner corner for accuracy.
    """
    iris   = _pt(landmarks, iris_idx, w, h)
    outer  = _pt(landmarks, eye_outer_idx, w, h)
    inner  = _pt(landmarks, eye_inner_idx, w, h)
    eye_w  = np.linalg.norm(outer - inner) + 1e-6
    return float(np.linalg.norm(iris - outer) / eye_w)


@dataclass
class WindowStats:
    samples: Deque[Tuple[float, bool, int, float]]

    def __init__(self):
        self.samples = deque()

    def prune(self, now, window_sec):
        while self.samples and (now - self.samples[0][0] > window_sec):
            self.samples.popleft()

    def add(self, ts, is_closed, blink_event, motion_norm):
        self.samples.append((ts, is_closed, blink_event, motion_norm))

    def blink_rate_per_min(self):
        if len(self.samples) < 2:
            return 0.0
        t0 = self.samples[0][0]
        t1 = self.samples[-1][0]
        dt = max(1e-6, t1 - t0)
        blinks = sum(s[2] for s in self.samples)
        return float(blinks * 60.0 / dt)

    def perclos(self):
        if not self.samples:
            return 0.0
        closed = sum(1 for s in self.samples if s[1])
        return float(closed / len(self.samples))

    def head_motion_var(self):
        if len(self.samples) < 3:
            return 0.0
        m = np.array([s[3] for s in self.samples], dtype=np.float32)
        return float(np.var(m))


def _safe_float(v, fallback=0.0):
    """Convert v to a plain Python float safe for JSON serialization.
    Returns fallback if v is None, NaN, or infinite."""
    try:
        f = float(v)
        if not np.isfinite(f):
            return fallback
        return f
    except (TypeError, ValueError):
        return fallback


def _sanitize_metrics(d: dict) -> dict:
    """Ensure all numeric values in an eye-metrics dict are JSON-safe floats."""
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, np.floating, np.integer)):
            out[k] = _safe_float(v)
        else:
            out[k] = v
    return out


async def receive_loop(ws):
    global stress_score, stress_state, _prev_stress_state
    while True:
        try:
            raw  = await ws.recv()
            msg  = json.loads(raw)
            if msg.get("type") == "stress_score":
                stress_score = msg.get("value", 0.0)
                stress_state = msg.get("state", "normal")

                # Play a random alert only on the transition INTO "stressed",
                # not on every update while already stressed.
                if stress_state == "stressed" and _prev_stress_state != "stressed":
                    _play_random_alert()
                    print("[camera_client] stress alert triggered")

                _prev_stress_state = stress_state

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[camera_client] receive_loop: connection closed ({e}), stopping receiver")
            break
        except Exception as e:
            print(f"[camera_client] receive_loop: unexpected error ({e}), stopping receiver")
            break


async def camera_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX, CAP_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    print(f"[camera_client] opened camera {CAMERA_INDEX} ({CAM_WIDTH}x{CAM_HEIGHT})")

    try:
        await _camera_loop_inner(cap)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[camera_client] shut down cleanly")


async def _camera_loop_inner(cap):
    global servo_filtered, last_servo_send, last_valid_eye_metrics

    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,       # required for iris landmarks
        min_detection_confidence=MIN_FACE_CONF,
        min_tracking_confidence=MIN_TRACK_CONF,
    )

    win       = WindowStats()
    last_send = 0.0

    was_closed  = False
    blink_armed = False

    ear_baseline_samples = []
    ear_baseline_start   = None
    ear_thresh           = EAR_FALLBACK_THRESH

    # Iris baseline: rolling buffer of recent iris ratios used to detect
    # relative dilation. Pupil dilation is meaningful only as a delta
    # from a personal baseline, not as an absolute value.
    iris_baseline_buf = deque(maxlen=150)   # ~5s at 30fps
    iris_ratio_smooth = None
    IRIS_ALPHA        = 0.1                 # slow-moving baseline

    prev_nose_xy = None

    rppg = RPPGEstimator(
        window_sec=RPPG_WINDOW_SEC,
        assumed_fps=RPPG_ASSUMED_FPS,
        hr_min_bpm=RPPG_HR_MIN_BPM,
        hr_max_bpm=RPPG_HR_MAX_BPM,
        min_window_sec=RPPG_MIN_WINDOW_SEC,
    ) if RPPG_ENABLED else None
    rppg_face_absent_since: float | None = None

    while True:  # reconnect loop
        try:
            async with websockets.connect(WS_INGEST_URL) as ws, \
                       websockets.connect(WS_SUBSCRIBE_URL) as ws_sub:
                asyncio.create_task(receive_loop(ws_sub))

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        await asyncio.sleep(0.01)
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w  = frame.shape[:2]
                    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    res = mesh.process(rgb)
                    now = time.time()

                    blink_event   = 0
                    ear           = None
                    is_closed     = False
                    motion_norm   = 0.0
                    face_detected = False
                    pupil_delta   = 0.0   # normalised deviation from personal iris baseline

                    if res.multi_face_landmarks:
                        face_detected = True
                        face = res.multi_face_landmarks[0]
                        lms  = face.landmark

                        ear_l = ear_from_eye(lms, L_EYE, w, h)
                        ear_r = ear_from_eye(lms, R_EYE, w, h)
                        ear   = (ear_l + ear_r) / 2.0

                        if ear_baseline_start is None:
                            ear_baseline_start = now
                        if (now - ear_baseline_start) <= EAR_BASELINE_SEC:
                            ear_baseline_samples.append(ear)
                        if (now - ear_baseline_start) > EAR_BASELINE_SEC and len(ear_baseline_samples) >= 10:
                            open_med   = float(np.median(ear_baseline_samples))
                            ear_thresh = max(0.10, EAR_THRESH_FRAC * open_med)

                        is_closed = ear < ear_thresh

                        # --- Iris / pupil delta ---
                        # Compute a scale-invariant iris size ratio for each eye, average them,
                        # then compare against a slow-updating personal baseline to get a delta.
                        # Positive delta = dilation (associated with stress/arousal).
                        try:
                            iris_l = iris_diameter_px(lms, LEFT_IRIS_IDX,  LEFT_EYE_OUTER,  LEFT_EYE_INNER,  w, h)
                            iris_r = iris_diameter_px(lms, RIGHT_IRIS_IDX, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, w, h)
                            iris_now = (iris_l + iris_r) / 2.0

                            iris_baseline_buf.append(iris_now)
                            baseline = float(np.median(iris_baseline_buf))

                            if iris_ratio_smooth is None:
                                iris_ratio_smooth = iris_now
                            else:
                                iris_ratio_smooth = IRIS_ALPHA * iris_now + (1 - IRIS_ALPHA) * iris_ratio_smooth

                            # Delta: how much larger/smaller the iris is vs personal baseline.
                            # Clamp to [-1, 1] for stable downstream use.
                            pupil_delta = float(np.clip((iris_ratio_smooth - baseline) / (baseline + 1e-6), -1.0, 1.0))
                        except Exception:
                            pupil_delta = 0.0

                        # --- Nose / head tracking ---
                        nose_xy = _pt(lms, NOSE_TIP_IDX, w, h)
                        cv2.circle(frame, (int(nose_xy[0]), int(nose_xy[1])), 4, (0, 255, 0), -1)

                        frame_center_x = w / 2
                        face_error = (nose_xy[0] - frame_center_x) / frame_center_x

                        if abs(face_error) < SERVO_DEAD_ZONE:
                            face_error = 0.0

                        target_speed   = SERVO_GAIN * face_error
                        servo_filtered = SERVO_ALPHA * target_speed + (1 - SERVO_ALPHA) * servo_filtered

                        if face_error != 0.0 and (now - last_servo_send) > (1.0 / SERVO_SEND_HZ):
                            send_servo_command(servo_filtered)
                            last_servo_send = now

                        l_outer    = _pt(lms, LEFT_EYE_OUTER, w, h)
                        r_outer    = _pt(lms, RIGHT_EYE_OUTER, w, h)
                        face_scale = float(np.linalg.norm(l_outer - r_outer) + 1e-6)

                        if prev_nose_xy is not None:
                            motion_norm = float(np.linalg.norm(nose_xy - prev_nose_xy) / face_scale)
                        prev_nose_xy = nose_xy

                        # --- Blink detection ---
                        if not was_closed and is_closed:
                            blink_armed = True
                        elif was_closed and not is_closed and blink_armed:
                            blink_event = 1
                            blink_armed = False
                        was_closed = is_closed

                        win.add(now, is_closed, blink_event, motion_norm)

                        # --- rPPG heart rate ---
                        if rppg is not None:
                            rppg.push_frame(frame, lms, w, h, now)
                            rppg_face_absent_since = None

                    else:
                        blink_armed  = False
                        was_closed   = False
                        prev_nose_xy = None
                        cv2.putText(frame, "Face not detected", (10, 145),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        if rppg is not None:
                            if rppg_face_absent_since is None:
                                rppg_face_absent_since = now
                            elif now - rppg_face_absent_since >= RPPG_RESET_ABSENT_SEC:
                                rppg.reset()
                                rppg_face_absent_since = None

                    win.prune(now, EYE_WINDOW_SEC)

                    blink_rate = win.blink_rate_per_min()
                    perclos    = win.perclos()
                    head_var   = win.head_motion_var()

                    # low_blink_rate: 1.0 when blink_rate == 0, falls linearly to 0.0 at
                    # LOW_BLINK_THRESH. The outer condition already guarantees the value
                    # is in [0, 1), so np.clip is not needed but kept for safety.
                    low_blink_rate = 1.0 - (blink_rate / LOW_BLINK_THRESH) if blink_rate < LOW_BLINK_THRESH else 0.0

                    current_eye_metrics = {
                        "blink_rate_per_min": blink_rate,
                        "low_blink_rate":     low_blink_rate,
                        "perclos":            perclos,
                        "head_motion_var":    head_var,
                        "pupil_delta":        pupil_delta,
                        "ear":                ear,
                        "ear_thresh":         ear_thresh,
                        "face_detected":      face_detected,
                        "window_sec":         EYE_WINDOW_SEC,
                        "camera_index":       CAMERA_INDEX,
                        "hr_bpm":             rppg.hr_bpm        if rppg else None,
                        "hr_quality":         rppg.signal_quality if rppg else 0.0,
                    }

                    if face_detected:
                        last_valid_eye_metrics = current_eye_metrics

                    send_metrics = last_valid_eye_metrics if last_valid_eye_metrics else current_eye_metrics

                    color = stress_color(stress_state)

                    # Draw a semi-transparent dark background panel behind the HUD text
                    # so it's readable against any background colour.
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (160, 165), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

                    cv2.putText(frame, "STRESS",             (20, 35),  cv2.FONT_HERSHEY_DUPLEX, 0.5, (220, 220, 220), 1)
                    cv2.putText(frame, f"{stress_score:.0f}", (20, 78),  cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
                    cv2.putText(frame, stress_state.upper(),  (20, 112), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)

                    hr_bpm_display = send_metrics.get("hr_bpm")
                    if hr_bpm_display:
                        cv2.putText(frame, "HR",                    (20, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (200, 200, 200), 1)
                        cv2.putText(frame, f"{hr_bpm_display:.0f} bpm", (20, 160), cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 1)
                    else:
                        cv2.putText(frame, "HR: --",                (20, 155), cv2.FONT_HERSHEY_DUPLEX, 0.5, (120, 120, 120), 1)
                    cv2.imshow("camera_client", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        return

                    if now - last_send >= (1.0 / EYE_SEND_HZ):
                        payload = {
                            "type":      "eye_metrics",
                            "data":      _sanitize_metrics(send_metrics),
                            "timestamp": now,
                        }
                        try:
                            await ws.send(json.dumps(payload))
                        except Exception as e:
                            print(f"[camera_client] send error: {e}")
                        last_send = now

                    await asyncio.sleep(0)

        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError) as e:
            print(f"[camera_client] connection lost ({e}), retrying in {RETRY_DELAY}s...")
            await asyncio.sleep(RETRY_DELAY)


if __name__ == "__main__":
    # SIGTERM (e.g. from a process manager) is treated like Ctrl+C so the
    # finally block in camera_loop() always runs and releases the camera.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        asyncio.run(camera_loop())
    except KeyboardInterrupt:
        pass