"""
Remote photoplethysmography (rPPG) — camera-based heart rate estimation.

Algorithm: POS (Plane-Orthogonal-to-Skin) — Wang et al., IEEE TBME 2017
  1. Collect per-frame mean RGB from forehead and cheek skin ROIs.
  2. Temporally normalise each channel by its windowed mean.
  3. Project onto the plane orthogonal to the skin-tone vector.
  4. Bandpass filter (0.75–3.0 Hz ≈ 45–180 bpm).
  5. FFT peak → BPM; spectral peak-to-band ratio → signal quality.

Signal quality is the fraction of in-band spectral power at the dominant peak.
Values above ~0.25 are generally reliable in a typical indoor environment.
scipy is used for the bandpass filter when available (strongly recommended);
otherwise a brick-wall FFT filter is used as a fallback.
"""

from collections import deque

import cv2
import numpy as np

try:
    from scipy.signal import butter, filtfilt
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Skin ROI landmark indices (MediaPipe 468-point face mesh) ─────────────────
# Three regions are averaged to reduce shot noise.  Selected to maximise skin
# area while excluding high-motion zones (eyes, mouth, hairline).

_FOREHEAD    = [10, 338, 297, 332, 284, 251, 389, 356,
                70,  63, 105,  66, 107,  55,   8]
_LEFT_CHEEK  = [234,  93, 132,  58, 172, 136, 150, 149, 176, 148, 152]
_RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
_ROI_SETS    = [_FOREHEAD, _LEFT_CHEEK, _RIGHT_CHEEK]

_MIN_ROI_PX = 30   # discard regions smaller than this many pixels


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _roi_mean_rgb(frame_bgr: np.ndarray, lms,
                  idx_list: list, w: int, h: int) -> np.ndarray | None:
    """
    Return the mean [R, G, B] of a face region defined by a landmark polygon,
    or None if the region is too small or falls outside the frame.
    """
    pts = np.array(
        [[int(lms[i].x * w), int(lms[i].y * h)] for i in idx_list],
        dtype=np.int32,
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    if cv2.countNonZero(mask) < _MIN_ROI_PX:
        return None

    b, g, r, _ = cv2.mean(frame_bgr, mask=mask)
    return np.array([r, g, b], dtype=np.float64)


def _bandpass(signal: np.ndarray, fps: float,
              low_hz: float, high_hz: float) -> np.ndarray:
    """
    Bandpass filter in [low_hz, high_hz].
    Uses a 3rd-order Butterworth (scipy) when available, otherwise an FFT
    brick-wall filter.
    """
    if _SCIPY:
        nyq = fps / 2.0
        lo  = max(low_hz  / nyq, 1e-4)
        hi  = min(high_hz / nyq, 1.0 - 1e-4)
        b, a = butter(3, [lo, hi], btype="band")
        return filtfilt(b, a, signal)

    fft   = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    fft[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    return np.fft.irfft(fft, n=len(signal))


# ── Main estimator ────────────────────────────────────────────────────────────

class RPPGEstimator:
    """
    Rolling-window heart rate estimator using POS rPPG.

    Call ``push_frame`` on every frame where a face is detected.
    ``hr_bpm`` and ``signal_quality`` are updated internally after each
    successful computation.  Call ``reset`` when the face has been absent
    long enough to make the buffer stale.
    """

    def __init__(
        self,
        window_sec:     float = 10.0,
        assumed_fps:    float = 25.0,
        hr_min_bpm:     float = 45.0,
        hr_max_bpm:     float = 180.0,
        min_window_sec: float = 5.0,
    ):
        self.window_sec     = window_sec
        self.assumed_fps    = assumed_fps
        self.hr_min_bpm     = hr_min_bpm
        self.hr_max_bpm     = hr_max_bpm
        self.min_window_sec = min_window_sec

        buf_len          = int(window_sec * assumed_fps)
        self._rgb_buf    = deque(maxlen=buf_len)
        self._ts_buf     = deque(maxlen=buf_len)

        self.hr_bpm:        float | None = None
        self.signal_quality: float       = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def push_frame(self, frame_bgr: np.ndarray, lms,
                   w: int, h: int, timestamp: float) -> None:
        """
        Ingest one frame.  Extracts skin ROI means, appends them to the
        buffer, and recomputes the HR estimate when enough data is available.
        """
        means = [_roi_mean_rgb(frame_bgr, lms, roi, w, h) for roi in _ROI_SETS]
        valid = [m for m in means if m is not None]
        if not valid:
            return

        self._rgb_buf.append(np.mean(valid, axis=0))
        self._ts_buf.append(timestamp)

        fps         = self._estimate_fps()
        min_samples = int(fps * self.min_window_sec)
        if len(self._rgb_buf) >= min_samples:
            self._compute_hr(fps)

    def reset(self) -> None:
        """Clear the buffer.  Call when the face has been absent too long."""
        self._rgb_buf.clear()
        self._ts_buf.clear()
        self.hr_bpm        = None
        self.signal_quality = 0.0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _estimate_fps(self) -> float:
        """Derive fps from recent frame timestamps; fall back to assumed_fps."""
        if len(self._ts_buf) < 10:
            return self.assumed_fps
        ts  = np.array(list(self._ts_buf)[-30:])
        idt = np.diff(ts)
        idt = idt[idt > 1e-4]
        if len(idt) == 0:
            return self.assumed_fps
        return float(1.0 / np.median(idt))

    def _compute_hr(self, fps: float) -> None:
        """Run POS → bandpass → FFT and update hr_bpm / signal_quality."""
        rgb = np.array(self._rgb_buf, dtype=np.float64)   # (N, 3)

        # Temporal normalisation makes the algorithm illumination-invariant.
        ch_mean = rgb.mean(axis=0)
        if np.any(ch_mean < 1.0):
            return
        Cn = rgb / ch_mean   # each channel ÷ its temporal mean

        # POS projection: [[0, 1, -1], [-2, 1, 1]] · [R_n, G_n, B_n]
        S1 = Cn[:, 1] - Cn[:, 2]                       # G_n - B_n
        S2 = -2.0 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]    # -2R_n + G_n + B_n

        std_s2 = float(np.std(S2))
        if std_s2 < 1e-9:
            return
        pulse = S1 + (float(np.std(S1)) / std_s2) * S2

        # Bandpass
        low_hz  = self.hr_min_bpm / 60.0
        high_hz = min(self.hr_max_bpm / 60.0, fps / 2.0 * 0.95)
        if high_hz <= low_hz:
            return
        try:
            pulse_f = _bandpass(pulse, fps, low_hz, high_hz)
        except Exception:
            return

        # FFT with Hamming window to reduce spectral leakage
        fft   = np.fft.rfft(pulse_f * np.hamming(len(pulse_f)))
        freqs = np.fft.rfftfreq(len(pulse_f), d=1.0 / fps)
        power = np.abs(fft) ** 2

        band = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(band):
            return

        band_power = power[band]
        total      = float(band_power.sum())
        if total < 1e-12:
            return

        peak_idx            = int(np.argmax(band_power))
        self.signal_quality = float(band_power[peak_idx] / total)
        self.hr_bpm         = float(freqs[band][peak_idx] * 60.0)
