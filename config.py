# config.py — central configuration for all components
#
# Edit this file to change server address, camera device, servo endpoint,
# and tunable behavioural parameters without touching source files.

# ── Server ───────────────────────────────────────────────────────────────────
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000

WS_INGEST_URL    = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/ingest"
WS_SUBSCRIBE_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws/subscribe"

RETRY_DELAY = 3.0   # seconds between reconnection attempts

# ── Camera ───────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0        # change if your webcam is not the default device
CAM_WIDTH    = 640
CAM_HEIGHT   = 480

# ── Eye-tracking ─────────────────────────────────────────────────────────────
EYE_SEND_HZ         = 1.0    # how often (per second) eye metrics are sent
EYE_WINDOW_SEC      = 15.0   # rolling window for blink/perclos statistics
EAR_BASELINE_SEC    = 2.0    # seconds of open-eye data used to calibrate EAR threshold
EAR_THRESH_FRAC     = 0.75   # fraction of baseline EAR that counts as "closed"
EAR_FALLBACK_THRESH = 0.21   # used before calibration completes
LOW_BLINK_THRESH    = 10.0   # blinks/min below which sustained fixation is flagged

# ── ESP32 pan-tilt servo ─────────────────────────────────────────────────────
# Set to None to disable face-tracking servo (e.g. when no ESP32 is connected).
ESP32_SERVO_URL = "http://10.48.126.77/servo"
SERVO_DEAD_ZONE = 0.08
SERVO_GAIN      = 0.6
SERVO_ALPHA     = 0.25
SERVO_SEND_HZ   = 10

# ── Keyboard / activity monitoring ───────────────────────────────────────────
KB_WINDOW_SIZE    = 30   # seconds of events kept in the sliding window
KB_SLIDE_INTERVAL = 5    # seconds between score recalculations
KB_IDLE_THRESHOLD = 3.0  # gap (seconds) between events that counts as idle time

# ── Fusion server ─────────────────────────────────────────────────────────────
FACE_ABSENT_FREEZE_SEC = 10.0   # hold score constant for this long after face disappears
FACE_ABSENT_DECAY_RATE = 1.0    # points/update the score decays toward 50 after freeze

# ── rPPG (Remote Photoplethysmography) ────────────────────────────────────────
RPPG_ENABLED          = True    # set False to disable without touching other code
RPPG_WINDOW_SEC       = 10.0   # seconds of RGB history used for each FFT
RPPG_MIN_WINDOW_SEC   = 5.0    # minimum seconds before an estimate is emitted
RPPG_ASSUMED_FPS      = 25.0   # starting fps estimate; auto-refined from timestamps
RPPG_HR_MIN_BPM       = 45.0   # lower bound of FFT search band
RPPG_HR_MAX_BPM       = 180.0  # upper bound of FFT search band
RPPG_RESET_ABSENT_SEC = 3.0    # seconds of face absence before buffer is cleared

# Fusion parameters for the HR contribution to the stress score
RPPG_HR_BASELINE      = 72.0   # bpm considered "neutral" resting HR
RPPG_HR_RANGE         = 30.0   # bpm deviation that saturates the normalised HR signal
RPPG_HR_WEIGHT        = 10.0   # max ±points the HR signal contributes to the stress score
RPPG_MIN_QUALITY_GATE = 0.25   # signal quality threshold below which HR is ignored
