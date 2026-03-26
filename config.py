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
