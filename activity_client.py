import time
import threading
import numpy as np
from pynput import keyboard, mouse

# =========================
# GLOBAL STATE
# =========================
keyboard_events = []  # {time, backspace, hold_duration, key}
mouse_events    = []  # {time, x, y, click, dwell_before_click}
scroll_events   = []  # {time, dx, dy}
_lock           = threading.Lock()

_key_press_times = {}

_last_mouse_pos  = None
_last_mouse_time = None
_pre_click_dwell = 0.0

WINDOW_SIZE      = 30
SLIDE_INTERVAL   = 5
IDLE_THRESHOLD   = 3.0
_prev_window_last_key_time = None
_latest_keyboard_score     = None  # raw score, no smoothing — fusion_server handles that

_score_thread_running = False
_score_thread         = None

FEATURE_COLS = [
    "avg_inter_key_delay", "typing_speed", "backspace_ratio",
    "long_pauses", "avg_hold_duration", "hold_duration_std",
    "bigram_timing_mean", "bigram_timing_std",
    "avg_velocity", "velocity_std", "path_efficiency",
    "avg_dwell_before_click", "click_rate",
    "scroll_reversal_rate", "idle_ratio",
]

# =========================
# KEYBOARD LISTENER
# =========================
_KEY_PRESS_STALE_SEC = 5.0  # discard press records older than this (handles stuck/unreleased keys)

def on_key_press(key):
    t = time.time()
    with _lock:
        is_backspace = (key == keyboard.Key.backspace)
        try:
            key_str = key.char
        except AttributeError:
            key_str = str(key)

        # Prune stale press records to prevent unbounded growth from keys
        # that were never released (e.g. modifier keys, crashes, focus loss).
        stale_keys = [k for k, ts in _key_press_times.items() if t - ts > _KEY_PRESS_STALE_SEC]
        for k in stale_keys:
            del _key_press_times[k]

        _key_press_times[key_str] = t
        keyboard_events.append({
            "time":          t,
            "backspace":     is_backspace,
            "key":           key_str,
            "hold_duration": None,
        })

def on_key_release(key):
    t = time.time()
    with _lock:
        try:
            key_str = key.char
        except AttributeError:
            key_str = str(key)
        press_time = _key_press_times.pop(key_str, None)
        if press_time is not None:
            hold = t - press_time
            for ev in reversed(keyboard_events):
                if ev["key"] == key_str and ev["hold_duration"] is None:
                    ev["hold_duration"] = hold
                    break

# =========================
# MOUSE LISTENER
# =========================
def on_move(x, y):
    global _last_mouse_pos, _last_mouse_time, _pre_click_dwell
    t = time.time()
    with _lock:
        if _last_mouse_pos is not None:
            dx = x - _last_mouse_pos[0]
            dy = y - _last_mouse_pos[1]
            dt = t - _last_mouse_time
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 5:
                _pre_click_dwell += dt
            else:
                _pre_click_dwell = 0.0
        _last_mouse_pos  = (x, y)
        _last_mouse_time = t
        mouse_events.append({"time": t, "x": x, "y": y, "click": False})

def on_click(x, y, button, pressed):
    if pressed:
        global _pre_click_dwell
        t = time.time()
        with _lock:
            dwell = _pre_click_dwell
            _pre_click_dwell = 0.0
            mouse_events.append({
                "time":               t,
                "x":                  x,
                "y":                  y,
                "click":              True,
                "button":             str(button),
                "dwell_before_click": dwell,
            })

def on_scroll(_x, _y, dx, dy):
    with _lock:
        scroll_events.append({"time": time.time(), "dx": dx, "dy": dy})

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(kb_snapshot, ms_snapshot, sc_snapshot, prev_last_key_time=None):
    """
    Extracts a 1x15 feature array from raw event snapshots.
    Returns None if fewer than 2 keyboard events — treat as 'no data', not low load.
    """
    if len(kb_snapshot) < 2:
        return None

    key_times = np.array([e["time"] for e in kb_snapshot])

    if prev_last_key_time is not None:
        gap = key_times[0] - prev_last_key_time
        if 0 < gap < WINDOW_SIZE * 2:
            key_times = np.concatenate([[prev_last_key_time], key_times])

    inter_key_delays    = np.diff(key_times)
    avg_inter_key_delay = float(np.mean(inter_key_delays))
    typing_speed        = len(kb_snapshot) / WINDOW_SIZE
    backspaces          = sum(1 for e in kb_snapshot if e["backspace"])
    backspace_ratio     = backspaces / len(kb_snapshot)
    long_pauses         = int(np.sum(inter_key_delays > 2.0))

    holds = [e["hold_duration"] for e in kb_snapshot if e["hold_duration"] is not None]
    if holds:
        avg_hold_duration = float(np.mean(holds))
        hold_duration_std = float(np.std(holds, ddof=1)) if len(holds) > 1 else 0.0
    else:
        avg_hold_duration = 0.0
        hold_duration_std = 0.0

    bigram_delays = []
    for i in range(1, len(kb_snapshot)):
        dt = kb_snapshot[i]["time"] - kb_snapshot[i - 1]["time"]
        if 0 < dt < 2.0:
            bigram_delays.append(dt)
    if bigram_delays:
        bigram_timing_mean = float(np.mean(bigram_delays))
        bigram_timing_std  = float(np.std(bigram_delays, ddof=1)) if len(bigram_delays) > 1 else 0.0
    else:
        bigram_timing_mean = avg_inter_key_delay
        bigram_timing_std  = 0.0

    all_times = sorted(
        [e["time"] for e in kb_snapshot] + [e["time"] for e in ms_snapshot]
    )
    if len(all_times) >= 2:
        all_gaps  = np.diff(np.array(all_times))
        idle_time = float(np.sum(all_gaps[all_gaps > IDLE_THRESHOLD]))
    else:
        idle_time = float(np.sum(inter_key_delays[inter_key_delays > IDLE_THRESHOLD]))
    idle_ratio = min(1.0, idle_time / WINDOW_SIZE)

    move_events = [e for e in ms_snapshot if not e["click"]]
    velocities  = []
    for i in range(1, len(move_events)):
        dx   = move_events[i]["x"] - move_events[i - 1]["x"]
        dy   = move_events[i]["y"] - move_events[i - 1]["y"]
        dt   = move_events[i]["time"] - move_events[i - 1]["time"]
        dist = np.sqrt(dx**2 + dy**2)
        if dt > 0:
            velocities.append(dist / dt)

    avg_velocity = float(np.mean(velocities))        if velocities          else 0.0
    velocity_std = float(np.std(velocities, ddof=1)) if len(velocities) > 1 else 0.0

    if len(move_events) >= 2:
        xs = np.array([e["x"] for e in move_events])
        ys = np.array([e["y"] for e in move_events])
        straight_line   = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        actual_path     = float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))
        path_efficiency = float(straight_line / actual_path) if actual_path > 0 else 1.0
    else:
        path_efficiency = 1.0

    click_events           = [e for e in ms_snapshot if e["click"]]
    dwells                 = [e.get("dwell_before_click", 0.0) for e in click_events]
    avg_dwell_before_click = float(np.mean(dwells)) if dwells else 0.0
    click_rate             = len(click_events) / WINDOW_SIZE

    if len(sc_snapshot) >= 2:
        dy_vals   = np.array([e["dy"] for e in sc_snapshot])
        signs     = np.sign(dy_vals[dy_vals != 0])
        reversals = int(np.sum(np.diff(signs) != 0)) if len(signs) > 1 else 0
        scroll_reversal_rate = reversals / WINDOW_SIZE
    else:
        scroll_reversal_rate = 0.0

    return np.array([
        avg_inter_key_delay,    # 0
        typing_speed,           # 1
        backspace_ratio,        # 2
        long_pauses,            # 3
        avg_hold_duration,      # 4
        hold_duration_std,      # 5
        bigram_timing_mean,     # 6
        bigram_timing_std,      # 7
        avg_velocity,           # 8
        velocity_std,           # 9
        path_efficiency,        # 10
        avg_dwell_before_click, # 11
        click_rate,             # 12
        scroll_reversal_rate,   # 13
        idle_ratio,             # 14
    ]).reshape(1, -1)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_load(features):
    """
    Returns a 0-1 cognitive load score.
      0 = low load  (fast, fluent, consistent input)
      1 = high load (hesitant, erratic, error-prone input)

    Output is NOT smoothed here — smoothing is handled centrally in fusion_server
    so that the keyboard signal isn't double-smoothed.
    """
    f = features[0]
    (avg_inter_key_delay, typing_speed, backspace_ratio, long_pauses,
     avg_hold_duration, hold_duration_std,
     bigram_timing_mean, bigram_timing_std,
     avg_velocity, velocity_std, path_efficiency,
     avg_dwell_before_click, click_rate,
     scroll_reversal_rate, idle_ratio) = f

    f_backspace   = min(1.0, backspace_ratio * 5)
    f_pauses      = min(1.0, long_pauses / 5)
    f_idle        = min(1.0, idle_ratio)
    f_mouse_std   = min(1.0, velocity_std / 400)
    f_mouse_avg   = min(1.0, avg_velocity / 800)
    f_clicks      = min(1.0, click_rate * 10)
    f_delay       = min(1.0, avg_inter_key_delay * 3)
    f_typing      = min(1.0, typing_speed / 8)       # subtracted: high speed = low load
    f_hold        = min(1.0, avg_hold_duration / 0.3)
    f_hold_std    = min(1.0, hold_duration_std / 0.15)
    f_bigram_mean = min(1.0, bigram_timing_mean / 0.5)  # subtracted: fast bigrams = low load
    f_bigram_std  = min(1.0, bigram_timing_std / 0.3)
    f_path_eff    = 1.0 - min(1.0, path_efficiency)  # inverted: straight path = low load
    f_dwell       = min(1.0, avg_dwell_before_click / 3.0)
    f_scroll_rev  = min(1.0, scroll_reversal_rate * 20)

    WEIGHT_BACKSPACE   = 10
    WEIGHT_PAUSES      = 10
    WEIGHT_IDLE        =  8
    WEIGHT_MOUSE_STD   =  6
    WEIGHT_TYPING      =  8  # subtracted
    WEIGHT_DELAY       =  5
    WEIGHT_BIGRAM_MEAN =  5  # subtracted
    WEIGHT_BIGRAM_STD  =  5
    WEIGHT_PATH_EFF    =  5
    WEIGHT_SCROLL_REV  =  5
    WEIGHT_MOUSE_AVG   =  4
    WEIGHT_CLICKS      =  4
    WEIGHT_HOLD        =  5
    WEIGHT_HOLD_STD    =  4
    WEIGHT_DWELL       =  4

    score = (
        WEIGHT_BACKSPACE  * f_backspace  +
        WEIGHT_PAUSES     * f_pauses     +
        WEIGHT_IDLE       * f_idle       +
        WEIGHT_MOUSE_STD  * f_mouse_std  +
        WEIGHT_CLICKS     * f_clicks     +
        WEIGHT_DELAY      * f_delay      +
        WEIGHT_HOLD       * f_hold       +
        WEIGHT_HOLD_STD   * f_hold_std   +
        WEIGHT_BIGRAM_STD * f_bigram_std +
        WEIGHT_PATH_EFF   * f_path_eff   +
        WEIGHT_DWELL      * f_dwell      +
        WEIGHT_SCROLL_REV * f_scroll_rev
        - WEIGHT_TYPING      * f_typing
        - WEIGHT_BIGRAM_MEAN * f_bigram_mean
        - WEIGHT_MOUSE_AVG   * f_mouse_avg  # high avg speed = fluent = low load
    )

    max_positive = (
        WEIGHT_BACKSPACE + WEIGHT_PAUSES + WEIGHT_IDLE +
        WEIGHT_MOUSE_STD + WEIGHT_CLICKS + WEIGHT_DELAY +
        WEIGHT_HOLD + WEIGHT_HOLD_STD + WEIGHT_BIGRAM_STD +
        WEIGHT_PATH_EFF + WEIGHT_DWELL + WEIGHT_SCROLL_REV
    )
    min_possible = -(WEIGHT_TYPING + WEIGHT_BIGRAM_MEAN + WEIGHT_MOUSE_AVG)
    total_range  = max_positive - min_possible

    normalized = (score - min_possible) / total_range
    return max(0.0, min(1.0, normalized))

# =========================
# SNAPSHOT HELPER
# =========================
def _take_snapshot():
    global _prev_window_last_key_time
    cutoff = time.time() - WINDOW_SIZE

    with _lock:
        kb_window = [e for e in keyboard_events if e["time"] >= cutoff]
        ms_window = [e for e in mouse_events    if e["time"] >= cutoff]
        sc_window = [e for e in scroll_events   if e["time"] >= cutoff]

        keyboard_events[:] = kb_window
        mouse_events[:]    = ms_window
        scroll_events[:]   = sc_window

        kb_snapshot = list(kb_window)
        ms_snapshot = list(ms_window)
        sc_snapshot = list(sc_window)

    prev_last = _prev_window_last_key_time
    if kb_snapshot:
        _prev_window_last_key_time = kb_snapshot[-1]["time"]

    return kb_snapshot, ms_snapshot, sc_snapshot, prev_last

# =========================
# BACKGROUND SCORE UPDATER
# =========================
def _score_updater_loop():
    global _latest_keyboard_score, _score_thread_running

    while _score_thread_running:
        time.sleep(SLIDE_INTERVAL)

        kb_snap, ms_snap, sc_snap, prev_last = _take_snapshot()
        features = extract_features(kb_snap, ms_snap, sc_snap, prev_last_key_time=prev_last)

        if features is None:
            continue

        # Raw score — no smoothing applied here. fusion_server smooths centrally.
        _latest_keyboard_score = evaluate_load(features)

# =========================
# GET CURRENT SCORE (external API)
# =========================
def get_keyboard_load_score():
    """
    Returns the current raw cognitive load score in [0, 1], or None if no data yet.
    Smoothing is intentionally omitted here and handled by fusion_server to avoid
    double-smoothing the keyboard signal.
    """
    return _latest_keyboard_score

# =========================
# START / STOP API
# =========================
_keyboard_listener = None
_mouse_listener    = None

def start_monitoring():
    global _keyboard_listener, _mouse_listener
    global _score_thread_running, _score_thread

    if _keyboard_listener is not None:
        return

    _keyboard_listener = keyboard.Listener(
        on_press=on_key_press,
        on_release=on_key_release
    )
    _mouse_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll
    )

    _keyboard_listener.start()
    _mouse_listener.start()

    _score_thread_running = True
    _score_thread = threading.Thread(target=_score_updater_loop, daemon=True)
    _score_thread.start()

    print("[activity_client] monitoring started")

def stop_monitoring():
    global _keyboard_listener, _mouse_listener, _score_thread_running

    _score_thread_running = False

    if _keyboard_listener:
        _keyboard_listener.stop()
    if _mouse_listener:
        _mouse_listener.stop()

    _keyboard_listener = None
    _mouse_listener    = None

    print("[activity_client] monitoring stopped")