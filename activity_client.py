import time
import threading
import numpy as np
from pynput import keyboard, mouse
from collections import deque
import os

# =========================
# GLOBAL STATE
# =========================
keyboard_events = []  # {time, backspace, hold_duration, key}
mouse_events    = []  # {time, x, y, click, dwell_before_click}
scroll_events   = []  # {time, dx, dy}
_lock           = threading.Lock()

# Key-hold tracking: key -> press time
_key_press_times = {}

# Mouse position tracking for dwell + trajectory
_last_mouse_pos  = None
_last_mouse_time = None
_pre_click_dwell = 0.0  # seconds cursor was stationary before last click

WINDOW_SIZE      = 30       # seconds per feature window
SLIDE_INTERVAL   = 5        # seconds between feature computations (sliding window)
IDLE_THRESHOLD   = 3.0      # seconds gap considered "idle"
ROLLING_WINDOW   = 120      # seconds of history shown in chart
SMOOTHING_ALPHA  = 0.3      # exponential smoothing factor

_prev_window_last_key_time = None
_smoothed_score            = None

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
def on_key_press(key):
    t = time.time()
    with _lock:
        is_backspace = (key == keyboard.Key.backspace)
        try:
            key_str = key.char
        except AttributeError:
            key_str = str(key)
        _key_press_times[key_str] = t
        keyboard_events.append({
            "time":         t,
            "backspace":    is_backspace,
            "key":          key_str,
            "hold_duration": None,  # filled on release
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
        mouse_events.append({
            "time":  t,
            "x":     x,
            "y":     y,
            "click": False,
        })

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
        scroll_events.append({
            "time": time.time(),
            "dx":   dx,
            "dy":   dy,
        })

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(kb_snapshot, ms_snapshot, sc_snapshot, prev_last_key_time=None):
    """
    Extracts a 1×15 feature array from raw event snapshots.

    Bridges the inter-key delay gap between the previous window and the current
    one using prev_last_key_time, so sliding windows don't lose cross-boundary
    timing information.

    Returns None if there are fewer than 2 keyboard events — callers should
    treat None as 'no data', not as a low-load signal.
    """
    if len(kb_snapshot) < 2:
        return None

    key_times = np.array([e["time"] for e in kb_snapshot])

    # Include the last keypress from the previous window so the first
    # inter-key delay of this window isn't artificially inflated.
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
        avg_hold_duration  = float(np.mean(holds))
        hold_duration_std  = float(np.std(holds, ddof=1)) if len(holds) > 1 else 0.0
    else:
        avg_hold_duration  = 0.0
        hold_duration_std  = 0.0

    # Bigram timing: consecutive keypress pairs, excluding inter-word pauses > 2s.
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

    # Idle ratio: fraction of the window spent in gaps longer than IDLE_THRESHOLD,
    # computed across both keyboard and mouse events combined.
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

    # Path efficiency: ratio of straight-line distance to actual cursor path length.
    # Values near 1 = direct movement; lower values = erratic/curved movement.
    if len(move_events) >= 2:
        xs = np.array([e["x"] for e in move_events])
        ys = np.array([e["y"] for e in move_events])
        straight_line   = np.sqrt((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2)
        actual_path     = float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))
        path_efficiency = float(straight_line / actual_path) if actual_path > 0 else 1.0
    else:
        path_efficiency = 1.0

    click_events = [e for e in ms_snapshot if e["click"]]
    dwells = [e.get("dwell_before_click", 0.0) for e in click_events]
    avg_dwell_before_click = float(np.mean(dwells)) if dwells else 0.0

    click_rate = len(click_events) / WINDOW_SIZE

    # Scroll reversal rate: sign changes in vertical scroll direction per second,
    # indicating back-and-forth searching behaviour.
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
    Returns a 0–1 cognitive load score derived from keyboard and mouse features.
      0 = low load  (fast, fluent, consistent input)
      1 = high load (hesitant, erratic, error-prone input)

    Each feature is clamped to [0, 1] using empirically chosen ceilings
    (e.g. 500ms bigram mean, 400px/s velocity std). The weighted sum is then
    linearly normalized against the theoretical min/max of the formula so the
    output reliably spans the full [0, 1] range.
    """
    f = features[0]
    (avg_inter_key_delay, typing_speed, backspace_ratio, long_pauses,
     avg_hold_duration, hold_duration_std,
     bigram_timing_mean, bigram_timing_std,
     avg_velocity, velocity_std, path_efficiency,
     avg_dwell_before_click, click_rate,
     scroll_reversal_rate, idle_ratio) = f

    # --- Normalize each feature to [0, 1] ---
    # Ceilings encode domain assumptions: tune these if scores feel mis-calibrated.
    f_backspace    = min(1.0, backspace_ratio * 5)        # 20%+ backspace rate = ceiling
    f_pauses       = min(1.0, long_pauses / 5)            # 5+ pauses >2s = ceiling
    f_idle         = min(1.0, idle_ratio)
    f_mouse_std    = min(1.0, velocity_std / 400)         # 400 px/s std = ceiling
    f_mouse_avg    = min(1.0, avg_velocity / 800)         # 800 px/s avg = ceiling
    f_clicks       = min(1.0, click_rate * 10)            # 6 clicks/30s = ceiling
    f_delay        = min(1.0, avg_inter_key_delay * 3)    # ~333ms avg delay = ceiling
    f_typing       = min(1.0, typing_speed / 8)           # 8 keys/s = ceiling; subtracted — high speed = low load
    f_hold         = min(1.0, avg_hold_duration / 0.3)    # 300ms avg hold = ceiling
    f_hold_std     = min(1.0, hold_duration_std / 0.15)   # erratic hold duration = higher load
    f_bigram_mean  = min(1.0, bigram_timing_mean / 0.5)   # slow avg bigram timing → higher load; 500ms = ceiling
    f_bigram_std   = min(1.0, bigram_timing_std / 0.3)    # rhythm variability = higher load; 300ms std = ceiling
    f_path_eff     = 1.0 - min(1.0, path_efficiency)      # invert: path_efficiency near 1 = straight = low load
    f_dwell        = min(1.0, avg_dwell_before_click / 3.0)  # 3s dwell before click = ceiling
    f_scroll_rev   = min(1.0, scroll_reversal_rate * 20)  # many reversals = searching behaviour

    # --- Weights ---
    # Additive terms increase the score (higher value = higher load).
    # WEIGHT_TYPING and WEIGHT_BIGRAM_MEAN are subtracted (higher value = lower load).
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
        WEIGHT_MOUSE_AVG  * f_mouse_avg  +
        WEIGHT_CLICKS     * f_clicks     +
        WEIGHT_DELAY      * f_delay      +
        WEIGHT_HOLD       * f_hold       +
        WEIGHT_HOLD_STD   * f_hold_std   +
        WEIGHT_BIGRAM_STD * f_bigram_std +
        WEIGHT_PATH_EFF   * f_path_eff   +
        WEIGHT_DWELL      * f_dwell      +
        WEIGHT_SCROLL_REV * f_scroll_rev
        - WEIGHT_TYPING     * f_typing
        - WEIGHT_BIGRAM_MEAN * f_bigram_mean
    )

    # Theoretical bounds for the formula above:
    #   max_positive: all additive terms at 1, subtracted terms at 0
    #   min_possible: all additive terms at 0, subtracted terms at 1
    max_positive = (
        WEIGHT_BACKSPACE + WEIGHT_PAUSES + WEIGHT_IDLE +
        WEIGHT_MOUSE_STD + WEIGHT_MOUSE_AVG + WEIGHT_CLICKS + WEIGHT_DELAY +
        WEIGHT_HOLD + WEIGHT_HOLD_STD + WEIGHT_BIGRAM_STD +
        WEIGHT_PATH_EFF + WEIGHT_DWELL + WEIGHT_SCROLL_REV
    )
    min_possible = -(WEIGHT_TYPING + WEIGHT_BIGRAM_MEAN)
    total_range  = max_positive - min_possible

    normalized = (score - min_possible) / total_range
    return max(0.0, min(1.0, normalized))

# =========================
# SNAPSHOT HELPER
# =========================
def _take_snapshot():
    """
    Returns copies of events within the last WINDOW_SIZE seconds and prunes
    older events from the global lists. Uses a sliding window — events are not
    cleared on every call, only events that have aged out are removed.
    """
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
# GET CURRENT SCORE (external API)
# =========================
def get_keyboard_load_score():
    """
    Returns the current smoothed cognitive load score in [0, 1].
    Returns None if there is no keyboard data in the current window —
    callers should not interpret None as low load.
    """
    global _smoothed_score

    kb_snap, ms_snap, sc_snap, prev_last = _take_snapshot()
    features = extract_features(kb_snap, ms_snap, sc_snap, prev_last_key_time=prev_last)

    if features is None:
        return None

    raw_score = evaluate_load(features)

    if _smoothed_score is None:
        _smoothed_score = raw_score
    else:
        _smoothed_score = SMOOTHING_ALPHA * raw_score + (1 - SMOOTHING_ALPHA) * _smoothed_score

    return _smoothed_score

# =========================
# ROLLING MONITOR
# =========================
def rolling_monitor():
    global _smoothed_score
    print("[Rolling Monitor] Real-time cognitive load chart active.\n")

    history = deque(maxlen=ROLLING_WINDOW // SLIDE_INTERVAL)

    while True:
        time.sleep(SLIDE_INTERVAL)

        kb_snap, ms_snap, sc_snap, prev_last = _take_snapshot()
        features = extract_features(kb_snap, ms_snap, sc_snap, prev_last_key_time=prev_last)

        if features is None:
            label = "(no input data)"
            score = _smoothed_score if _smoothed_score is not None else 0.0
        else:
            raw_score = evaluate_load(features)
            if _smoothed_score is None:
                _smoothed_score = raw_score
            else:
                _smoothed_score = SMOOTHING_ALPHA * raw_score + (1 - SMOOTHING_ALPHA) * _smoothed_score
            score = _smoothed_score
            label = ""

        history.append(score)

        os.system("cls" if os.name == "nt" else "clear")
        window_secs = len(history) * SLIDE_INTERVAL
        print(f"Cognitive Load — Last {window_secs}s  (sliding {WINDOW_SIZE}s window, α={SMOOTHING_ALPHA})\n")

        bar_len = 40
        for s in history:
            filled = int(s * bar_len)
            bar    = "█" * filled + "░" * (bar_len - filled)
            print(f"{bar} {s:.0%}")

        if features is not None:
            f = features[0]
            print(f"\n  Backspace ratio  : {f[2]:.1%}   Hold std    : {f[5]*1000:.0f}ms")
            print(f"  Bigram timing std: {f[7]*1000:.0f}ms      Path efficiency: {f[10]:.2f}")
            print(f"  Scroll reversals : {f[13]*WINDOW_SIZE:.0f}   Dwell before click: {f[11]:.2f}s")
        if label:
            print(f"\n  ⓘ  {label}")

        if score > 0.75:
            print("\n⚠  HIGH LOAD — Consider a short break.")
        elif score > 0.50:
            print("\n⚠  Approaching overload.")
        else:
            print("\n✓  Normal load.")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cognitive Load Monitor")
    args = parser.parse_args()

    print("=" * 56)
    print("  Cognitive Load Monitor (Enhanced)")
    print(f"  Window size      : {WINDOW_SIZE}s (sliding, every {SLIDE_INTERVAL}s)")
    print(f"  Idle threshold   : {IDLE_THRESHOLD}s")
    print(f"  Smoothing alpha  : {SMOOTHING_ALPHA}")
    print(f"  Features         : {len(FEATURE_COLS)}")
    print("=" * 56 + "\n")

    keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener    = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
    keyboard_listener.start()
    mouse_listener.start()

    try:
        rolling_monitor()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        keyboard_listener.stop()
        mouse_listener.stop()
