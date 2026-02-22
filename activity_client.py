import time
import threading
import numpy as np
from pynput import keyboard, mouse
from collections import deque
import os

keyboard_events = []
mouse_events    = []
scroll_events   = []
_lock           = threading.Lock()

_key_press_times = {}

_last_mouse_pos  = None
_last_mouse_time = None
_pre_click_dwell = 0.0

WINDOW_SIZE      = 30
SLIDE_INTERVAL   = 5
IDLE_THRESHOLD   = 3.0
ROLLING_WINDOW   = 120
SMOOTHING_ALPHA  = 0.3

_prev_window_last_key_time = None
_smoothed_score            = None
_latest_keyboard_score     = None

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
            "time": t,
            "backspace": is_backspace,
            "key": key_str,
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
            "time": t,
            "x": x,
            "y": y,
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
                "time": t,
                "x": x,
                "y": y,
                "click": True,
                "button": str(button),
                "dwell_before_click": dwell,
            })

def on_scroll(_x, _y, dx, dy):
    with _lock:
        scroll_events.append({
            "time": time.time(),
            "dx": dx,
            "dy": dy,
        })

def extract_features(kb_snapshot, ms_snapshot, sc_snapshot, prev_last_key_time=None):
    if len(kb_snapshot) < 2:
        return None

    key_times = np.array([e["time"] for e in kb_snapshot])

    if prev_last_key_time is not None:
        gap = key_times[0] - prev_last_key_time
        if 0 < gap < WINDOW_SIZE * 2:
            key_times = np.concatenate([[prev_last_key_time], key_times])

    inter_key_delays = np.diff(key_times)
    avg_inter_key_delay = float(np.mean(inter_key_delays))
    typing_speed = len(kb_snapshot) / WINDOW_SIZE
    backspaces = sum(1 for e in kb_snapshot if e["backspace"])
    backspace_ratio = backspaces / len(kb_snapshot)
    long_pauses = int(np.sum(inter_key_delays > 2.0))

    holds = [e["hold_duration"] for e in kb_snapshot if e["hold_duration"] is not None]
    avg_hold_duration = float(np.mean(holds)) if holds else 0.0
    hold_duration_std = float(np.std(holds, ddof=1)) if len(holds) > 1 else 0.0

    bigram_delays = []
    for i in range(1, len(kb_snapshot)):
        dt = kb_snapshot[i]["time"] - kb_snapshot[i-1]["time"]
        if 0 < dt < 2.0:
            bigram_delays.append(dt)

    if bigram_delays:
        bigram_timing_mean = float(np.mean(bigram_delays))
        bigram_timing_std  = float(np.std(bigram_delays, ddof=1)) if len(bigram_delays) > 1 else 0.0
    else:
        bigram_timing_mean = avg_inter_key_delay
        bigram_timing_std  = 0.0

    idle_ratio = 0.0

    move_events = [e for e in ms_snapshot if not e["click"]]
    velocities  = []
    for i in range(1, len(move_events)):
        dx = move_events[i]["x"] - move_events[i-1]["x"]
        dy = move_events[i]["y"] - move_events[i-1]["y"]
        dt = move_events[i]["time"] - move_events[i-1]["time"]
        dist = np.sqrt(dx**2 + dy**2)
        if dt > 0:
            velocities.append(dist / dt)

    avg_velocity = float(np.mean(velocities)) if velocities else 0.0
    velocity_std = float(np.std(velocities, ddof=1)) if len(velocities) > 1 else 0.0

    path_efficiency = 1.0
    avg_dwell_before_click = 0.0
    click_rate = 0.0
    scroll_reversal_rate = 0.0

    return np.array([
        avg_inter_key_delay,
        typing_speed,
        backspace_ratio,
        long_pauses,
        avg_hold_duration,
        hold_duration_std,
        bigram_timing_mean,
        bigram_timing_std,
        avg_velocity,
        velocity_std,
        path_efficiency,
        avg_dwell_before_click,
        click_rate,
        scroll_reversal_rate,
        idle_ratio,
    ]).reshape(1, -1)

def evaluate_load(features):
    f = features[0]
    typing_speed = f[1]
    backspace_ratio = f[2]
    idle_ratio = f[14]

    score = (
        backspace_ratio * 0.5
        + idle_ratio * 0.3
        + (1 - min(1.0, typing_speed / 8)) * 0.2
    )

    return max(0.0, min(1.0, score))

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

    prev_last = _prev_window_last_key_time
    if kb_window:
        _prev_window_last_key_time = kb_window[-1]["time"]

    return kb_window, ms_window, sc_window, prev_last

def _score_updater_loop():
    global _latest_keyboard_score, _smoothed_score, _score_thread_running

    while _score_thread_running:
        time.sleep(SLIDE_INTERVAL)

        kb, ms, sc, prev_last = _take_snapshot()
        features = extract_features(kb, ms, sc, prev_last)

        if features is None:
            continue

        raw = evaluate_load(features)

        if _smoothed_score is None:
            _smoothed_score = raw
        else:
            _smoothed_score = (
                SMOOTHING_ALPHA * raw
                + (1 - SMOOTHING_ALPHA) * _smoothed_score
            )

        _latest_keyboard_score = _smoothed_score

def get_keyboard_load_score():
    return _latest_keyboard_score

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
    _score_thread = threading.Thread(
        target=_score_updater_loop,
        daemon=True
    )
    _score_thread.start()

    print("[activity_client] monitoring started")

def stop_monitoring():
    global _keyboard_listener, _mouse_listener
    global _score_thread_running

    _score_thread_running = False

    if _keyboard_listener:
        _keyboard_listener.stop()
    if _mouse_listener:
        _mouse_listener.stop()

    _keyboard_listener = None
    _mouse_listener = None

    print("[activity_client] monitoring stopped")

if __name__ == "__main__":
    start_monitoring()
    try:
        while True:
            time.sleep(2)
            print("Keyboard score:", get_keyboard_load_score())
    except KeyboardInterrupt:
        stop_monitoring()