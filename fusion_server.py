import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from config import (
    FACE_ABSENT_DECAY_RATE, FACE_ABSENT_FREEZE_SEC, SERVER_PORT,
    RPPG_HR_BASELINE, RPPG_HR_RANGE, RPPG_HR_WEIGHT, RPPG_MIN_QUALITY_GATE,
)

app = FastAPI()


class FusionState:
    def __init__(self):
        self.eye_metrics     = None
        self.keyboard_load   = None
        self.hr_bpm:     float | None = None
        self.hr_quality: float        = 0.0
        self.smoothed_stress = 50.0  # persistent smoothed output
        self.face_absent_since: float | None = None  # timestamp when face was last lost


state       = FusionState()
subscribers = set()   # display/consumer clients on /ws/subscribe


def stress_label(score: float) -> str:
    if score < 45:
        return "calm"
    elif score < 70:
        return "normal"
    else:
        return "stressed"


def fuse(state: FusionState) -> float:
    """
    Fuse eye and keyboard signals into a single stress score on [0, 100].

    Returns the raw (unsmoothed) instantaneous stress estimate.
    Smoothing is applied by the caller so that fuse() remains a pure function
    with no hidden state mutation.
    """
    eye = state.eye_metrics or {}
    kb  = state.keyboard_load

    # ---- Keyboard contribution: bidirectional, centred on 0 ----
    if kb is None:
        kb_effect = 0.0
    else:
        kb_norm   = float(kb) * 2.0 - 1.0
        kb_effect = kb_norm * 15.0

    # ---- Eye metric contributions ----
    blink_rate     = eye.get("blink_rate_per_min", 0.0) or 0.0
    low_blink_rate = eye.get("low_blink_rate",     0.0) or 0.0
    perclos        = eye.get("perclos",            0.0) or 0.0
    pupil_delta    = eye.get("pupil_delta",        0.0) or 0.0
    face_detected  = eye.get("face_detected",      False)

    eye_effect = 0.0
    blink_high_diff = max(0.0, blink_rate - 40.0)
    eye_effect += (blink_high_diff ** 1.3) * 0.07
    eye_effect += low_blink_rate * 19.0   # strong, reliable cognitive load marker
    eye_effect += perclos * 12.0          # reduced: PERCLOS is fatigue-specific, not cognitive load
    eye_effect += max(0.0, pupil_delta) * 24.0  # gold standard for cognitive load

    # ---- rPPG heart rate contribution ----
    # Gated by signal quality so that noisy/absent HR has zero effect.
    hr_bpm     = state.hr_bpm
    hr_quality = state.hr_quality
    if hr_bpm and hr_quality >= RPPG_MIN_QUALITY_GATE:
        hr_norm   = max(-1.0, min(1.0, (hr_bpm - RPPG_HR_BASELINE) / RPPG_HR_RANGE))
        hr_effect = hr_norm * RPPG_HR_WEIGHT
    else:
        hr_effect = 0.0

    stress = 50.0 + eye_effect + kb_effect + hr_effect
    return float(max(0.0, min(100.0, stress)))


async def broadcast(payload: dict):
    """Push a stress update to all subscribed display/consumer clients."""
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json(payload)
        except Exception as e:
            print(f"[fusion_server] broadcast: dropping subscriber after error: {e}")
            dead.append(ws)
    for ws in dead:
        subscribers.discard(ws)


# FACE_ABSENT_FREEZE_SEC and FACE_ABSENT_DECAY_RATE are loaded from config.py


def _process_message(msg: dict):
    """
    Update fusion state from an inbound sensor message.
    Returns a broadcast payload dict if state was updated, else None.

    Face-absent policy
    ------------------
    When the face is not detected:
      - For the first FACE_ABSENT_FREEZE_SEC seconds: freeze the score.
        The last measured value is the best estimate — don't inject noise.
      - After that: decay slowly toward neutral (50) at FACE_ABSENT_DECAY_RATE
        points/second. A long absence likely means the session has been
        interrupted, so neutral is a more honest default than holding a stale peak.
    When the face returns: resume normal fusion immediately.
    """
    msg_type = msg.get("type")

    if msg_type == "eye_metrics":
        data              = msg.get("value") or msg.get("data") or {}
        state.eye_metrics = data
        if isinstance(data, dict):
            state.hr_bpm     = data.get("hr_bpm") or None
            state.hr_quality = float(data.get("hr_quality") or 0.0)

    elif msg_type == "keyboard_load":
        val = msg.get("value")
        if val is not None:
            try:
                state.keyboard_load = float(val)
            except (TypeError, ValueError):
                return None
    else:
        return None  # unknown message type — ignore without crashing

    now          = time.monotonic()
    eye          = state.eye_metrics or {}
    face_detected = eye.get("face_detected", False)

    if face_detected:
        # Face is present — clear the absent timer and run normal fusion.
        state.face_absent_since = None

        raw_stress            = fuse(state)
        alpha                 = 0.25
        state.smoothed_stress = alpha * raw_stress + (1 - alpha) * state.smoothed_stress

    else:
        # Face is absent — start or continue the absence timer.
        if state.face_absent_since is None:
            state.face_absent_since = now

        absent_for = now - state.face_absent_since

        if absent_for < FACE_ABSENT_FREEZE_SEC:
            # Within freeze window: hold the score exactly where it is.
            pass
        else:
            # Beyond freeze window: decay toward neutral at a fixed rate.
            # We apply the decay in proportion to one update cycle (~1s at SEND_HZ=1).
            # Using a fixed step rather than alpha-smooth so the rate is predictable.
            decay_step = FACE_ABSENT_DECAY_RATE  # points per update cycle
            current    = state.smoothed_stress
            if current > 50.0:
                state.smoothed_stress = max(50.0, current - decay_step)
            elif current < 50.0:
                state.smoothed_stress = min(50.0, current + decay_step)
            # If exactly 50, nothing to do.

    stress_score = state.smoothed_stress
    label        = stress_label(stress_score)

    eye                = state.eye_metrics or {}
    absent_for_display = (now - state.face_absent_since) if state.face_absent_since else 0.0

    print("\n--- UPDATE ---")
    print("source        :", msg_type)
    print("face_detected :", face_detected,
          f"| absent for {absent_for_display:.1f}s" if not face_detected else "")
    print("keyboard_load :", round(state.keyboard_load, 3) if state.keyboard_load is not None else None)
    print("eye ► blink   :", round(eye.get("blink_rate_per_min", 0.0), 1), "blinks/min",
          "| low_blink:", round(eye.get("low_blink_rate", 0.0), 2))
    print("eye ► perclos :", round(eye.get("perclos", 0.0), 3))
    print("eye ► pupil Δ :", round(eye.get("pupil_delta", 0.0), 3))
    print("eye ► EAR     :", round(eye.get("ear", 0.0) or 0.0, 3),
          "| thresh:", round(eye.get("ear_thresh", 0.0) or 0.0, 3))
    print("rPPG ► HR     :", round(state.hr_bpm or 0.0, 1), "bpm",
          "| quality:", round(state.hr_quality or 0.0, 3))
    print("stress_score  :", round(stress_score, 1), "| state:", label)

    return {"type": "stress_score", "value": stress_score, "state": label}


@app.websocket("/ws/ingest")
async def ingest_endpoint(ws: WebSocket):
    """
    Send-only endpoint for sensor clients (camera_client, run.py).

    The server never pushes stress scores back down this connection, so there
    is no receive-buffer back-pressure and no cross-client interference.

    Previously all clients shared /ws and the server broadcast stress scores
    back to run.py, which had no receive loop to drain them. This caused
    buffer build-up and triggered 'no close frame' disconnects whenever the
    camera hiccupped and sent a burst of rapid messages.
    """
    await ws.accept()
    print("[fusion_server] ingest client connected")

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                print("[fusion_server] bad JSON on ingest, skipping")
                continue

            payload = _process_message(msg)
            if payload:
                await broadcast(payload)

    except WebSocketDisconnect:
        print("[fusion_server] ingest client disconnected cleanly")
    except Exception as e:
        print(f"[fusion_server] ingest client error: {e}")


@app.websocket("/ws/subscribe")
async def subscribe_endpoint(ws: WebSocket):
    """
    Receive-only endpoint for display/consumer clients that want stress score
    updates but never send sensor data (e.g. a dashboard UI).
    """
    await ws.accept()
    subscribers.add(ws)
    print(f"[fusion_server] subscriber connected ({len(subscribers)} total)")

    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        print("[fusion_server] subscriber disconnected cleanly")
    except Exception as e:
        print(f"[fusion_server] subscriber error: {e}")
    finally:
        subscribers.discard(ws)


if __name__ == "__main__":
    uvicorn.run("fusion_server:app", host="0.0.0.0", port=SERVER_PORT, reload=True)