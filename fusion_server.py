from fastapi import FastAPI, WebSocket
import uvicorn
import math

app = FastAPI()

# ============================================================
# GLOBAL STATE
# ============================================================

state = {
    "eye_metrics": None,
    "keyboard_load": None,
}


# ============================================================
# HELPERS
# ============================================================

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _clamp01(x):
    return _clamp(x, 0.0, 1.0)


# ============================================================
# FUSION LOGIC
# ============================================================

def compute_eye_score(eye):
    if not eye:
        return 35.0

    blink = eye.get("blink_rate_per_min", 0) or 0
    perclos = eye.get("perclos", 0) or 0
    motion = eye.get("head_motion_var", 0) or 0
    pupil_delta = eye.get("pupil_delta", 0) or 0

    # Normalize signals (tuned for your ranges)
    blink_n = _clamp01(blink / 120.0)
    perclos_n = _clamp01(perclos / 0.5)
    motion_n = _clamp01(motion / 0.002)
    pupil_n = _clamp01(abs(pupil_delta) / 0.1)

    eye_score = (
        20
        + blink_n * 25
        + perclos_n * 30
        + motion_n * 15
        + pupil_n * 10
    )

    return eye_score


def compute_stress(state):
    eye = state["eye_metrics"]
    kb = state["keyboard_load"]

    # -------------------------
    # EYE COMPONENT
    # -------------------------
    eye_score = compute_eye_score(eye)

    # -------------------------
    # KEYBOARD COMPONENT
    # -------------------------
    if kb is not None:
        try:
            kb = float(kb)
            kb = _clamp01(kb)

            # nonlinear shaping
            kb_activation = (kb ** 1.8)

        except Exception:
            kb_activation = 0.0
    else:
        kb_activation = 0.0

    # -------------------------
    # SMART FUSION
    # -------------------------
    # typing when calm ≠ stress
    if eye_score < 35:
        kb_gain = 0.25
    elif eye_score < 50:
        kb_gain = 0.6
    else:
        kb_gain = 1.0

    total = eye_score * (1.0 + kb_gain * kb_activation)

    # smooth scale into 0–100 range
    stress = _clamp(total, 0, 100)

    return round(stress, 1)


# ============================================================
# WEBSOCKET
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("Client connected")

    try:
        while True:
            msg = await websocket.receive_json()

            source = msg.get("type")

            # -------------------------
            # UPDATE SHARED STATE
            # -------------------------
            if source == "eye_metrics":
                state["eye_metrics"] = msg["data"]

            elif source == "keyboard_load":
                state["keyboard_load"] = msg["value"]

            # -------------------------
            # FUSE ONCE
            # -------------------------
            stress_score = compute_stress(state)

            print("\n--- UPDATE ---")
            print("source:", source)
            print("eye_metrics:", state["eye_metrics"])
            print("keyboard_load:", state["keyboard_load"])
            print("stress_score:", stress_score)

            await websocket.send_json({
                "type": "fusion_update",
                "stress_score": stress_score,
                "eye_metrics": state["eye_metrics"],
                "keyboard_load": state["keyboard_load"]
            })

    except Exception:
        print("Client disconnected cleanly")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run("fusion_server:app", host="0.0.0.0", port=8000, reload=True)