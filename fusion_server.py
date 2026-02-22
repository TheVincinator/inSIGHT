import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()


class FusionState:
    def __init__(self):
        self.eye_metrics = None
        self.keyboard_load = None
        self.smoothed_stress = 50.0  # persistent smoothing

state = FusionState()
clients = set()

def fuse(state: FusionState) -> float:
    """
    Human-calibrated stress fusion.

    Eye metrics ≈ dominant signal (~70%)
    Keyboard ≈ stabilizer (~30%)

    Designed to feel natural rather than linear.
    """

    eye = state.eye_metrics or {}
    kb  = state.keyboard_load

    if kb is None:
        kb_score = 0.0
    else:
        try:
            kb_score = (float(kb) ** 1.8) * 2.5
        except:
            kb_score = 0.0

    blink_rate    = eye.get("blink_rate_per_min", 0.0) or 0.0
    perclos       = eye.get("perclos", 0.0) or 0.0
    pupil_delta   = eye.get("pupil_delta", 0.0) or 0.0
    face_detected = eye.get("face_detected", False)

    eye_effect = 0.0

    blink_diff = max(0.0, blink_rate - 40.0)
    eye_effect += (blink_diff ** 1.3) * 0.07
    eye_effect += perclos * 32.0
    eye_effect += pupil_delta * 18.0
    keyboard_effect = kb_score * 10.0
    stress = 50.0 + eye_effect + keyboard_effect

    if not face_detected:
        stress -= 0.4

    stress = max(0.0, min(100.0, stress))

    alpha = 0.25
    state.smoothed_stress = (
        alpha * stress + (1 - alpha) * state.smoothed_stress
    )

    return float(state.smoothed_stress)

async def broadcast(payload):
    dead = []

    for ws in clients:
        try:
            await ws.send_json(payload)
        except:
            dead.append(ws)

    for ws in dead:
        clients.discard(ws)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    print("Client connected")

    try:
        while True:
            raw = await ws.receive_text()

            try:
                msg = json.loads(raw)
            except:
                print("Bad JSON received")
                continue

            msg_type = msg.get("type")

            if msg_type == "eye_metrics":
                state.eye_metrics = msg.get("value") or msg.get("data")

            elif msg_type == "keyboard_load":
                val = msg.get("value")
                if val is not None:
                    try:
                        state.keyboard_load = float(val)
                    except:
                        pass

            stress_score = fuse(state)

            payload = {
                "type": "stress_score",
                "value": stress_score
            }

            print("\n--- UPDATE ---")
            print("source:", msg_type)
            print("eye_metrics:", "OK" if state.eye_metrics else None)
            print("keyboard_load:", state.keyboard_load)
            print("stress_score:", round(stress_score, 1))

            await broadcast(payload)

    except WebSocketDisconnect:
        print("Client disconnected cleanly")
        clients.discard(ws)

if __name__ == "__main__":
    uvicorn.run("fusion_server:app", host="0.0.0.0", port=8000, reload=True)