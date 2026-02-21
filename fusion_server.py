from fastapi import FastAPI, WebSocket
import json
import time

app = FastAPI()

state = {
    "heart_rate": None,
    "activity": None,
    "eye_metrics": None,
    "last_update": None,
}

def compute_stress_simple():
    """
    Placeholder fusion:
    - blink_rate_per_min high -> more stress/fatigue
    - perclos high -> more fatigue
    - head_motion_var high -> agitation
    """
    if not state["eye_metrics"]:
        return None

    em = state["eye_metrics"]
    blink = em.get("blink_rate_per_min", 0.0) or 0.0
    perclos = em.get("perclos", 0.0) or 0.0
    headv = em.get("head_motion_var", 0.0) or 0.0

    # Very rough demo scoring; tune later
    score = 0.0
    score += max(0.0, (blink - 15.0) / 15.0)      # above ~15 blinks/min adds
    score += perclos * 2.0                        # 0..1 scaled up
    score += min(2.0, headv * 10.0)               # depends on camera scale

    return round(100.0 * (1.0 - (1.0 / (1.0 + score))), 1)  # squash-ish 0..100

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    while True:
        data = await ws.receive_text()
        msg = json.loads(data)

        mtype = msg.get("type")
        value = msg.get("value")

        state[mtype] = value
        state["last_update"] = time.time()

        stress = compute_stress_simple()
        print("\n--- UPDATE ---")
        print("eye_metrics:", state["eye_metrics"])
        print("stress_score:", stress)