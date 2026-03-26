# Real-Time Cognitive Load Monitor

Estimates cognitive load in real time by fusing signals from a webcam (eye tracking) and the keyboard/mouse. A central server combines the signals and broadcasts a live stress score to all connected clients.

---

## Architecture

```
camera_client.py  ──┐
                    ├──► fusion_server.py ──► ws/subscribe ──► camera_client.py (HUD)
run.py            ──┘
```

| Component | Role |
|---|---|
| `fusion_server.py` | FastAPI WebSocket server — fuses eye and keyboard signals, broadcasts a stress score on `[0, 100]` |
| `camera_client.py` | Reads the webcam, extracts eye metrics (blink rate, PERCLOS, pupil dilation), renders a live HUD overlay |
| `run.py` | Monitors keyboard and mouse activity, computes a cognitive load score, and forwards it to the server |

All tunable parameters (thresholds, camera index, server address, ESP32 IP) live in **`config.py`**.

---

## Requirements

- Python 3.10+
- A webcam
- macOS (default capture backend is `CAP_AVFOUNDATION`; change `CAP_BACKEND` in `camera_client.py` for Linux/Windows)
- *(Optional)* An ESP32 with a pan-tilt servo for face tracking

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running

Start each component in a separate terminal:

```bash
# 1. Fusion server (must be first)
python fusion_server.py

# 2. Keyboard / mouse activity sender
python run.py

# 3. Camera + eye tracking + HUD
python camera_client.py
```

Press **q** in the camera window to exit `camera_client.py`. Use **Ctrl+C** to stop any component cleanly.

---

## Configuration

Edit `config.py` to change:

- `SERVER_HOST` / `SERVER_PORT` — where the fusion server listens
- `CAMERA_INDEX` — which webcam to use (`0` = default)
- `ESP32_SERVO_URL` — URL of the pan-tilt servo endpoint; set to `None` to disable servo tracking
- Eye-tracking thresholds, window sizes, and fusion parameters

---

## Stress Score

The score runs from **0** (very calm) to **100** (high stress/cognitive load).

| Range | Label |
|---|---|
| < 45 | calm |
| 45 – 69 | normal |
| ≥ 70 | stressed |

When stress transitions into the *stressed* state, an audio alert plays from the `stress_alerts/` folder.

---

## Project Structure

```
.
├── config.py            — all tunable parameters
├── fusion_server.py     — WebSocket fusion server
├── camera_client.py     — eye-tracking client + HUD
├── run.py               — keyboard/mouse activity client
├── activity_client.py   — keyboard/mouse feature extraction library
├── requirements.txt
└── stress_alerts/       — .wav/.mp3 alert clips
```
