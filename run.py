import asyncio
import json
import websockets
import activity_client

FUSION_URI   = "ws://localhost:8000/ws/ingest"
RETRY_DELAY  = 3.0   # seconds to wait before reconnecting after a dropped connection


async def main():
    print("[run.py] starting keyboard sender...")
    activity_client.start_monitoring()  # start here, not at import time

    while True:
        try:
            async with websockets.connect(FUSION_URI) as ws:
                print("[run.py] connected to fusion_server")

                while True:
                    await asyncio.sleep(2)

                    score = activity_client.get_keyboard_load_score()
                    print("LOCAL KEYBOARD SCORE =", score)

                    if score is not None:
                        await ws.send(json.dumps({
                            "type":  "keyboard_load",
                            "value": float(score),
                        }))
                        print("[run.py] sent keyboard_load:", score)

        except (websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError) as e:
            print(f"[run.py] connection lost ({e}), retrying in {RETRY_DELAY}s...")
            await asyncio.sleep(RETRY_DELAY)


asyncio.run(main())