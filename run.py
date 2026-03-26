import asyncio
import json
import signal

import websockets

import activity_client
from config import RETRY_DELAY, WS_INGEST_URL


async def main():
    print("[run.py] starting keyboard/mouse sender...")
    activity_client.start_monitoring()

    try:
        while True:
            try:
                async with websockets.connect(WS_INGEST_URL) as ws:
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

    finally:
        activity_client.stop_monitoring()
        print("[run.py] shut down cleanly")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
