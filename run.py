import asyncio
import json
import websockets
import activity_client

FUSION_URI = "ws://localhost:8000/ws"

activity_client.start_monitoring()


async def main():
    print("[run.py] starting keyboard sender...")

    async with websockets.connect(FUSION_URI) as ws:
        print("[run.py] connected to fusion_server")

        while True:
            await asyncio.sleep(2)

            score = activity_client.get_keyboard_load_score()
            print("LOCAL KEYBOARD SCORE =", score)

            if score is not None:
                await ws.send(json.dumps({
                    "type": "keyboard_load",
                    "value": float(score)
                }))
                print("[run.py] sent keyboard_load:", score)


asyncio.run(main())