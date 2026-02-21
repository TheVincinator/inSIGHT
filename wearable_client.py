# wearable_client.py
import asyncio, websockets, json, random

async def send():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        while True:
            bpm = random.randint(60, 110)

            msg = {
                "type": "heart_rate",
                "value": bpm
            }

            await ws.send(json.dumps(msg))
            await asyncio.sleep(1)

asyncio.run(send())