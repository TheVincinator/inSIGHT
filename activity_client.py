# activity_client.py
import asyncio, websockets, json, random

async def send():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        while True:
            activity = random.random()

            msg = {
                "type": "activity",
                "value": activity
            }

            await ws.send(json.dumps(msg))
            await asyncio.sleep(1)

asyncio.run(send())