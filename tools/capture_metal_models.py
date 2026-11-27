"""Capture public weights from two Suprnova model notifications for offline kernel replay."""
import argparse
import asyncio
import base64
import json
from pathlib import Path

from test_metal_stratum import ROOT, open_pool, request


async def capture(args):
    reader, writer = await open_pool(args.pool)
    models, captured = {}, {0: set(), 1: set()}
    try:
        request(writer, 1, "mining.subscribe", ["neb-miner-metal/replay"])
        login = (ROOT / ".neb-payout-address").read_text().strip() + "." + args.worker
        request(writer, 2, "mining.authorize", [login, "x"])
        await writer.drain()
        while any(len(seen) < args.per_slot for seen in captured.values()):
            line = await reader.readline()
            if not line:
                raise ConnectionError("pool closed model capture")
            msg = json.loads(line)
            if msg.get("id") == 2 and msg.get("result") is not True:
                raise ValueError("capture authorization failed")
            if msg.get("method") == "mining.set_model":
                key, slot, data = msg["params"]
                raw = base64.b64decode(data, validate=True)
                if slot in (0, 1) and len(raw) == 13000:
                    models[slot] = (key, raw)
            if msg.get("method") == "mining.notify":
                job = msg["params"]
                height, key, slot = job[9:12]
                if slot in captured and len(captured[slot]) < args.per_slot and key not in captured[slot] and models.get(slot, (None,))[0] == key:
                    # Keep the original single-pair replay filenames.
                    dest = args.directory if args.per_slot == 1 else args.directory / str(len(captured[slot]))
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / f"slot-{slot}.bin").write_bytes(models[slot][1])
                    (dest / f"slot-{slot}.hash").write_text(key)
                    (dest / f"slot-{slot}.height").write_text(str(height))
                    captured[slot].add(key)
                    print(f"captured slot {slot} at height {height}", flush=True)
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--per-slot", type=int, default=1, choices=(1,2,3))
    args=parser.parse_args()
    asyncio.run(asyncio.wait_for(capture(args), 240 * args.per_slot))
