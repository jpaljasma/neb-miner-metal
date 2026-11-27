"""Offline Stratum fault tests. No external socket is opened."""
import asyncio
from collections import Counter
import json
import struct
import unittest
from unittest.mock import patch

import test_metal_stratum as bridge


class Sink:
    def __init__(self):
        self.sent = asyncio.Queue()

    def write(self, data):
        self.sent.put_nowait(data)

    async def drain(self):
        pass

    def close(self):
        pass

    @property
    def transport(self):
        return self

    def get_write_buffer_size(self):
        return 0

    def is_closing(self):
        return False


class Events:
    duration = 1800
    started = None
    pause_file = None
    reconnect_test = False

    def __init__(self):
        self.slots = Counter()
        self.events = []

    def event(self, name, **details):
        self.events.append((name, details))

    def remaining(self):
        return 1800


class Recovery(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pool, self.worker = asyncio.StreamReader(), asyncio.StreamReader()
        self.to_pool, self.to_worker = Sink(), Sink()
        self.events = Events()
        self.hints = Sink()
        self.worker.feed_data(bridge.frame(1, struct.pack("<HB", 1, 15) + b"neb-miner-metal"))
        # This login never leaves the in-memory transport.
        address_file = bridge.ROOT / ".neb-payout-address"
        self.login = (address_file.read_text().strip() if address_file.exists() else "OFFLINE") + ".neb-miner-metal"
        self.task = asyncio.create_task(bridge.session((self.pool, self.to_pool),
            (self.worker, self.to_worker), self.login, self.events, self.hints))
        await self.recv_pool()

    async def asyncTearDown(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    def send(self, **msg):
        self.pool.feed_data(json.dumps(msg).encode() + b"\n")

    async def recv_pool(self):
        return json.loads(await asyncio.wait_for(self.to_pool.sent.get(), 1))

    async def job(self):
        self.send(id=1, result=[[], "12"*8, 8], error=None)
        auth = await self.recv_pool()
        self.assertTrue(auth["params"][0] == self.login, "payout worker changed")
        self.send(method="mining.set_difficulty", params=[1000000])
        self.send(method="mining.set_model", params=["ab"*32, 0, "A"*17334+"=="])
        self.notify("first", True)
        self.send(id=2, result=True, error=None)
        await asyncio.wait_for(self.to_worker.sent.get(), 1)  # model
        await asyncio.wait_for(self.to_worker.sent.get(), 1)  # job

    def notify(self, job_id, clean):
        self.send(method="mining.notify", params=[job_id, "01"*32, "", "", [], "00000001",
            "dd800000", "00000001", clean, 12, "ab"*32, 0, "00000001", "02"*32, "00000060"])

    def hit(self, job, nonce):
        self.worker.feed_data(bridge.frame(6, struct.pack("<QI", job, nonce)))

    async def test_eof_with_pending_share_is_unknown(self):
        await self.job()
        self.hit(1, 3)
        req = await self.recv_pool()
        self.pool.feed_eof()
        with self.assertRaises(ConnectionError):
            await self.task
        unknown = [d for e,d in self.events.events if e == "unconfirmed"]
        self.assertEqual([d["request"] for d in unknown], [req["id"]])
        self.assertFalse(self.events.slots)

    async def test_early_tess_model_is_advisory_until_matching_job(self):
        await self.job()
        key = "cd"*32
        self.send(method="mining.set_model", params=[key, 1, "A"*17334+"=="])
        hint = await asyncio.wait_for(self.hints.sent.get(), 1)
        self.assertEqual(hint, bytes.fromhex(key) + bytes(13000))
        self.assertTrue(self.to_worker.sent.empty())
        # A later announcement must not displace a job's still-cached weights.
        self.send(method="mining.set_model", params=["ef"*32, 1, "A"*17334+"=="])
        self.send(method="mining.notify", params=["next", "01"*32, "", "", [], "00000001",
                  "dd800000", "00000001", True, 13, key, 1, "00000001", "02"*32, "00000060"])
        model = await asyncio.wait_for(self.to_worker.sent.get(), 1)
        self.assertEqual(model[4:38], bytes([7, 1]) + bytes.fromhex(key))
        await asyncio.wait_for(self.to_worker.sent.get(), 1)
        await asyncio.sleep(.12)
        self.assertTrue(self.hints.sent.empty())

    async def test_silent_pool_releases_unacknowledged_share(self):
        with patch.object(bridge,"ACK_TIMEOUT_S",0.02):
            await self.job()
            self.hit(1, 2)
            await self.recv_pool()
            with self.assertRaisesRegex(bridge.Reconnect,"acknowledgement timeout"):
                await asyncio.wait_for(self.task, 1)
        self.assertEqual(sum(e=="unconfirmed" for e,_ in self.events.events),1)

    async def test_silent_job_feed_reconnects(self):
        with patch.object(bridge,"JOB_TIMEOUT_S",0.02):
            await self.job()
            with self.assertRaisesRegex(bridge.Reconnect,"no fresh job"):
                await asyncio.wait_for(self.task,1)

    async def test_out_of_order_replies_and_stale_jobs(self):
        await self.job()
        self.notify("refresh", False)
        await asyncio.wait_for(self.to_worker.sent.get(), 1)
        self.hit(1, 7)
        self.hit(2, 7)
        a, b = await self.recv_pool(), await self.recv_pool()
        self.assertNotEqual(a["params"][2], b["params"][2])
        self.send(id=b["id"], result=True)
        self.send(id=a["id"], result=False, error=[21, "stale", None])
        self.notify("clean", True)
        await asyncio.wait_for(self.to_worker.sent.get(), 1)
        self.hit(2, 8)
        self.hit(3, 8)
        c = await self.recv_pool()
        self.assertEqual(c["params"][1], "clean")
        self.assertEqual(self.events.slots[0], 1)
        self.assertEqual([d["code"] for e,d in self.events.events if e=="rejected"], [21])
        self.assertIn("stale_local_hit", [e for e,_ in self.events.events])

    async def test_request_reconnect_discards_peer_redirect(self):
        self.send(method="client.reconnect", params=["untrusted.invalid", 1234])
        with self.assertRaisesRegex(bridge.Reconnect, "pool requested reconnect"):
            await self.task

    async def test_exhausted_range_rolls_same_pool_job_without_reusing_en2(self):
        await self.job()
        self.hit(1, 17)
        first = await self.recv_pool()
        self.worker.feed_data(bridge.frame(8, struct.pack("<Q", 1)))
        await asyncio.wait_for(self.to_worker.sent.get(), 1)
        self.hit(2, 17)
        second = await self.recv_pool()
        self.assertEqual(first["params"][1], second["params"][1])
        self.assertNotEqual(first["params"][2], second["params"][2])
        self.assertEqual(len(second["params"][2]), 16)
        self.assertEqual(sum(e == "extranonce_roll" for e, _ in self.events.events), 1)

    async def test_malformed_notification_requires_fresh_subscription(self):
        self.pool.feed_data(b"[]\n")
        with self.assertRaisesRegex(bridge.Reconnect, "invalid protocol frame"):
            await self.task

    async def test_bad_share_fails_closed(self):
        await self.job()
        self.hit(1, 9)
        r = await self.recv_pool()
        self.send(id=r["id"], result=False, error=[23, "bad", None])
        with self.assertRaisesRegex(RuntimeError,"code 23"):
            await self.task

    async def test_authorization_refusal_stops(self):
        self.send(id=2, result=False, error=[24, "invalid", None])
        with self.assertRaisesRegex(RuntimeError, "refused payout"):
            await self.task

    async def test_timeout_drains_last_ack(self):
        await self.job()
        self.hit(1, 10)
        r = await self.recv_pool()
        self.events.remaining = lambda: 0
        self.send(id=r["id"], result=True)
        await asyncio.wait_for(self.task, 2)
        self.assertEqual(self.events.slots[0], 1)
        self.assertNotIn("unconfirmed", [e for e,_ in self.events.events])


if __name__ == "__main__":
    unittest.main()
