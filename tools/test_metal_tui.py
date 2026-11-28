import io
import json
import unittest
from unittest.mock import patch

from metal_tui import Session, compact_difficulty, draw, miner_cpu_usage, read_events, work_time


class Dashboard(unittest.TestCase):
    def test_display_units(self):
        self.assertEqual(compact_difficulty("2340000000"), "2.34G")
        self.assertEqual(compact_difficulty("1234567"), "1.23M")
        self.assertEqual(compact_difficulty("999999999"), "1.00G")
        self.assertEqual(compact_difficulty("12.5"), "12.5")
        self.assertEqual(compact_difficulty(None), "—")
        self.assertEqual(work_time(0), "0s")
        self.assertEqual(work_time(1_200_000_000), "1.2s")
        self.assertEqual(work_time(65 * 10**9), "1m 05s")
        self.assertEqual(work_time(3605 * 10**9), "1h 00m 05s")
        self.assertEqual(work_time(90061 * 10**9), "1d 01h 01m 01s")

    def test_sky_config_survives_reconnect_and_process_stats_reset(self):
        s = Session()
        def event(name, conn, **fields):
            s.update(dict(utc="2026-11-27T12:00:00+00:00", connection=conn, event=name, **fields))
        event("start", 0)
        self.assertEqual(s.sky_threads, 0)
        event("start", 0, sky_threads=2)
        event("miner_started", 1, pid=111)
        event("sky_sample", 1, curves=11, batches=1, refused=0, errors=0)
        self.assertEqual((s.pid, s.sky_threads, s.sky_sample["curves"]), (111, 2, 11))
        event("miner_reaped", 1)
        self.assertIsNone(s.pid)
        event("tcp_connected", 2)
        self.assertEqual((s.pid, s.sky_threads, s.sky_sample), (None, 2, None))

    @patch("metal_tui.subprocess.run")
    def test_cpu_usage_is_per_miner_and_can_exceed_one_core(self, run):
        self.assertEqual(miner_cpu_usage(None), "—")
        run.assert_not_called()
        run.return_value.stdout = "124.3 /a path/neb-miner-metal\n"
        self.assertEqual(miner_cpu_usage(17), "124.3%")
        run.return_value.stdout = "12.3 /another-process"
        self.assertEqual(miner_cpu_usage(17), "—")

    def test_shows_selected_pool_transport(self):
        state = Session()
        for event, details, expected in (("tcp_connected", {}, "TCP"), ("tls_connected", {"tls": "TLSv1.3"}, "TLSv1.3")):
            state.update(dict(event=event, utc="2026-11-27T12:00:00+00:00", **details))
            self.assertEqual(state.transport, expected)

    def test_slot_rate_peak_and_replies_stay_with_original_model(self):
        s=Session()
        def add(event,**kw):
            s.update(dict(event=event,utc="2026-11-27T12:00:00+00:00",connection=1,**kw))
        add("job",slot=0,height=10,difficulty="123")
        add("submitted",slot=0,request=3)
        add("slot_sample",slot=0,evaluated=10_000_000,work_ns=1_000_000_000)
        add("job",slot=1,height=11,difficulty="456")
        add("rejected",slot=0,request=3,code=21,latency_ms=143)
        add("block_candidate",slot=0)
        add("block_candidate")  # Older sessions did not record the model.
        add("slot_sample",slot=0,evaluated=10_000_000,work_ns=1_000_000_000)
        add("slot_sample",slot=1,evaluated=100_000_000,work_ns=500_000_000)
        kepler, tess = s.models
        self.assertEqual((kepler.peak, tess.peak), (10e6, 200e6))
        self.assertEqual((kepler.last_rate, tess.last_rate), (10e6, 200e6))
        self.assertEqual((kepler.difficulty, tess.difficulty), ("123", "456"))
        self.assertEqual((kepler.rejected, kepler.stale, tess.rejected), (1, 1, 0))
        self.assertEqual((kepler.ack_ms, tess.ack_ms), (143, None))
        self.assertEqual((kepler.candidates, tess.candidates), (1, 0))

    def test_compact_and_full_views_show_both_model_rates(self):
        class Screen:
            def __init__(self, size):
                self.size = size
                self.lines = []
                self.writes = []

            def getmaxyx(self):
                return self.size

            def addnstr(self, y, x, value, n, style):
                self.lines.append(value[:n])
                self.writes.append((y,x,value[:n]))

            def erase(self):
                pass

            def refresh(self):
                pass

        state = Session()
        for slot, model in enumerate(state.models):
            model.add_work(10_000_000 * (slot + 1), 1_000_000_000)
        state.sky_threads = 2
        state.models[0].difficulty = "2340000000"
        for size in ((12, 40), (24, 80), (40, 140)):
            with self.subTest(size=size):
                screen = Screen(size)
                draw(screen, state, "72%   1.00 GiB", (0, 0, 0), 0, "23.4%")
                content = "\n".join(screen.lines).upper()
                gpu_y, gpu_x = next((y,x) for y,x,v in screen.writes if v.startswith("GPU"))
                cpu_y, cpu_x = next((y,x) for y,x,v in screen.writes if v.startswith("CPU"))
                self.assertEqual((cpu_y, cpu_x), (gpu_y + 1, gpu_x))
                if size[1] >= 58:
                    self.assertIn("2.34G", content)
                for value in ("KEPLER", "TESS", "10.00 MH/S", "20.00 MH/S", "23.4%", "SKY SEARCH  ON"):
                    self.assertIn(value, content)

    def test_partial_record_and_reconnect_accounting(self):
        def line(name, conn, **fields):
            return json.dumps(dict(utc="2026-11-27T12:00:00+00:00",event=name,connection=conn,**fields)).encode()+b"\n"
        data = (line("sample",1,rate=20,attempts=70)
                +line("slot_sample",1,slot=0,evaluated=70,work_ns=100)
                +line("slot_final",1,slot=0,evaluated=80,work_ns=110)
                +line("miner_final",1,attempts=80)
                +line("slot_sample",2,slot=0,evaluated=15,work_ns=20)
                +line("accepted",2,slot=0,latency_ms=150))
        stream=io.BytesIO(data[:-3])
        state=Session()
        read_events(stream,state)
        self.assertEqual(state.models[0].accepted, 0)
        pos=stream.tell()
        stream.seek(0,2)
        stream.write(data[-3:])
        stream.seek(pos)
        read_events(stream,state)
        kepler = state.models[0]
        self.assertEqual((kepler.tickets, kepler.work_ns), (95, 130))
        self.assertEqual(kepler.accepted, 1)
        self.assertEqual(kepler.ack_ms, 150)


if __name__=="__main__":
    unittest.main()
