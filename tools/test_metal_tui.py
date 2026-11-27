import io
import json
import unittest

from metal_tui import Session, draw, read_events


class Dashboard(unittest.TestCase):
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

            def getmaxyx(self):
                return self.size

            def addnstr(self, y, x, value, n, style):
                self.lines.append(value[:n])

            def erase(self):
                pass

            def refresh(self):
                pass

        state = Session()
        for slot, model in enumerate(state.models):
            model.add_work(10_000_000 * (slot + 1), 1_000_000_000)
        for size in ((12, 40), (24, 80), (35, 140)):
            with self.subTest(size=size):
                screen = Screen(size)
                draw(screen, state, "—", (0, 0, 0), 0)
                content = "\n".join(screen.lines).upper()
                for value in ("KEPLER", "TESS", "10.00 MH/S", "20.00 MH/S"):
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
