#!/usr/bin/env python3
"""Watch a Metal miner JSONL session in a separate terminal."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import curses
import json
from pathlib import Path
import plistlib
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOGO = ("            █", "            █", "█▀▀▄ ▄▀▀█▀▀ █▀▀█", "█  █ █▀▀▀▀▀ █  █", "▀  ▀ ▀▀▀▀▀▀ ▀▀▀▀")
SLOTS = ("Kepler", "TESS")


def mh(rate):
    return "—" if rate is None else f"{rate / 1e6:,.2f} MH/s"


@dataclass
class ModelStats:
    tickets: int = 0
    work_ns: int = 0
    last_rate: float | None = None
    peak: float | None = None
    height: int | None = None
    difficulty: str = "—"
    engine: str = "—"
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    stale: int = 0
    unconfirmed: int = 0
    candidates: int = 0
    ack_ms: float | None = None

    @property
    def average(self):
        return self.tickets * 1e9 / self.work_ns if self.work_ns else None

    def add_work(self, tickets, work_ns):
        self.tickets += tickets
        self.work_ns += work_ns
        if work_ns > 0:
            self.last_rate = tickets * 1e9 / work_ns
            self.peak = max(self.peak or 0, self.last_rate)


class Session:
    def __init__(self):
        self.status = "Waiting"
        self.worker = self.transport = self.difficulty = "—"
        self.slot = None
        self.last = 0
        self.models = [ModelStats() for _ in SLOTS]
        self.previous = {}
        self.sent_at = {}
        self.conn = 0
        self.recent = deque(maxlen=12)

    def update(self, r):
        name = r["event"]
        dt = datetime.fromisoformat(r["utc"])
        self.last = dt.timestamp()
        conn = r.get("connection", 0)
        if conn != self.conn:
            self.slot = None
            self.previous.clear()
            self.sent_at.clear()
            self.difficulty = "—"
            self.conn = conn
        slot = r.get("slot")
        model = self.models[slot] if slot in (0, 1) else None
        label = SLOTS[slot] if model else "Unknown model"
        message = None
        if name == "start":
            self.worker = r.get("worker") or "unnamed"
        elif name == "tcp_connected":
            self.status, self.transport = "Connected", "TCP"
        elif name == "tls_connected":
            self.status, self.transport = "Connected", r.get("tls", "TLS")
        elif name == "authorized":
            self.status, message = "Authorized", "Pool login accepted"
        elif name == "job" and model:
            if self.slot != slot or model.height != r["height"]:
                message = f"{label} job  #{r['height']:,}"
            self.slot, model.height, self.status = slot, r["height"], "Mining"
            if not r.get("rolled") or "difficulty" in r:
                model.difficulty = r.get("difficulty", self.difficulty)
        elif name == "difficulty":
            self.difficulty = r["value"]
        elif name in ("slot_sample", "slot_final") and model:
            current = (r["evaluated"], r["work_ns"])
            prev = self.previous.get(slot, (0, 0))
            model.add_work(current[0] - prev[0], current[1] - prev[1])
            self.previous[slot] = current
        elif name == "submitted" and model:
            model.submitted += 1
            self.sent_at[r["request"]] = self.last
        elif name == "unconfirmed" and model:
            model.unconfirmed += 1
            self.sent_at.pop(r.get("request"), None)
        elif name in ("accepted", "rejected") and model:
            sent = self.sent_at.pop(r.get("request"), None)
            model.ack_ms = r.get("latency_ms")
            if model.ack_ms is None and sent is not None:
                model.ack_ms = max(0, (self.last - sent) * 1000)
            message = f"{label} share {name}"
            if name == "accepted":
                model.accepted += 1
            else:
                model.rejected += 1
                model.stale += r.get("code") == 21
                message += f" · code {r.get('code')}"
        elif name == "block_candidate":
            if model:
                model.candidates += 1
            message = f"{label} block candidate"
        elif name == "miner_message":
            msg = r.get("message", "")
            if msg.startswith("active scanner slot "):
                fields = msg.split()
                if len(fields) == 5 and fields[3] in ("0","1") and fields[4] in ("selective","exhaustive"):
                    self.models[int(fields[3])].engine = fields[4]
                    message = f"{SLOTS[int(fields[3])]} · {fields[4]}"
        elif name in ("paused", "reconnect", "error", "mining_deadline", "summary"):
            self.status = str(r.get("outcome", name)).capitalize()
            message = self.status
        if message:
            self.recent.append((dt.astimezone().strftime("%H:%M:%S"), name, message))


def read_events(stream, state):
    # The miner may still be writing the last line when the viewer reads it.
    for _ in range(4096):
        pos = stream.tell()
        line = stream.readline()
        if not line.endswith(b"\n"):
            stream.seek(pos)
            return
        state.update(json.loads(line))


def latest_log(directory):
    for path in sorted(directory.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        if path.name == "gpu.jsonl":
            continue
        with path.open("rb") as stream:
            try:
                first = json.loads(stream.readline(8192))
            except ValueError:
                continue
        if isinstance(first, dict) and first.get("event") == "start":
            return path
    return None


def device_usage():
    try:
        data = subprocess.check_output(["/usr/sbin/ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"],
                                       stderr=subprocess.DEVNULL, timeout=1)
        stats = plistlib.loads(data)[0]["PerformanceStatistics"]
        return f"{stats['Device Utilization %']}%   {stats['In use system memory']/2**30:.2f} GiB"
    except (OSError, ValueError, KeyError, IndexError, subprocess.SubprocessError, plistlib.InvalidFileException):
        return "unavailable"


def draw(screen, s, gpu, colors, now):
    height, width = screen.getmaxyx()
    screen.erase()
    def put(y, x, value, style=0, length=None):
        if 0 <= y < height and 0 <= x < width-1:
            text = "".join(c if c.isprintable() else " " for c in str(value))
            screen.addnstr(y, x, text, max(0, min(width-x-1, length if length is not None else width)), style)
    green, dim, alert = colors
    stale = s.status == "Mining" and now - s.last > 20
    status = "No recent data" if stale else s.status
    if width < 58 or height < 18:
        put(0, 0, f"NEB · {status}", alert if stale else green)
        for slot, model in enumerate(s.models):
            active = "*" if slot == s.slot and status == "Mining" else " "
            put(slot + 1, 0, f"{active}{SLOTS[slot]:6} {mh(model.last_rate):>12}  {model.accepted} ok  {model.rejected} rej")
        put(4, 0, "58×18 for full dashboard · q closes viewer", dim)
        screen.refresh()
        return
    for y, line in enumerate(LOGO):
        put(y, 1, line, green)
    put(1, 23, "METAL  /  NEB", curses.A_BOLD)
    put(2, 23, s.worker, dim, width-42)
    put(3, 23, f"● {status}  ·  {s.transport}", alert if stale else green)
    if s.last:
        stamp = datetime.fromtimestamp(s.last).strftime("%H:%M:%S")
        put(4, 23, f"Last event {stamp}  ·  {max(0, int(now-s.last))}s ago", dim)
    def panel(x, w, title, rows):
        put(5, x, "╭─ "+title+" "+"─"*max(0,w-len(title)-5)+"╮", green, w)
        for y, (k,v) in enumerate(rows, 6):
            put(y,x,"│",dim)
            put(y,x+2,k,dim,11)
            put(y,x+14,v,curses.A_BOLD,w-16)
            put(y,x+w-1,"│",dim)
        put(6+len(rows), x, "╰"+"─"*(w-2)+"╯", dim,w)
    put(1, max(48,width-31), f"GPU   {gpu}",dim)
    panel_width = min(width - 2, 140)
    left = (panel_width - 1) // 2
    for i, model in enumerate(s.models):
        active = s.slot == i and status == "Mining"
        state = "ACTIVE" if active else "LAST" if model.height is not None else "WAITING"
        decided = model.accepted + model.rejected
        rejected = f"{model.rejected} ({100 * model.rejected / decided:.1f}%)" if decided else "0"
        rows = [
            ("LAST RATE", mh(model.last_rate)),
            ("AVG ACTIVE", mh(model.average)),
            ("PEAK", mh(model.peak)),
            ("TICKETS", f"{model.tickets:,}"),
            ("WORK TIME", f"{model.work_ns / 1e9:,.1f} s"),
            ("ACCEPTED", model.accepted),
            ("REJECTED", rejected),
            ("STALE", model.stale),
            ("PENDING", model.submitted - decided - model.unconfirmed),
            ("LAST ACK", "—" if model.ack_ms is None else f"{model.ack_ms:.0f} ms"),
            ("DIFFICULTY", model.difficulty),
            ("JOB HEIGHT", f"#{model.height:,}" if model.height is not None else "—"),
            ("LAST ENGINE", model.engine),
            ("CANDIDATES", model.candidates),
            ("UNCONFIRMED", model.unconfirmed),
        ][:height - 8]
        panel(0 if i == 0 else left + 1, left if i == 0 else panel_width - left - 1,
              f"{SLOTS[i].upper()}  {state}", rows)
    activity=8+len(rows)
    if height>=activity+3:
        put(activity,2,"RECENT POOL ACTIVITY",green)
        for j,(ts,name,msg) in enumerate(reversed(s.recent)):
            if activity+1+j >= height-1: break
            put(activity+1+j,2,f"{ts}   {msg}",alert if name in ("error","reconnect","rejected") else 0)
    put(height-1,1,"q  close viewer   ·   miner runs independently",dim)
    screen.refresh()


def watch(screen, stream):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.timeout(500)
    colors = (curses.A_BOLD,curses.A_DIM,curses.A_REVERSE)
    if curses.has_colors():
        curses.start_color()
        curses.init_pair(1,120 if curses.COLORS >= 256 else curses.COLOR_GREEN,curses.COLOR_BLACK)
        curses.init_pair(2,curses.COLOR_YELLOW,curses.COLOR_BLACK)
        curses.init_pair(3,250 if curses.COLORS >= 256 else curses.COLOR_WHITE,curses.COLOR_BLACK)
        colors=(curses.color_pair(1)|curses.A_BOLD,curses.color_pair(3),curses.color_pair(2))
    state=Session()
    next_gpu=0
    gpu="—"
    while True:
        read_events(stream,state)
        now=time.time()
        if state.status=="Mining" and now-state.last<20:
            if time.monotonic()>=next_gpu:
                gpu=device_usage()
                next_gpu=time.monotonic()+5
        else:
            gpu="—"
        draw(screen,state,gpu,colors,now)
        if screen.getch() in (ord("q"),ord("Q"),27): return


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",nargs="?",type=Path,help="default: most recently modified mining log under build/")
    path=parser.parse_args().log
    if path is None:
        path=latest_log(ROOT/"build")
    if path is None: parser.exit(1,"No session log found. Start the pool adapter with --log FILE.\n")
    if not sys.stdout.isatty(): parser.exit(1,"Open the dashboard in a terminal.\n")
    try:
        with path.open("rb") as stream:
            curses.wrapper(watch, stream)
    except KeyboardInterrupt:
        pass
    except (OSError,ValueError,curses.error) as exc:
        parser.exit(1,f"Dashboard: {exc}\n")
