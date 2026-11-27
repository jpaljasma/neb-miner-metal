#!/usr/bin/env python3
"""Mine one verified Suprnova share or run a timed, reconnecting Metal session."""
import argparse
import asyncio
import base64
import contextlib
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, DecimalException
import json
import math
from pathlib import Path
import re
import secrets
import ssl
import struct
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
U256_MAX = (1 << 256) - 1
ACK_TIMEOUT_S = 30
JOB_TIMEOUT_S = 95


class Reconnect(ConnectionError):
    """Transport or stale-work condition requiring a fresh subscription."""


class Pause(Exception):
    """Operator requested exclusive GPU access for a benchmark."""


class Monitor:
    def __init__(self, args):
        self.duration = args.duration
        self.started = None
        self.connection = 0
        self.reconnect_test = args.test_reconnect
        self.pause_file = args.pause_file
        self.paused_s = 0
        self.counts = Counter()
        self.slots = Counter()
        self.attempts = {}
        self.log = args.log.open("x", buffering=1) if args.log else None

    def event(self, event, **fields):
        self.counts[event] += 1
        record = dict(utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                      connection=self.connection, event=event, **fields)
        if self.log:
            self.log.write(json.dumps(record, allow_nan=False) + "\n")
        if event in {"authorized", "reconnect", "error", "summary", "first_job",
                     "rejected", "unconfirmed", "unknown_method", "unexpected_response"}:
            print(json.dumps(record), flush=True)

    def remaining(self):
        if self.duration is not None and self.started is not None:
            return max(0, self.duration - (time.monotonic() - self.started - self.paused_s))
        return float("inf")

    def summary(self, outcome):
        elapsed = 0 if self.started is None else time.monotonic() - self.started - self.paused_s
        total = sum(self.attempts.values())
        self.event("summary", outcome=outcome, elapsed_s=round(elapsed, 2), attempts=total,
                   attempts_per_s=round(total / elapsed) if elapsed else 0,
                   counts=dict(self.counts), accepted_by_slot=dict(self.slots), benchmark_pause_s=round(self.paused_s, 2))


async def miner_log(proc, monitor):
    while line := await proc.stderr.readline():
        msg = line.decode(errors="replace").strip()
        sample = re.fullmatch(r"(\d+) attempts/s, shares (\d+), accepted blocks (\d+), mismatches (\d+), fallbacks (\d+), attempts (\d+)", msg)
        final = re.fullmatch(r"final attempts (\d+), mismatches (\d+), fallbacks (\d+)", msg)
        slot = re.fullmatch(r"slot (sample|final) (\d+) evaluated (\d+) screened (\d+) work_ns (\d+)", msg)
        if slot:
            kind, i, evaluated, screened, ns = slot.groups()
            monitor.event("slot_"+kind, slot=int(i), evaluated=int(evaluated),screened=int(screened),work_ns=int(ns))
            continue
        if sample:
            rate, shares, blocks, mismatches, fallbacks, attempts = map(int, sample.groups())
            monitor.attempts[monitor.connection] = attempts
            monitor.event("sample", rate=rate, local_shares=shares, local_block_acks=blocks,
                          mismatches=mismatches, fallbacks=fallbacks, attempts=attempts)
        elif final:
            attempts, mismatches, fallbacks = map(int, final.groups())
            monitor.attempts[monitor.connection] = attempts
            monitor.event("miner_final", attempts=attempts, mismatches=mismatches, fallbacks=fallbacks)
        else:
            monitor.event("miner_message", message=re.sub(r"neb1[a-zA-Z0-9.\-_]+", "[redacted]", msg)[:300])
            if "switching to the fallback scanner" in msg:
                raise RuntimeError("miner switched to CPU fallback")
            if "mismatch" in msg.lower():
                raise RuntimeError("native scanner reported a verification mismatch")
            continue
        if mismatches or fallbacks:
            raise RuntimeError("Metal verification fault")


def compact(n):
    if n < 253:
        return bytes([n])
    return b"\xfd" + struct.pack("<H", n) if n <= 65535 else b"\xfe" + struct.pack("<I", n)


def frame(kind, payload):
    return struct.pack("<IB", len(payload) + 1, kind) + payload


def hash_bytes(text):
    raw = bytes.fromhex(text)
    if len(raw) != 32:
        raise ValueError("invalid hash length")
    return raw


def job_frame(serial, job, extranonce1, size, difficulty):
    """Convert a Suprnova job into a work-service job with the same reference header."""
    _, parent, prefix, suffix, branch, version, bits, timestamp, clean, height, _, _, count, root, end = job
    bits, timestamp, end = (int(x, 16) for x in (bits, timestamp, end))
    block_target = (bits & 0xffffff) << (bits >> 24)
    scaled = round(difficulty * 1_000_000)
    if scaled < 1_000_000 or len(branch) > 4096:
        raise ValueError("invalid difficulty or branch length")
    pool_target = U256_MAX * 1_000_000 // scaled
    # The upstream client fills the remaining extranonce (en2) with zeros.
    prefix, suffix = bytes.fromhex(prefix) + extranonce1, bytes.fromhex(suffix)
    payload = b"".join((
        struct.pack("<QBI", serial, int(clean), int(version, 16)), hash_bytes(parent),
        struct.pack("<III", timestamp, end - timestamp, bits),
        block_target.to_bytes(32, "little"), pool_target.to_bytes(32, "little"),
        compact(len(prefix)), prefix, compact(len(suffix)), suffix, bytes([size]),
        compact(len(branch)), b"".join(map(hash_bytes, branch)),
        struct.pack("<I", int(count, 16)), hash_bytes(root), struct.pack("<II", height, end),
    ))
    return frame(2, payload)


def request(writer, number, method, params):
    writer.write((json.dumps(dict(id=number, method=method, params=params)) + "\n").encode())


async def read_miner(reader):
    size, = struct.unpack("<I", await reader.readexactly(4))
    if not 1 <= size <= 4096:
        raise ValueError("invalid miner message size")
    message = await reader.readexactly(size)
    return message[0], message[1:]


async def session(pool, worker, login, monitor):
    reader, writer = pool
    miner_in, miner_out = worker
    kind, hello = await asyncio.wait_for(read_miner(miner_in), timeout=10)
    name = b"neb-miner-metal"
    if kind != 1 or hello != struct.pack("<HB", 1, len(name)) + name:
        raise RuntimeError("expected neb-miner-metal work-service greeting")
    request(writer, 1, "mining.subscribe", ["neb-miner-metal/1.0.9"])
    await writer.drain()
    authorized = False
    en1, en2_size, difficulty = None, None, None
    models, jobs, pending = {}, {}, {}
    nonce_prefix = secrets.randbits(56)
    serial, next_id = 0, 3
    last_model = None
    buffered_job = None
    pool_read = asyncio.create_task(reader.readline())
    miner_read = asyncio.create_task(read_miner(miner_in))
    job_deadline = time.monotonic() + 20
    draining = False
    latest_job = None

    async def dispatch(job, target_diff, rolled=False):
        nonlocal serial, last_model, job_deadline
        digest, model = models[job[11]]
        if digest != job[10]:
            raise ValueError("pool job references an unknown model")
        if job[8] and not rolled:
            jobs.clear()
        if len(jobs) >= 16:
            del jobs[next(iter(jobs))]
            monitor.event("evicted_job")
        serial += 1
        # Put seven unique bytes in the prefix and let the node client
        # supply the final zero. This also separates reissued templates.
        tag = ((nonce_prefix + serial) % (1 << 56)).to_bytes(7, "big")
        en2_hex = (tag + b"\0").hex()
        jobs[serial] = (job[0], job[7], job[11], en2_hex, target_diff)
        if last_model != (job[11], digest):
            miner_out.write(model)
            last_model = (job[11], digest)
        miner_out.write(job_frame(serial, job, en1 + tag, en2_size-len(tag), target_diff))
        await miner_out.drain()
        if monitor.started is None:
            monitor.started = time.monotonic()
            monitor.event("first_job")
        job_deadline = time.monotonic() + JOB_TIMEOUT_S
        monitor.event("job", slot=job[11], height=job[9], clean=bool(job[8]) and not rolled, sequence=serial, rolled=rolled)

    try:
        while True:
            now = time.monotonic()
            if monitor.pause_file and monitor.pause_file.exists() and not pending and not draining:
                raise Pause()
            if monitor.remaining() == 0 and not draining:
                draining = True
                miner_out.close()
                monitor.event("mining_deadline", awaiting=len(pending))
                job_deadline = now + 4
            if draining and (not pending or now >= job_deadline):
                return
            if now >= job_deadline:
                raise Reconnect("no fresh job within watchdog interval")
            ack_left = min((v[1] + ACK_TIMEOUT_S - now for v in pending.values()), default=float("inf"))
            if ack_left <= 0:
                raise Reconnect("pool share acknowledgement timeout")
            reads = [pool_read, miner_read] if len(pending) < 16 and not draining else [pool_read]
            timeout = min(job_deadline-now, ack_left, 1, float("inf") if draining else monitor.remaining())
            ready, _ = await asyncio.wait(reads, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if pool_read in ready:
                line = pool_read.result()
                if not line:
                    raise ConnectionError("pool disconnected")
                msg = json.loads(line, parse_float=Decimal)
                if not isinstance(msg, dict):
                    raise ValueError("pool message must be a JSON object")
                pool_read = asyncio.create_task(reader.readline())
                method, params, number = msg.get("method"), msg.get("params"), msg.get("id")
                if number == 1 and not method:
                    if msg.get("error"):
                        raise RuntimeError("pool refused subscription")
                    _, en1_text, en2_size = msg["result"]
                    en1 = bytes.fromhex(en1_text)
                    if len(en1) != 8 or en2_size != 8:
                        raise ValueError("unexpected NEB extranonce size")
                    request(writer, 2, "mining.authorize", [login, "d=1000000"])
                elif number == 2 and not method:
                    if msg.get("result") is not True or msg.get("error"):
                        raise RuntimeError("pool refused payout address")
                    authorized = True
                    monitor.event("authorized")
                elif number in pending and not method:
                    slot, sent, old_job_id, share_diff = pending.pop(number)
                    if msg.get("result") is True and not msg.get("error"):
                        monitor.slots[slot] += 1
                        monitor.event("accepted", slot=slot, request=number, difficulty=str(share_diff),
                                      latency_ms=round((time.monotonic()-sent)*1000, 2))
                        if not monitor.duration:
                            return
                        if monitor.reconnect_test and not pending and time.monotonic()-monitor.started >= 60:
                            monitor.reconnect_test = False
                            raise Reconnect("controlled disconnect after accepted share")
                    else:
                        error = msg.get("error")
                        code = error[0] if isinstance(error, list) and error and type(error[0]) is int else None
                        monitor.event("rejected", slot=slot, request=number, code=code)
                        if code == 21:
                            for local_id in [i for i, v in jobs.items() if v[0] == old_job_id]:
                                del jobs[local_id]
                        else:
                            raise RuntimeError(f"pool rejected share (code {code})")
                elif method == "mining.set_difficulty":
                    difficulty = Decimal(params[0])
                    if not difficulty.is_finite() or not 1 <= difficulty <= U256_MAX:
                        raise ValueError("invalid pool difficulty")
                    monitor.event("difficulty", value=str(difficulty))
                elif method == "mining.set_model":
                    digest, slot, encoded = params
                    packed = base64.b64decode(encoded, validate=True)
                    if slot not in (0, 1) or len(packed) != 13000:
                        raise ValueError("unexpected mainnet model")
                    # decode_model in the miner verifies this hash before accepting the model frame.
                    models[slot] = (digest, frame(7, bytes([slot]) + hash_bytes(digest) + packed))
                    monitor.event("model", slot=slot)
                elif method == "mining.notify":
                    if len(params) != 15 or difficulty is None or en1 is None:
                        raise ValueError("incomplete Suprnova NEB job")
                    buffered_job = (params, difficulty)
                elif method:
                    if method == "client.reconnect":
                        raise Reconnect("pool requested reconnect")
                    if method == "mining.set_extranonce":
                        raise Reconnect("pool changed extranonce")
                    monitor.event("unknown_method", method=re.sub(r"[^a-z_.]", "?", str(method)[:64]))
                else:
                    monitor.event("unexpected_response", request=number if type(number) is int else None)
                if authorized and buffered_job is not None and not draining:
                    latest_job = buffered_job
                    buffered_job = None
                    await dispatch(*latest_job)
                await writer.drain()
            if miner_read in ready:
                kind, payload = miner_read.result()
                miner_read = asyncio.create_task(read_miner(miner_in))
                # The miner sends a Share for every hit, including block candidates.
                # Its following Submit for a block is the same nonce.
                if kind == 6:
                    local_id, nonce = struct.unpack("<QI", payload)
                    if local_id in jobs:
                        job_id, timestamp, slot, en2_hex, job_diff = jobs[local_id]
                        pending[next_id] = (slot, time.monotonic(), job_id, job_diff)
                        monitor.event("submitted", slot=slot, request=next_id)
                        request(writer, next_id, "mining.submit",
                                [login, job_id, en2_hex, timestamp, f"{nonce:08x}"])
                        next_id += 1
                        await writer.drain()
                    else:
                        monitor.event("stale_local_hit", sequence=local_id)
                elif kind == 8:
                    last_id, = struct.unpack("<Q", payload)
                    if latest_job and last_id == serial and last_id in jobs:
                        # Keep the original pool freshness deadline. Rolling en2
                        # locally must not mask a silent/dead pool connection.
                        deadline = job_deadline
                        await dispatch(*latest_job, rolled=True)
                        job_deadline = deadline
                        monitor.event("extranonce_roll")
                elif kind == 4:
                    monitor.event("block_candidate")
                else:
                    monitor.event("unexpected_miner_frame", kind=kind)
    except (ValueError, KeyError, TypeError, IndexError, OverflowError, DecimalException, struct.error) as error:
        # A broken or truncated notification invalidates the job/target context;
        # discard it and establish a new subscription at the configured TLS endpoint.
        raise Reconnect(f"invalid protocol frame ({type(error).__name__})") from error
    finally:
        for number, (slot, _, _, _) in pending.items():
            monitor.event("unconfirmed", request=number, slot=slot)
        for task in (pool_read, miner_read):
            task.cancel()
        await asyncio.gather(pool_read, miner_read, return_exceptions=True)


async def run(args, monitor):
    url = urlsplit(args.pool)
    if url.scheme != "stratum+ssl" or not url.hostname or not url.port or url.username or url.password or url.path:
        raise RuntimeError("expected stratum+ssl://host:port")
    login = args.address_file.read_text().strip()
    if not re.fullmatch(r"neb1[a-z0-9]{58}", login):
        raise RuntimeError("invalid NEB address in local file")
    if args.worker:
        login += "." + args.worker
    ctx = ssl.create_default_context()
    if not ctx.get_ca_certs() and Path("/etc/ssl/cert.pem").is_file():
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    pool = await asyncio.wait_for(asyncio.open_connection(
        url.hostname, url.port, ssl=ctx, server_hostname=url.hostname, limit=262144), 12)
    monitor.event("tls_connected", tls=pool[1].get_extra_info("ssl_object").version())
    connected = asyncio.get_running_loop().create_future()

    def connect(reader, writer):
        if not connected.done():
            connected.set_result((reader, writer))
        else:
            writer.close()

    proc = None
    log_reader = None
    try:
        async with await asyncio.start_server(connect, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            proc = await asyncio.create_subprocess_exec(str(args.miner), "--dataset", str(args.dataset),
                *(["--stratum-adapter"] if args.selective else []), "--node", f"127.0.0.1:{port}", stderr=asyncio.subprocess.PIPE)
            monitor.event("miner_started", pid=proc.pid)
            log_reader = asyncio.create_task(miner_log(proc, monitor))
            exited = asyncio.create_task(proc.wait())
            try:
                await asyncio.wait([connected, exited], timeout=10, return_when=asyncio.FIRST_COMPLETED)
                if exited.done():
                    raise RuntimeError(f"Metal miner exited before connecting (status {exited.result()})")
                if not connected.done():
                    raise TimeoutError("Metal miner did not connect")
                worker = connected.result()
            finally:
                exited.cancel()
                await asyncio.gather(exited, return_exceptions=True)
        try:
            active = asyncio.create_task(session(pool, worker, login, monitor))
            try:
                done, _ = await asyncio.wait([log_reader, active], return_when=asyncio.FIRST_COMPLETED)
                if log_reader in done and not active.done():
                    await log_reader
                    if monitor.remaining() > 0:
                        raise ConnectionError("local miner ended")
                await active
            finally:
                active.cancel()
                await asyncio.gather(active, return_exceptions=True)
        finally:
            worker[1].close()
    finally:
        pool[1].close()
        if proc is not None:
            try:
                await asyncio.wait_for(proc.wait(), 2)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), 2)
                except asyncio.TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
            monitor.event("miner_reaped", pid=proc.pid, status=proc.returncode)
            if log_reader is not None:
                await log_reader


async def supervise(args, monitor):
    failures = 0
    startup_deadline = time.monotonic() + 60
    while monitor.remaining() > 0:
        if monitor.started is None and time.monotonic() >= startup_deadline:
            raise RuntimeError("no mining job within startup deadline")
        monitor.connection += 1
        accepted = monitor.counts["accepted"]
        try:
            await run(args, monitor)
            return
        except Pause:
            start = time.monotonic()
            monitor.event("paused")
            try:
                while monitor.pause_file.exists():
                    await asyncio.sleep(0.25)
            finally:
                monitor.paused_s += time.monotonic()-start
            monitor.event("resuming", paused_s=round(monitor.paused_s, 2))
        except ssl.SSLCertVerificationError:
            raise
        except (OSError, asyncio.IncompleteReadError) as error:
            if not args.duration or isinstance(error, FileNotFoundError):
                raise
            failures = 0 if monitor.counts["accepted"] > accepted else failures + 1
            delay = min(monitor.remaining(), 2 ** min(failures, 5))
            monitor.event("reconnect", reason=safe_error(error), retry_s=delay)
            await asyncio.sleep(delay)


def safe_error(error):
    return type(error).__name__ + ": " + re.sub(r"neb1[a-zA-Z0-9._-]+", "[redacted]", str(error))[:240]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--duration", type=float, help="mine for this many seconds from first dispatched job")
    parser.add_argument("--selective", action="store_true", help="validate and select GPU sparse screening per model")
    parser.add_argument("--worker", help="Suprnova worker suffix")
    parser.add_argument("--log", type=Path, help="new JSONL event log")
    parser.add_argument("--test-reconnect", action="store_true", help="exercise one reconnect after a confirmed share")
    parser.add_argument("--pause-file", type=Path, help="disconnect and stop the GPU while this file exists")
    parser.add_argument("--address-file", type=Path, default=ROOT / ".neb-payout-address")
    parser.add_argument("--dataset", type=Path, default=ROOT / "build/neb-datasets.bin")
    parser.add_argument("--miner", type=Path, default=ROOT / "node/target/release/neb-miner-metal")
    args = parser.parse_args()
    if args.worker and not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args.worker):
        parser.error("invalid worker name")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("duration must be positive")
    monitor = Monitor(args)
    monitor.event("start", worker=args.worker, duration_s=args.duration)
    outcome = "failed"
    try:
        watchdog = args.timeout if not args.duration else args.duration + 60 + (3600 if args.pause_file else 0)
        asyncio.run(asyncio.wait_for(supervise(args, monitor), watchdog))
        outcome = "completed" if monitor.counts["accepted"] else "no_pool_acceptance"
        if outcome != "completed":
            sys.exit(1)
        if not args.duration:
            print("CPU-verified Metal share accepted by pool", flush=True)
    except KeyboardInterrupt:
        outcome = "interrupted"
        sys.exit(130)
    except Exception as error:
        monitor.event("error", reason=safe_error(error))
        sys.exit(1)
    finally:
        monitor.summary(outcome)
        if monitor.log:
            monitor.log.close()
