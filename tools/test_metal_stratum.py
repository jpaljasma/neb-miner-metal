#!/usr/bin/env python3
"""Run the release Metal miner until Suprnova accepts a share, or the test times out."""
import argparse
import asyncio
import base64
import contextlib
from decimal import Decimal
import json
from pathlib import Path
import re
import ssl
import struct
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
U256_MAX = (1 << 256) - 1


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


async def session(pool, worker, login):
    reader, writer = pool
    miner_in, miner_out = worker
    kind, hello = await read_miner(miner_in)
    name = b"neb-miner-metal"
    if kind != 1 or hello != struct.pack("<HB", 1, len(name)) + name:
        raise RuntimeError("expected neb-miner-metal work-service greeting")
    request(writer, 1, "mining.subscribe", ["neb-miner-metal/1.0.9"])
    await writer.drain()
    authorized = False
    en1, en2_size, difficulty = None, None, None
    models, jobs, pending = {}, {}, set()
    serial, next_id = 0, 3
    last_model = None
    buffered_job = None
    pool_read = asyncio.create_task(reader.readline())
    miner_read = asyncio.create_task(read_miner(miner_in))
    try:
        while True:
            reads = [pool_read, miner_read] if len(pending) < 16 else [pool_read]
            ready, _ = await asyncio.wait(reads, return_when=asyncio.FIRST_COMPLETED)
            if pool_read in ready:
                line = pool_read.result()
                if not line:
                    raise ConnectionError("pool disconnected")
                msg = json.loads(line, parse_float=Decimal)
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
                    print("TLS subscription and payout login accepted", flush=True)
                elif number in pending and not method:
                    pending.remove(number)
                    if msg.get("result") is True and not msg.get("error"):
                        return
                    error = msg.get("error")
                    code = error[0] if isinstance(error, list) and error else None
                    if code != 21:
                        raise RuntimeError(f"pool rejected share (code {code if type(code) is int else 'unknown'})")
                elif method == "mining.set_difficulty":
                    difficulty = Decimal(params[0])
                    if not difficulty.is_finite() or not 1 <= difficulty <= U256_MAX:
                        raise ValueError("invalid pool difficulty")
                elif method == "mining.set_model":
                    digest, slot, encoded = params
                    packed = base64.b64decode(encoded, validate=True)
                    if slot not in (0, 1) or len(packed) != 13000:
                        raise ValueError("unexpected mainnet model")
                    # decode_model in the miner verifies this hash before accepting the model frame.
                    models[slot] = (digest, frame(7, bytes([slot]) + hash_bytes(digest) + packed))
                elif method == "mining.notify":
                    if len(params) != 15 or difficulty is None or en1 is None:
                        raise ValueError("incomplete Suprnova NEB job")
                    buffered_job = (params, difficulty)
                if authorized and buffered_job is not None:
                    job, target_diff = buffered_job
                    buffered_job = None
                    digest, model = models[job[11]]
                    if digest != job[10]:
                        raise ValueError("pool job references an unknown model")
                    if job[8]:
                        jobs.clear()
                    if len(jobs) >= 16:
                        del jobs[next(iter(jobs))]
                    serial += 1
                    jobs[serial] = (job[0], job[7])
                    if last_model != (job[11], digest):
                        miner_out.write(model)
                        last_model = (job[11], digest)
                    miner_out.write(job_frame(serial, job, en1, en2_size, target_diff))
                    await miner_out.drain()
                await writer.drain()
            if miner_read in ready:
                kind, payload = miner_read.result()
                miner_read = asyncio.create_task(read_miner(miner_in))
                # The miner sends a Share for every hit, including block candidates.
                # Its following Submit for a block is the same nonce.
                if kind == 6:
                    local_id, nonce = struct.unpack("<QI", payload)
                    if local_id in jobs:
                        job_id, timestamp = jobs[local_id]
                        request(writer, next_id, "mining.submit",
                                [login, job_id, "00" * en2_size, timestamp, f"{nonce:08x}"])
                        pending.add(next_id)
                        next_id += 1
                        await writer.drain()
    finally:
        for task in (pool_read, miner_read):
            task.cancel()
        await asyncio.gather(pool_read, miner_read, return_exceptions=True)


async def run(args):
    url = urlsplit(args.pool)
    if url.scheme != "stratum+ssl" or not url.hostname or not url.port or url.username or url.password or url.path:
        raise ValueError("expected stratum+ssl://host:port")
    login = args.address_file.read_text().strip()
    if not re.fullmatch(r"neb1[a-z0-9]{58}", login):
        raise ValueError("invalid NEB address in local file")
    ctx = ssl.create_default_context()
    if not ctx.get_ca_certs() and Path("/etc/ssl/cert.pem").is_file():
        ctx.load_verify_locations("/etc/ssl/cert.pem")
    pool = await asyncio.open_connection(url.hostname, url.port, ssl=ctx, server_hostname=url.hostname, limit=262144)
    connected = asyncio.get_running_loop().create_future()

    def connect(reader, writer):
        if not connected.done():
            connected.set_result((reader, writer))
        else:
            writer.close()

    proc = None
    try:
        async with await asyncio.start_server(connect, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            proc = await asyncio.create_subprocess_exec(str(args.miner), "--dataset", str(args.dataset),
                "--node", f"127.0.0.1:{port}", stderr=asyncio.subprocess.PIPE)
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
            await session(pool, worker, login)
        finally:
            worker[1].close()
    finally:
        pool[1].close()
        if proc is not None:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
            finished = asyncio.create_task(proc.communicate())
            try:
                _, log = await asyncio.wait_for(asyncio.shield(finished), 3)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                _, log = await finished
            if b"switching to the fallback scanner" in log:
                raise RuntimeError("miner switched to CPU fallback")
    print("CPU-verified Metal share accepted by pool", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--address-file", type=Path, default=ROOT / ".neb-payout-address")
    parser.add_argument("--dataset", type=Path, default=ROOT / "build/neb-datasets.bin")
    parser.add_argument("--miner", type=Path, default=ROOT / "node/target/release/neb-miner-metal")
    args = parser.parse_args()
    try:
        asyncio.run(asyncio.wait_for(run(args), args.timeout))
    except (OSError, ValueError, RuntimeError, asyncio.IncompleteReadError) as error:
        sys.exit(f"Stratum test failed: {type(error).__name__}: {error}")
