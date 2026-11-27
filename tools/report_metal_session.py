"""Summarize completed local pool event logs, including an explicitly identified handoff."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, median


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def timestamp(event):
    return datetime.fromisoformat(event["utc"]).timestamp()


def percentiles(values):
    if not values:
        return None
    values = sorted(values)
    return {k: round(v, 3) for k, v in {
        "min": values[0], "median": median(values), "p95":values[round(.95*(len(values)-1))],
        "max": values[-1], "mean":mean(values)}.items()}


def phase(folder, handoff, live=False):
    events = read(folder / "events.jsonl")
    first = next(e for e in events if e["event"] == "first_job")
    summary = next((e for e in reversed(events) if e["event"] == "summary"), None)
    if summary is None:
        if not live:
            raise ValueError("session still running; pass --live for a snapshot")
        summary = dict(events[-1],outcome="running")
    stop = (next(e for e in reversed(events) if e["event"] == "paused") if handoff
            else next((e for e in reversed(events) if e["event"] == "mining_deadline"), summary))
    start, end = timestamp(first), timestamp(stop)
    pauses, begin = 0, None
    slot, last = None, start
    windows = []
    for e in events:
        now = timestamp(e)
        if now > end: break
        if e["event"] == "paused": begin = now
        elif e["event"] == "resuming" and begin is not None:
            pauses += now-begin
            begin = None
        if e["event"] in ("job", "miner_final", "miner_reaped", "mining_deadline"):
            if slot is not None:
                windows.append((last, now, slot))
            slot = e.get("slot") if e["event"] == "job" else None
            last = now
    if slot is not None:
        windows.append((last,end,slot))
    if begin is not None:
        pauses += end - begin
    duration = end-start-pauses
    samples = read(folder / "gpu.jsonl") if (folder / "gpu.jsonl").exists() else []
    gpu = [e for e in samples if "device_pct" in e and any(a<=timestamp(e)<b for a,b,_ in windows)]
    counts = Counter(e["event"] for e in events)
    slots = {}
    for i, name in enumerate(("Kepler","TESS")):
        wall = sum(b-a for a,b,s in windows if s==i)
        finals = list({e["connection"]: e for e in events if e["event"] in ("slot_sample","slot_final") and e["slot"]==i}.values())
        evaluated = sum(e["evaluated"] for e in finals)
        screened = sum(e["screened"] for e in finals)
        work_s = sum(e["work_ns"] for e in finals)/1e9
        subset = [e for e in events if e.get("slot")==i]
        accepts = [e for e in subset if e["event"]=="accepted"]
        slots[name] = dict(job_seconds=round(wall,3), submitted=sum(e["event"]=="submitted" for e in subset),
            accepted=len(accepts), rejected_by_code=dict(Counter(e["code"] for e in subset if e["event"]=="rejected")),
            unconfirmed=sum(e["event"]=="unconfirmed" for e in subset),
            acknowledgement_ms=percentiles([e["latency_ms"] for e in accepts]))
        if finals:
            slots[name].update(evaluated=evaluated,screened=screened,worker_seconds=round(work_s,3),
                attempts_per_job_s=round(evaluated/wall) if wall else None,
                attempts_per_worker_s=round(evaluated/work_s) if work_s else None)
        if screened:
            slots[name]["evaluated_pct_of_screened"] = round(100*evaluated/screened,3)
        scoped_gpu = [e["device_pct"] for e in gpu if any(a<=timestamp(e)<b and s==i for a,b,s in windows)]
        slots[name]["gpu_device_pct"] = percentiles(scoped_gpu)
        slots[name]["in_flight_or_unaccounted"] = (slots[name]["submitted"] - len(accepts)
            - sum(slots[name]["rejected_by_code"].values()) - slots[name]["unconfirmed"])
        if accepts and wall and all("difficulty" in e for e in accepts):
            slots[name]["accepted_difficulty_per_job_s_estimate"] = round(sum(float(e["difficulty"]) for e in accepts)/wall)
    finals = list({e["connection"]:e for e in events if e["event"] in ("sample","miner_final")}.values())
    total = sum(e["attempts"] for e in finals)
    edges = [e for e in events if e["event"] in {
        "reconnect","error","rejected","unconfirmed","stale_local_hit","unexpected_response","unknown_method",
        "unexpected_miner_frame","block_candidate"}]
    policy_full = [e for e in events if e["event"]=="miner_message" and
                   ("selected false" in e["message"] or "engine unavailable" in e["message"])]
    submitted = {(e["connection"], e["request"]): e for e in events if e["event"] == "submitted"}
    clean_jobs = [e for e in events if e["event"] == "job" and e.get("clean") and not e.get("rolled")]
    for e in edges:
        if e["event"] != "rejected":
            continue
        sent = submitted.get((e["connection"],e["request"]))
        if sent:
            e["response_ms"] = round(1000*(timestamp(e)-timestamp(sent)),2)
        clean = min((j for j in clean_jobs if j["connection"] == e["connection"]),
                    key=lambda j: abs(timestamp(j)-timestamp(e)), default=None)
        if clean:
            e["ms_after_nearest_clean_notify"] = round(1000*(timestamp(e)-timestamp(clean)),2)
            e["nearest_clean_height"] = clean["height"]
    # Keep output compact; the JSONL has every individual job, sample and event.
    return dict(folder=str(folder),outcome="planned_handoff" if handoff else summary["outcome"],
        first_job_utc=first["utc"], stop_utc=stop["utc"], duration_s=round(duration,3),
        benchmark_pause_s=round(pauses,3), worker=events[0]["worker"], evaluated=total,
        attempts_per_session_s=round(total/duration), events=dict(counts),slots=slots,
        mismatches=sum(e["mismatches"] for e in finals),fallbacks=sum(e["fallbacks"] for e in finals),
        gpu_device_pct=percentiles([e["device_pct"] for e in gpu]),
        acknowledgement_ms=percentiles([e["latency_ms"] for e in events if e["event"]=="accepted"]),
        peak_device_mem_gib=round(max((e["device_bytes"] for e in gpu if e["device_bytes"]),default=0)/2**30,2),
        reaped_all=counts["miner_started"]==counts["miner_reaped"],edge_cases=edges,full_path_decisions=policy_full)


def markdown(reports):
    print("# Metal pool session\n")
    for r in reports:
        counts=r["events"]
        print(f"Worker `{r['worker']}` · {r['outcome']} · {r['duration_s']:.1f} seconds. "
              f"First job {r['first_job_utc']}. End {r['stop_utc']}.\n")
        print("| Slot | Job seconds | Completed tickets | Completed MH/s | Screened nonces | Submitted | Accepted | Rejected | Unconfirmed | Median ACK |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name,s in r["slots"].items():
            tickets=f"{s['evaluated']:,}" if 'evaluated' in s else "—"
            rate=f"{s['attempts_per_job_s']/1e6:.3f}" if s.get("attempts_per_job_s") is not None else "—"
            latency=f"{s['acknowledgement_ms']['median']:.1f} ms" if s["acknowledgement_ms"] else "—"
            screened=f"{s['screened']:,}" if "screened" in s else "—"
            print(f"| {name} | {s['job_seconds']:.2f} | {tickets} | {rate} | {screened} | {s['submitted']} | {s['accepted']} | "
                  f"{sum(s['rejected_by_code'].values())} | {s['unconfirmed']} | {latency} |")
        print(f"\n{r['evaluated']:,} completed tickets in total. {r['mismatches']} verifier mismatches, "
              f"{r['fallbacks']} fallbacks, {counts.get('reconnect',0)} reconnects, "
              f"{counts.get('extranonce_roll',0)} extranonce rolls. "
              f"Miner processes started/reaped: {counts.get('miner_started',0)}/{counts.get('miner_reaped',0)}.\n")
        if r["gpu_device_pct"]:
            g=r["gpu_device_pct"]
            print(f"Device-wide GPU utilization: median {g['median']:.1f}%, p95 {g['p95']:.1f}%. "
                  f"Peak device-wide memory in use: {r['peak_device_mem_gib']:.2f} GiB.\n")
        print("## Errors and exceptional events\n")
        if not r["edge_cases"]: print("None recorded.\n")
        for e in r["edge_cases"]:
            detail=e.get('reason') or e.get('code') or e.get('kind') or ""
            slot={0:"Kepler",1:"TESS"}.get(e.get("slot"),"")
            if "response_ms" in e:
                detail = f"code {detail}, reply in {e['response_ms']} ms"
            if "ms_after_nearest_clean_notify" in e:
                detail += f", Δ to nearest clean job (height {e['nearest_clean_height']}): {e['ms_after_nearest_clean_notify']:+.1f} ms"
            print(f"- {e['utc']} · {e['event']} · connection {e['connection']} · {slot} {detail}")
        print("\nCode 21 means stale or unknown job. Extranonce rolls reuse the TLS connection. "
              "Completed MH/s uses completed tickets divided by the slot’s job time. "
              "Accepted shares measure pool acknowledgements. Balances and payouts follow the pool's accounting.\n")
        for e in r["full_path_decisions"]:
            print(f"- Full-path model decision at {e['utc']}: {e['message']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--first-ended-at-handoff", action="store_true")
    parser.add_argument("--live", action="store_true", help="allow an in-progress session snapshot")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()
    reports = [phase(p,args.first_ended_at_handoff and i==0,args.live) for i,p in enumerate(args.folders)]
    if args.markdown:
        markdown(reports)
        raise SystemExit(0)
    counters = sum((Counter(r["events"]) for r in reports),Counter())
    dur = sum(r["duration_s"] for r in reports)
    attempts = sum(r["evaluated"] for r in reports)
    print(json.dumps(dict(duration_s=round(dur,3),evaluated=attempts,attempts_per_s=round(attempts/dur),
        accepted=counters["accepted"],rejected=counters["rejected"],unconfirmed=counters["unconfirmed"],
        phases=reports),indent=2))
