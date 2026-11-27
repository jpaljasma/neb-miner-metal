"""Rank the factorial screen, or check its nominated mask on untouched confirm blocks."""
import csv
from collections import defaultdict
import json
from math import exp, log
import random
from statistics import mean
import sys


def interval(values):
    rng = random.Random(28412)
    sims = sorted(mean(rng.choices(values, k=len(values))) for _ in range(6000))
    return [round(100 * (exp(x) - 1), 3) for x in (sims[300], mean(values), sims[5699])]


def analyze(path):
    with open(path) as stream:
        rows = list(csv.DictReader(stream))
    phase, = {r["phase"] for r in rows}
    blocks = list(range(1, (24 if phase.endswith("confirm") else 12) + 1))
    masks = sorted({int(r["mask"]) for r in rows})
    if ((phase == "screen" and masks != list(range(32))) or
            (phase == "queue" and masks != list(range(9))) or
            (phase == "host" and masks != list(range(8))) or
            (phase == "compact" and masks != list(range(6))) or
            (phase == "tables" and masks != list(range(4))) or
            (phase.endswith("confirm") and (len(masks) != 2 or masks[0] != 0))):
        raise ValueError("incomplete variant set")
    times = {}
    gpu = defaultdict(float)
    wall = defaultdict(float)
    for row in rows:
        key = tuple(int(row[k]) for k in ("block", "slot", "mask"))
        if key in times:
            raise ValueError("duplicate timing row")
        t = float(row["wall_ms"])
        times[key] = log(t)
        gpu[key[-1]] += float(row["gpu_ms"])
        wall[key[-1]] += t
    if len(times) != len(blocks) * len(masks) * 2:
        raise ValueError("incomplete block")
    results = {}
    for mask in masks:
        per_slot = [[times[b, s, 0] - times[b, s, mask] for b in blocks] for s in (0, 1)]
        paired = [mean(v) for v in zip(*per_slot)]
        results[mask] = {"gain_pct_ci90": interval(paired), "slot_gain_pct": [
            round(100*(exp(mean(v))-1), 3) for v in per_slot],
            "gpu_time_pct": round(100*gpu[mask]/wall[mask], 2)}
    selected = max(masks, key=lambda m: results[m]["gain_pct_ci90"][1])
    effects = {}
    if phase in ("screen", "host"):
        factors = (["row_offsets", "planar_factors", "short_mac", "zero_salt", "width_256"]
                   if phase == "screen" else ["header_midstate", "batched_blake3", "single_row_lookup"])
        for bit, name in enumerate(factors):
            effects[name] = interval([mean(times[b,s,m]-times[b,s,m | (1 << bit)]
                for s in (0,1) for m in masks if not (m & (1 << bit))) for b in blocks])
    print(json.dumps({"phase":phase, "selected":selected, "effects_pct_ci90":effects,
                      "variants":results}, indent=2))


if __name__ == "__main__":
    analyze(sys.argv[1])
