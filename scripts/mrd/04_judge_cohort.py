#!/usr/bin/env python
"""Step 04: the fixed instance file every served judge runs on (Nov 5, 2026).

Served judges cost seconds per request, and unresolved Walmart-Amazon pools (8,500 of the 12,000 evaluation
instances, almost all B->A) enter no recall, N-FMR or routing-utility denominator. Judges therefore run on every
*scorable* instance of the partitions that are evaluated or route (dev, test, transfer): matchable queries and
certified no-match pools, natural and constructed, plus a fixed seeded sample of unresolved pools per partition for
descriptive NONE rates. Calibration partitions are not needed: judges emit final actions, not probabilities.
Recorded deviation from the runbook's "whole cohorts" judge commands.

  python scripts/mrd/04_judge_cohort.py           -> runs/cohorts/judge_eval_v1.jsonl (+ .meta.json)
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from _common import RUNS_DIR, add_common, benchmark_instances, dataset_list, dump_json, log, setup_logging
from mrd.eval.outcomes import cohort_of, is_certified_negative
from mrd.hashing import sha256_file
from mrd.io import write_instances

p = argparse.ArgumentParser()
p.add_argument("--partitions", default="dev,test,transfer")
p.add_argument("--unresolved-sample", type=int, default=200, help="per partition")
p.add_argument("--seed", type=int, default=20261105)
p.add_argument("--out", default=str(RUNS_DIR / "cohorts" / "judge_eval_v1.jsonl"))
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
parts = a.partitions.split(",")
insts, man = benchmark_instances(a.benchmark_dir, set(parts), dataset_list(a), ("natural", "constructed"))
rng = random.Random(a.seed)
chosen, counts = [], Counter()
for part in parts:  # dev first, then test, then transfer
    pi = [i for i in insts if i.partition == part]
    scorable = [i for i in pi if i.is_matchable or is_certified_negative(i)]
    unresolved = sorted((i for i in pi if not (i.is_matchable or is_certified_negative(i))), key=lambda i: i.instance_id)
    sample = rng.sample(unresolved, min(a.unresolved_sample, len(unresolved)))
    for i in scorable + sample:
        i.extra["judge_cohort"] = "scorable" if i in scorable else "unresolved_sample"
        chosen.append(i)
        counts[(part, i.dataset, cohort_of(i), "match" if i.is_matchable else ("neg" if is_certified_negative(i) else "unresolved"))] += 1
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
write_instances(out, chosen)
meta = {"n": len(chosen), "sha256": sha256_file(out), "partitions": parts, "unresolved_sample_per_partition": a.unresolved_sample, "seed": a.seed,
        "counts": {"|".join(k): v for k, v in sorted(counts.items())}}
dump_json(out.with_suffix(".meta.json"), meta)
log.info("judge cohort: %d instances -> %s", len(chosen), out)
for k, v in sorted(counts.items()):
    log.info("  %s %d", k, v)
