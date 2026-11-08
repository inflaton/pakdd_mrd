#!/usr/bin/env python
"""Quick scorer for development checks: score decision files on an instance file, by dataset and cohort.

Refuses instance files that contain test or transfer instances unless operating points are frozen (plan §7).
  python scripts/mrd/score_decisions.py --instances runs/cohorts/judge_pilot_dev200.jsonl --decisions name=path.jsonl ...
"""
from __future__ import annotations

import argparse
import json

from _common import dump_json
from mrd.config import load_config
from mrd.eval.metrics import compute_metrics
from mrd.eval.error_categories import categorize_errors
from mrd.eval.outcomes import score_run
from mrd.io import read_decisions, read_instances

p = argparse.ArgumentParser()
p.add_argument("--instances", required=True)
p.add_argument("--decisions", action="append", required=True, help="name=decisions.jsonl")
p.add_argument("--partition", default=None, help="restrict to one partition")
p.add_argument("--out", default=None)
a = p.parse_args()
insts = read_instances(a.instances)
if a.partition:
    insts = [i for i in insts if i.partition == a.partition]
if {i.partition for i in insts} & {"test", "transfer"} and not load_config("operating_points.yaml").get("frozen"):
    raise SystemExit("instances include test/transfer and operating points are not frozen; pass --partition dev")
KEYS = ("n_matchable", "n_certified_nomatch", "match_recall", "precision", "n_fmr", "f05", "false_none_rate", "failure_rate")
out = {}
for spec in a.decisions:
    name, path = spec.split("=", 1)
    decs = read_decisions(path)
    sc = score_run(insts, decs)
    groups = {"ALL": sc} | {f"{ds}": [s for s in sc if s.dataset == ds] for ds in sorted({s.dataset for s in sc})}
    groups |= {f"cohort:{c}": [s for s in sc if s.cohort == c] for c in sorted({s.cohort for s in sc})}
    out[name] = {g: compute_metrics(v) for g, v in groups.items()}
    print(f"== {name}  ({path})")
    print(f"{'group':24s} " + " ".join(f"{k[:10]:>10s}" for k in KEYS))
    for g, m in out[name].items():
        print(f"{g:24s} " + " ".join(f"{m[k]:10.3f}" if isinstance(m[k], float) else f"{m[k]:10}" for k in KEYS))
    out[name]["error_categories"] = categorize_errors(insts, decs)
    print("false accepts by category:", out[name]["error_categories"])
if a.out:
    dump_json(a.out, out)
