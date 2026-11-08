#!/usr/bin/env python3
"""Post hoc decision-head diagnostics on calibration fit (no test labels): how thinking changes the Qwen3-1.7B head's
argmax behavior, and how often its reasoning closes within the budget on each partition."""

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/mrd"))

from _common import benchmark_instances  # noqa: E402
from mrd.eval.outcomes import is_certified_negative  # noqa: E402
from mrd.io import read_decisions  # noqa: E402


DATASETS = ("abt_buy", "dblp_scholar", "walmart_amazon", "wdc_products")
MODES = {"off": "decoder_head_qwen3_1p7b_off_seed0", "on": "decoder_head_qwen3_1p7b_on_seed0"}
TRACES = ("calib_fit_traces.jsonl", "confirm_dev_test_traces.jsonl")
SYMBOLS = [chr(65 + i) for i in range(26)]
# The answer a trace states near its end ("the answer is B", "so the answer should be NONE"); a heuristic.
STATED = re.compile(r"\b(?:answer|choose|select|match(?:es)?|pick)\b[^.\n]{0,40}?\b(NONE|[A-T])\b")


def read_jsonl(path):
    path = Path(path)
    gz = path.with_name(path.name + ".gz")
    opener = (lambda: gzip.open(gz, "rt", encoding="utf-8")) if gz.exists() and not path.exists() else (lambda: open(path, encoding="utf-8"))
    with opener() as source:
        for line in source:
            yield json.loads(line)


def scaled(probs, temperature):
    logits = np.log(np.clip(np.asarray(probs), 1e-12, 1)) / temperature
    weights = np.exp(logits - logits.max())
    return weights / weights.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=REPO / "data/mrd/public/benchmark")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    runs = args.runs_dir

    traces = {row["instance_id"]: row for path in TRACES for row in read_jsonl(runs / "preds" / "dhq3_on_parts" / path)}
    pools, _ = benchmark_instances(args.benchmark_dir, {"calib_fit", "calib_confirm", "dev", "test"}, list(DATASETS), ("natural", "constructed"))
    partition_of = {pool.instance_id: pool.partition for pool in pools}
    fit = [pool for pool in pools if pool.partition == "calib_fit"]
    decisions = {mode: read_decisions(runs / "preds" / f"{name}.jsonl.gz") for mode, name in MODES.items()}
    reports = {mode: json.loads((runs / "calib" / name / "calibration_report.json").read_text(encoding="utf-8")) for mode, name in MODES.items()}

    def temperature(mode, dataset):
        # the temperature of the policy that decides the pool: Walmart-Amazon uses the pooled policy
        return reports[mode]["temperature"]["pooled" if dataset == "walmart_amazon" else dataset]

    by_dataset = {}
    for dataset in DATASETS:
        members = [pool for pool in fit if pool.dataset == dataset]
        row = {}
        for mode in MODES:
            c = Counter()
            for pool in members:
                acceptable = set(pool.acceptable_indices())
                negative = is_certified_negative(pool)
                if not (acceptable or negative):
                    continue
                probs = scaled(decisions[mode][pool.instance_id].probs, temperature(mode, dataset))
                k = pool.k
                top = min(range(k), key=lambda j: (-probs[j], pool.candidates[j].row_id))
                argmax_none = bool(probs[k] > probs[top])
                if acceptable:
                    c["matchable"] += 1
                    c["forced_top1"] += top in acceptable
                    c["argmax_none"] += argmax_none
                    if top in acceptable and not argmax_none:
                        c["argmax_correct"] += 1
                        c["argmax_correct_conf_ge_0.9"] += bool(probs[top] >= 0.9)
                    if mode == "on":
                        answer = traces[pool.instance_id]["stated_answer"]
                        if answer is not None:
                            c["trace_states_answer"] += 1
                            head = "NONE" if argmax_none else SYMBOLS[top]
                            c["head_follows_trace"] += head == answer
                else:
                    c["negatives"] += 1
                    c["argmax_link_on_negative"] += not argmax_none
            row[mode] = dict(c)
        by_dataset[dataset] = row

    status = {}
    for instance_id, row in traces.items():
        part = partition_of.get(instance_id)
        if part is None:
            raise ValueError(f"trace for unknown pool {instance_id}")
        s = status.setdefault(part, {"closed": 0, "forced": 0, "skipped": 0, "tokens": []})
        s[row["status"]] += 1
        s["tokens"].append(row["reasoning_tokens"])
    for s in status.values():
        tokens = sorted(s.pop("tokens"))
        s.update(pools=len(tokens), median_tokens=statistics.median_low(tokens), p90_tokens=tokens[int(0.9 * len(tokens))], max_tokens=tokens[-1])

    summary = {
        "partition": "calib_fit",
        "arms": MODES,
        "temperature": {mode: reports[mode]["temperature"] for mode in MODES},
        "by_dataset": by_dataset,
        "trace_status": status,
        "stated_answer_pattern": STATED.pattern,
    }
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
