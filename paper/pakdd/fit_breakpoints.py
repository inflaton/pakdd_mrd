#!/usr/bin/env python3
"""Replay the frozen selection grid and exact score breakpoints on fit pools only."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sigir"))

from _common import benchmark_instances
from sigir.eval.outcomes import is_certified_negative
from sigir.io import read_decisions


DATASETS = ("abt_buy", "dblp_scholar", "walmart_amazon", "wdc_products")
WDC_ARMS = ("pairwise_clean", "pairwise_interv", "set_joint_clean", "set_joint_interv")
TRANSFER_ARMS = ("pairwise_clean", "set_joint_interv")


def replay(pools, prediction_path, policy_path):
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    decisions = read_decisions(prediction_path)
    temperature = policy["temperature"]
    # Equal scores enter together, since the selection gate includes equality.
    by_score = {}
    n_matchable = n_negative = 0
    for pool in pools:
        negative = is_certified_negative(pool)
        if not (pool.is_matchable or negative):
            continue
        n_matchable += pool.is_matchable
        n_negative += negative
        probs = decisions[pool.instance_id].probs
        if probs is None or len(probs) != pool.k + 1:
            raise ValueError(f"Expected candidate and None probabilities for {pool.instance_id}")
        logits = np.log(np.clip(np.asarray(probs), 1e-12, 1)) / temperature
        weights = np.exp(logits - logits.max())
        scores = weights / weights.sum()
        top = min(range(pool.k), key=lambda j: (-scores[j], pool.candidates[j].row_id))
        if scores[top] < scores[-1]:
            continue
        counts = by_score.setdefault(float(scores[top]), [0, 0, 0])
        counts[0] += 1
        counts[1] += pool.candidates[top].row_id in pool.acceptable_row_ids
        counts[2] += negative

    frozen_fit = policy["fit_metrics"]
    if (n_matchable, n_negative) != (frozen_fit["n_matchable"], frozen_fit["n_certified_nomatch"]):
        raise ValueError(f"Fit cohort counts disagree with {policy_path}")

    def metrics(threshold, accepted, correct, spurious):
        return {
            "threshold": threshold,
            "accepted": accepted,
            "correct": correct,
            "spurious": spurious,
            "recall": correct / n_matchable,
            "precision": correct / accepted if accepted else None,
            "nfmr": spurious / n_negative,
        }

    def at_threshold(threshold):
        rows = [counts for score, counts in by_score.items() if score >= threshold]
        return metrics(threshold, *(sum(row[i] for row in rows) for i in range(3)))

    frozen = at_threshold(policy["tau_select"])
    if (frozen["correct"], frozen["accepted"], frozen["spurious"]) != (
        frozen_fit["CM"], frozen_fit["CM"] + frozen_fit["WC"] + frozen_fit["SM"], frozen_fit["SM"],
    ):
        raise ValueError(f"Frozen cutoff does not reproduce the fit counts in {policy_path}")

    lower, upper, size = policy["grid"]["tau_select"]
    grid = [at_threshold(round(float(t), 12)) for t in np.linspace(lower, upper, size)]
    max_grid_precision = max((row["precision"] for row in grid if row["accepted"]), default=None)

    def qualifies(row):
        return (
            row["accepted"] > 0
            and row["precision"] >= policy["precision_floor"]
            and row["nfmr"] <= policy["nfmr_budget"]
        )

    accepted = correct = spurious = 0
    qualifying = []
    for threshold, (links, matches, false_links) in sorted(by_score.items(), reverse=True):
        accepted += links
        correct += matches
        spurious += false_links
        row = metrics(threshold, accepted, correct, spurious)
        if qualifies(row):
            qualifying.append(row)
    best = max(qualifying, key=lambda row: row["correct"], default=None)
    in_range = [row for row in qualifying if lower <= row["threshold"] <= upper]
    in_range.extend(row for row in (grid[0], grid[-1]) if qualifies(row))
    best_in_range = max(in_range, key=lambda row: row["correct"], default=None)
    return {
        "n_matchable": n_matchable,
        "n_negative": n_negative,
        "temperature": temperature,
        "frozen_tau_select": policy["tau_select"],
        "grid_max_precision": max_grid_precision,
        "best": best,
        "best_in_range": best_in_range,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=REPO / "data/sigir/public/benchmark")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()

    pools, _ = benchmark_instances(args.benchmark_dir, {"calib_fit"}, list(DATASETS), ("natural", "constructed"))
    wdc = [pool for pool in pools if pool.dataset == "wdc_products"]
    runs = args.runs_dir
    summary = {
        "partition": "calib_fit",
        "budget": 0.02,
        "precision_floor": 0.98,
        "wdc": {},
        "transfer": {},
    }
    for arm in WDC_ARMS:
        name = f"{arm}_seed0"
        summary["wdc"][arm] = replay(
            wdc, runs / "preds" / f"{name}.jsonl.gz",
            runs / "calib" / name / "policies/wdc_products_nfmr0.02.json",
        )
    for target in DATASETS:
        source = [pool for pool in pools if pool.dataset != target]
        summary["transfer"][target] = {}
        for arm in TRANSFER_ARMS:
            name = f"lodo_{target}_{arm}_seed0"
            policies = list(runs.glob(f"*_calibrate_{name}_*/policies/pooled_nfmr0.02.json"))
            if len(policies) != 1:
                raise ValueError(f"Expected one frozen pooled policy for {name}, found {len(policies)}")
            summary["transfer"][target][arm] = replay(
                source, runs / "preds_lodo" / f"{name}.jsonl.gz", policies[0],
            )
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
