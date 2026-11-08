#!/usr/bin/env python3
"""Replay LODO selection breakpoints on source fit, then score fixed policies."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fit_breakpoints import DATASETS, REPO, TRANSFER_ARMS
from _common import COHORT_FILES, benchmark_instances
from sigir.calibration.calibrate import OperatingPolicy, evaluate_frozen_policy, select_policy
from sigir.io import read_decisions


COHORTS = ("natural", "constructed")


def recorded_file(path, base):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(base).as_posix(), "sha256": digest.hexdigest()}


def scaled_probabilities(pools, decisions, temperature):
    scaled = {}
    for pool in pools:
        probs = decisions[pool.instance_id].probs
        if probs is None or len(probs) != pool.k + 1:
            raise ValueError(f"Expected candidate and None probabilities for {pool.instance_id}")
        logits = np.log(np.clip(np.asarray(probs), 1e-12, 1)) / temperature
        weights = np.exp(logits - logits.max())
        scaled[pool.instance_id] = (weights / weights.sum()).tolist()
    return scaled


def expanded_policy(pools, probs, original):
    grid = {
        key: [round(float(value), 12) for value in np.linspace(*original.grid[key])]
        for key in ("tau_select", "tau_none")
    }
    select_thresholds = set(grid["tau_select"])
    min_select = min(select_thresholds)
    for scores in probs.values():
        if len(scores) > 1:
            top = max(scores[:-1])
            if top >= min_select and top >= scores[-1]:
                select_thresholds.add(top)
    grid["tau_select"] = sorted(select_thresholds)
    # Use the original fitting objective, constraints, and tie order.
    return select_policy(
        pools, probs, grid=grid, temperature=original.temperature,
        nfmr_budget=original.nfmr_budget, precision_floor=original.precision_floor,
        min_recall=original.min_recall,
    )


def nullable(value):
    return value if math.isfinite(value) else None


def outcome_summary(metrics, policy, evaluation=None, *, confirmation=False):
    correct, wrong, false_link = (int(metrics[key]) for key in ("CM", "WC", "SM"))
    precision = nullable(metrics["accepted_link_precision"])
    nfmr = nullable(metrics["n_fmr"])
    summary = {
        "correct": correct,
        "wrong": wrong,
        "false_link": false_link,
        "accepted": correct + wrong + false_link,
        "matchable": int(metrics["n_matchable"]),
        "negative": int(metrics["n_certified_nomatch"]),
        "unresolved": int(metrics["n_unresolved"]),
        "recall": nullable(metrics["match_recall"]),
        "precision": precision,
        "nfmr": nfmr,
        "meets_point_targets": precision is not None and nfmr is not None
        and precision >= policy.precision_floor and nfmr <= policy.nfmr_budget,
    }
    if evaluation is not None:
        family = evaluation["family_level"]
        error_upper = evaluation["accepted_link_error_upper_bound"]
        summary.update(
            negative_families=family["n_families"],
            false_link_families=family["families_with_sm"],
            query_nfmr_upper=nullable(evaluation["nfmr_upper_bound"]),
            family_nfmr_upper=nullable(family["upper_bound_family"]),
            accepted_error_upper=nullable(error_upper),
            alpha_per_bound=evaluation["alpha_corrected"],
            meets_confirmation_bounds=confirmation
            and family["upper_bound_family"] <= policy.nfmr_budget
            and 1 - error_upper >= policy.precision_floor,
        )
    return summary


def policy_summary(policy):
    return {
        "tau_select": policy.tau_select,
        "tau_none": policy.tau_none,
        "temperature": policy.temperature,
        "nfmr_budget": policy.nfmr_budget,
        "precision_floor": policy.precision_floor,
        "grid_points": policy.grid["n_points"],
        "fit": outcome_summary(policy.fit_metrics, policy),
    }


def replay(benchmark, runs):
    fit, _ = benchmark_instances(benchmark, {"calib_fit"}, list(DATASETS), COHORTS)
    policies = []
    rows = []
    for target in DATASETS:
        source_fit = [pool for pool in fit if pool.dataset != target]
        for arm in TRANSFER_ARMS:
            name = f"lodo_{target}_{arm}_seed0"
            paths = list(runs.glob(f"*_calibrate_{name}_*/policies/pooled_nfmr0.02.json"))
            if len(paths) != 1:
                raise ValueError(f"Expected one frozen pooled policy for {name}, found {len(paths)}")
            original = OperatingPolicy.load(paths[0])
            prediction_path = runs / "preds_lodo" / f"{name}.jsonl.gz"
            decisions = read_decisions(prediction_path)
            fit_probs = scaled_probabilities(source_fit, decisions, original.temperature)
            replayed_fit = evaluate_frozen_policy(original, source_fit, fit_probs)["metrics"]
            for key in ("CM", "WC", "SM", "CR", "n_matchable", "n_certified_nomatch", "n_unresolved"):
                if replayed_fit[key] != original.fit_metrics[key]:
                    raise ValueError(f"Archived fit count {key} disagrees for {name}")
            expanded = expanded_policy(source_fit, fit_probs, original)
            rows.append({
                "target": target,
                "arm": arm,
                "predictions": recorded_file(prediction_path, runs),
                "original_policy": recorded_file(paths[0], runs),
                "original": policy_summary(original),
                "expanded": policy_summary(expanded),
            })
            policies.append((original, expanded, decisions))

    # Neither confirmation nor transfer labels are used to choose the new thresholds.
    confirm, _ = benchmark_instances(benchmark, {"calib_confirm"}, list(DATASETS), COHORTS)
    transfer, _ = benchmark_instances(benchmark, {"transfer"}, list(DATASETS), COHORTS)
    for row, (original, expanded, decisions) in zip(rows, policies, strict=True):
        source_confirm = [pool for pool in confirm if pool.dataset != row["target"]]
        target_transfer = [pool for pool in transfer if pool.dataset == row["target"]]
        # Confirm uses a two-constraint correction at alpha=.05. Use .10 with the
        # same implementation to obtain descriptive pointwise alpha=.05 on transfer.
        for key, pools, alpha in (("confirm", source_confirm, .05), ("transfer", target_transfer, .10)):
            probs = scaled_probabilities(pools, decisions, original.temperature)
            for version, policy in (("original", original), ("expanded", expanded)):
                evaluation = evaluate_frozen_policy(policy, pools, probs, alpha=alpha, correction="constraints")
                row[version][key] = outcome_summary(
                    evaluation["metrics"], policy, evaluation, confirmation=key == "confirm",
                )

    benchmark_inputs = [benchmark / "manifest.json"]
    for dataset in DATASETS:
        for filename in (*COHORT_FILES.values(), "label_screen.jsonl"):
            path = benchmark / dataset / filename
            if path.exists():
                benchmark_inputs.append(path)
    return {
        "seed": 0,
        "post_hoc": True,
        "cohorts": COHORTS,
        "fit_partition": "calib_fit on source datasets only",
        "confirm_partition": "calib_confirm on source datasets only",
        "transfer_partition": "transfer on held-out dataset only",
        "benchmark_inputs": [recorded_file(path, benchmark) for path in benchmark_inputs],
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=REPO / "data/sigir/public/benchmark")
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--output", type=Path, help="write JSON here instead of stdout")
    args = parser.parse_args()
    encoded = json.dumps(replay(args.benchmark_dir, args.runs_dir), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
