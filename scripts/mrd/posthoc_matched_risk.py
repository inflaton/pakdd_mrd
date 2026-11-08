#!/usr/bin/env python
"""Fit a pooled compact policy at each LLM's development risks, then score held-out pools.

Run `fit` before `score`. This post hoc experiment does not write any locked
evaluation, calibration, or manuscript artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from _common import REPO, benchmark_instances, load_config
from mrd.calibration.calibrate import OperatingPolicy, TemperatureScaler, evaluate_frozen_policy, select_policy
from mrd.eval.bootstrap import paired_cluster_bootstrap
from mrd.eval.metrics import compute_metrics, per_group_metrics
from mrd.eval.outcomes import cohort_of, is_certified_negative, score_run
from mrd.hashing import sha256_file, sha256_obj
from mrd.io import read_decisions, read_probs

RUNS = REPO / "runs"
ARM = "set_joint_interv_seed0"
COMPACT = RUNS / "preds" / f"{ARM}.jsonl.gz"
POOLED_POLICY = RUNS / "calib" / ARM / "policies" / "pooled_nfmr0.02.json"
COMEM_TEST = RUNS / "20261107T160617Z_comem_frozen_test" / "decisions.jsonl.gz"
JUDGES = {
    name: RUNS / "judges" / f"{name}.jsonl.gz"
    for name in ("judge_120b_rich", "judge_12b_minimal_json", "judge_12b_rich", "judge_12b_symbols")
}
COHORTS = ("natural", "constructed")
CONTRAST_METRICS = ("match_recall", "precision", "n_fmr", "defer_rate", "false_none_rate")


def scorable(instances, expected):
    pools = [i for i in instances if i.is_matchable or is_certified_negative(i)]
    counts = (sum(i.is_matchable for i in pools),) + tuple(
        sum(is_certified_negative(i) and cohort_of(i) == cohort for i in pools)
        for cohort in COHORTS
    )
    if counts != expected or len(pools) != len({i.instance_id for i in pools}):
        raise ValueError(f"Unexpected pool counts or duplicate IDs: {counts}, expected {expected}")
    return pools


def read_covered_decisions(path, pools):
    decisions = read_decisions(path)
    missing = sum(i.instance_id not in decisions for i in pools)
    if missing:
        raise ValueError(f"{path.name} is missing {missing} scorable pools")
    return decisions


def scaled_probs(pools, raw, temperature):
    scaler = TemperatureScaler(temperature)
    return {
        i.instance_id: scaler.apply(np.log(np.clip(raw[i.instance_id], 1e-12, 1))).tolist()
        for i in pools
    }


def write_new(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return None if isinstance(value, float) and not math.isfinite(value) else value

    payload = json.dumps(clean(data), allow_nan=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def partition(benchmark_dir, name):
    return benchmark_instances(benchmark_dir, {name}, None, COHORTS)[0]


def benchmark_pool_files_sha256(benchmark_dir):
    """Bind replay to rebuilt pools and label screens independently of manifest paths."""
    manifest = json.loads((benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    pool_files = ["instances.jsonl", "instances_constructed.jsonl", "label_screen.jsonl"]
    return sha256_obj([
        (dataset, {name: sha256_file(benchmark_dir / dataset / name)
                   for name in pool_files if (benchmark_dir / dataset / name).is_file()})
        for dataset in manifest["datasets"]
    ])


def fit(args, config):
    dev = scorable(partition(args.benchmark_dir, "dev"), (811, 37, 581))
    fit_pools = partition(args.benchmark_dir, "calib_fit")
    raw = read_probs(COMPACT)
    temperature = OperatingPolicy.load(POOLED_POLICY).temperature
    probs = scaled_probs(fit_pools, raw, temperature)
    grid = {
        name: [round(x, 3) for x in np.arange(spec["start"], spec["stop"] + 1e-9, spec["step"])]
        for name, spec in config["grid"].items()
    }
    paths = JUDGES | ({"comem": args.comem_dev} if args.comem_dev else {})
    comparisons = {}
    for name, path in paths.items():
        dev_metrics = compute_metrics(score_run(dev, read_covered_decisions(path, dev)))
        policy = select_policy(
            fit_pools, probs,
            nfmr_budget=dev_metrics["n_fmr"],
            precision_floor=dev_metrics["precision"],
            min_recall=config["min_useful_recall"],
            temperature=temperature,
            grid=grid,
            provenance={"targets": "baseline dev", "thresholds": "calib_fit", "scope": "pooled"},
        )
        comparisons[name] = {"dev_metrics": dev_metrics, "dev_predictions_sha256": sha256_file(path),
                             "frozen_policy": asdict(policy)}

    write_new(args.output_dir / "frozen_fit.json", {
        "analysis": "Post hoc, separate from the primary locked comparison",
        "compact_arm": ARM,
        "compact_predictions_sha256": sha256_file(COMPACT),
        "benchmark_pool_files_sha256": benchmark_pool_files_sha256(args.benchmark_dir),
        "dev_pool_ids_sha256": sha256_obj(sorted(i.instance_id for i in dev)),
        "calib_fit_pool_ids_sha256": sha256_obj(sorted(i.instance_id for i in fit_pools)),
        "comparisons": comparisons,
    })


def score(args, config):
    fit_file = args.output_dir / "frozen_fit.json"
    frozen = json.loads(fit_file.read_text(encoding="utf-8"))
    if frozen["compact_predictions_sha256"] != sha256_file(COMPACT):
        raise ValueError("Compact prediction archive differs from the frozen fit")
    if frozen["benchmark_pool_files_sha256"] != benchmark_pool_files_sha256(args.benchmark_dir):
        raise ValueError("Benchmark pools or label screens differ from the frozen fit")
    policies = {k: OperatingPolicy(**v["frozen_policy"]) for k, v in frozen["comparisons"].items()}
    temperatures = {p.temperature for p in policies.values()}
    temperature, = temperatures
    raw = read_probs(COMPACT)

    confirm = partition(args.benchmark_dir, "calib_confirm")
    confirm_probs = scaled_probs(confirm, raw, temperature)
    confirmations = {
        name: evaluate_frozen_policy(
            policy, confirm, confirm_probs, alpha=config["alpha"],
            correction=config["confirm_alpha_correction"],
        )
        for name, policy in policies.items()
    }
    freeze_sha256 = sha256_file(fit_file)
    write_new(args.output_dir / "confirmation.json", {
        "partition": "calib_confirm", "frozen_fit_sha256": freeze_sha256, "results": confirmations,
    })

    # Every frozen policy is reported, regardless of its confirmation outcome.
    test = scorable(partition(args.benchmark_dir, "test"), (1095, 42, 929))
    probs = scaled_probs(test, raw, temperature)
    paths = JUDGES | {"comem": COMEM_TEST}
    comparisons = {}
    for name, policy in policies.items():
        path = paths[name]
        baseline = score_run(test, read_covered_decisions(path, test))
        ours = score_run(test, {i.instance_id: policy.decide(i, probs[i.instance_id]) for i in test})
        bm, om = compute_metrics(baseline), compute_metrics(ours)
        comparisons[name] = {
            "baseline_predictions_sha256": sha256_file(path),
            "baseline": bm,
            "ours": om,
            "baseline_by_cohort": per_group_metrics(baseline, "cohort"),
            "ours_by_cohort": per_group_metrics(ours, "cohort"),
            "ours_meets_dev_targets": om["n_fmr"] <= policy.nfmr_budget and om["precision"] >= policy.precision_floor,
            "ours_meets_both_realized_baseline_risks": om["n_fmr"] <= bm["n_fmr"] and om["precision"] >= bm["precision"],
            "paired_ours_minus_baseline": paired_cluster_bootstrap(
                ours, baseline, metric_keys=CONTRAST_METRICS, **config["bootstrap"],
            ),
        }
    write_new(args.output_dir / "heldout_test.json", {
        "analysis": "Post hoc evaluation of off-test frozen policies on the existing locked benchmark pools",
        "partition": "test",
        "frozen_fit_sha256": freeze_sha256,
        "pool_ids_sha256": sha256_obj(sorted(i.instance_id for i in test)),
        "comparisons": comparisons,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "score"))
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comem-dev", type=Path, help="off-test ComEM decisions for the same development pools")
    args = parser.parse_args()
    config = load_config("operating_points.yaml")
    if not config["frozen"]:
        parser.error("the primary operating-point configuration must be frozen")
    (fit if args.phase == "fit" else score)(args, config)


if __name__ == "__main__":
    main()
