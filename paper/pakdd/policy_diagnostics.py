#!/usr/bin/env python3
"""Reproduce the archived-prediction and frozen-pool diagnostics for the PAKDD paper."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/mrd"))

from _common import benchmark_instances
from mrd.calibration import OperatingPolicy, TemperatureScaler, evaluate_frozen_policy, select_policy
from mrd.data.duplicates import dblp_probable_duplicate
from mrd.eval.outcomes import cohort_of, is_certified_negative
from mrd.io import read_decisions
from mrd.schema import decision_from_probs

DATASETS = ["abt_buy", "dblp_scholar", "walmart_amazon", "wdc_products"]
SET_ARMS = ("set_joint_clean_seed0", "set_joint_interv_seed0")
LOCKED_RUN = "20261107T130412Z_test_compare_b0.02_test_0886bf"


def scaled_scores(pool, decision, temperature):
    if decision.probs is None or len(decision.probs) != pool.k + 1:
        raise ValueError(f"Expected candidate and None probabilities for {pool.instance_id}")
    logits = np.log(np.clip(decision.probs, 1e-12, 1))
    return TemperatureScaler(temperature).apply(logits)


def top_candidate(pool, scores, candidates):
    return min(candidates, key=lambda j: (-scores[j], pool.candidates[j].row_id))


def recall_drop(pools, predictions, locked):
    counts = dict.fromkeys(
        ("matchable", "clean_correct", "interv_correct", "clean_top1", "interv_top1",
         "lost", "gained", "changed_correct_top1_both", "lost_defer", "lost_none",
         "lost_defer_below_select_only"),
        0,
    )
    lost_datasets, gained_datasets = Counter(), Counter()
    for pool in pools:
        if not pool.is_matchable:
            continue
        decisions, top1 = [], []
        for arm in SET_ARMS:
            policy = locked["thresholds"][f"{arm}/{pool.dataset}"]
            scores = scaled_scores(pool, predictions[arm][pool.instance_id], policy["temperature"])
            top = top_candidate(pool, scores, range(pool.k))
            top1.append(pool.candidates[top].row_id in pool.acceptable_row_ids)
            decisions.append(decision_from_probs(
                pool, scores, tau_select=policy["tau_select"], tau_none=policy["tau_none"],
            ))
        clean, interv = (dec.row_id in pool.acceptable_row_ids for dec in decisions)
        counts["matchable"] += 1
        counts["clean_correct"] += clean
        counts["interv_correct"] += interv
        counts["clean_top1"] += top1[0]
        counts["interv_top1"] += top1[1]
        if clean != interv:
            counts["changed_correct_top1_both"] += all(top1)
        if clean and not interv:
            counts["lost"] += 1
            lost_datasets[pool.dataset] += 1
            after = decisions[1]
            counts["lost_defer"] += after.kind.value == "defer"
            counts["lost_none"] += after.kind.value == "none"
            counts["lost_defer_below_select_only"] += (
                after.kind.value == "defer" and after.max_candidate_prob >= after.p_none
            )
        if interv and not clean:
            counts["gained"] += 1
            gained_datasets[pool.dataset] += 1
    counts.update(lost_by_dataset=dict(sorted(lost_datasets.items())),
                  gained_by_dataset=dict(sorted(gained_datasets.items())))
    for arm, key in zip(SET_ARMS, ("clean_correct", "interv_correct")):
        metrics = locked["results"][arm]
        if (counts[key], counts["matchable"]) != (metrics["CM"], metrics["n_matchable"]):
            raise ValueError(f"Locked counts disagree with evaluation for {arm}")
    return counts


def group_scores(pool, scores):
    """Use candidate fields alone. Score each complete-link group's highest-scoring row."""
    groups = []
    order = sorted(range(pool.k), key=lambda j: pool.candidates[j].row_id)
    for j in order:
        for group in groups:
            if all(dblp_probable_duplicate(pool.candidates[j].fields, pool.candidates[g].fields)
                   for g in group):
                group.append(j)
                break
        else:
            groups.append([j])
    aggregated = np.zeros_like(scores)
    aggregated[-1] = scores[-1]
    for group in groups:
        representative = top_candidate(pool, scores, group)
        aggregated[representative] = scores[group].sum()
    return aggregated


def grouping(pools, predictions, frozen):
    fit = [p for p in pools if p.dataset == "dblp_scholar" and p.partition == "calib_fit"]
    confirm = [p for p in pools if p.dataset == "dblp_scholar" and p.partition == "calib_confirm"]
    singleton = {p.instance_id: scaled_scores(p, predictions[p.instance_id], frozen.temperature)
                 for p in fit + confirm}
    grouped = {p.instance_id: group_scores(p, singleton[p.instance_id]) for p in fit + confirm}
    grid = {}
    for name in ("tau_select", "tau_none"):
        lower, upper, size = frozen.grid[name]
        grid[name] = [round(float(t), 12) for t in np.linspace(lower, upper, size)]
    output = {}
    for name, scores in (("ungrouped", singleton), ("grouped", grouped)):
        policy = select_policy(
            fit, scores, nfmr_budget=frozen.nfmr_budget, precision_floor=frozen.precision_floor,
            min_recall=frozen.min_recall, temperature=frozen.temperature, grid=grid,
        )
        if name == "ungrouped" and (
            policy.tau_select, policy.tau_none, policy.fit_metrics["CM"], policy.fit_metrics["SM"]
        ) != (frozen.tau_select, frozen.tau_none, frozen.fit_metrics["CM"], frozen.fit_metrics["SM"]):
            raise ValueError("Singleton grid replay disagrees with the archived fit")
        output[name] = {"tau_select": policy.tau_select, "tau_none": policy.tau_none}
        for split, subset in (("fit", fit), ("confirm", confirm)):
            result = evaluate_frozen_policy(policy, subset, scores, correction="constraints")
            metrics, family = result["metrics"], result["family_level"]
            output[name][split] = {
                "matchable": int(metrics["n_matchable"]),
                "negatives": int(metrics["n_certified_nomatch"]),
                "correct": int(metrics["CM"]),
                "accepted": int(metrics["CM"] + metrics["WC"] + metrics["SM"]),
                "false_links": int(metrics["SM"]),
                "families": family["n_families"],
                "affected_families": family["families_with_sm"],
                "upper_link_error": result["accepted_link_error_upper_bound"],
                "upper_family": family["upper_bound_family"],
            }
    return output


def backfill(pools):
    natural = {p.instance_id: p for p in pools if cohort_of(p) == "natural"}
    constructed = [p for p in pools if cohort_of(p) == "constructed"]
    pairs, score_shifts, rank_shifts = 0, 0, 0
    for pool in constructed:
        original = natural[pool.variant_of]
        original_by_row = {c.row_id: c for c in original.candidates}
        orig_max = max(c.retrieval_score for c in original.candidates)
        pool_max = max(c.retrieval_score for c in pool.candidates)
        shared = [(original_by_row[c.row_id], c) for c in pool.candidates
                  if c.row_id in original_by_row]
        pairs += 1
        if any(old.retrieval_score != new.retrieval_score for old, new in shared):
            raise ValueError(f"Backfill changed the raw score of a retained row in {pool.instance_id}")
        score_shifts += any(old.retrieval_score / orig_max != new.retrieval_score / pool_max
                            for old, new in shared)
        rank_shifts += any(old.retrieval_rank != new.retrieval_rank for old, new in shared)
    output = {"paired": {"pools": pairs, "relative_score_shifts": score_shifts,
                         "rank_shifts": rank_shifts}}
    for dataset in ("dblp_scholar", "wdc_products"):
        output[dataset] = {}
        for cohort in ("natural", "constructed"):
            tails = []
            for pool in pools:
                if pool.dataset == dataset and cohort_of(pool) == cohort and is_certified_negative(pool):
                    scores = [c.retrieval_score for c in pool.candidates]
                    tails.append(min(scores) / max(scores))
            output[dataset][f"{cohort}_count"] = len(tails)
            output[dataset][f"{cohort}_median_tail"] = median(tails)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path, default=REPO / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pools, _ = benchmark_instances(args.benchmark_dir, {"calib_fit", "calib_confirm", "test"},
                                   DATASETS, ("natural", "constructed"))
    test = [p for p in pools if p.partition == "test"]
    predictions = {arm: read_decisions(args.runs_dir / f"preds/{arm}.jsonl.gz") for arm in SET_ARMS}
    locked = json.loads((args.runs_dir / LOCKED_RUN / "evaluation.json").read_text(encoding="utf-8"))
    frozen = OperatingPolicy.load(
        args.runs_dir / f"calib/{SET_ARMS[1]}/policies/dblp_scholar_nfmr0.02.json"
    )
    output = {
        "recall_drop": recall_drop(test, predictions, locked),
        "grouping": grouping(pools, predictions[SET_ARMS[1]], frozen),
        "backfill": backfill(test),
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
