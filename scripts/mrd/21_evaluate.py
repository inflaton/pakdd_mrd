#!/usr/bin/env python
"""Step 21: evaluate arms on a partition with frozen policies; per-dataset + macro metrics, cluster-bootstrap CIs,
paired differences, and probabilistic metrics. Refuses the test partition until operating points are frozen.

  python scripts/mrd/21_evaluate.py --partition test \
     --arm set_joint=runs/<pred>/decisions.jsonl:runs/<calib>/policies \
     --arm pairwise=runs/<pred2>/decisions.jsonl:runs/<calib2>/policies \
     --arm judge_12b_rich=runs/<judge>/decisions.jsonl --primary set_joint --baseline pairwise
Arms without a policies dir are scored on their emitted final action (judges, argmax selectors).
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from _common import REPO, add_common, benchmark_instances, dataset_list, dump_json, log, screened_instances_file, setup_logging, cohort_list
from mrd.calibration.calibrate import OperatingPolicy
from mrd.config import load_config
from mrd.eval.bootstrap import cluster_bootstrap, paired_cluster_bootstrap
from mrd.eval.metrics import compute_metrics, macro_summary, per_group_metrics, probabilistic_metrics
from mrd.eval.error_categories import categorize_errors
from mrd.eval.families import family_level_nfmr
from mrd.eval.outcomes import cohort_of, is_certified_negative, score_run
from mrd.io import read_decisions
from mrd.ledger import RunLedger
from mrd.report.tables import results_table
from mrd.schema import Decision

p = argparse.ArgumentParser()
p.add_argument("--partition", default="dev")
p.add_argument("--arm", action="append", required=True, help="name=decisions.jsonl[:policies_dir]")
p.add_argument("--budget", type=float, default=None, help="N-FMR budget to use (default primary from operating_points.yaml)")
p.add_argument("--policy-scope", default="per_dataset", choices=["per_dataset", "pooled"])
p.add_argument("--primary", default=None)
p.add_argument("--baseline", default=None)
p.add_argument("--op-config", default="operating_points.yaml")
p.add_argument("--allow-unfrozen", action="store_true", help="dev-only: evaluate without frozen operating points")
p.add_argument("--n-boot", type=int, default=None)
p.add_argument("--instances-file", default=None, help="evaluate this cohort file's instances of --partition (e.g. the retriever-2 pools) instead of the benchmark's")
p.add_argument("--name", default="evaluate")
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
op = load_config(a.op_config)
if a.partition in ("test", "transfer") and not op.get("frozen") and not a.allow_unfrozen:
    raise SystemExit("operating_points.yaml is not frozen. Run 31_freeze_operating_points.py first (plan §7). Refusing to evaluate on test.")
budget = a.budget if a.budget is not None else float(op["primary_nfmr_budget"])
n_boot = a.n_boot or int(op["bootstrap"]["n_boot"])
if a.instances_file:
    insts = screened_instances_file(a.instances_file, a.benchmark_dir, {a.partition}, dataset_list(a))
    hashes = {"instances_file": a.instances_file, "sha256": hashlib.sha256(Path(a.instances_file).read_bytes()).hexdigest()}
else:
    insts, man = benchmark_instances(a.benchmark_dir, {a.partition}, dataset_list(a), cohort_list(a))
    hashes = {d: v["instances_hash"] for d, v in man["datasets"].items()}
if not insts:
    raise SystemExit(f"no instances for partition {a.partition}")
ledger = RunLedger(a.runs_dir, f"{a.name}_{a.partition}", {"args": vars(a), "budget": budget})
ledger.set("inputs", partition=a.partition, n_instances=len(insts), instances_hashes=hashes)
arms: dict[str, dict[str, Decision]] = {}
thresholds = {}
for spec in a.arm:
    name, rest = spec.split("=", 1)
    dec_path, pol_dir = (rest.split(":", 1) + [None])[:2]
    decs = read_decisions(dec_path)
    if pol_dir:
        out = {}
        for inst in insts:
            d = decs.get(inst.instance_id)
            if d is None or d.probs is None:
                continue
            scope = inst.dataset if a.policy_scope == "per_dataset" else "pooled"
            pp = Path(pol_dir) / f"{scope}_nfmr{budget}.json"
            if not pp.exists():
                pp = Path(pol_dir) / f"pooled_nfmr{budget}.json"
            pol = OperatingPolicy.load(pp)
            thresholds[f"{name}/{scope}"] = {"tau_select": pol.tau_select, "tau_none": pol.tau_none, "temperature": pol.temperature, "path": str(pp)}
            lp = np.log(np.clip(np.asarray(d.probs), 1e-12, 1)) / pol.temperature
            e = np.exp(lp - lp.max())
            out[inst.instance_id] = pol.decide(inst, (e / e.sum()).tolist(), latency_ms=d.latency_ms, input_tokens=d.input_tokens)
        arms[name] = out
    else:
        arms[name] = decs
ledger.set("thresholds", **thresholds)
results, per_ds, cis, scored_all, by_cohort, errors, families = {}, {}, {}, {}, {}, {}, {}
for name, decs in arms.items():
    scored = score_run(insts, decs)
    scored_all[name] = scored
    m = compute_metrics(scored)
    m.update(probabilistic_metrics(insts, decs))
    results[name] = m
    per_ds[name] = per_group_metrics(scored, "dataset")
    per_ds[name]["__macro__"] = macro_summary({k: v for k, v in per_ds[name].items()})
    if len(cohort_list(a)) > 1:
        # natural and constructed no-match cohorts are reported separately (plan §5); the union is the headline only
        # for recall, which does not depend on negatives
        by_cohort.setdefault(name, {})["__all__"] = per_group_metrics(scored, "cohort")
        for ds in sorted({s.dataset for s in scored}):
            by_cohort[name][ds] = per_group_metrics([s for s in scored if s.dataset == ds], "cohort")
    cis[name] = cluster_bootstrap(scored, n_boot=n_boot, seed=int(op["bootstrap"]["seed"]))
    # review note R3: named false-accept categories and family-level N-FMR (negatives are not independent)
    errors[name] = categorize_errors(insts, decs)
    sm_ids = {s.instance_id for s in scored if s.outcome.value == "SM"}
    fam = {}
    for ds in sorted({i.dataset for i in insts}):
        for coh in cohort_list(a):
            negs = [i for i in insts if i.dataset == ds and cohort_of(i) == coh and is_certified_negative(i)]
            if negs:
                fam[f"{ds}/{coh}"] = family_level_nfmr(negs, sm_ids, alpha=float(op["alpha"]))
    families[name] = fam
    log.info("%s: recall %.4f precision %.4f nfmr %.4f f05 %.2f coverage %.3f defer %.3f fail %.3f", name, m["match_recall"], m["precision"], m["n_fmr"], m["f05"], m["coverage_auto"], m["defer_rate"], m["failure_rate"])
paired = None
if a.primary and a.baseline and a.primary in scored_all and a.baseline in scored_all:
    paired = paired_cluster_bootstrap(scored_all[a.primary], scored_all[a.baseline], n_boot=n_boot, seed=int(op["bootstrap"]["seed"]))
    margin = float(op.get("non_inferiority_margin_recall", 0.01))
    paired["non_inferiority_recall"] = {"margin": margin, "lower_ci_of_diff": paired["match_recall"]["lo"], "non_inferior": paired["match_recall"]["lo"] > -margin}
    log.info("paired %s - %s: %s", a.primary, a.baseline, {k: round(v["diff"], 4) for k, v in paired.items() if isinstance(v, dict) and "diff" in v})
out = {"partition": a.partition, "budget": budget, "cohorts": list(cohort_list(a)), "results": results, "per_dataset": per_ds, "by_cohort": by_cohort, "error_categories": errors, "family_level_nfmr": families, "bootstrap_ci": cis, "paired": paired, "thresholds": thresholds, "n_instances": len(insts)}
dump_json(ledger.path("evaluation.json"), out)
md = results_table(results, ci={k: v for k, v in cis.items()})
(ledger.path("evaluation.md")).write_text(f"# {a.name} on {a.partition} (N-FMR budget {budget})\n\n{md}\n", encoding="utf-8")
ledger.set("outputs", evaluation=str(ledger.path("evaluation.json")))
ledger.finish("complete")
print(md)
