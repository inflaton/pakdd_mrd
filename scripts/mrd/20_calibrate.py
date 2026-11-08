#!/usr/bin/env python
"""Step 20: fit temperature + frozen operating policies on calib_fit; confirm on calib_confirm with one-sided bounds.

Per dataset (no pooling for per-domain risk claims) and a pooled policy, for every N-FMR budget in
operating_points.yaml. Writes policies/<dataset>_<budget>.json and a calibration report.

  python scripts/mrd/20_calibrate.py --decisions runs/<predict_run>/decisions.jsonl --name set_joint_seed0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from _common import REPO, add_common, benchmark_instances, dataset_list, dump_json, log, setup_logging, cohort_list
from mrd.calibration.calibrate import TemperatureScaler, evaluate_frozen_policy, select_policy
from mrd.eval.outcomes import cohort_of, is_certified_negative
from mrd.config import load_config
from mrd.eval.metrics import probabilistic_metrics, reliability_bins, ece_from_bins
from mrd.io import read_probs
from mrd.ledger import RunLedger

p = argparse.ArgumentParser()
p.add_argument("--decisions", required=True, help="decisions.jsonl with probs covering calib_fit and calib_confirm")
p.add_argument("--op-config", default="operating_points.yaml")
p.add_argument("--name", required=True)
p.add_argument("--no-temperature", action="store_true")
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
op = load_config(a.op_config)
probs_raw = read_probs(a.decisions)
fit, man = benchmark_instances(a.benchmark_dir, {"calib_fit"}, dataset_list(a), cohort_list(a))
conf, _ = benchmark_instances(a.benchmark_dir, {"calib_confirm"}, dataset_list(a), cohort_list(a))
fit = [i for i in fit if i.instance_id in probs_raw]
conf = [i for i in conf if i.instance_id in probs_raw]
log.info("calib_fit %d, calib_confirm %d instances with probabilities", len(fit), len(conf))
grid = {k: [round(x, 3) for x in np.arange(v["start"], v["stop"] + 1e-9, v["step"])] for k, v in op["grid"].items()}
ledger = RunLedger(a.runs_dir, f"calibrate_{a.name}", {"op": op, "args": vars(a)})
report = {"temperature": {}, "policies": {}, "operating_points_frozen": op.get("frozen", False), "cohorts": list(cohort_list(a)), "confirm_alpha_correction": op.get("confirm_alpha_correction", "constraints_x_grid")}
pol_dir = ledger.path("policies", "x").parent
scopes = {"pooled": None} | {ds: ds for ds in sorted({i.dataset for i in fit})}
for scope, ds in scopes.items():
    f_s = [i for i in fit if ds is None or i.dataset == ds]
    c_s = [i for i in conf if ds is None or i.dataset == ds]
    if len(f_s) < 20:
        log.warning("scope %s has only %d calib_fit instances; skipping", scope, len(f_s))
        continue
    # temperature scaling on log-probs (equivalent to logits up to a constant)
    T = 1.0
    if not a.no_temperature:
        lg = [np.log(np.clip(np.asarray(probs_raw[i.instance_id]), 1e-12, 1)) for i in f_s]
        T = TemperatureScaler().fit(lg, [i.acceptable_indices() for i in f_s], [i.k for i in f_s]).temperature
    def scaled(i):
        lp = np.log(np.clip(np.asarray(probs_raw[i.instance_id]), 1e-12, 1)) / T
        e = np.exp(lp - lp.max())
        return (e / e.sum()).tolist()
    pf = {i.instance_id: scaled(i) for i in f_s}
    pc = {i.instance_id: scaled(i) for i in c_s}
    report["temperature"][scope] = T
    n_neg_fit = sum(1 for i in f_s if is_certified_negative(i))
    if ds is not None and n_neg_fit == 0:
        # no certified negatives (Walmart-Amazon without adjudication): no per-dataset risk policy; 21_evaluate falls
        # back to the pooled policy and no per-dataset N-FMR claim is made (plan §12)
        report["policies"][f"{scope}@fallback"] = {"fallback": "pooled", "reason": "no certified no-match instances in calib_fit"}
        log.warning("%s: no certified negatives in calib_fit; per-dataset policy not written (pooled fallback)", scope)
        continue
    for b in op["nfmr_budgets"]:
        pol = select_policy(f_s, pf, nfmr_budget=float(b), precision_floor=float(op["accepted_link_precision_floor"]), min_recall=float(op["min_useful_recall"]), temperature=T, grid=grid, provenance={"decisions": a.decisions, "scope": scope, "op_config_hash": ledger.manifest["config_hash"], "n_fit": len(f_s)})
        path = pol_dir / f"{scope}_nfmr{b}.json"
        pol.save(path)
        corr = op.get("confirm_alpha_correction", "constraints_x_grid")
        ev = evaluate_frozen_policy(pol, c_s, pc, alpha=float(op["alpha"]), correction=corr) if c_s else {}
        if c_s and len(cohort_list(a)) > 1:
            ev["by_cohort"] = {c: evaluate_frozen_policy(pol, [i for i in c_s if cohort_of(i) == c], pc, alpha=float(op["alpha"]), correction=corr) for c in cohort_list(a)}
        report["policies"][f"{scope}@{b}"] = {"path": str(path), "tau_select": pol.tau_select, "tau_none": pol.tau_none, "feasible_on_fit": pol.feasible, "n_certified_negatives_fit": n_neg_fit, "fit": {k: pol.fit_metrics[k] for k in ("match_recall", "precision", "n_fmr", "coverage_auto", "accepted_link_precision")}, "confirm": ev}
        log.info("%s budget %.3f: tau_sel %.3f tau_none %.3f feasible=%s fit recall %.3f | confirm nfmr %.4f (UB %.4f) recall %.3f", scope, b, pol.tau_select, pol.tau_none, pol.feasible, pol.fit_metrics["match_recall"], ev.get("metrics", {}).get("n_fmr", float("nan")), ev.get("nfmr_upper_bound", float("nan")), ev.get("metrics", {}).get("match_recall", float("nan")))
    rb = reliability_bins(c_s, {i.instance_id: __import__("mrd.schema", fromlist=["decision_from_probs"]).decision_from_probs(i, pc[i.instance_id]) for i in c_s})
    report.setdefault("reliability", {})[scope] = {"bins": rb, "ece_descriptive": ece_from_bins(rb)}
dump_json(ledger.path("calibration_report.json"), report)
ledger.set("outputs", report=str(ledger.path("calibration_report.json")), policies_dir=str(pol_dir))
ledger.set("thresholds", source="calib_fit grid search", confirm_partition="calib_confirm", frozen_operating_points=op.get("frozen", False))
ledger.finish("complete")
log.info("calibration report: %s", ledger.path("calibration_report.json"))
