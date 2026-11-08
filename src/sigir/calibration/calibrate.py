"""Probability calibration and frozen operating-point selection (plan §4, §7).

Procedure
---------
1. Split calibration data *by entity* into `calib_fit` and `calib_confirm` (done upstream in splits).
2. Fit temperature scaling on `calib_fit` logits (set NLL objective).
3. Search the predefined (tau_select, tau_none) grid on `calib_fit` only: maximise match recall subject to
   N-FMR <= budget and accepted-link precision >= floor.
4. Freeze. Evaluate on `calib_confirm` with one-sided bounds and Bonferroni correction for the two
   constraints and the policy-family size. Never tune on `calib_confirm` or on the test set.
5. If no grid point meets the constraints on `calib_fit`, report infeasibility.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from ..schema import Decision, DecisionInstance, decision_from_probs
from ..eval.metrics import compute_metrics
from ..eval.families import family_level_nfmr
from ..eval.outcomes import is_certified_negative, score_run
from .bounds import bonferroni_alpha, clopper_pearson_upper, independence_caveat


# ------------------------------------------------------------------ temperature scaling
def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: list[np.ndarray], acceptable: list[list[int]], ks: list[int]) -> "TemperatureScaler":
        """logits[i]: length K_i+1 (NONE last). acceptable[i]: acceptable indices (empty -> NONE)."""

        def nll(t: float) -> float:
            tot = 0.0
            for lg, acc, k in zip(logits, acceptable, ks):
                p = _softmax_masked(np.asarray(lg) / t, np.ones(k + 1, bool))
                tot += -math.log(max(p[acc].sum() if acc else p[k], 1e-12))
            return tot / max(1, len(logits))

        res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def apply(self, logits: np.ndarray) -> np.ndarray:
        return _softmax_masked(np.asarray(logits) / self.temperature, np.ones(len(logits), bool))


# ------------------------------------------------------------------ operating policy
@dataclass
class OperatingPolicy:
    tau_select: float
    tau_none: float
    nfmr_budget: float
    precision_floor: float
    min_recall: float
    temperature: float
    fit_metrics: dict
    grid: dict
    feasible: bool
    provenance: dict

    def decide(self, inst: DecisionInstance, probs: Sequence[float], **meta) -> Decision:
        return decision_from_probs(inst, probs, tau_select=self.tau_select, tau_none=self.tau_none, **meta)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "OperatingPolicy":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


def default_grid() -> dict:
    return {"tau_select": [round(x, 3) for x in np.arange(0.50, 0.9951, 0.005)], "tau_none": [round(x, 3) for x in np.arange(0.50, 0.9951, 0.005)]}


def select_policy(
    instances: Sequence[DecisionInstance],
    probs: dict[str, Sequence[float]],
    *,
    nfmr_budget: float,
    precision_floor: float,
    min_recall: float,
    temperature: float = 1.0,
    grid: dict | None = None,
    provenance: dict | None = None,
) -> OperatingPolicy:
    """Grid-search thresholds on calib_fit. Objective: max recall s.t. N-FMR <= budget and precision >= floor."""
    grid = grid or default_grid()
    insts = [i for i in instances if i.instance_id in probs]
    # ---- vectorised precomputation: per instance top prob, p_none, whether top is acceptable
    top_p = np.zeros(len(insts))
    p_none = np.zeros(len(insts))
    top_ok = np.zeros(len(insts), bool)
    matchable = np.zeros(len(insts), bool)
    cert_neg = np.zeros(len(insts), bool)
    for n, inst in enumerate(insts):
        d = decision_from_probs(inst, probs[inst.instance_id])  # argmax with stable tie-break
        top_p[n] = d.max_candidate_prob
        p_none[n] = d.p_none
        acc = set(inst.acceptable_indices())
        top_ok[n] = d.index in acc if d.index >= 0 else False
        matchable[n] = bool(acc)
        cert_neg[n] = is_certified_negative(inst)
    n_match = matchable.sum()
    n_neg = cert_neg.sum()

    def fast_metrics(ts: float, tn: float) -> tuple[float, float, float, float]:
        sel = (top_p >= ts) & (top_p >= p_none)
        none = (~sel) & (p_none >= tn)
        cm = (sel & matchable & top_ok).sum()
        wc = (sel & matchable & ~top_ok).sum()
        sm = (sel & cert_neg).sum()
        cr = (none & cert_neg).sum()
        recall = cm / n_match if n_match else float("nan")
        acc_prec = cm / (cm + wc + sm) if (cm + wc + sm) else float("nan")
        nfmr = sm / n_neg if n_neg else float("nan")
        cov = (cm + wc + sm + cr) / len(insts) if insts else float("nan")
        return recall, acc_prec, nfmr, cov

    best = None
    n_points = 0
    for ts, tn in itertools.product(grid["tau_select"], grid["tau_none"]):
        n_points += 1
        recall, acc_prec, nfmr, cov = fast_metrics(ts, tn)
        ok = (not math.isnan(nfmr) and nfmr <= nfmr_budget) and (not math.isnan(acc_prec) and acc_prec >= precision_floor)
        if not ok:
            continue
        key = (recall, cov, -ts)  # prefer higher recall, then coverage, then looser select
        if best is None or key > best[0]:
            best = (key, ts, tn)
    if best is None:
        ts, tn = 1.1, 1.1  # all-defer: never accept anything automatically
    else:
        _, ts, tn = best
    best_metrics = compute_metrics(
        score_run(insts, {i.instance_id: decision_from_probs(i, probs[i.instance_id], tau_select=ts, tau_none=tn) for i in insts})
    )
    feasible = best is not None and best_metrics["match_recall"] >= min_recall
    return OperatingPolicy(
        tau_select=float(ts),
        tau_none=float(tn),
        nfmr_budget=nfmr_budget,
        precision_floor=precision_floor,
        min_recall=min_recall,
        temperature=temperature,
        fit_metrics=best_metrics,
        grid={"n_points": n_points, **{k: [v[0], v[-1], len(v)] for k, v in grid.items()}},
        feasible=bool(feasible),
        provenance=provenance or {},
    )


def evaluate_frozen_policy(
    policy: OperatingPolicy,
    instances: Sequence[DecisionInstance],
    probs: dict[str, Sequence[float]],
    *,
    alpha: float = 0.05,
    n_constraints: int = 2,
    correction: str = "constraints_x_grid",
) -> dict:
    """Apply the frozen policy to an untouched partition and report one-sided bounds.

    correction: "constraints_x_grid" divides alpha by 2 x grid size (plan §7 as written); "constraints" divides by
    the number of constraints only, valid because the policy was chosen on calib_fit and is confirmed once here."""
    insts = [i for i in instances if i.instance_id in probs]
    decs = {i.instance_id: policy.decide(i, probs[i.instance_id]) for i in insts}
    scored = score_run(insts, decs)
    m = compute_metrics(scored)
    family = policy.grid.get("n_points", 1) if correction == "constraints_x_grid" else 1
    a = bonferroni_alpha(alpha, n_constraints, family)
    n_neg = int(m["n_certified_nomatch"])
    sm = int(m["SM"])
    accepted = int(m["CM"] + m["WC"] + m["SM"])
    err = int(m["WC"] + m["SM"])
    clusters = len({s.entity_group for s in scored})
    return {
        "metrics": m,
        "nfmr_upper_bound": clopper_pearson_upper(sm, n_neg, a),
        "accepted_link_error_upper_bound": clopper_pearson_upper(err, accepted, a),
        "alpha_corrected": a,
        "alpha_correction": correction,
        "constraints_met_point": (not math.isnan(m["n_fmr"]) and m["n_fmr"] <= policy.nfmr_budget)
        and (m["accepted_link_precision"] >= policy.precision_floor),
        "constraints_met_bound": (clopper_pearson_upper(sm, n_neg, a) <= policy.nfmr_budget)
        and (1 - clopper_pearson_upper(err, accepted, a) >= policy.precision_floor),
        "independence_caveat": independence_caveat(len(scored), clusters),
        # review note R3: negatives from one product family / entity group are not independent trials
        "family_level": family_level_nfmr([i for i in insts if is_certified_negative(i)], {s.instance_id for s in scored if s.outcome.value == "SM"}, alpha=a),
    }
