"""Metrics defined in plan §7.

Precision     = CM / (CM + WC + SM)
Match recall  = CM / (CM + WC + MM)          (every gold-matchable query in the denominator)
N-FMR         = SM / #certified no-match queries
F0.5          on a 0-100 scale
Coverage      automatic decisions (accepted matches + accepted NONE) / all queries, plus the two parts
Also: false-NONE rate on matchable, accepted-link error, overall automatic-decision error,
escalation rate, failure/timeout rate, marginal set NLL, matchability Brier.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from ..schema import Decision, DecisionInstance
from .outcomes import Outcome, ScoredInstance


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    return 0.0 if denom == 0 else (1 + b2) * precision * recall / denom


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else float("nan")


def compute_metrics(scored: Sequence[ScoredInstance], weighted: bool = False) -> dict[str, float]:
    w = (lambda s: s.weight) if weighted else (lambda s: 1.0)
    c: Counter[str] = Counter()
    n_match = n_neg = n_all = 0.0
    n_defer = n_fail = n_escal = n_false_none = 0.0
    for s in scored:
        if s.outcome == Outcome.UNRESOLVED:
            c["UNRESOLVED"] += w(s)
            continue
        n_all += w(s)
        c[s.outcome.value] += w(s)
        if s.matchable:
            n_match += w(s)
            if s.false_none:
                n_false_none += w(s)
        if s.certified_negative:
            n_neg += w(s)
        n_defer += w(s) * s.deferred
        n_fail += w(s) * s.failed
        n_escal += w(s) * s.escalated
    cm, wc, mm, sm, cr = (c[k] for k in ("CM", "WC", "MM", "SM", "CR"))
    precision = _safe_div(cm, cm + wc + sm)
    recall = _safe_div(cm, cm + wc + mm)
    nfmr = _safe_div(sm, n_neg)
    accepted = cm + wc + sm
    out = {
        "n": n_all,
        "n_matchable": n_match,
        "n_certified_nomatch": n_neg,
        "n_unresolved": c["UNRESOLVED"],
        "CM": cm,
        "WC": wc,
        "MM": mm,
        "SM": sm,
        "CR": cr,
        "precision": precision,
        "match_recall": recall,
        "n_fmr": nfmr,
        "f05": 100.0 * f_beta(precision, recall, 0.5) if not (math.isnan(precision) or math.isnan(recall)) else float("nan"),
        "coverage_auto": _safe_div(accepted + cr, n_all),
        "coverage_match": _safe_div(accepted, n_all),
        "coverage_none": _safe_div(cr, n_all),
        "false_none_rate": _safe_div(n_false_none, n_match),
        "accepted_link_error": _safe_div(wc + sm, accepted),
        "accepted_link_precision": _safe_div(cm, accepted),
        "auto_decision_error": _safe_div(wc + sm + n_false_none, accepted + cr + n_false_none),
        "escalation_rate": _safe_div(n_escal, n_all),
        "defer_rate": _safe_div(n_defer, n_all),
        "failure_rate": _safe_div(n_fail, n_all),
    }
    return out


def per_group_metrics(scored: Sequence[ScoredInstance], key: str = "dataset") -> dict[str, dict[str, float]]:
    groups: dict[str, list[ScoredInstance]] = {}
    for s in scored:
        groups.setdefault(getattr(s, key), []).append(s)
    return {g: compute_metrics(v) for g, v in sorted(groups.items())}


def macro_summary(per_dataset: dict[str, dict[str, float]], keys: Iterable[str] = ("precision", "match_recall", "n_fmr", "f05", "coverage_auto")) -> dict[str, float]:
    """Unweighted mean across datasets: the 'clearly defined macro summary' of plan §7."""
    out = {}
    for k in keys:
        vals = [m[k] for m in per_dataset.values() if not math.isnan(m.get(k, float("nan")))]
        out[f"macro_{k}"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------- probabilistic metrics
def marginal_set_nll(inst: DecisionInstance, probs: Sequence[float], eps: float = 1e-12) -> float:
    """-log sum_{j in Y} p[j] for matchable; -log p[NONE] otherwise (plan §4)."""
    k = inst.k
    acc = inst.acceptable_indices()
    if acc:
        return -math.log(max(sum(probs[j] for j in acc), eps))
    return -math.log(max(probs[k], eps))


def matchability_brier(inst: DecisionInstance, probs: Sequence[float]) -> float:
    """Binary Brier score of P(some candidate acceptable) = 1 - p[NONE] against Y non-empty."""
    a = 1.0 - probs[inst.k]
    y = 1.0 if inst.is_matchable else 0.0
    return (a - y) ** 2


def probabilistic_metrics(instances: Sequence[DecisionInstance], decisions: dict[str, Decision]) -> dict[str, float]:
    nll, brier = [], []
    for inst in instances:
        d = decisions.get(inst.instance_id)
        if d is None or d.probs is None:
            continue
        nll.append(marginal_set_nll(inst, d.probs))
        brier.append(matchability_brier(inst, d.probs))
    return {
        "set_nll": float(np.mean(nll)) if nll else float("nan"),
        "matchability_brier": float(np.mean(brier)) if brier else float("nan"),
        "n_scored_prob": len(nll),
    }


def reliability_bins(instances: Sequence[DecisionInstance], decisions: dict[str, Decision], n_bins: int = 10) -> list[dict[str, float]]:
    """Reliability of the matchability probability (1 - p_none) vs Y non-empty; ECE is descriptive only."""
    conf, hit = [], []
    for inst in instances:
        d = decisions.get(inst.instance_id)
        if d is None or d.probs is None:
            continue
        conf.append(1.0 - d.probs[inst.k])
        hit.append(1.0 if inst.is_matchable else 0.0)
    conf_a, hit_a = np.asarray(conf), np.asarray(hit)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf_a >= lo) & (conf_a < hi if hi < 1 else conf_a <= hi)
        if m.sum() == 0:
            rows.append({"lo": lo, "hi": hi, "n": 0, "conf": float("nan"), "acc": float("nan")})
        else:
            rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "conf": float(conf_a[m].mean()), "acc": float(hit_a[m].mean())})
    return rows


def ece_from_bins(rows: list[dict[str, float]]) -> float:
    n = sum(r["n"] for r in rows)
    if n == 0:
        return float("nan")
    return float(sum(r["n"] / n * abs(r["conf"] - r["acc"]) for r in rows if r["n"] > 0))
