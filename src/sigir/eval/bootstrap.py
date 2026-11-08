"""Paired entity-cluster bootstrap (plan §7).

Resampling unit = entity_group. All perturbation variants of a query share its cluster, so replicas
never count as new independent test entities. Paired differences resample the same clusters for both
systems.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .metrics import compute_metrics
from .outcomes import ScoredInstance


def _clusters(scored: Sequence[ScoredInstance]) -> dict[str, list[int]]:
    cl: dict[str, list[int]] = {}
    for i, s in enumerate(scored):
        cl.setdefault(s.entity_group, []).append(i)
    return cl


def cluster_bootstrap(
    scored: Sequence[ScoredInstance],
    metric_keys: Sequence[str] = ("match_recall", "precision", "n_fmr", "f05", "coverage_auto"),
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    cl = _clusters(scored)
    keys = list(cl.keys())
    idx_lists = [cl[k] for k in keys]
    point = compute_metrics(scored)
    samples: dict[str, list[float]] = {k: [] for k in metric_keys}
    n = len(keys)
    for _ in range(n_boot):
        draw = rng.integers(0, n, n)
        sub = [scored[i] for c in draw for i in idx_lists[c]]
        m = compute_metrics(sub)
        for k in metric_keys:
            samples[k].append(m[k])
    out = {}
    for k in metric_keys:
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        out[k] = {
            "point": point[k],
            "lo": float(np.quantile(arr, alpha / 2)) if arr.size else float("nan"),
            "hi": float(np.quantile(arr, 1 - alpha / 2)) if arr.size else float("nan"),
            "n_clusters": n,
        }
    return out


def paired_cluster_bootstrap(
    scored_a: Sequence[ScoredInstance],
    scored_b: Sequence[ScoredInstance],
    metric_keys: Sequence[str] = ("match_recall", "precision", "n_fmr", "f05"),
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """CI for metric(A) - metric(B), resampling the same entity clusters for both."""
    by_id_b = {s.instance_id: s for s in scored_b}
    pairs = [(a, by_id_b[a.instance_id]) for a in scored_a if a.instance_id in by_id_b]
    cl: dict[str, list[int]] = {}
    for i, (a, _) in enumerate(pairs):
        cl.setdefault(a.entity_group, []).append(i)
    keys = list(cl.keys())
    idx_lists = [cl[k] for k in keys]
    rng = np.random.default_rng(seed)
    pa = compute_metrics([a for a, _ in pairs])
    pb = compute_metrics([b for _, b in pairs])
    diffs: dict[str, list[float]] = {k: [] for k in metric_keys}
    n = len(keys)
    for _ in range(n_boot):
        draw = rng.integers(0, n, n)
        ia = [pairs[i][0] for c in draw for i in idx_lists[c]]
        ib = [pairs[i][1] for c in draw for i in idx_lists[c]]
        ma, mb = compute_metrics(ia), compute_metrics(ib)
        for k in metric_keys:
            diffs[k].append(ma[k] - mb[k])
    out = {}
    for k in metric_keys:
        arr = np.asarray(diffs[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        out[k] = {
            "diff": pa[k] - pb[k],
            "lo": float(np.quantile(arr, alpha / 2)) if arr.size else float("nan"),
            "hi": float(np.quantile(arr, 1 - alpha / 2)) if arr.size else float("nan"),
            "n_pairs": len(pairs),
            "n_clusters": n,
        }
    return out


def seed_variability(metric_by_seed: dict[int, dict[str, float]], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    """Training-seed variability, reported separately from bootstrap intervals."""
    out = {}
    for k in keys:
        vals = np.asarray([m[k] for m in metric_by_seed.values()], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0, "n_seeds": len(vals)}
    return out
