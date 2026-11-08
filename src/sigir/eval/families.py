"""Negative families and family-level risk bounds (review note R3, Nov 5, 2026).

Certified no-match queries are not independent: WDC's 17 dev negatives are offers from 6 product clusters (six
LaCie 2 TB offers, six RAM suction-cup mounts, ...), with near-identical pools. A query-level Clopper-Pearson bound
counts them as 17 trials. A family groups negatives that share an entity group or whose candidate pools overlap
(row-id Jaccard >= 0.5); the family-level bound treats each family as one trial that fails if any member is falsely
accepted, which is conservative and does not assume independence within a family.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from ..calibration.bounds import clopper_pearson_upper
from ..schema import DecisionInstance

POOL_JACCARD = 0.5


def negative_families(instances: Sequence[DecisionInstance], pool_jaccard: float = POOL_JACCARD) -> dict[str, int]:
    """instance_id -> family id over the given instances (union-find on entity group and pool overlap)."""
    insts = list(instances)
    parent = list(range(len(insts)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    by_group: dict[str, int] = {}
    pools = [frozenset(c.row_id for c in i.candidates) for i in insts]
    by_row: dict[str, list[int]] = defaultdict(list)
    for n, i in enumerate(insts):
        if i.entity_group in by_group:
            union(n, by_group[i.entity_group])
        else:
            by_group[i.entity_group] = n
        for r in pools[n]:
            by_row[r].append(n)
    for n in range(len(insts)):  # only pairs sharing at least one candidate can reach the threshold
        seen = set()
        for r in pools[n]:
            for m in by_row[r]:
                if m > n and m not in seen:
                    seen.add(m)
                    if pools[n] and len(pools[n] & pools[m]) / len(pools[n] | pools[m]) >= pool_jaccard:
                        union(n, m)
    return {i.instance_id: find(n) for n, i in enumerate(insts)}


def family_level_nfmr(negatives: Sequence[DecisionInstance], spurious: set[str], alpha: float = 0.05) -> dict:
    """negatives: certified no-match instances in scope; spurious: ids scored SM. Query- and family-level counts."""
    fam = negative_families(negatives)
    fams = set(fam.values())
    bad = {fam[i.instance_id] for i in negatives if i.instance_id in spurious}
    n, k = len(negatives), sum(1 for i in negatives if i.instance_id in spurious)
    return {
        "n_negatives": n,
        "sm": k,
        "n_families": len(fams),
        "families_with_sm": len(bad),
        "nfmr_family": len(bad) / len(fams) if fams else float("nan"),
        "upper_bound_query": clopper_pearson_upper(k, n, alpha),
        "upper_bound_family": clopper_pearson_upper(len(bad), len(fams), alpha),
        "alpha": alpha,
    }
