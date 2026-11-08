"""Entity grouping with union-find across both source tables (plan §5).

Known equivalent records are grouped BEFORE splitting. Query-ID separation does not imply entity
separation; groups are the unit for splits, exposure audits and bootstrap clusters.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: lexicographically smaller id
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def build_entity_groups(all_row_ids: Iterable[str], links: pd.DataFrame, dataset: str, cluster_of: dict[str, str] | None = None) -> dict[str, str]:
    """Return row_id -> entity_group id. Singletons get their own group."""
    uf = UnionFind()
    ids = list(all_row_ids)
    for rid in ids:
        uf.find(rid)
    for a, b in zip(links["a_id"], links["b_id"]):
        uf.union(str(a), str(b))
    if cluster_of:
        by_cl: dict[str, str] = {}
        for rid, cl in cluster_of.items():
            if cl in by_cl:
                uf.union(rid, by_cl[cl])
            else:
                by_cl[cl] = rid
    groups: dict[str, str] = {}
    for rid in ids:
        groups[rid] = f"{dataset}:g:{uf.find(rid)}"
    return groups


def group_sizes(groups: dict[str, str]) -> pd.Series:
    return pd.Series(list(groups.values())).value_counts()


def inventory(groups: dict[str, str], links: pd.DataFrame) -> dict[str, int]:
    sizes = group_sizes(groups)
    return {
        "records": len(groups),
        "entities": int(sizes.size),
        "singleton_entities": int((sizes == 1).sum()),
        "multi_record_entities": int((sizes > 1).sum()),
        "max_group_size": int(sizes.max()) if sizes.size else 0,
        "positive_links": int(len(links)),
    }
