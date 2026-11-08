"""Build DecisionInstances from a SourceBundle + entity groups + partitions + frozen retrieval snapshot.

Provenance tiers are kept distinct (plan §5):
  released_link      acceptable set = released links present in the pool; exhaustive only for complete mappings
  adjudicated        acceptable set from the two-reviewer pool adjudication file (overrides released links)
  unresolved         adjudication left the pool unresolved: never a negative
Retrieval misses (gold exists but none in pool) are recorded in `extra.retrieval_miss` and are
NONE-within-pool instances; they are certified negatives only when the acceptable set is exhaustive.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..hashing import pool_hash
from ..io import read_jsonl
from ..schema import Candidate, DecisionInstance, LabelProvenance
from .retrieval import Snapshot
from .sources import SourceBundle


def _row_lookup(df: pd.DataFrame, fields: list[str]) -> dict[str, dict[str, Any]]:
    cols = ["row_id"] + [f for f in fields if f in df.columns]
    return {r["row_id"]: {f: r[f] for f in cols if f != "row_id"} for r in df[cols].to_dict("records")}


def acceptable_map(bundle: SourceBundle, direction: str) -> dict[str, set[str]]:
    """query_row_id -> set of acceptable catalog row_ids for the given direction."""
    out: dict[str, set[str]] = {}
    if direction == "A->B":
        for a, b in zip(bundle.links["a_id"], bundle.links["b_id"]):
            out.setdefault(a, set()).add(b)
    elif direction == "B->A":
        for a, b in zip(bundle.links["a_id"], bundle.links["b_id"]):
            out.setdefault(b, set()).add(a)
    else:
        raise ValueError(direction)
    return out


def load_adjudications(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Adjudication file: JSONL rows {instance_id | (query_row_id, pool_hash), acceptable_row_ids, status}.

    status in {"resolved", "unresolved"}; resolved rows with empty acceptable set certify pool-level NONE.
    """
    if not path or not Path(path).exists():
        return {}
    out = {}
    for r in read_jsonl(path):
        key = r.get("instance_id") or f'{r["query_row_id"]}|{r["pool_hash"]}'
        out[key] = r
    return out


def wdc_role(row_id: str, seed: int) -> str:
    """Deterministic query/catalog role assignment for WDC offers (plan §5)."""
    h = hashlib.sha256(f"{row_id}|{seed}|role".encode()).hexdigest()
    return "query" if int(h[:8], 16) % 2 == 0 else "catalog"


def build_instances(
    bundle: SourceBundle,
    groups: dict[str, str],
    partition_of_group: dict[str, str],
    snapshot: Snapshot,
    *,
    k: int,
    direction: str,
    adjudications: dict[str, dict[str, Any]] | None = None,
    direction_weight: float = 1.0,
    entity_disjoint_pools: bool = False,
    train_groups: set[str] | None = None,
) -> Iterable[DecisionInstance]:
    adjudications = adjudications or {}
    exhaustive_links = bundle.link_completeness in ("complete", "cluster")
    qa, qb = (bundle.table_a, bundle.table_b) if direction == "A->B" else (bundle.table_b, bundle.table_a)
    fq, fc = (bundle.fields_a, bundle.fields_b) if direction == "A->B" else (bundle.fields_b, bundle.fields_a)
    q_rows = _row_lookup(qa, fq)
    c_rows = _row_lookup(qb, fc)
    acc_map = acceptable_map(bundle, direction)
    snapshot_hash = snapshot.hash
    for qid, pool in snapshot.pools.items():
        if qid not in q_rows:
            continue
        g = groups[qid]
        partition = partition_of_group[g]
        cands = []
        for c in pool:
            rid = c["row_id"]
            if rid == qid:
                continue  # self-match (WDC single table)
            if entity_disjoint_pools and train_groups and partition != "train" and groups.get(rid) in train_groups:
                continue  # stronger entity-disjoint setting: drop train-entity distractors
            cands.append(Candidate(row_id=rid, fields=c_rows.get(rid, {}), retrieval_rank=c["rank"], retrieval_score=c["score"], entity_group=groups.get(rid)))
            if len(cands) >= k:
                break
        gold = acc_map.get(qid, set())
        ph = pool_hash(qid, [c.row_id for c in cands])
        inst_id = f"{bundle.dataset}|{direction}|{qid}|{ph[:10]}"
        acceptable = sorted(gold & {c.row_id for c in cands})
        provenance = LabelProvenance.RELEASED_LINK.value
        exhaustive = exhaustive_links
        extra: dict[str, Any] = {
            "gold_total": len(gold),
            "retrieval_miss": bool(gold) and not acceptable,
            "sampling_weight": direction_weight,
            "snapshot_hash": snapshot_hash,
            "identity_policy": bundle.identity_policy,
        }
        adj = adjudications.get(inst_id) or adjudications.get(f"{qid}|{ph}")
        if adj:
            if adj.get("status") == "unresolved":
                provenance = LabelProvenance.UNRESOLVED.value
                acceptable = []
                exhaustive = False
            else:
                provenance = LabelProvenance.ADJUDICATED.value
                acceptable = sorted(set(map(str, adj.get("acceptable_row_ids", []))) & {c.row_id for c in cands})
                exhaustive = True
                extra["adjudication"] = {k2: v for k2, v in adj.items() if k2 not in ("acceptable_row_ids",)}
        inst = DecisionInstance(
            instance_id=inst_id,
            dataset=bundle.dataset,
            direction=direction,
            query_row_id=qid,
            query_fields=q_rows[qid],
            candidates=cands,
            acceptable_row_ids=acceptable,
            acceptable_exhaustive=exhaustive,
            label_provenance=provenance,
            entity_group=g,
            partition=partition,
            pool_hash=ph,
            extra=extra,
        )
        inst.validate()
        yield inst


def instance_inventory(instances: Iterable[DecisionInstance]) -> dict[str, Any]:
    """Counts the plan asks to report separately: entities, query records, variants, matchability, by dataset/direction/partition."""
    rows = []
    for i in instances:
        rows.append({"dataset": i.dataset, "direction": i.direction, "partition": i.partition, "entity_group": i.entity_group, "query": i.query_row_id, "matchable": i.is_matchable, "variant": i.variant_of is not None, "provenance": i.label_provenance, "retrieval_miss": i.extra.get("retrieval_miss", False), "k": i.k})
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    g = df.groupby(["dataset", "direction", "partition"])
    table = g.agg(instances=("query", "size"), unique_entities=("entity_group", "nunique"), unique_queries=("query", "nunique"), variants=("variant", "sum"), matchable=("matchable", "sum"), retrieval_miss=("retrieval_miss", "sum"), mean_k=("k", "mean")).reset_index()
    return {"by_dataset_direction_partition": json.loads(table.to_json(orient="records")), "provenance": df["provenance"].value_counts().to_dict()}
