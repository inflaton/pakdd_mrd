"""Auxiliary pools per instance: verified distractors and missing gold records.

distractors   candidates from the frozen snapshot beyond the presented K that are verified non-matches
              (only when the acceptable set is exhaustive: complete mappings / clusters / adjudicated pools)
missing_gold  gold records for the query that retrieval did not place in the presented pool
Used by intervention training (add_hard_nonmatch, add_acceptable) and the robustness diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

from ..io import read_jsonl, write_jsonl
from ..schema import Candidate, DecisionInstance
from .build_instances import acceptable_map, _row_lookup
from .retrieval import Snapshot
from .sources import SourceBundle


def build_aux_pools(bundle: SourceBundle, groups: dict[str, str], snapshot: Snapshot, instances: Iterable[DecisionInstance], direction: str, max_distractors: int = 10, max_missing_gold: int = 5) -> list[dict[str, Any]]:
    catalog = bundle.table_b if direction == "A->B" else bundle.table_a
    fc = bundle.fields_b if direction == "A->B" else bundle.fields_a
    c_rows = _row_lookup(catalog, fc)
    acc_map = acceptable_map(bundle, direction)
    out = []
    for inst in instances:
        gold = acc_map.get(inst.query_row_id, set())
        present = {c.row_id for c in inst.candidates}
        distractors = []
        if inst.acceptable_exhaustive:
            for c in snapshot.pools.get(inst.query_row_id, []):
                rid = c["row_id"]
                if rid in present or rid in gold or rid == inst.query_row_id or rid not in c_rows:
                    continue
                distractors.append(asdict(Candidate(rid, c_rows[rid], c["rank"], c["score"], groups.get(rid))))
                if len(distractors) >= max_distractors:
                    break
        missing = [asdict(Candidate(g, c_rows[g], None, None, groups.get(g))) for g in sorted(gold - present) if g in c_rows][:max_missing_gold]
        out.append({"instance_id": inst.instance_id, "distractors": distractors, "missing_gold": missing})
    return out


def load_aux_pools(path) -> dict[str, dict[str, list[Candidate]]]:
    out = {}
    for r in read_jsonl(path):
        out[r["instance_id"]] = {"distractors": [Candidate(**c) for c in r["distractors"]], "missing_gold": [Candidate(**c) for c in r["missing_gold"]]}
    return out


def save_aux_pools(path, rows: list[dict[str, Any]]) -> int:
    return write_jsonl(path, rows)
