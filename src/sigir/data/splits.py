"""Grouped partitions with legacy-exposure handling and an exposure audit (plan §5).

Partitions: train, dev, calib_fit, calib_confirm, test, transfer, legacy_wa_test.
Assignment is a deterministic hash of (entity_group, seed) so re-running never moves a group, and a
new seed does not erase exposure: groups listed in a legacy exposure inventory can only land in
train/dev.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PARTITIONS = ["train", "dev", "calib_fit", "calib_confirm", "test", "transfer"]


def _bucket(group: str, seed: int) -> float:
    h = hashlib.sha256(f"{group}|{seed}".encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def assign_partitions(
    groups: dict[str, str],
    fractions: dict[str, float],
    seed: int,
    exposed_groups: set[str] | None = None,
    exposed_target: tuple[str, ...] = ("train", "dev"),
) -> dict[str, str]:
    """Return entity_group -> partition."""
    total = sum(fractions[p] for p in PARTITIONS)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1, got {total}")
    cum, acc = [], 0.0
    for p in PARTITIONS:
        acc += fractions[p]
        cum.append((p, acc))
    exposed_groups = exposed_groups or set()
    exp_total = sum(fractions[p] for p in exposed_target)
    out: dict[str, str] = {}
    for g in sorted(set(groups.values())):
        u = _bucket(g, seed)
        if g in exposed_groups:
            # redistribute exposed groups only among exposed_target, proportional to their fractions
            acc2 = 0.0
            for p in exposed_target:
                acc2 += fractions[p] / exp_total
                if u <= acc2:
                    out[g] = p
                    break
            continue
        for p, c in cum:
            if u <= c:
                out[g] = p
                break
    return out


class LegacyExposureError(ValueError):
    """The legacy exposure inventory cannot be read or mapped safely. Never fall back to 'nothing exposed'."""


def _title_key(s: Any) -> str:
    # The raw archive keeps original case and HTML entities; DeepMatcher's exp_data is lower-cased and
    # re-tokenised. Comparing alphanumerics only matches all 2,554 Walmart titles (checked 4 Nov 2026).
    t = html.unescape(unicodedata.normalize("NFKC", str(s))).lower()
    return re.sub(r"[^0-9a-z]", "", t)


def load_legacy_exposure(
    path: str | Path,
    table_a: Any,
    raw_table_a: Any,
    *,
    exposed_splits: tuple[str, ...] = ("test", "dev"),
    id_offset: int = -1,
    prefix: str = "A:",
) -> tuple[set[str], dict[str, Any]]:
    """Read the Walmart-Amazon exposure reservation and return (exposed row_ids, summary).

    The reservation lists the raw archive's 1-based ``custom_id`` per split under ``membership``
    (test 1000, dev 100, unused 1454). The earlier study used test and dev queries, so they are
    exposed; ``unused`` stays eligible for evaluation. DeepMatcher ``exp_data`` ids are 0-based, so
    ``row_id = prefix + str(custom_id + id_offset)``. Every record's title is checked across the two
    tables before any id is trusted; any failure raises instead of returning an empty set.

    table_a: DeepMatcher table A as loaded (columns row_id, title).
    raw_table_a: raw archive tableA.csv (columns custom_id, title).
    """
    path = Path(path)
    if not path.exists():
        raise LegacyExposureError(f"legacy exposure file {path} is configured but missing")
    obj = json.load(open(path, encoding="utf-8"))
    membership = obj.get("membership") if isinstance(obj, dict) else None
    if not isinstance(membership, dict) or not all(isinstance(v, list) for v in membership.values()):
        raise LegacyExposureError(f"{path}: expected a 'membership' dict of id lists, found keys {sorted(obj) if isinstance(obj, dict) else type(obj).__name__}")
    missing = [sp for sp in exposed_splits if sp not in membership]
    if missing:
        raise LegacyExposureError(f"{path}: membership has no split(s) {missing}; has {sorted(membership)}")

    split_of: dict[int, str] = {}
    for sp, ids in membership.items():
        for i in ids:
            cid = int(i)
            if cid in split_of:
                raise LegacyExposureError(f"{path}: custom_id {cid} is in both {split_of[cid]} and {sp}")
            split_of[cid] = sp
    n_q = obj.get("query_count")
    if n_q is not None and len(split_of) != int(n_q):
        raise LegacyExposureError(f"{path}: membership covers {len(split_of)} ids but query_count is {n_q}")

    raw_title = {int(c): t for c, t in zip(raw_table_a["custom_id"], raw_table_a["title"])}
    dm_title = dict(zip(table_a["row_id"], table_a["title"]))
    if set(raw_title) != set(split_of):
        raise LegacyExposureError(f"raw tableA custom_ids ({len(raw_title)}) differ from the reservation's ({len(split_of)})")
    mismatches = []
    for cid, t in sorted(raw_title.items()):
        rid = prefix + str(cid + id_offset)
        if rid not in dm_title:
            mismatches.append((cid, rid, t, "<missing in DeepMatcher table A>"))
        elif _title_key(t) != _title_key(dm_title[rid]):
            mismatches.append((cid, rid, t, dm_title[rid]))
    if mismatches:
        head = "; ".join(f"custom_id {c} -> {r}: {a!r} vs {b!r}" for c, r, a, b in mismatches[:5])
        raise LegacyExposureError(f"{len(mismatches)} of {len(raw_title)} Walmart records do not match at custom_id{id_offset:+d}: {head}")

    exposed = {prefix + str(cid + id_offset) for cid, sp in split_of.items() if sp in exposed_splits}
    summary = {
        "file": str(path),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "exposed_splits": list(exposed_splits),
        "id_offset": id_offset,
        "by_split": {sp: len(ids) for sp, ids in membership.items()},
        "n_exposed_records": len(exposed),
        "title_check": f"{len(raw_title)}/{len(raw_title)} titles match (alphanumeric, case-folded)",
    }
    return exposed, summary


def check_exposed_placement(exposed_groups: set[str], part_of_group: dict[str, str], allowed: tuple[str, ...] = ("train", "dev")) -> dict[str, int]:
    """Raise if any legacy-exposed group landed outside `allowed`; return the exposed groups' partition counts."""
    bad = sorted(g for g in exposed_groups if part_of_group.get(g) not in allowed)
    if bad:
        raise LegacyExposureError(f"{len(bad)} legacy-exposed groups outside {allowed}, e.g. {bad[:5]}")
    return dict(Counter(part_of_group[g] for g in exposed_groups))


def exposure_audit(
    instances_by_partition: dict[str, list[dict[str, Any]]],
    groups: dict[str, str],
) -> dict[str, Any]:
    """Audit target-entity and candidate-distractor exposure between train and each evaluation partition.

    instances_by_partition[p] = list of dicts with keys: query_row_id, candidate_row_ids, entity_group.
    """
    train = instances_by_partition.get("train", [])
    train_groups = {i["entity_group"] for i in train}
    train_cand_groups = {groups.get(c) for i in train for c in i["candidate_row_ids"]}
    report: dict[str, Any] = {}
    for p, insts in instances_by_partition.items():
        if p == "train":
            continue
        tgt_overlap = sum(1 for i in insts if i["entity_group"] in train_groups)
        distractor_overlap = 0
        pools = 0
        for i in insts:
            pools += 1
            if any(groups.get(c) in train_groups for c in i["candidate_row_ids"]):
                distractor_overlap += 1
        cand_in_train_distractors = sum(1 for i in insts if groups.get(i["query_row_id"]) in train_cand_groups)
        report[p] = {
            "n": len(insts),
            "target_entity_in_train": tgt_overlap,  # must be 0 for entity-held-out
            "pools_with_train_entity_distractors": distractor_overlap,  # >0 means shared-catalog setting
            "targets_seen_as_train_distractors": cand_in_train_distractors,
        }
    return report


def partition_counts(partition_of_group: dict[str, str]) -> dict[str, int]:
    return dict(Counter(partition_of_group.values()))
