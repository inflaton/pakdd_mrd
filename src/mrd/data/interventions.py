"""Candidate-set interventions (plan §8).

Families
--------
permutation            reorder candidates; targets unchanged; always eligible (even for weak labels)
truncate               keep top-k' (5/10/20/50 study); disclose whether acceptable candidates remain
add_hard_nonmatch      append verified non-matching records (eligible only if the non-match fact is verified)
remove_all_acceptable  drop every acceptable candidate -> target NONE; valid only with exhaustive gold
add_acceptable         insert a known acceptable record that retrieval missed (recovery test)
duplicate_record       duplicate a candidate as a new exact row_id with the same entity identity
missing_field          blank one field of the acceptable (or top) candidate            (evaluation only)
contradictory_field    overwrite a field with a value from another candidate              (evaluation only)

Training recipe (`InterventionMixer`): 50% originals / 50% transformed per epoch, uniform over eligible
families; indices are remapped after every transform; realized mixture weights are recorded.
"""
from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any, Callable, Sequence

from ..hashing import pool_hash
from .duplicates import probable_duplicates
from ..schema import Candidate, DecisionInstance, LabelProvenance, apply_permutation

EVAL_ONLY = {"missing_field", "contradictory_field"}
TRAIN_FAMILIES = ["permutation", "add_hard_nonmatch", "duplicate_record", "remove_all_acceptable"]


def _clone(inst: DecisionInstance, cands: list[Candidate], *, intervention: str, acceptable: list[str] | None = None, provenance: str | None = None, exhaustive: bool | None = None, extra: dict[str, Any] | None = None, suffix: str) -> DecisionInstance:
    d = inst.to_dict()
    d["candidates"] = [asdict(c) for c in cands]
    d["acceptable_row_ids"] = inst.acceptable_row_ids if acceptable is None else acceptable
    d["label_provenance"] = provenance or inst.label_provenance
    d["acceptable_exhaustive"] = inst.acceptable_exhaustive if exhaustive is None else exhaustive
    d["variant_of"] = inst.variant_of or inst.instance_id
    d["intervention"] = intervention
    d["permutation"] = None
    d["pool_hash"] = pool_hash(inst.query_row_id, [c.row_id for c in cands])
    d["instance_id"] = f"{inst.instance_id}#{suffix}"
    d["extra"] = {**inst.extra, **(extra or {})}
    out = DecisionInstance.from_dict(d)
    out.validate()
    return out


# ---------------------------------------------------------------- families
def permutation(inst: DecisionInstance, rng: random.Random, **_: Any) -> DecisionInstance:
    perm = list(range(inst.k))
    rng.shuffle(perm)
    return apply_permutation(inst, perm, new_id=f"{inst.instance_id}#perm{rng.randint(0, 10**9)}")


def truncate(inst: DecisionInstance, rng: random.Random, *, k: int, **_: Any) -> DecisionInstance:
    cands = inst.candidates[:k]
    acc_before = set(inst.acceptable_row_ids)
    acc_after = [c.row_id for c in cands if c.row_id in acc_before]
    return _clone(inst, cands, intervention=f"truncate_{k}", acceptable=acc_after, extra={"acceptable_remaining": len(acc_after), "acceptable_before": len(acc_before)}, suffix=f"k{k}")


def add_hard_nonmatch(inst: DecisionInstance, rng: random.Random, *, distractors: Sequence[Candidate], n_add: int = 2, **_: Any) -> DecisionInstance | None:
    """distractors: verified non-matching candidates for this query (e.g. ranks beyond K under exhaustive gold)."""
    present = {c.row_id for c in inst.candidates}
    pool = [d for d in distractors if d.row_id not in present]
    if not pool:
        return None
    add = rng.sample(pool, min(n_add, len(pool)))
    cands = inst.candidates + add
    perm = list(range(len(cands)))
    rng.shuffle(perm)
    cands = [cands[j] for j in perm]
    return _clone(inst, cands, intervention="add_hard_nonmatch", suffix=f"hard{len(add)}")


def remove_all_acceptable(inst: DecisionInstance, rng: random.Random, *, distractors: Sequence[Candidate] = (), missing_gold: Sequence[Candidate] = (), suffix: str = "rmacc", **_: Any) -> DecisionInstance | None:
    """Catalog-absence counterfactual: the pool the retriever would return if no acceptable record were in the catalog.

    Every acceptable record leaves the pool; the next verified non-matches from the frozen snapshot (`distractors`,
    in rank order) backfill the freed slots so K is unchanged; each remaining record's rank drops by the number of
    removed acceptable records retrieved above it (scores are unchanged). Without backfill and re-ranking the variant
    carried two shortcuts no natural no-match pool has: K one short, and no rank-0 candidate (found Nov 5, 2026).
    Gold beyond K (`missing_gold`) shifts ranks only when its snapshot rank is known.
    """
    if not inst.is_matchable or not inst.acceptable_exhaustive:
        return None  # certified counterfactual requires an exhaustive acceptable set
    acc = set(inst.acceptable_row_ids)
    # review note R1: records the identity policy calls the same publication but the mapping does not link
    # (probable unlinked duplicates) leave the catalog with the acceptable ones and never backfill
    pud = set(probable_duplicates(inst.dataset, inst.query_fields, [c for c in inst.candidates if c.row_id not in acc]))
    gone = acc | pud
    gold_ranks = [c.retrieval_rank for c in inst.candidates if c.row_id in gone] + [g.retrieval_rank for g in missing_gold if g.row_id not in acc]
    gold_ranks = [r for r in gold_ranks if r is not None]
    kept = [c for c in inst.candidates if c.row_id not in gone]
    present = {c.row_id for c in inst.candidates} | {g.row_id for g in missing_gold}
    fill = [d for d in distractors if d.row_id not in present and not probable_duplicates(inst.dataset, inst.query_fields, [d])][: inst.k - len(kept)]

    def shifted(c: Candidate) -> Candidate:
        if c.retrieval_rank is None:
            return c
        return Candidate(c.row_id, c.fields, c.retrieval_rank - sum(1 for r in gold_ranks if r < c.retrieval_rank), c.retrieval_score, c.entity_group)

    cands = [shifted(c) for c in kept + fill]
    return _clone(inst, cands, intervention="remove_all_acceptable", acceptable=[], provenance=LabelProvenance.SYNTHETIC_REMOVED.value, exhaustive=True, extra={"removed_acceptable": len(acc & {c.row_id for c in inst.candidates}), "removed_probable_duplicates": len(pud), "backfilled": len(fill), "k_short": inst.k - len(cands)}, suffix=suffix)


def add_acceptable(inst: DecisionInstance, rng: random.Random, *, missing_gold: Sequence[Candidate], **_: Any) -> DecisionInstance | None:
    present = {c.row_id for c in inst.candidates}
    pool = [g for g in missing_gold if g.row_id not in present]
    if not pool:
        return None
    g = rng.choice(pool)
    pos = rng.randint(0, inst.k)
    cands = inst.candidates[:pos] + [g] + inst.candidates[pos:]
    return _clone(inst, cands, intervention="add_acceptable", acceptable=sorted(set(inst.acceptable_row_ids) | {g.row_id}), extra={"recovered_retrieval_miss": not inst.is_matchable}, suffix="addacc")


def duplicate_record(inst: DecisionInstance, rng: random.Random, **_: Any) -> DecisionInstance | None:
    if inst.k == 0:
        return None
    j = rng.randrange(inst.k)
    src = inst.candidates[j]
    dup = Candidate(row_id=f"{src.row_id}__dup1", fields=dict(src.fields), retrieval_rank=src.retrieval_rank, retrieval_score=src.retrieval_score, entity_group=src.entity_group)
    cands = inst.candidates[: j + 1] + [dup] + inst.candidates[j + 1 :]
    acc = list(inst.acceptable_row_ids)
    if src.row_id in acc:
        acc.append(dup.row_id)  # equivalent record: selecting either is an entity-level correct match
    return _clone(inst, cands, intervention="duplicate_record", acceptable=sorted(acc), extra={"duplicated_row_id": src.row_id, "duplicate_is_acceptable": src.row_id in inst.acceptable_row_ids}, suffix="dup")


def missing_field(inst: DecisionInstance, rng: random.Random, *, field: str | None = None, **_: Any) -> DecisionInstance | None:
    if inst.k == 0:
        return None
    acc = inst.acceptable_indices()
    j = acc[0] if acc else 0
    c = inst.candidates[j]
    fields = [f for f in c.fields if c.fields.get(f)]
    if not fields:
        return None
    f = field if field in fields else rng.choice(fields)
    new = Candidate(c.row_id, {**c.fields, f: ""}, c.retrieval_rank, c.retrieval_score, c.entity_group)
    cands = list(inst.candidates)
    cands[j] = new
    return _clone(inst, cands, intervention="missing_field", extra={"field": f, "target_index": j}, suffix=f"miss_{f}")


def contradictory_field(inst: DecisionInstance, rng: random.Random, *, field: str | None = None, **_: Any) -> DecisionInstance | None:
    """Evaluation diagnostic only: identity may change, so the label is marked unresolved unless annotated."""
    if inst.k < 2:
        return None
    acc = inst.acceptable_indices()
    j = acc[0] if acc else 0
    c = inst.candidates[j]
    other = inst.candidates[(j + 1) % inst.k]
    fields = [f for f in c.fields if c.fields.get(f) and other.fields.get(f) and c.fields[f] != other.fields[f]]
    if not fields:
        return None
    f = field if field in fields else rng.choice(fields)
    new = Candidate(c.row_id, {**c.fields, f: other.fields[f]}, c.retrieval_rank, c.retrieval_score, c.entity_group)
    cands = list(inst.candidates)
    cands[j] = new
    return _clone(inst, cands, intervention="contradictory_field", provenance=LabelProvenance.UNRESOLVED.value, exhaustive=False, extra={"field": f, "target_index": j, "note": "label unresolved under frozen identity policy unless separately annotated"}, suffix=f"contra_{f}")


FAMILIES: dict[str, Callable[..., DecisionInstance | None]] = {
    "permutation": permutation,
    "truncate": truncate,
    "add_hard_nonmatch": add_hard_nonmatch,
    "remove_all_acceptable": remove_all_acceptable,
    "add_acceptable": add_acceptable,
    "duplicate_record": duplicate_record,
    "missing_field": missing_field,
    "contradictory_field": contradictory_field,
}


# ---------------------------------------------------------------- eligibility + training mixer
def eligible_families(inst: DecisionInstance, families: Sequence[str], *, have_distractors: bool, have_missing_gold: bool, weak_labels: bool) -> list[str]:
    out = []
    for f in families:
        if f in EVAL_ONLY:
            continue
        if weak_labels and f != "permutation":
            continue  # for weak labels only permutation is eligible (plan §8)
        if f == "remove_all_acceptable" and not (inst.is_matchable and inst.acceptable_exhaustive):
            continue
        if f == "add_hard_nonmatch" and not (have_distractors and inst.acceptable_exhaustive):
            continue
        if f == "add_acceptable" and not have_missing_gold:
            continue
        if f == "duplicate_record" and inst.k == 0:
            continue
        out.append(f)
    return out


class InterventionMixer:
    """Per-epoch sampler: 50% original, 50% transformed uniformly over eligible families."""

    def __init__(self, families: Sequence[str] = TRAIN_FAMILIES, p_transform: float = 0.5, seed: int = 0, weak_labels: bool = False, ablate_label_changing: bool = False):
        self.families = [f for f in families if not (ablate_label_changing and f == "remove_all_acceptable")]
        self.p_transform = p_transform
        self.rng = random.Random(seed)
        self.weak_labels = weak_labels
        self.realized: dict[str, int] = {"original": 0}
        self.eligibility: dict[str, int] = {}

    def __call__(self, inst: DecisionInstance, distractors: Sequence[Candidate] = (), missing_gold: Sequence[Candidate] = ()) -> DecisionInstance:
        elig = eligible_families(inst, self.families, have_distractors=bool(distractors), have_missing_gold=bool(missing_gold), weak_labels=self.weak_labels)
        for f in elig:
            self.eligibility[f] = self.eligibility.get(f, 0) + 1
        if not elig or self.rng.random() >= self.p_transform:
            self.realized["original"] += 1
            return inst
        fam = self.rng.choice(elig)
        out = FAMILIES[fam](inst, self.rng, distractors=distractors, missing_gold=missing_gold)
        if out is None:
            self.realized["original"] += 1
            return inst
        self.realized[fam] = self.realized.get(fam, 0) + 1
        return out

    def report(self) -> dict[str, Any]:
        tot = sum(self.realized.values()) or 1
        return {"realized_counts": dict(self.realized), "realized_weights": {k: v / tot for k, v in self.realized.items()}, "eligibility_counts": dict(self.eligibility)}


# ---------------------------------------------------------------- evaluation diagnostics
def diagnostic_variants(inst: DecisionInstance, *, n_perms: int = 10, counts: Sequence[int] = (5, 10, 20, 50), seed: int = 0, distractors: Sequence[Candidate] = (), missing_gold: Sequence[Candidate] = ()) -> list[DecisionInstance]:
    """Evaluation-time variant set for the robustness study. Variants stay in the source query's partition/cluster."""
    rng = random.Random(f"{inst.instance_id}|{seed}")
    out: list[DecisionInstance] = []
    for i in range(n_perms):
        perm = list(range(inst.k))
        rng.shuffle(perm)
        out.append(apply_permutation(inst, perm, new_id=f"{inst.instance_id}#perm{i}"))
    for k in counts:
        if k < inst.k:
            out.append(truncate(inst, rng, k=k))
    for fam in ("add_hard_nonmatch", "remove_all_acceptable", "add_acceptable", "duplicate_record", "missing_field", "contradictory_field"):
        v = FAMILIES[fam](inst, rng, distractors=distractors, missing_gold=missing_gold)
        if v is not None:
            out.append(v)
    return out
