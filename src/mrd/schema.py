"""Decision schema: exact-ID candidate mapping, NONE / DEFER / FAIL semantics, permutations, masks.

Design rules (Research plan §4 and §10, integration issue 1):

* Every candidate row_id is kept as an exact *string*. Catalog IDs can exceed float precision, so
  nothing in this module ever converts IDs through float or int(float(...)).
* The public decision interface maps NONE -> -1 and candidate c_j -> j (0 <= j < K). For K = 21 that
  is 22 categories, not a regression over integers.
* NONE (no supplied candidate acceptable), DEFER (not confident enough to decide automatically) and
  FAIL (stronger judge unresolved / timeout) are distinct. FAIL is never converted to -1.
* The legacy gold sentinel -20 ("any selected ID acceptable") must be resolved or excluded before
  labels are constructed. `resolve_legacy_sentinel` does that explicitly.
* Permutations are stored as explicit index maps and are invertible; `invert_permutation` and
  `remap_decision` are used by both intervention training and robustness evaluation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

NONE_INDEX = -1
LEGACY_ANY_SENTINEL = -20


class DecisionKind(str, Enum):
    SELECT = "select"  # an index 0..K-1 was chosen
    NONE = "none"  # automatic "no supplied match"
    DEFER = "defer"  # system abstained; needs more computation / human
    FAIL = "fail"  # request failed / unresolved / timed out


class LabelProvenance(str, Enum):
    RELEASED_LINK = "released_link"  # links released with the public dataset
    ADJUDICATED = "adjudicated"  # two-reviewer pool adjudication
    TEACHER_SINGLE = "teacher_single"  # single teacher label (training only)
    TEACHER_DUAL = "teacher_dual"  # gated dual-teacher agreement (training only)
    SYNTHETIC_REMOVED = "synthetic_removed"  # all acceptable candidates removed; certified only if exhaustive
    UNRESOLVED = "unresolved"  # not a negative; excluded from risk denominators


@dataclass
class Candidate:
    row_id: str
    fields: dict[str, Any]
    retrieval_rank: int | None = None  # rank in the frozen retrieval snapshot (0-based)
    retrieval_score: float | None = None
    entity_group: str | None = None  # entity-level identity where established

    def __post_init__(self) -> None:
        if not isinstance(self.row_id, str):
            raise TypeError(f"row_id must be str, got {type(self.row_id).__name__}: {self.row_id!r}")


@dataclass
class DecisionInstance:
    """One query with its candidate set and the acceptable-ID set.

    `acceptable_row_ids` is set-valued. `acceptable_exhaustive` records whether the acceptable set is
    known to be complete for this pool (required before "remove all acceptable" interventions can
    create a certified no-match counterfactual).
    """

    instance_id: str
    dataset: str
    direction: str  # e.g. "A->B" (query from table A, catalog = table B)
    query_row_id: str
    query_fields: dict[str, Any]
    candidates: list[Candidate]
    acceptable_row_ids: list[str]
    acceptable_exhaustive: bool
    label_provenance: str
    entity_group: str  # cluster key used for splits and bootstrap
    partition: str  # train / dev / calib_fit / calib_confirm / test / transfer / legacy_wa_test
    pool_hash: str = ""
    variant_of: str | None = None  # source instance_id if this is a counterfactual variant
    intervention: str | None = None  # intervention family name if a variant
    permutation: list[int] | None = None  # presented_index -> original_index, if permuted
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    @property
    def k(self) -> int:
        return len(self.candidates)

    @property
    def is_matchable(self) -> bool:
        return len(self.acceptable_indices()) > 0

    def acceptable_indices(self) -> list[int]:
        acc = set(self.acceptable_row_ids)
        return [j for j, c in enumerate(self.candidates) if c.row_id in acc]

    def index_to_row_id(self, j: int) -> str:
        if j == NONE_INDEX:
            raise ValueError("NONE has no row_id")
        if not (0 <= j < self.k):
            raise IndexError(f"candidate index {j} out of bounds for K={self.k}")
        return self.candidates[j].row_id

    def validate(self) -> None:
        if not isinstance(self.query_row_id, str):
            raise TypeError("query_row_id must be str")
        ids = [c.row_id for c in self.candidates]
        if len(set(ids)) != len(ids):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate candidate row_ids in pool {self.instance_id}: {dup}")
        for r in self.acceptable_row_ids:
            if not isinstance(r, str):
                raise TypeError("acceptable_row_ids must be strings")
        if self.permutation is not None:
            if sorted(self.permutation) != list(range(self.k)):
                raise ValueError("permutation must be a bijection over 0..K-1")
        for c in self.candidates:
            if c.row_id == self.query_row_id and self.direction.split("->")[0] == self.direction.split("->")[1]:
                raise ValueError("self-match in candidate pool")

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DecisionInstance":
        d = dict(d)
        d["candidates"] = [Candidate(**c) for c in d["candidates"]]
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


@dataclass
class Decision:
    """Model output for one instance, already mapped back to the *presented* candidate order."""

    instance_id: str
    kind: DecisionKind
    index: int  # 0..K-1 for SELECT, -1 otherwise
    row_id: str | None  # exact string; None for NONE / DEFER / FAIL
    probs: list[float] | None = None  # length K+1: candidates 0..K-1 then NONE at position K
    p_none: float | None = None
    max_candidate_prob: float | None = None
    margin: float | None = None  # top1 - top2 candidate prob
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    escalated: bool = False
    failure_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def public_code(self) -> int | None:
        """-1 for NONE, j for SELECT, None for DEFER / FAIL (never coerced to -1)."""
        if self.kind == DecisionKind.SELECT:
            return self.index
        if self.kind == DecisionKind.NONE:
            return NONE_INDEX
        return None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        d = dict(d)
        d["kind"] = DecisionKind(d["kind"])
        return cls(**d)


# ---------------------------------------------------------------------- helpers
def decision_from_probs(
    inst: DecisionInstance,
    probs: Sequence[float],
    *,
    tau_select: float | None = None,
    tau_none: float | None = None,
    **meta: Any,
) -> Decision:
    """Turn a (K+1)-vector of probabilities into a Decision.

    With no thresholds this is argmax (standalone selector quality). With thresholds it is the
    calibrated policy: select if max candidate prob >= tau_select, NONE if p_none >= tau_none,
    otherwise DEFER.
    """
    k = inst.k
    if len(probs) != k + 1:
        raise ValueError(f"expected K+1={k + 1} probabilities, got {len(probs)}")
    cand = list(probs[:k])
    p_none = float(probs[k])
    if k == 0:
        return Decision(inst.instance_id, DecisionKind.NONE, NONE_INDEX, None, list(probs), 1.0, 0.0, 0.0, **meta)
    # stable tie-break by row_id (plan §4: tie-breaking by stable identity)
    order = sorted(range(k), key=lambda j: (-cand[j], inst.candidates[j].row_id))
    top = order[0]
    top_p = cand[top]
    second_p = cand[order[1]] if k > 1 else 0.0
    margin = top_p - second_p
    if tau_select is None and tau_none is None:
        if p_none >= top_p:
            kind, idx = DecisionKind.NONE, NONE_INDEX
        else:
            kind, idx = DecisionKind.SELECT, top
    else:
        ts = tau_select if tau_select is not None else 1.1
        tn = tau_none if tau_none is not None else 1.1
        if top_p >= ts and top_p >= p_none:
            kind, idx = DecisionKind.SELECT, top
        elif p_none >= tn:
            kind, idx = DecisionKind.NONE, NONE_INDEX
        else:
            kind, idx = DecisionKind.DEFER, NONE_INDEX
    row_id = inst.index_to_row_id(idx) if kind == DecisionKind.SELECT else None
    return Decision(
        inst.instance_id, kind, idx, row_id, [float(p) for p in probs], p_none, float(top_p), float(margin), **meta
    )


def resolve_legacy_sentinel(gold_indices: Iterable[int], k: int) -> list[int] | None:
    """Resolve the legacy -20 sentinel.

    Returns the explicit acceptable index list, or None when the sentinel is present and cannot be
    resolved (caller must exclude the example). -20 means "any selected ID acceptable" in the legacy
    merchant scorer; under the new schema that is only representable as "all K candidates acceptable",
    which is what we return when K > 0. It is neither candidate 20 nor NONE.
    """
    g = list(gold_indices)
    if LEGACY_ANY_SENTINEL in g:
        if k <= 0:
            return None
        return list(range(k))
    for j in g:
        if j != NONE_INDEX and not (0 <= j < k):
            raise IndexError(f"legacy gold index {j} out of bounds for K={k}")
    return [j for j in g if j != NONE_INDEX]


def apply_permutation(inst: DecisionInstance, perm: Sequence[int], *, new_id: str | None = None) -> DecisionInstance:
    """Return a copy with candidates presented in order `perm` (presented_index -> original_index).

    Retrieval rank/score travel with the candidate (explicit features), not with the slot.
    """
    perm = list(perm)
    if sorted(perm) != list(range(inst.k)):
        raise ValueError("perm must be a bijection over 0..K-1")
    cands = [inst.candidates[j] for j in perm]
    d = inst.to_dict()
    d["candidates"] = [asdict(c) for c in cands]
    d["permutation"] = perm
    d["variant_of"] = inst.variant_of or inst.instance_id
    d["intervention"] = "permutation"
    d["instance_id"] = new_id or f"{inst.instance_id}#perm"
    return DecisionInstance.from_dict(d)


def invert_permutation(perm: Sequence[int]) -> list[int]:
    inv = [0] * len(perm)
    for presented, original in enumerate(perm):
        inv[original] = presented
    return inv


def remap_decision(dec: Decision, perm: Sequence[int] | None) -> Decision:
    """Map a decision made on a permuted presentation back to original indices.

    perm: presented_index -> original_index. probs are reordered so that position j refers to the
    original candidate j; NONE stays last.
    """
    if perm is None:
        return dec
    k = len(perm)
    idx = perm[dec.index] if dec.kind == DecisionKind.SELECT else NONE_INDEX
    probs = None
    if dec.probs is not None:
        probs = [0.0] * (k + 1)
        for presented, original in enumerate(perm):
            probs[original] = dec.probs[presented]
        probs[k] = dec.probs[k]
    d = dec.to_dict()
    d["index"] = idx
    d["probs"] = probs
    return Decision.from_dict(d)


def candidate_mask(k: int, k_max: int) -> list[bool]:
    """Mask for padded candidate slots plus the NONE slot (always valid)."""
    return [j < k for j in range(k_max)] + [True]


def public_code_to_kind(code: int | None) -> DecisionKind:
    if code is None:
        return DecisionKind.FAIL
    if code == NONE_INDEX:
        return DecisionKind.NONE
    if code < 0:
        raise ValueError(f"invalid public decision code {code}")
    return DecisionKind.SELECT
