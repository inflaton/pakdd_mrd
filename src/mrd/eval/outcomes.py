"""Canonical outcome scorer based on the paper's CM / WC / MM / SM taxonomy (plan §7, §10 issue 2).

The scorer scores the *final action* of a model directly. It does not fabricate legacy name/location
flags. Matchable queries have a non-empty acceptable set; true no-match queries have an empty set
AND a label provenance that certifies pool-level NONE (released/adjudicated/exhaustive-removal).
Unresolved labels are reported separately and never enter the N-FMR denominator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ..schema import Decision, DecisionInstance, DecisionKind, LabelProvenance


class Outcome(str, Enum):
    CM = "CM"  # correct match: selected index in acceptable set
    WC = "WC"  # wrong candidate: selected index not in acceptable set (matchable query)
    MM = "MM"  # missed match: NONE / DEFER / FAIL on a matchable query
    SM = "SM"  # spurious match: selected a candidate on a true no-match query
    CR = "CR"  # correct reject: NONE on a true no-match query
    DN = "DN"  # deferred on a no-match query (not an automatic decision)
    FN = "FN"  # failure/timeout on a no-match query
    UNRESOLVED = "UNRESOLVED"  # label not certified; excluded from denominators


@dataclass
class ScoredInstance:
    instance_id: str
    dataset: str
    entity_group: str
    partition: str
    matchable: bool
    certified_negative: bool
    outcome: Outcome
    kind: DecisionKind
    false_none: bool  # NONE emitted on a matchable query (subset of MM)
    deferred: bool
    failed: bool
    escalated: bool
    weight: float = 1.0  # sampling weight if strata were enriched
    intervention: str | None = None
    variant_of: str | None = None
    cohort: str = "natural"


NEGATIVE_CERTIFYING = {
    LabelProvenance.RELEASED_LINK.value,
    LabelProvenance.ADJUDICATED.value,
    LabelProvenance.SYNTHETIC_REMOVED.value,
}


def cohort_of(inst: DecisionInstance) -> str:
    """natural (benchmark pools) or constructed (catalog-absence counterfactuals from 03_constructed_cohort.py)."""
    return "constructed" if inst.extra.get("cohort") == "constructed" else "natural"


def is_certified_negative(inst: DecisionInstance) -> bool:
    """A no-match query counts for N-FMR only when its NONE label is certified for this pool."""
    if inst.is_matchable:
        return False
    if inst.label_provenance == LabelProvenance.UNRESOLVED.value:
        return False
    if inst.label_provenance == LabelProvenance.SYNTHETIC_REMOVED.value:
        return bool(inst.acceptable_exhaustive)
    # released-link negatives: certified only when the pool was fully reviewed / exhaustive
    if inst.label_provenance == LabelProvenance.RELEASED_LINK.value:
        return bool(inst.acceptable_exhaustive)
    return inst.label_provenance in NEGATIVE_CERTIFYING


def score_instance(inst: DecisionInstance, dec: Decision) -> ScoredInstance:
    acc = set(inst.acceptable_indices())
    matchable = len(acc) > 0
    certified_neg = is_certified_negative(inst)
    kind = dec.kind
    false_none = False
    if matchable:
        if kind == DecisionKind.SELECT:
            out = Outcome.CM if dec.index in acc else Outcome.WC
        else:
            out = Outcome.MM
            false_none = kind == DecisionKind.NONE
    elif certified_neg:
        if kind == DecisionKind.SELECT:
            out = Outcome.SM
        elif kind == DecisionKind.NONE:
            out = Outcome.CR
        elif kind == DecisionKind.DEFER:
            out = Outcome.DN
        else:
            out = Outcome.FN
    else:
        out = Outcome.UNRESOLVED
    return ScoredInstance(
        instance_id=inst.instance_id,
        dataset=inst.dataset,
        entity_group=inst.entity_group,
        partition=inst.partition,
        matchable=matchable,
        certified_negative=certified_neg,
        outcome=out,
        kind=kind,
        false_none=false_none,
        deferred=kind == DecisionKind.DEFER,
        failed=kind == DecisionKind.FAIL,
        escalated=dec.escalated,
        weight=float(inst.extra.get("sampling_weight", 1.0)),
        intervention=inst.intervention,
        variant_of=inst.variant_of,
        cohort=cohort_of(inst),
    )


def score_run(instances: Iterable[DecisionInstance], decisions: dict[str, Decision]) -> list[ScoredInstance]:
    """Score every instance. Missing decisions are FAIL (missed decisions stay in the denominator)."""
    scored = []
    for inst in instances:
        dec = decisions.get(inst.instance_id)
        if dec is None:
            dec = Decision(inst.instance_id, DecisionKind.FAIL, -1, None, failure_reason="missing_output")
        scored.append(score_instance(inst, dec))
    return scored
