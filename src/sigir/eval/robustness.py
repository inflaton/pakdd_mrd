"""Candidate-set robustness analysis (plan §8).

Given decisions on original instances and on their diagnostic variants, compute per intervention:
  * exact-record flip rate      (selected row_id differs after undoing permutation / across variants)
  * entity-level flip rate      (selected entity_group differs; switches between equivalent acceptable records
                                 are NOT counted as errors)
  * NONE-probability shift      (mean |delta p_none|)
  * metrics on the variant set  (CM/WC/MM/SM etc.) alongside the clean metrics
  * stochastic variation        from repeated identical requests when provided
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from ..schema import Decision, DecisionInstance, DecisionKind, remap_decision
from .metrics import compute_metrics
from .outcomes import score_run


def _selected_entity(inst: DecisionInstance, dec: Decision) -> str | None:
    if dec.kind != DecisionKind.SELECT:
        return f"__{dec.kind.value}__"
    c = inst.candidates[dec.index]
    return c.entity_group or c.row_id


def _selected_record(inst: DecisionInstance, dec: Decision) -> str | None:
    if dec.kind != DecisionKind.SELECT:
        return f"__{dec.kind.value}__"
    rid = inst.candidates[dec.index].row_id
    return rid.split("__dup")[0]  # duplicates of a record count as the same record


def robustness_report(
    originals: Sequence[DecisionInstance],
    variants: Sequence[DecisionInstance],
    decisions: dict[str, Decision],
    repeated: dict[str, list[Decision]] | None = None,
) -> dict[str, Any]:
    orig_by_id = {i.instance_id: i for i in originals}
    by_family: dict[str, list[tuple[DecisionInstance, DecisionInstance]]] = defaultdict(list)
    for v in variants:
        src = orig_by_id.get(v.variant_of or "")
        if src is None or v.instance_id not in decisions or src.instance_id not in decisions:
            continue
        fam = v.intervention or "unknown"
        if fam.startswith("truncate_"):
            fam = fam  # keep count in the name
        by_family[fam].append((src, v))
    out: dict[str, Any] = {"clean": compute_metrics(score_run(originals, decisions)), "families": {}}
    for fam, pairs in sorted(by_family.items()):
        rec_flips = ent_flips = 0
        dp_none = []
        acceptable_remaining = []
        for src, v in pairs:
            d_src = decisions[src.instance_id]
            d_v = decisions[v.instance_id]
            if v.permutation is not None:
                d_v_cmp = remap_decision(d_v, v.permutation)
                v_cmp = src  # after undoing the permutation, compare in the original frame
            else:
                d_v_cmp, v_cmp = d_v, v
            if _selected_record(src, d_src) != _selected_record(v_cmp, d_v_cmp):
                rec_flips += 1
            if _selected_entity(src, d_src) != _selected_entity(v_cmp, d_v_cmp):
                ent_flips += 1
            if d_src.p_none is not None and d_v.p_none is not None:
                dp_none.append(abs(d_src.p_none - d_v.p_none))
            if "acceptable_remaining" in v.extra:
                acceptable_remaining.append(v.extra["acceptable_remaining"] > 0)
        var_insts = [v for _, v in pairs]
        m = compute_metrics(score_run(var_insts, decisions))
        out["families"][fam] = {
            "n_pairs": len(pairs),
            "record_flip_rate": rec_flips / len(pairs),
            "entity_flip_rate": ent_flips / len(pairs),
            "mean_abs_delta_p_none": float(np.mean(dp_none)) if dp_none else float("nan"),
            "acceptable_present_fraction": float(np.mean(acceptable_remaining)) if acceptable_remaining else None,
            "metrics": m,
        }
    if repeated:
        flips = []
        for iid, decs in repeated.items():
            inst = orig_by_id.get(iid)
            if inst is None or len(decs) < 2:
                continue
            sel = {_selected_record(inst, d) for d in decs}
            flips.append(len(sel) > 1)
        out["stochastic_variation"] = {"n_instances": len(flips), "fraction_with_any_flip": float(np.mean(flips)) if flips else float("nan")}
    return out
