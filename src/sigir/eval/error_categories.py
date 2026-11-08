"""Named categories for false accepts (review note R3b, Nov 5, 2026).

Variant confusion is the H1 target: most confident false matches on WDC negatives were capacity or model variants the
identity policy calls different products ("LaCie 2TB Mobile Drive" -> "LaCie Mobile Drive 5 TB"). Every spurious match
(SM, on a certified no-match pool) and wrong candidate (WC) is labelled by comparing the query with the selected
record, deterministically and from text only:

  capacity_variant   both mention capacities / sizes / pack counts (number + unit) and the sets differ
  model_variant      same brand (brand field or first title word) or disjoint model-number-like codes, titles overlap
                     (token Jaccard >= 0.2), and no capacity difference: another model or product line
  same_title_other   publications: title token Jaccard >= 0.5 (versions, columns, citation variants)
  other              anything else
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

UNIT = r"(?:tb|gb|mb|kb|mah|wh|w|v|mm|cm|in|inch|inches|oz|ml|l|lb|lbs|kg|g|ft|hz|mhz|ghz|mp|pack|pk|ct|count|pcs|pc|x)"
QTY = re.compile(r"(\d+(?:[.,]\d+)?)\s*" + UNIT + r"\b")
CODE = re.compile(r"\b(?=[a-z0-9-]*\d)(?=[a-z0-9-]*[a-z])[a-z0-9]+(?:-[a-z0-9]+)*\b")
TITLE_FIELDS = ("title", "name")


def _text(fields: dict[str, Any]) -> str:
    vals = [str(fields.get(k) or "") for k in TITLE_FIELDS if fields.get(k)]
    if not vals:
        vals = [" ".join(str(v) for v in fields.values() if v)]
    return unicodedata.normalize("NFKC", " ".join(vals)).lower()


def quantities(fields: dict[str, Any]) -> set[str]:
    out = set()
    for num, unit in re.findall(r"(\d+(?:[.,]\d+)?)\s*(" + UNIT[3:-1] + r")\b", _text(fields)):
        out.add(f"{float(num.replace(',', '.')):g}{unit}")
    return out


def codes(fields: dict[str, Any]) -> set[str]:
    txt = _text(fields) + " " + unicodedata.normalize("NFKC", str(fields.get("modelno") or "")).lower()
    return {c.replace("-", "") for c in CODE.findall(txt) if len(c.replace("-", "")) >= 3 and not QTY.fullmatch(c)}


def _tokens(fields: dict[str, Any]) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _text(fields)) if len(w) > 1}


def classify(dataset: str, query: dict[str, Any], selected: dict[str, Any]) -> str:
    tq, ts = _tokens(query), _tokens(selected)
    jac = len(tq & ts) / len(tq | ts) if tq | ts else 0.0
    if dataset == "dblp_scholar":
        return "same_title_other" if jac >= 0.5 else "other"
    qq, qs = quantities(query), quantities(selected)
    if qq and qs and qq != qs:
        return "capacity_variant"
    cq, cs = codes(query), codes(selected)
    if jac >= 0.2 and ((cq and cs and not (cq & cs)) or _brand(query) == _brand(selected) != ""):
        return "model_variant"
    return "other"


def _brand(fields: dict[str, Any]) -> str:
    b = unicodedata.normalize("NFKC", str(fields.get("brand") or fields.get("manufacturer") or "")).lower().strip()
    if b:
        return b.split()[0]
    t = re.findall(r"[a-z0-9]+", _text(fields))
    return t[0] if t else ""


def categorize_errors(instances, decisions) -> dict[str, dict[str, dict[str, int]]]:
    """dataset -> {"SM": {category: n}, "WC": {category: n}} for decisions that select a candidate wrongly."""
    from ..eval.outcomes import is_certified_negative

    out: dict[str, dict[str, dict[str, int]]] = {}
    for inst in instances:
        d = decisions.get(inst.instance_id)
        if d is None or d.kind.value != "select" or not (0 <= d.index < inst.k):
            continue
        acc = set(inst.acceptable_indices())
        if inst.is_matchable and d.index not in acc:
            kind = "WC"
        elif not inst.is_matchable and is_certified_negative(inst):
            kind = "SM"
        else:
            continue
        cat = classify(inst.dataset, inst.query_fields, inst.candidates[d.index].fields)
        slot = out.setdefault(inst.dataset, {}).setdefault(kind, {})
        slot[cat] = slot.get(cat, 0) + 1
    return out
