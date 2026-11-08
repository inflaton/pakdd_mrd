"""Record serialization: rich (all fields, named) versus compact (short, fewer fields), separately versioned.

The same fields and candidate information are used across model comparisons within an arm; the rich
versus compact contrast is its own experiment (plan §6/§9).
"""
from __future__ import annotations

import unicodedata
from typing import Any, Sequence

# Model inputs are NFKC-normalised (Nov 5, 2026, review note R2). SentencePiece applies nmt_nfkc, the transformers 5
# fast conversion of deberta-v3 does not: non-breaking spaces inside part numbers and symbols such as ™ became [UNK]
# (30 of 4,000 texts); after NFKC the fast tokenizer equals SentencePiece on all 4,000. Applied to every arm's input
# (encoders, decision head, judges) so comparisons see identical text. Retrieval (record_text) is not affected.
TEXT_NORMALIZATION = "NFKC"

SERIALIZERS: dict[str, dict[str, Any]] = {
    "rich_v1": {"max_chars_per_field": 400, "include_empty": True},
    "compact_v1": {"max_chars_per_field": 80, "include_empty": False},
}


def serialize_record(fields: dict[str, Any], order: Sequence[str] | None = None, version: str = "rich_v1", compact_fields: Sequence[str] | None = None, normalize: str | None = TEXT_NORMALIZATION) -> str:
    cfg = SERIALIZERS[version]
    keys = list(order) if order else list(fields.keys())
    if version.startswith("compact") and compact_fields:
        keys = [k for k in keys if k in compact_fields]
    parts = []
    for k in keys:
        v = str(fields.get(k, "") or "")
        v = (unicodedata.normalize(normalize, v) if normalize else v).strip()
        if not v and not cfg["include_empty"]:
            continue
        v = v[: cfg["max_chars_per_field"]]
        parts.append(f"{k}: {v}")
    return " | ".join(parts)


SYMBOLS = [chr(ord("A") + i) for i in range(26)] + [f"A{chr(ord('A') + i)}" for i in range(26)]  # A..Z, AA..AZ
NONE_SYMBOL = "NONE"


def candidate_symbol(j: int) -> str:
    return SYMBOLS[j]


def pair_text(query: str, candidate: str, sep: str = " [SEP] ") -> str:
    return f"{query}{sep}{candidate}"
