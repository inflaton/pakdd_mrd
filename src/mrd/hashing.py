"""Stable content hashes for files, JSON-able objects, candidate pools and configs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Hash of canonical JSON (sorted keys, no whitespace)."""
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pool_hash(query_row_id: str, candidate_row_ids: Iterable[str]) -> str:
    """Order-sensitive hash of a presented candidate pool (query + candidate IDs as exact strings)."""
    return sha256_obj({"q": query_row_id, "c": list(candidate_row_ids)})


def short(h: str, n: int = 12) -> str:
    return h[:n]


def sha256_jsonl(path: str | Path) -> str:
    """Hash of a JSONL file's content (identical to sha256_file, named for clarity)."""
    return sha256_file(path)
