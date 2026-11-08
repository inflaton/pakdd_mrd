"""JSONL helpers for instances and decisions."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import Decision, DecisionInstance


def _open(path: str | Path, mode: str):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, mode + "t", encoding="utf-8")
    return open(p, mode, encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with _open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_instances(path: str | Path, insts: Iterable[DecisionInstance]) -> int:
    return write_jsonl(path, (i.to_dict() for i in insts))


def read_instances(path: str | Path, partitions: set[str] | None = None, datasets: set[str] | None = None) -> list[DecisionInstance]:
    out = []
    for d in read_jsonl(path):
        if partitions and d["partition"] not in partitions:
            continue
        if datasets and d["dataset"] not in datasets:
            continue
        out.append(DecisionInstance.from_dict(d))
    return out


def write_decisions(path: str | Path, decs: Iterable[Decision]) -> int:
    return write_jsonl(path, (d.to_dict() for d in decs))


def read_decisions(path: str | Path) -> dict[str, Decision]:
    return {d["instance_id"]: Decision.from_dict(d) for d in read_jsonl(path)}


def read_probs(path: str | Path) -> dict[str, list[float]]:
    """Read decisions and return instance_id -> probs for those with full probabilities."""
    out = {}
    for d in read_jsonl(path):
        if d.get("probs") is not None:
            out[d["instance_id"]] = d["probs"]
    return out
