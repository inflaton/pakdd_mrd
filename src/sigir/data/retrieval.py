"""Frozen candidate retrieval snapshots (plan §5).

Default retriever: BM25 (bm25s) over the serialized catalog record. Optional dense retriever
(sentence-transformers) and reciprocal-rank-fusion hybrid. The snapshot stores, per query, the ordered
list of (row_id, rank, score) up to `pool_size` (default 50 so count interventions 5/10/20/50 come from
one frozen pool). Candidate recall is reported separately from selector quality.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..hashing import sha256_obj
from ..io import read_jsonl, write_jsonl

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def record_text(row: pd.Series | dict[str, Any], fields: Sequence[str]) -> str:
    return " ".join(str(row.get(f, "") or "") for f in fields)


SNAPSHOT_FORMAT = 1  # bump when retrieval code changes the pools for the same inputs


@dataclass
class Snapshot:
    dataset: str
    direction: str
    retriever: str
    pool_size: int
    pools: dict[str, list[dict[str, Any]]]  # query_row_id -> [{row_id, rank, score}]
    config: dict[str, Any]
    inputs_key: str | None = None  # snapshot_inputs_key() of what the pools were built from

    @property
    def hash(self) -> str:
        return sha256_obj({"ds": self.dataset, "dir": self.direction, "ret": self.retriever, "pools": {q: [c["row_id"] for c in cs] for q, cs in self.pools.items()}})

    def save(self, path: Path) -> None:
        rows = [{"query_row_id": q, "candidates": cs} for q, cs in self.pools.items()]
        write_jsonl(path, rows)
        meta = {"dataset": self.dataset, "direction": self.direction, "retriever": self.retriever, "pool_size": self.pool_size, "config": self.config, "snapshot_hash": self.hash, "n_queries": len(self.pools), "inputs_key": self.inputs_key}
        with open(path.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Snapshot":
        meta = json.load(open(path.with_suffix(".meta.json"), encoding="utf-8"))
        pools = {r["query_row_id"]: r["candidates"] for r in read_jsonl(path)}
        return cls(meta["dataset"], meta["direction"], meta["retriever"], meta["pool_size"], pools, meta["config"], meta.get("inputs_key"))


def snapshot_inputs_key(
    dataset: str,
    direction: str,
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    fields_q: Sequence[str],
    fields_c: Sequence[str],
    cfg: dict[str, Any],
    exclude_self: bool = False,
) -> str:
    """Hash of everything a snapshot's pools depend on: the exact text each retriever sees for every query and
    catalog record (row_id + serialized fields), the field lists, the retrieval config, self-exclusion and
    SNAPSHOT_FORMAT. A reused snapshot must carry the same key, so stale pools can never be reused silently."""

    def texts(df: pd.DataFrame, fields: Sequence[str]) -> list[list[str]]:
        cols = [df[f].astype(str).tolist() if f in df.columns else [""] * len(df) for f in fields]
        return [[rid, " ".join(vals)] for rid, vals in zip(df["row_id"].astype(str).tolist(), zip(*cols))] if cols else [[r, ""] for r in df["row_id"]]

    return sha256_obj({
        "format": SNAPSHOT_FORMAT,
        "dataset": dataset,
        "direction": direction,
        "cfg": {k: v for k, v in cfg.items() if k != "device"},
        "exclude_self": bool(exclude_self),
        "fields_q": list(fields_q),
        "fields_c": list(fields_c),
        "queries": sha256_obj(texts(queries, fields_q)),
        "catalog": sha256_obj(texts(catalog, fields_c)),
    })


def load_or_build_snapshot(path: Path, inputs_key: str, build) -> tuple["Snapshot", str]:
    """Reuse `path` only if it was built from the same inputs; otherwise rebuild it with `build()`.

    Returns (snapshot, status); status says whether it was reused, built, or rebuilt and why, and for a
    rebuild whether the pools actually changed (compared by snapshot_hash).
    """
    path = Path(path)
    existed = path.exists()
    old_key = old_hash = None
    if existed:
        meta_p = path.with_suffix(".meta.json")
        meta = json.load(open(meta_p, encoding="utf-8")) if meta_p.exists() else {}
        old_key, old_hash = meta.get("inputs_key"), meta.get("snapshot_hash")
        if old_key == inputs_key:
            return Snapshot.load(path), "reused"
    snap = build()
    snap.inputs_key = inputs_key
    snap.save(path)
    if not existed:
        return snap, "built"
    why = "built before the inputs check" if old_key is None else "inputs changed"
    same = "pools unchanged" if old_hash == snap.hash else f"POOLS CHANGED ({str(old_hash)[:12]} -> {snap.hash[:12]})"
    return snap, f"rebuilt ({why}); {same}"


# ---------------------------------------------------------------- retrievers
def bm25_retrieve(queries: pd.DataFrame, catalog: pd.DataFrame, fields_q: Sequence[str], fields_c: Sequence[str], pool_size: int, k1: float = 1.2, b: float = 0.75) -> dict[str, list[tuple[str, float]]]:
    import bm25s

    corpus_ids = catalog["row_id"].tolist()
    corpus_tok = [tokenize(record_text(r, fields_c)) for _, r in catalog.iterrows()]
    retriever = bm25s.BM25(k1=k1, b=b)
    retriever.index(corpus_tok, show_progress=False)
    q_tok = [tokenize(record_text(r, fields_q)) for _, r in queries.iterrows()]
    k = min(pool_size, len(corpus_ids))
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, toks in zip(queries["row_id"].tolist(), q_tok):
        if not toks:
            out[qid] = []
            continue
        res, scores = retriever.retrieve([toks], k=k, show_progress=False)
        out[qid] = [(corpus_ids[int(i)], float(s)) for i, s in zip(res[0], scores[0]) if s > 0]
    return out


def dense_retrieve(queries: pd.DataFrame, catalog: pd.DataFrame, fields_q: Sequence[str], fields_c: Sequence[str], pool_size: int, model_name: str, batch_size: int = 256, device: str | None = None) -> dict[str, list[tuple[str, float]]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    c_text = [record_text(r, fields_c) for _, r in catalog.iterrows()]
    q_text = [record_text(r, fields_q) for _, r in queries.iterrows()]
    c_emb = model.encode(c_text, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    q_emb = model.encode(q_text, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    sims = q_emb @ c_emb.T
    k = min(pool_size, sims.shape[1])
    ids = catalog["row_id"].tolist()
    out = {}
    for qi, qid in enumerate(queries["row_id"].tolist()):
        top = np.argpartition(-sims[qi], k - 1)[:k]
        top = top[np.argsort(-sims[qi][top])]
        out[qid] = [(ids[int(i)], float(sims[qi][i])) for i in top]
    return out


def rrf_fuse(runs: list[dict[str, list[tuple[str, float]]]], pool_size: int, k: int = 60) -> dict[str, list[tuple[str, float]]]:
    out = {}
    qids = set().union(*[r.keys() for r in runs])
    for q in qids:
        score: dict[str, float] = {}
        for run in runs:
            for rank, (rid, _) in enumerate(run.get(q, [])):
                score[rid] = score.get(rid, 0.0) + 1.0 / (k + rank + 1)
        ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))[:pool_size]
        out[q] = ranked
    return out


def build_snapshot(
    dataset: str,
    direction: str,
    queries: pd.DataFrame,
    catalog: pd.DataFrame,
    fields_q: Sequence[str],
    fields_c: Sequence[str],
    cfg: dict[str, Any],
    exclude_self: bool = False,
) -> Snapshot:
    pool_size = int(cfg.get("pool_size", 50))
    name = cfg.get("retriever", "bm25")
    fetch = pool_size + (1 if exclude_self else 0)
    runs = []
    if name in ("bm25", "hybrid"):
        runs.append(bm25_retrieve(queries, catalog, fields_q, fields_c, fetch, **cfg.get("bm25", {})))
    if name in ("dense", "hybrid"):
        runs.append(dense_retrieve(queries, catalog, fields_q, fields_c, fetch, cfg.get("dense_model", "BAAI/bge-small-en-v1.5"), device=cfg.get("device")))
    fused = runs[0] if len(runs) == 1 else rrf_fuse(runs, fetch, **cfg.get("rrf", {}))
    pools = {}
    for q, cands in fused.items():
        rows = []
        for rid, s in cands:
            if exclude_self and rid == q:
                continue
            rows.append({"row_id": rid, "rank": len(rows), "score": round(float(s), 6)})
            if len(rows) >= pool_size:
                break
        pools[q] = rows
    return Snapshot(dataset, direction, name, pool_size, pools, cfg)


def candidate_recall(snapshot: Snapshot, acceptable: dict[str, set[str]], ks: Sequence[int] = (5, 10, 20, 50)) -> dict[str, Any]:
    """Fraction of gold-matchable queries with >= 1 acceptable candidate in the top-k (reported separately)."""
    matchable = [q for q, acc in acceptable.items() if acc and q in snapshot.pools]
    out: dict[str, Any] = {"n_matchable_queries": len(matchable)}
    for k in ks:
        hit = sum(1 for q in matchable if any(c["row_id"] in acceptable[q] for c in snapshot.pools[q][:k]))
        out[f"recall@{k}"] = hit / len(matchable) if matchable else float("nan")
    return out
