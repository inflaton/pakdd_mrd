#!/usr/bin/env python
"""Run the published CoMEM selector on FLAN-T5-XL shortlists from 14_comem_rank.py."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import re
import time

from openai import OpenAI

from _common import RUNS_DIR, log, setup_logging
from comem_inputs import FROZEN_MANIFEST, load_pool_rows, record_fields, render_record
from sigir.hashing import sha256_file, sha256_obj
from sigir.io import read_decisions
from sigir.ledger import RunLedger
from sigir.schema import Decision, DecisionKind

MODEL = "gpt-4o-mini-2024-07-18"
SEED = 42
PROMPT = """Select a record from the following candidates that refers to the same real-world entity as the given record. Answer with the corresponding record number surrounded by "[]" or "[0]" if there is none.

Given entity record:
{anchor}

Candidate records:{candidates}"""
SOURCE = "https://github.com/tshu-w/ComEM/blob/main/src/selecting.py"


def read_ranks(path: Path, pools: dict) -> dict[str, list[int]]:
    ranks = {}
    with path.open(encoding="utf-8") as stream:
        for row in map(json.loads, stream):
            key = row["instance_id"]
            indices = row["top_indices"]
            pool = pools[key]
            if key in ranks or len(indices) != min(4, pool.k) or len(set(indices)) != len(indices):
                raise ValueError(f"Invalid or duplicate shortlist for {key}")
            if any(not isinstance(i, int) or not 0 <= i < pool.k for i in indices):
                raise ValueError(f"Shortlist index outside the original pool for {key}")
            ranks[key] = indices
    if ranks.keys() != pools.keys():
        raise ValueError(f"Missing shortlists for {len(pools.keys() - ranks.keys())} pools")
    return ranks


def select(client: OpenAI, inst, indices: list[int], order: dict) -> Decision:
    candidates = [render_record(inst, inst.candidates[index].fields, order) for index in indices]
    prompt = PROMPT.format(
        anchor=render_record(inst, inst.query_fields, order),
        candidates="".join(f"\n[{position}] {record}" for position, record in enumerate(candidates, 1)),
    )
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        seed=SEED, temperature=0.0, max_tokens=2048, logprobs=True, top_logprobs=3,
    )
    elapsed = (time.perf_counter() - start) * 1000
    if response.model != MODEL:
        raise RuntimeError(f"Expected selector {MODEL}, received {response.model}")
    choice = response.choices[0]
    reply = choice.message.content or ""
    bracket = re.search(r"\[(\d+)\]", reply)
    label = int(bracket[1]) if bracket else -1
    kind, index, row_id, reason = DecisionKind.FAIL, -1, None, "invalid_selection"
    if choice.finish_reason == "stop" and label == 0:
        kind, reason = DecisionKind.NONE, None
    elif choice.finish_reason == "stop" and 1 <= label <= len(indices):
        kind, index, reason = DecisionKind.SELECT, indices[label - 1], None
        row_id = inst.candidates[index].row_id
    return Decision(
        inst.instance_id, kind, index, row_id,
        latency_ms=elapsed,
        input_tokens=response.usage.prompt_tokens if response.usage else None,
        output_tokens=response.usage.completion_tokens if response.usage else None,
        failure_reason=reason,
        raw={
            "reply": reply, "resolved_model": response.model, "top_indices": indices,
            "finish_reason": choice.finish_reason, "request_id": response.id,
            "system_fingerprint": response.system_fingerprint,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instances", required=True, type=Path)
    p.add_argument("--ranks", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    setup_logging("INFO")
    for logger_name in ("httpx", "httpx2"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if args.workers < 1:
        p.error("--workers must be positive")
    rank_meta_path = args.ranks.with_suffix(args.ranks.suffix + ".meta.json")
    rank_meta = json.loads(rank_meta_path.read_text(encoding="utf-8"))
    input_hash = sha256_file(args.instances)
    manifest_hash = sha256_file(FROZEN_MANIFEST)
    if (rank_meta["instances_sha256"], rank_meta["manifest_sha256"], rank_meta["top_k"]) != (input_hash, manifest_hash, 4):
        raise ValueError("Ranker inputs or shortlist size differ from the selector inputs")
    pools = load_pool_rows(args.instances)
    ranks = read_ranks(args.ranks, pools)
    order = record_fields()
    provenance = {
        "instances_sha256": input_hash, "ranks_sha256": sha256_file(args.ranks),
        "rank_metadata_sha256": sha256_file(rank_meta_path),
        "benchmark_manifest_sha256": manifest_hash,
        "prompt_sha256": sha256_obj(PROMPT), "model": MODEL, "seed": SEED,
        "source": SOURCE, "serialization": "rich_v1", "shortlist_size": 4,
        "temperature": 0.0,
    }
    run_dir = args.runs_dir / args.run_id
    manifest = run_dir / "manifest.json"
    out = run_dir / "decisions.jsonl"
    if manifest.exists() and json.loads(manifest.read_text())["config"] != provenance:
        raise ValueError(f"Resume input or configuration changed for {args.run_id}")
    done = read_decisions(out) if out.exists() else {}
    if set(done) - pools.keys():
        raise ValueError("Existing decisions include pools outside this run")
    ledger = RunLedger(args.runs_dir, "comem_select", provenance, run_id=args.run_id)
    ledger.set("inputs", n_instances=len(pools), instances=str(args.instances), ranks=str(args.ranks))
    ledger.set("model", requested_model=MODEL, prompt_hash=provenance["prompt_sha256"])
    todo = [inst for inst in pools.values() if inst.instance_id not in done]
    log.info("CoMEM selector: %d remaining, %d already saved", len(todo), len(done))
    try:
        with OpenAI(timeout=45, max_retries=4) as client, ThreadPoolExecutor(args.workers) as workers, out.open("a", encoding="utf-8") as stream:
            for start in range(0, len(todo), 2 * args.workers):
                batch = todo[start : start + 2 * args.workers]
                futures = [workers.submit(select, client, inst, ranks[inst.instance_id], order) for inst in batch]
                errors = []
                for future in as_completed(futures):
                    try:
                        decision = future.result()
                    except Exception as exc:
                        errors.append(exc)
                        continue
                    stream.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
                    done[decision.instance_id] = decision
                stream.flush()
                if errors:
                    raise errors[0]
                if start % (10 * args.workers) == 0 or len(done) == len(pools):
                    log.info("CoMEM selector: %d/%d saved", len(done), len(pools))
        ledger.set("model", resolved_models=sorted({d.raw["resolved_model"] for d in done.values()}))
        ledger.set(
            "outputs", decisions=str(out), sha256=sha256_file(out), n_decisions=len(done),
            failures=sum(d.kind == DecisionKind.FAIL for d in done.values()),
            input_tokens=sum(d.input_tokens or 0 for d in done.values()),
            output_tokens=sum(d.output_tokens or 0 for d in done.values()),
        )
        ledger.finish("complete")
    except Exception as exc:
        ledger.failure("selector", type(exc).__name__, status_code=getattr(exc, "status_code", None))
        ledger.finish("failed")
        raise


if __name__ == "__main__":
    main()
