#!/usr/bin/env python
"""Rank frozen candidate pools with the ComEM MatchingSQ Flan-T5-XL stage.

The output keeps zero-based indices into the supplied pool. Multi-query blocks
record amortized timing. No outcome metrics or label-dependent decisions are made.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import comem_inputs
from comem_inputs import FROZEN_MANIFEST, load_pool_rows, record_fields, render_record
from mrd.data.serialization import TEXT_NORMALIZATION
from mrd.hashing import sha256_file, sha256_text
from mrd.io import read_jsonl
from mrd.schema import DecisionInstance


MODEL = "google/flan-t5-xl"
REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
SERIALIZATION = "rich_v1"
TOP_K = 4
CHECKPOINT_QUERIES = 32
# MatchingSQ's Jinja template, rendered without the stripped final newline.
PROMPT = (
    'Do the two entity records refer to the same real-world entity? '
    'Answer "Yes" if they do and "No" if they do not.\n\n'
    'Record 1: {query}\nRecord 2: {candidate}'
)


def replace_text(path: Path, lines) -> None:
    """Keep the previous complete checkpoint if writing is interrupted."""
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.writelines(lines)
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


class MatchingSQRanker:
    def __init__(self, device: str, dtype, batch_size: int, local_files_only: bool):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers.modeling_outputs import BaseModelOutput

        self.torch = torch
        self.encoder_output_type = BaseModelOutput
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=local_files_only)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL, revision=REVISION, local_files_only=local_files_only, torch_dtype=dtype,
        ).to(device).eval()
        labels = self.tokenizer(text_target=["Yes", "No"], padding=True, return_tensors="pt").input_ids.to(device)
        self.labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        self.decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=self.labels)

    def score_batch(self, prompts: list[str]) -> list[float]:
        torch = self.torch
        encoded = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model.get_encoder()(**encoded, return_dict=True).last_hidden_state
            # The same source is used for Yes and No. Reuse its encoder states
            # while teacher-forcing both answer sequences, including their EOS.
            encoder_outputs = self.encoder_output_type(last_hidden_state=hidden.repeat_interleave(2, dim=0))
            logits = self.model(
                encoder_outputs=encoder_outputs,
                attention_mask=encoded.attention_mask.repeat_interleave(2, dim=0),
                decoder_input_ids=self.decoder_input_ids.repeat(len(prompts), 1),
                use_cache=False,
                return_dict=True,
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), self.labels.repeat(len(prompts), 1).reshape(-1), reduction="none",
            )
            log_probs = -loss.reshape(len(prompts), 2, -1).mean(dim=-1)
        scores = [math.exp(yes) if yes >= no else -math.exp(no) for yes, no in log_probs.tolist()]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("non-finite MatchingSQ score")
        return scores

    def rank_block(self, block: list[DecisionInstance], order: dict[str, list[str]], batched: bool) -> list[dict]:
        start = perf_counter()
        prompts_by_pool = []
        for inst in block:
            query = render_record(inst, inst.query_fields, order)
            prompts_by_pool.append([
                PROMPT.format(query=query, candidate=render_record(inst, candidate.fields, order))
                for candidate in inst.candidates
            ])
        prompts = list(dict.fromkeys(prompt for pool in prompts_by_pool for prompt in pool))
        if batched:
            prompts.sort(key=len)
        scores = {}
        for start_index in range(0, len(prompts), self.batch_size):
            batch = prompts[start_index:start_index + self.batch_size]
            scores.update(zip(batch, self.score_batch(batch), strict=True))
        rows = []
        for inst, pool in zip(block, prompts_by_pool, strict=True):
            # MatchingSQ.pointwise_rank sorts (score, original index) in reverse order.
            ranked = sorted(range(inst.k), key=lambda index: (scores[pool[index]], index), reverse=True)
            rows.append({"instance_id": inst.instance_id, "top_indices": ranked[:TOP_K]})
        elapsed = (perf_counter() - start) * 1000
        for row in rows:
            row["latency_ms"] = None if batched else elapsed
            if batched:
                row["amortized_latency_ms"] = elapsed / len(block)
        return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True, help="frozen DecisionInstance JSONL or JSONL.gz")
    parser.add_argument("--out", type=Path, required=True, help="resumable output JSONL")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"))
    parser.add_argument("--batch-size", type=int, default=8, help="pair prompts per model forward pass")
    parser.add_argument("--query-batch-size", type=int, default=1, help="query pools per ranking block; >1 amortizes latency")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.query_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.out.suffix == ".gz":
        parser.error("--out must be a plain JSONL file")
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    if args.instances.resolve() in {args.out.resolve(), meta_path.resolve()}:
        parser.error("input and output paths must be distinct")

    import torch
    import transformers

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float32 if device == "cpu" else torch.float16
    meta = {
        "instances_sha256": sha256_file(args.instances),
        "manifest_sha256": sha256_file(FROZEN_MANIFEST),
        "script_sha256": sha256_file(__file__),
        "input_helper_sha256": sha256_file(comem_inputs.__file__),
        "model": MODEL,
        "revision": REVISION,
        "score": "comem_matching_sq_signed_sequence_probability_including_eos",
        "prompt_sha256": sha256_text(PROMPT),
        "serialization": SERIALIZATION,
        "text_normalization": TEXT_NORMALIZATION,
        "top_k": TOP_K,
        "device": device,
        "dtype": str(dtype),
        "batch_size": args.batch_size,
        "query_batch_size": args.query_batch_size,
        "timing": "amortized_query_block_wall_time" if args.query_batch_size > 1 else "query_wall_time",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != meta:
            parser.error("existing output metadata differs from this run; choose a new --out")
    elif args.out.exists():
        parser.error("existing output is missing its metadata; choose a new --out")

    instances = load_pool_rows(args.instances)
    rows = list(read_jsonl(args.out)) if args.out.exists() else []
    done = {row["instance_id"] for row in rows}
    if len(done) != len(rows) or not done <= instances.keys():
        parser.error("existing output contains duplicate or unknown instance IDs")
    todo = [inst for inst in instances.values() if inst.instance_id not in done]
    batched = args.query_batch_size > 1
    if batched:
        # Keep natural and constructed pools of the same query close enough to
        # reuse duplicate pair prompts within a block.
        todo.sort(key=lambda inst: (inst.dataset, inst.direction, inst.query_row_id))
    print(f"Ranking {len(todo)} pools on {device} ({len(rows)} already done)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not meta_path.exists():
        replace_text(meta_path, [json.dumps(meta, indent=2) + "\n"])
    if not todo:
        if not args.out.exists():
            replace_text(args.out, [])
        return

    order = record_fields()
    ranker = MatchingSQRanker(device, dtype, args.batch_size, args.local_files_only)
    for start in range(0, len(todo), args.query_batch_size):
        block = todo[start:start + args.query_batch_size]
        rows.extend(ranker.rank_block(block, order, batched))
        count = start + len(block)
        if count // CHECKPOINT_QUERIES > start // CHECKPOINT_QUERIES or count == len(todo):
            replace_text(args.out, (json.dumps(row) + "\n" for row in rows))
            print(f"Saved {len(rows)}/{len(instances)} rankings", flush=True)


if __name__ == "__main__":
    main()
