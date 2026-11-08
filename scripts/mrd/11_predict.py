#!/usr/bin/env python
"""Step 11: run a trained compact checkpoint on partitions (or an explicit instances file) -> decisions JSONL with probabilities.

  python scripts/mrd/11_predict.py --checkpoint models/mrd/set_joint_seed0/best.pt --partitions dev,calib_fit,calib_confirm,test
  python scripts/mrd/11_predict.py --checkpoint ... --instances-file runs/<robustness>/variants.jsonl
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import add_common, benchmark_instances, dataset_list, field_maps, log, setup_logging, cohort_list
from mrd.data.serialization import TEXT_NORMALIZATION
from mrd.io import read_instances, write_decisions
from mrd.ledger import RunLedger
from mrd.models.compact import CompactConfig, CompactScorer, load_tokenizer
from mrd.train.dataset import Collator
from mrd.train.trainer import device_of, predict_compact

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--partitions", default="dev,calib_fit,calib_confirm,test")
p.add_argument("--instances-file", default=None, help="predict on this file instead of benchmark partitions")
p.add_argument("--serialization", default="rich_v1")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--precision", default="bf16")
p.add_argument("--name", default=None)
p.add_argument("--out", default=None, help="decisions path; default runs/<run>/decisions.jsonl")
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
mcfg = CompactConfig(**ck["config"])
name = a.name or Path(a.checkpoint).parent.name
ledger = RunLedger(a.runs_dir, f"predict_{name}", {"checkpoint": a.checkpoint, "args": vars(a)})
try:
    if a.instances_file:
        insts = read_instances(a.instances_file)
        from _common import benchmark_instances as _bi

        _, man = _bi(a.benchmark_dir, set(), None)
    else:
        insts, man = benchmark_instances(a.benchmark_dir, set(a.partitions.split(",")), dataset_list(a), cohort_list(a))
    order, compact, _ = field_maps(man)
    model = CompactScorer(mcfg)
    model.load_state_dict(ck["state_dict"])
    device = device_of()
    model.to(device)
    tok = load_tokenizer(mcfg.backbone, mcfg.pin().get("revision"))
    collator = Collator(tok, mcfg.encoder_mode, mcfg.max_len, a.serialization, order, compact, use_features=mcfg.use_retrieval_features)
    ledger.set("model", checkpoint=a.checkpoint, backbone=mcfg.backbone, config=mcfg.to_dict(), precision=a.precision, epoch=ck.get("epoch"), step=ck.get("step"), text_normalization=TEXT_NORMALIZATION)
    ledger.set("inputs", n_instances=len(insts), partitions=a.partitions if not a.instances_file else None, instances_file=a.instances_file, instances_hashes={d: v["instances_hash"] for d, v in man["datasets"].items()})
    decs = predict_compact(model, collator, insts, a.batch_size, device, a.precision, time_each=True)
    out = Path(a.out) if a.out else ledger.path("decisions.jsonl")
    n = write_decisions(out, decs.values())
    ledger.set("outputs", decisions=str(out), n=n)
    ledger.finish("complete")
    log.info("wrote %d decisions to %s", n, out)
except Exception as e:
    ledger.failure("predict", repr(e))
    ledger.finish("failed")
    raise
