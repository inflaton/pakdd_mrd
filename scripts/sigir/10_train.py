#!/usr/bin/env python
"""Step 10: train a compact scorer arm (pairwise / set joint / set factorized / bi) with provenance.

Examples
  python scripts/sigir/10_train.py --config models/set_model_joint.yaml --seed 0
  python scripts/sigir/10_train.py --config models/set_model_joint.yaml --override train.intervention_training=false --name set_joint_clean
  python scripts/sigir/10_train.py --config models/pairwise_encoder.yaml --n-groups 1000 --variants-per-group 1 --max-steps 1200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import MODELS_DIR, REPO, add_common, benchmark_instances, dataset_list, dump_json, field_maps, log, setup_logging
from sigir.data.serialization import TEXT_NORMALIZATION
from sigir.config import load_config
from sigir.data.aux_pools import load_aux_pools
from sigir.data.interventions import InterventionMixer
from sigir.hashing import sha256_obj
from sigir.ledger import RunLedger
from sigir.models.compact import CompactConfig, CompactScorer, count_parameters, load_tokenizer
from sigir.train.dataset import Collator, subsample_groups
from sigir.train.trainer import train_compact

p = argparse.ArgumentParser()
p.add_argument("--config", required=True, help="e.g. models/set_model_joint.yaml")
p.add_argument("--override", nargs="*", default=[])
p.add_argument("--name", default=None)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n-groups", type=int, default=None, help="learning curve: number of independent training groups")
p.add_argument("--variants-per-group", type=int, default=None)
p.add_argument("--max-steps", type=int, default=None, help="matched optimizer-step budget")
p.add_argument("--dev-limit", type=int, default=None)
p.add_argument("--out-dir", default=None)
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
cfg = load_config(a.config, a.override)
name = a.name or cfg["name"]
tag = f"{name}_seed{a.seed}" + (f"_g{a.n_groups}" if a.n_groups else "") + (f"_v{a.variants_per_group}" if a.variants_per_group else "")
out_dir = Path(a.out_dir) if a.out_dir else MODELS_DIR / tag
out_dir.mkdir(parents=True, exist_ok=True)
ledger = RunLedger(a.runs_dir, f"train_{tag}", {"model_config": cfg, "args": vars(a)})
try:
    train, man = benchmark_instances(a.benchmark_dir, {"train"}, dataset_list(a))
    dev, _ = benchmark_instances(a.benchmark_dir, {"dev"}, dataset_list(a))
    if a.dev_limit:
        dev = dev[: a.dev_limit]
    train = subsample_groups(train, a.n_groups, a.variants_per_group, a.seed)
    log.info("train instances %d (groups %d), dev %d", len(train), len({i.entity_group for i in train}), len(dev))
    ledger.set("inputs", benchmark_manifest_hash=man.get("benchmark_config_hash"), instances_hashes={d: v["instances_hash"] for d, v in man["datasets"].items()}, n_train=len(train), n_train_groups=len({i.entity_group for i in train}), n_dev=len(dev), train_ids_hash=sha256_obj(sorted(i.instance_id for i in train)))
    order, compact, _ = field_maps(man)
    mcfg = CompactConfig(**cfg["model"])
    model = CompactScorer(mcfg)
    tok = load_tokenizer(mcfg.backbone, mcfg.pin().get("revision"))
    collator = Collator(tok, mcfg.encoder_mode, mcfg.max_len, cfg["train"].get("serialization", "rich_v1"), order, compact, use_features=mcfg.use_retrieval_features)
    ledger.set("model", backbone=mcfg.backbone, config=mcfg.to_dict(), parameters=count_parameters(model), tokenizer=tok.name_or_path, tokenizer_vocab_hash=sha256_obj(sorted(tok.get_vocab().items()))[:16], precision=cfg["train"].get("precision", "bf16"), weight_dtype=str(next(model.encoder.parameters()).dtype), backbone_pin=mcfg.pin() or None, backbone_commit=getattr(model.encoder.config, "_commit_hash", None), text_normalization=TEXT_NORMALIZATION, backend="torch")
    ledger.set("seeds", training=a.seed)
    tcfg = cfg["train"]
    mixer = None
    distractor_fn = missing_gold_fn = None
    if tcfg.get("intervention_training"):
        aux = {}
        for ds in man["datasets"]:
            if dataset_list(a) and ds not in dataset_list(a):
                continue
            ap = Path(a.benchmark_dir) / ds / "aux_pools.jsonl"
            if ap.exists():
                aux.update(load_aux_pools(ap))
        weak = any(i.label_provenance.startswith("teacher") for i in train)
        mixer = InterventionMixer(tcfg.get("intervention_families", ["permutation", "add_hard_nonmatch", "duplicate_record", "remove_all_acceptable"]), tcfg.get("p_transform", 0.5), a.seed, weak_labels=weak, ablate_label_changing=bool(tcfg.get("ablate_label_changing")))
        distractor_fn = lambda inst: aux.get(inst.instance_id, {}).get("distractors", [])  # noqa: E731
        missing_gold_fn = lambda inst: aux.get(inst.instance_id, {}).get("missing_gold", [])  # noqa: E731
        ledger.note(f"intervention training on; families {mixer.families}; weak_labels={weak}")
    result = train_compact(
        model, collator, train, dev, out_dir=out_dir, epochs=int(tcfg["epochs"]), batch_size=int(tcfg["batch_size"]), lr=float(tcfg["lr"]), head_lr=float(tcfg.get("head_lr", 1e-4)), weight_decay=float(tcfg.get("weight_decay", 0.01)), warmup_ratio=float(tcfg.get("warmup_ratio", 0.06)), grad_accum=int(tcfg.get("grad_accum", 1)), max_steps=a.max_steps or tcfg.get("max_steps"), precision=tcfg.get("precision", "bf16"), seed=a.seed, mixer=mixer, distractor_fn=distractor_fn, missing_gold_fn=missing_gold_fn, direction_weighting=bool(tcfg.get("direction_weighting", True)), log=log.info
    )
    dump_json(out_dir / "train_config.json", {"config": cfg, "args": vars(a), "run_id": ledger.run_id, "result": {k: v for k, v in result.items() if k != "history"}})
    ledger.set("outputs", checkpoint=str(out_dir / "best.pt"), last=str(out_dir / "last.pt"), steps=result["steps"], best_dev_set_nll=result["best_dev"])
    ledger.set("timing", train_seconds=result["train_seconds"], peak_gpu_mem_gb=result.get("peak_gpu_mem_gb"))
    ledger.set("outputs", direction_weighting=result.get("direction_weighting"), expected_epoch_size=result.get("expected_epoch_size"))
    if mixer is not None:
        ledger.set("outputs", mixture=mixer.report())
    ledger.finish("complete")
    log.info("done: %s (run %s)", out_dir, ledger.run_id)
except Exception as e:
    ledger.failure("train", repr(e))
    ledger.finish("failed")
    raise
