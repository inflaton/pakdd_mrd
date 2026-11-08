"""Trainer for the compact scorers and the decoder decision head; prediction utilities.

Matched optimizer steps across arms: the trainer counts optimizer steps and stops at `max_steps` when
set, so clean-only and intervention-trained models are compared at equal budgets (plan §8).
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval.metrics import compute_metrics, probabilistic_metrics
from ..eval.outcomes import score_run
from ..schema import Decision, DecisionInstance, decision_from_probs
from .dataset import Collator, InstanceDataset, batches_by_k, expected_epoch_size, move_batch


def peak_gpu_mem_gb(device: torch.device) -> float | None:
    """Peak memory allocated by PyTorch tensors since the last reset (GiB); None on CPU."""
    return round(torch.cuda.max_memory_allocated(device) / 2**30, 2) if device.type == "cuda" else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ListBatchSampler:
    def __init__(self, batches: list[list[int]]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _amp_dtype(precision: str) -> torch.dtype | None:
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)


@torch.no_grad()
def predict_compact(model, collator: Collator, instances: Sequence[DecisionInstance], batch_size: int, device: torch.device, precision: str = "bf16", time_each: bool = False) -> dict[str, Decision]:
    model.to(device)  # train_compact leaves the model on device_of() (CUDA when visible); predict where asked
    model.eval()
    out: dict[str, Decision] = {}
    amp = _amp_dtype(precision)
    order = sorted(range(len(instances)), key=lambda i: instances[i].k)
    for s in range(0, len(order), batch_size):
        idx = order[s : s + batch_size]
        insts = [instances[i] for i in idx]
        batch = move_batch(collator(insts), device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None and device.type == "cuda"):
            logp = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000 / len(insts)
        probs = logp.exp().float().cpu().numpy()
        for b, inst in enumerate(insts):
            p = list(probs[b, : inst.k]) + [float(probs[b, -1])]
            z = sum(p)
            p = [x / z for x in p] if z > 0 else p
            out[inst.instance_id] = decision_from_probs(inst, p, latency_ms=dt if time_each else None)
    return out


def evaluate_dev(model, collator, dev: Sequence[DecisionInstance], batch_size: int, device, precision: str) -> dict[str, float]:
    decs = predict_compact(model, collator, dev, batch_size, device, precision)
    m = compute_metrics(score_run(dev, decs))
    m.update(probabilistic_metrics(dev, decs))
    return m


def train_compact(
    model,
    collator: Collator,
    train: Sequence[DecisionInstance],
    dev: Sequence[DecisionInstance],
    *,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    grad_accum: int = 1,
    max_steps: int | None = None,
    precision: str = "bf16",
    seed: int = 0,
    eval_every_epoch: bool = True,
    select_metric: str = "set_nll",
    mixer=None,
    distractor_fn: Callable | None = None,
    missing_gold_fn: Callable | None = None,
    direction_weighting: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    set_seed(seed)
    device = device_of()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    ds = InstanceDataset(train, mixer=mixer, distractor_fn=distractor_fn, missing_gold_fn=missing_gold_fn)
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") or n.startswith("lm.")]
    head_params = [p for n, p in model.named_parameters() if not (n.startswith("encoder.") or n.startswith("lm."))]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": lr}, {"params": head_params, "lr": head_lr}], weight_decay=weight_decay)
    steps_per_epoch = math.ceil(expected_epoch_size(train, direction_weighting) / (batch_size * grad_accum))
    total = max_steps or steps_per_epoch * epochs
    warm = int(total * warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / max(1, warm))
    amp = _amp_dtype(precision)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16" and device.type == "cuda"))
    from ..models.compact import set_nll_loss

    history: list[dict[str, Any]] = []
    best = None
    step = 0
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        loader = DataLoader(ds, batch_sampler=ListBatchSampler(batches_by_k(train, batch_size, seed + epoch, weighted=direction_weighting)), collate_fn=collator, num_workers=0)
        running = 0.0
        n_b = 0
        for i, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None and device.type == "cuda"):
                logp = model(batch)
            loss = set_nll_loss(logp.float(), batch["targets"]) / grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * grad_accum
            n_b += 1
            if (i + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if step % 50 == 0:
                    log(f"epoch {epoch} step {step}/{total} loss {running / n_b:.4f} lr {sched.get_last_lr()[0]:.2e}")
                if max_steps and step >= max_steps:
                    break
        rec: dict[str, Any] = {"epoch": epoch, "step": step, "train_loss": running / max(1, n_b), "elapsed_s": time.time() - t_start}
        if mixer is not None:
            rec["mixture"] = mixer.report()
        if eval_every_epoch and dev:
            m = evaluate_dev(model, collator, dev, batch_size, device, precision)
            rec["dev"] = m
            score = m[select_metric] if select_metric == "set_nll" else -m[select_metric]
            if best is None or score < best:
                best = score
                torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict(), "epoch": epoch, "step": step, "dev": m}, out_dir / "best.pt")
                rec["saved_best"] = True
            log(f"epoch {epoch} dev set_nll {m['set_nll']:.4f} recall {m['match_recall']:.4f} precision {m['precision']:.4f} nfmr {m['n_fmr']:.4f} f05 {m['f05']:.2f}")
        history.append(rec)
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, default=str)
        if max_steps and step >= max_steps:
            break
    torch.save({"state_dict": model.state_dict(), "config": model.cfg.to_dict(), "epoch": epochs, "step": step}, out_dir / "last.pt")
    return {"history": history, "steps": step, "best_dev": best, "train_seconds": time.time() - t_start, "peak_gpu_mem_gb": peak_gpu_mem_gb(device), "direction_weighting": direction_weighting, "expected_epoch_size": expected_epoch_size(train, direction_weighting)}
