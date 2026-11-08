"""Torch datasets and collation for variable-K candidate sets.

`InstanceDataset` holds DecisionInstances; an optional `InterventionMixer` transforms items on the fly
(50/50 recipe). `Collator` tokenizes per batch (cross or bi mode) and builds slot masks, retrieval
features and multi-hot targets (NONE column when the acceptable set is empty).
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

import torch
from torch.utils.data import Dataset

from ..data.interventions import InterventionMixer
from ..data.serialization import pair_text, serialize_record
from ..schema import Candidate, DecisionInstance


class InstanceDataset(Dataset):
    def __init__(self, instances: Sequence[DecisionInstance], mixer: InterventionMixer | None = None, distractor_fn: Callable[[DecisionInstance], Sequence[Candidate]] | None = None, missing_gold_fn: Callable[[DecisionInstance], Sequence[Candidate]] | None = None):
        self.instances = list(instances)
        self.mixer = mixer
        self.distractor_fn = distractor_fn
        self.missing_gold_fn = missing_gold_fn

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, i: int) -> DecisionInstance:
        inst = self.instances[i]
        if self.mixer is None:
            return inst
        d = self.distractor_fn(inst) if self.distractor_fn else ()
        g = self.missing_gold_fn(inst) if self.missing_gold_fn else ()
        return self.mixer(inst, distractors=d, missing_gold=g)


def targets_multi_hot(insts: Sequence[DecisionInstance], k_max: int) -> torch.Tensor:
    y = torch.zeros(len(insts), k_max + 1)
    for b, inst in enumerate(insts):
        acc = inst.acceptable_indices()
        if acc:
            y[b, acc] = 1.0
        else:
            y[b, k_max] = 1.0
    return y


def retrieval_features(insts: Sequence[DecisionInstance], k_max: int) -> torch.Tensor:
    """[B, Kmax, 2]: normalized retrieval score within the pool and 1/(1+rank). Travel with the candidate."""
    f = torch.zeros(len(insts), k_max, 2)
    for b, inst in enumerate(insts):
        scores = [c.retrieval_score if c.retrieval_score is not None else 0.0 for c in inst.candidates]
        mx = max(scores) if scores else 1.0
        mx = mx if mx else 1.0
        for j, c in enumerate(inst.candidates):
            f[b, j, 0] = scores[j] / mx
            f[b, j, 1] = 1.0 / (1.0 + (c.retrieval_rank if c.retrieval_rank is not None else j))
    return f


class Collator:
    def __init__(self, tokenizer, mode: str, max_len: int, serialization: str, field_order: dict[str, list[str]] | None = None, compact_fields: dict[str, list[str]] | None = None, use_features: bool = True):
        self.tok = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.serialization = serialization
        self.field_order = field_order or {}
        self.compact_fields = compact_fields or {}
        self.use_features = use_features

    def _ser(self, ds: str, fields: dict[str, Any]) -> str:
        return serialize_record(fields, self.field_order.get(ds), self.serialization, self.compact_fields.get(ds))

    def __call__(self, insts: Sequence[DecisionInstance]) -> dict[str, Any]:
        B = len(insts)
        k_max = max(1, max(i.k for i in insts))
        slot_mask = torch.zeros(B, k_max, dtype=torch.bool)
        for b, inst in enumerate(insts):
            slot_mask[b, : inst.k] = True
        batch: dict[str, Any] = {"slot_mask": slot_mask, "targets": targets_multi_hot(insts, k_max), "instances": insts}
        if self.use_features:
            batch["features"] = retrieval_features(insts, k_max)
        q_text = [self._ser(i.dataset, i.query_fields) for i in insts]
        if self.mode == "cross":
            pairs = []
            for b, inst in enumerate(insts):
                for j in range(k_max):
                    pairs.append(pair_text(q_text[b], self._ser(inst.dataset, inst.candidates[j].fields)) if j < inst.k else "")
            enc = self.tok(pairs, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
            batch["pair_input_ids"], batch["pair_attention_mask"] = enc["input_ids"], enc["attention_mask"]
        else:
            qe = self.tok(q_text, padding=True, truncation=True, max_length=self.max_len // 2, return_tensors="pt")
            batch["q_input_ids"], batch["q_attention_mask"] = qe["input_ids"], qe["attention_mask"]
            cands = []
            for b, inst in enumerate(insts):
                for j in range(k_max):
                    cands.append(self._ser(inst.dataset, inst.candidates[j].fields) if j < inst.k else "")
            ce = self.tok(cands, padding=True, truncation=True, max_length=self.max_len // 2, return_tensors="pt")
            batch["c_input_ids"], batch["c_attention_mask"] = ce["input_ids"], ce["attention_mask"]
        return batch


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def subsample_groups(instances: Sequence[DecisionInstance], n_groups: int | None, variants_per_group: int | None, seed: int) -> list[DecisionInstance]:
    """Learning-curve control: vary independent training groups separately from variants per group."""
    rng = random.Random(seed)
    by_g: dict[str, list[DecisionInstance]] = {}
    for i in instances:
        by_g.setdefault(i.entity_group, []).append(i)
    groups = sorted(by_g)
    rng.shuffle(groups)
    if n_groups is not None:
        groups = groups[:n_groups]
    out: list[DecisionInstance] = []
    for g in groups:
        items = sorted(by_g[g], key=lambda x: x.instance_id)
        if variants_per_group is not None:
            items = items[:variants_per_group]
        out.extend(items)
    return out


def sampling_weight(inst: DecisionInstance) -> float:
    return float(inst.extra.get("sampling_weight", 1.0))


def epoch_indices(instances: Sequence[DecisionInstance], seed: int) -> list[int]:
    """Frozen direction weights as per-epoch sampling (plan §5): weight >= 1 always kept, weight w < 1 kept with
    probability w, redrawn every epoch. Walmart-Amazon B->A (weight 0.25, 12,204 of 17,353 training queries) would
    otherwise dominate training."""
    rng = random.Random(f"direction-weights|{seed}")
    return [i for i, inst in enumerate(instances) if (w := sampling_weight(inst)) >= 1.0 or rng.random() < w]


def expected_epoch_size(instances: Sequence[DecisionInstance], weighted: bool) -> float:
    return sum(min(1.0, sampling_weight(i)) for i in instances) if weighted else float(len(instances))


def batches_by_k(instances: Sequence[DecisionInstance], batch_size: int, seed: int, weighted: bool = False) -> list[list[int]]:
    """Group indices with similar K to limit padding; shuffle within and across buckets.

    weighted=True applies the direction sampling weights (training only; never for prediction)."""
    rng = random.Random(seed)
    idx = epoch_indices(instances, seed) if weighted else list(range(len(instances)))
    rng.shuffle(idx)
    idx.sort(key=lambda i: instances[i].k)
    batches = [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]
    rng.shuffle(batches)
    return batches
