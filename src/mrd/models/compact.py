"""Compact decision models over a candidate set (plan §4).

One module covers the two required trainable arms and the two ablations:

  arm                            encoder_mode   set_attention   formulation
  compact pairwise + rejection   cross | bi     False           joint
  compact set model (main)       cross | bi     True            joint
  factorized ablation            cross | bi     True            factorized

* cross: a shared query-candidate cross-encoder produces h_j = CLS([q ; c_j]).
* bi:    separate encodings u = f(q), v_j = f(c_j); h_j = MLP([u, v_j, u*v_j, |u-v_j|]). Candidate
         encodings are cacheable (efficiency experiment).
* Set attention: Transformer encoder layers over the K slots with NO positional encoding. A learned
  NONE element joins the set; its output vector produces the NONE logit, so NONE can "look at" every
  candidate. Retrieval rank/score are explicit per-candidate features (optional) and travel with the
  candidate under permutation.
* Joint: softmax over [cand_0..cand_{K-1}, NONE]. Factorized: a = sigmoid(matchability), r = softmax
  over candidates; P(j) = a r_j, P(NONE) = 1 - a.

Outputs are log-probabilities of shape [B, Kmax+1] with padded slots at -inf.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


@dataclass
class CompactConfig:
    backbone: str = "microsoft/deberta-v3-small"
    encoder_mode: str = "cross"  # cross | bi
    set_attention: bool = True
    n_set_layers: int = 2
    n_set_heads: int = 8
    set_dropout: float = 0.1
    formulation: str = "joint"  # joint | factorized
    use_retrieval_features: bool = True
    max_len: int = 256
    proj_dim: int | None = None  # project encoder hidden to this size before set layers (None = keep)
    gradient_checkpointing: bool = False
    attn_implementation: str = "sdpa"
    # backbone -> {revision, use_safetensors}: exact weights per backbone, so a --override of model.backbone
    # never inherits another backbone's pin. Unpinned backbones load the hub default (recorded in the ledger).
    backbone_pins: dict[str, dict[str, Any]] | None = None

    def pin(self) -> dict[str, Any]:
        return dict((self.backbone_pins or {}).get(self.backbone, {}))

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class CompactScorer(nn.Module):
    def __init__(self, cfg: CompactConfig, encoder_config: Any | None = None):
        """encoder_config (a transformers config): build the backbone architecture only, without downloading its
        pretrained weights, for when a checkpoint's state dict supplies every weight (demo/export_checkpoints.py)."""
        super().__init__()
        self.cfg = cfg
        pin = {k: v for k, v in cfg.pin().items() if k in ("revision", "use_safetensors")}
        kwargs = dict(pin)
        if cfg.attn_implementation:
            kwargs["attn_implementation"] = cfg.attn_implementation
        if encoder_config is not None:
            attn = {"attn_implementation": cfg.attn_implementation} if cfg.attn_implementation else {}
            try:  # copies: a refused attn_implementation is left set on the config it was given
                self.encoder = AutoModel.from_config(copy.deepcopy(encoder_config), **attn)
            except (ValueError, TypeError):
                self.encoder = AutoModel.from_config(copy.deepcopy(encoder_config))
        else:
            try:
                self.encoder = AutoModel.from_pretrained(cfg.backbone, **kwargs)
            except (ValueError, TypeError):
                self.encoder = AutoModel.from_pretrained(cfg.backbone, **pin)
        # Master weights in fp32. transformers 5 loads checkpoints in their stored dtype, and
        # microsoft/deberta-v3-small is stored in fp16: AdamW on fp16 weights turned every encoder
        # parameter into NaN after one update (pilot, 4 Nov 2026). bf16/fp16 compute comes from autocast.
        self.encoder = self.encoder.float()
        if cfg.gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        h_enc = self.encoder.config.hidden_size
        self.hidden = cfg.proj_dim or h_enc
        if cfg.encoder_mode == "cross":
            self.pair_proj = nn.Linear(h_enc, self.hidden) if cfg.proj_dim else nn.Identity()
        elif cfg.encoder_mode == "bi":
            self.pair_proj = nn.Sequential(nn.Linear(4 * h_enc, self.hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(self.hidden, self.hidden))
        else:
            raise ValueError(cfg.encoder_mode)
        self.feat_proj = nn.Linear(2, self.hidden) if cfg.use_retrieval_features else None
        self.none_token = nn.Parameter(torch.randn(self.hidden) * 0.02)
        if cfg.set_attention:
            layer = nn.TransformerEncoderLayer(self.hidden, cfg.n_set_heads, dim_feedforward=4 * self.hidden, dropout=cfg.set_dropout, batch_first=True, norm_first=True, activation="gelu")
            self.set_layers = nn.TransformerEncoder(layer, cfg.n_set_layers, enable_nested_tensor=False)
        else:
            self.set_layers = None
        self.cand_head = nn.Linear(self.hidden, 1)
        self.none_head = nn.Linear(self.hidden, 1)
        self.match_head = nn.Linear(self.hidden, 1)  # factorized: P(some candidate acceptable)

    # ------------------------------------------------------------------ encoding
    def _cls(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        if getattr(self.encoder.config, "model_type", "") in ("bert", "roberta", "deberta", "deberta-v2", "modernbert", "electra", "xlm-roberta", "mpnet"):
            return hs[:, 0]
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        return (hs * m).sum(1) / m.sum(1).clamp(min=1)

    def encode_candidates(self, cand_ids: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        """bi mode only: [N, L] -> [N, H]. Used for the cached-candidate-encoding experiment."""
        assert self.cfg.encoder_mode == "bi"
        return self._cls(cand_ids, cand_mask)

    def slot_vectors(self, batch: dict[str, torch.Tensor], cached_cand_vecs: torch.Tensor | None = None) -> torch.Tensor:
        """Return h: [B, Kmax, H]."""
        B, K = batch["slot_mask"].shape
        if self.cfg.encoder_mode == "cross":
            ids, am = batch["pair_input_ids"], batch["pair_attention_mask"]  # [B*K, L]
            flat = batch["slot_mask"].reshape(-1)
            h = torch.zeros(B * K, self.encoder.config.hidden_size, device=ids.device, dtype=self.encoder.dtype if hasattr(self.encoder, "dtype") else torch.float32)
            if flat.any():
                h_valid = self._cls(ids[flat], am[flat])
                h = h.to(h_valid.dtype)
                h[flat] = h_valid
            h = self.pair_proj(h).view(B, K, -1)
        else:
            u = self._cls(batch["q_input_ids"], batch["q_attention_mask"])  # [B, H]
            if cached_cand_vecs is not None:
                v = cached_cand_vecs
            else:
                flat = batch["slot_mask"].reshape(-1)
                v = torch.zeros(B * K, u.shape[-1], device=u.device, dtype=u.dtype)
                if flat.any():
                    v[flat] = self._cls(batch["c_input_ids"][flat], batch["c_attention_mask"][flat])
                v = v.view(B, K, -1)
            uu = u.unsqueeze(1).expand_as(v)
            h = self.pair_proj(torch.cat([uu, v, uu * v, (uu - v).abs()], dim=-1))
        if self.feat_proj is not None and "features" in batch:
            h = h + self.feat_proj(batch["features"].to(h.dtype))
        return h

    # ------------------------------------------------------------------ forward
    def forward(self, batch: dict[str, torch.Tensor], cached_cand_vecs: torch.Tensor | None = None) -> torch.Tensor:
        slot_mask = batch["slot_mask"]  # [B, K] bool
        h = self.slot_vectors(batch, cached_cand_vecs)  # [B, K, H]
        B, K, H = h.shape
        none = self.none_token.to(h.dtype).view(1, 1, H).expand(B, 1, H)
        x = torch.cat([h, none], dim=1)  # [B, K+1, H]
        full_mask = torch.cat([slot_mask, torch.ones(B, 1, dtype=torch.bool, device=h.device)], dim=1)
        if self.set_layers is not None:
            x = self.set_layers(x, src_key_padding_mask=~full_mask)
        else:
            # pairwise arm: NONE element sees a masked mean of candidate vectors (no inter-candidate interaction)
            m = slot_mask.unsqueeze(-1).to(x.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            x = torch.cat([h, (pooled + self.none_token.to(h.dtype)).unsqueeze(1)], dim=1)
        cand_logits = self.cand_head(x[:, :K]).squeeze(-1).float()  # [B, K]
        cand_logits = cand_logits.masked_fill(~slot_mask, float("-inf"))
        none_vec = x[:, K]
        if self.cfg.formulation == "joint":
            none_logit = self.none_head(none_vec).float()  # [B, 1]
            logits = torch.cat([cand_logits, none_logit], dim=1)
            # K = 0 -> only NONE is finite -> log_softmax gives 0 for NONE
            return F.log_softmax(logits, dim=-1)
        # factorized
        a_logit = self.match_head(none_vec).squeeze(-1).float()  # [B]
        log_a = F.logsigmoid(a_logit)
        log_1ma = F.logsigmoid(-a_logit)
        has_cand = slot_mask.any(1)
        log_r = F.log_softmax(cand_logits, dim=-1)  # -inf on padded
        log_pj = log_a.unsqueeze(1) + log_r
        log_none = torch.where(has_cand, log_1ma, torch.zeros_like(log_1ma))
        log_pj = torch.where(has_cand.unsqueeze(1), log_pj, torch.full_like(log_pj, float("-inf")))
        return torch.cat([log_pj, log_none.unsqueeze(1)], dim=1)


def set_nll_loss(log_probs: torch.Tensor, target_multi_hot: torch.Tensor) -> torch.Tensor:
    """-log sum_{j in Y} p_j, with Y = {NONE} for non-matchable rows. target_multi_hot: [B, K+1] in {0,1}."""
    masked = log_probs.masked_fill(target_multi_hot <= 0, float("-inf"))
    return -(torch.logsumexp(masked, dim=-1)).mean()


def load_tokenizer(name: str, revision: str | None = None):
    return AutoTokenizer.from_pretrained(name, use_fast=True, revision=revision)


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
