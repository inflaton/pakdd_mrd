"""YAML config loading with `key.sub=value` overrides and repository-relative paths."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(os.environ.get("SIGIR_REPO_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = REPO_ROOT / "configs" / "sigir"
DATA_DIR = Path(os.environ.get("SIGIR_DATA_DIR", REPO_ROOT / "data" / "sigir"))
MODELS_DIR = Path(os.environ.get("SIGIR_MODELS_DIR", REPO_ROOT / "models" / "sigir"))
RUNS_DIR = Path(os.environ.get("SIGIR_RUNS_DIR", REPO_ROOT / "runs"))


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists() and (CONFIG_DIR / path).exists():
        p = CONFIG_DIR / path
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _coerce(v: str) -> Any:
    try:
        return yaml.safe_load(v)
    except Exception:
        return v


def apply_overrides(cfg: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_overrides(_load_with_extends(path), overrides)


def _load_with_extends(path: str | Path, depth: int = 0) -> dict[str, Any]:
    """`extends: other.yaml` resolved recursively (Nov 6, 2026: single-level resolution left
    set_model_factorized / set_model_bi_cached / open_decision_model without their grandparent's keys)."""
    if depth > 8:
        raise ValueError(f"config inheritance deeper than 8 levels at {path}")
    cfg = load_yaml(path)
    if "extends" in cfg:
        merged = copy.deepcopy(_load_with_extends(cfg.pop("extends"), depth + 1))
        _deep_update(merged, cfg)
        cfg = merged
    return cfg


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def ensure_dirs() -> None:
    for d in (DATA_DIR / "public", DATA_DIR / "private", MODELS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
