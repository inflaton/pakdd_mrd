"""Shared utilities for the manuscript results notebook."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
DATASETS = ["abt_buy", "dblp_scholar", "walmart_amazon", "wdc_products"]
DS_LABEL = {"abt_buy": "Abt-Buy", "dblp_scholar": "DBLP-Scholar", "walmart_amazon": "Walmart-Amazon", "wdc_products": "WDC Products"}
ARM_LABEL = {
    "set_joint_interv_seed0": "main (set + interventions, joint), seed 0",
    "set_joint_interv_seed1": "main, seed 1",
    "set_joint_interv_seed2": "main, seed 2",
    "pairwise_clean_seed0": "clean pairwise, seed 0",
    "pairwise_clean_seed1": "clean pairwise, seed 1",
    "pairwise_clean_seed2": "clean pairwise, seed 2",
    "set_joint_clean_seed0": "H2 control: set, no interventions",
    "pairwise_interv_seed0": "pairwise + interventions",
    "set_factorized_interv_seed0": "factorized, seed 0",
    "set_factorized_interv_seed1": "factorized, seed 1",
    "set_factorized_interv_seed2": "factorized, seed 2",
    "set_joint_interv_nolbl_seed0": "no label-changing family",
    "set_joint_bi_seed0": "set bi-encoder",
    "decoder_head_seed0": "decision head (Qwen2.5-1.5B LoRA)",
    "judge_12b_minimal_json": "12B judge, minimal JSON",
    "judge_12b_symbols": "12B judge, symbols",
    "judge_12b_rich": "12B judge, rich",
    "judge_120b_rich": "120B judge, rich",
}


def load(path: str | Path):
    p = Path(path)
    p = p if p.is_absolute() else REPO / p
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def evaluation(run: str | Path) -> dict:
    """evaluation.json of a 21_evaluate run directory (relative to the repo)."""
    return load(Path(run) / "evaluation.json")


def run_status(run: str | Path) -> str:
    m = load(Path(run) / "manifest.json")
    return m.get("status", "?")


def f3(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.3f}"


def ci(c: dict | None) -> str:
    if not c:
        return "–"
    return f"{c['point']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def diff(d: dict | None, digits: int = 3) -> str:
    if not d:
        return "–"
    if any(isinstance(d.get(k), float) and math.isnan(d[k]) for k in ("diff", "lo", "hi")):
        return "–"
    return f"{d['diff']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"


def results_frame(ev: dict, arms: list[str] | None = None, per_dataset: bool = True) -> pd.DataFrame:
    """One row per arm: overall recall (with CI when present), precision, N-FMR, defer, false-NONE, per-dataset recall."""
    rows = []
    for arm in arms or list(ev["results"]):
        if arm not in ev["results"]:
            continue
        m = ev["results"][arm]
        c = ev.get("bootstrap_ci", {}).get(arm, {})
        row = {"arm": ARM_LABEL.get(arm, arm),
               "recall": ci(c.get("match_recall")) if c.get("match_recall") else f3(m["match_recall"]),
               "precision": f3(m["precision"]), "N-FMR": f3(m["n_fmr"]), "defer": f3(m.get("defer_rate")),
               "false NONE": f3(m.get("false_none_rate"))}
        if per_dataset:
            for ds in DATASETS:
                pd_ = ev["per_dataset"].get(arm, {}).get(ds)
                row[DS_LABEL[ds]] = f3(pd_["match_recall"]) if pd_ else "–"
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def risk_frame(ev: dict, arms: list[str]) -> pd.DataFrame:
    """Per-dataset false matches on certified negatives (SM / n), N-FMR and precision."""
    rows = []
    for arm in arms:
        row = {"arm": ARM_LABEL.get(arm, arm)}
        for ds in ["abt_buy", "dblp_scholar", "wdc_products"]:
            m = ev["per_dataset"][arm][ds]
            row[f"{DS_LABEL[ds]} false matches"] = f"{int(m['SM'])} / {int(m['n_certified_nomatch'])}"
            row[f"{DS_LABEL[ds]} precision"] = f3(m["precision"])
        rows.append(row)
    return pd.DataFrame(rows).set_index("arm")


def md(df: pd.DataFrame) -> None:
    """Render a frame as a Markdown table (readable in a notebook viewer and in plain text)."""
    from IPython.display import Markdown, display
    display(Markdown(df.to_markdown()))
