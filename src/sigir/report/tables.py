"""Markdown tables for the main results, ablations, robustness and efficiency."""
from __future__ import annotations

import math
from typing import Any, Sequence


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"
    return str(v)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


MAIN_COLS = ["match_recall", "precision", "n_fmr", "f05", "coverage_auto", "coverage_match", "coverage_none", "false_none_rate", "accepted_link_error", "defer_rate", "failure_rate"]


def results_table(results: dict[str, dict[str, Any]], cols: Sequence[str] = MAIN_COLS, ci: dict[str, dict[str, dict[str, float]]] | None = None) -> str:
    """results: arm -> metrics dict. ci: arm -> metric -> {lo, hi}."""
    headers = ["arm"] + list(cols)
    rows = []
    for arm, m in results.items():
        row: list[Any] = [arm]
        for c in cols:
            v = m.get(c, float("nan"))
            if ci and arm in ci and c in ci[arm]:
                row.append(f"{_fmt(v)} [{_fmt(ci[arm][c]['lo'])}, {_fmt(ci[arm][c]['hi'])}]")
            else:
                row.append(v)
        rows.append(row)
    return markdown_table(headers, rows)


def robustness_table(rep: dict[str, Any]) -> str:
    headers = ["intervention", "n", "record flip", "entity flip", "|d p_none|", "recall", "n_fmr", "coverage"]
    rows = []
    for fam, r in rep.get("families", {}).items():
        m = r["metrics"]
        rows.append([fam, r["n_pairs"], r["record_flip_rate"], r["entity_flip_rate"], r["mean_abs_delta_p_none"], m["match_recall"], m["n_fmr"], m["coverage_auto"]])
    return markdown_table(headers, rows)


def bench_table(results: Sequence[dict[str, Any]]) -> str:
    headers = ["system", "conc", "p50 ms", "p95 ms", "req/s", "in tok", "out tok", "weights GB", "peak GB", "J/query", "fail"]
    rows = [[r["name"], r["concurrency"], r["latency_p50_ms"], r["latency_p95_ms"], r["requests_per_s"], r.get("input_tokens_mean"), r.get("output_tokens_mean"), r.get("loaded_weight_mem_gb"), r.get("peak_mem_gb"), r.get("energy_j_per_query"), r["failures"]] for r in results]
    return markdown_table(headers, rows)


def routing_table(sweep: dict[str, Any]) -> str:
    headers = ["policy", "budget", "escalated", "recall", "precision", "n_fmr", "f05", "deployed ms"]
    rows = []
    for pol, rs in sweep["policies"].items():
        for r in rs:
            rows.append([pol, r["budget"], r["n_escalated"], r["match_recall"], r["precision"], r["n_fmr"], r["f05"], r["deployed_mean_latency_ms"]])
    return markdown_table(headers, rows)
