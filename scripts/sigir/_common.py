"""Shared helpers for the scripts: import path, logging, instance loading, config paths."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sigir.config import CONFIG_DIR, DATA_DIR, MODELS_DIR, RUNS_DIR, REPO_ROOT, load_config, ensure_dirs  # noqa: E402,F401
from sigir.io import read_instances  # noqa: E402

log = logging.getLogger("sigir")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--benchmark-dir", default=str(DATA_DIR / "public" / "benchmark"), help="output of 02_build_benchmark.py")
    p.add_argument("--datasets", default=None, help="comma list; default = all datasets in the benchmark")
    p.add_argument("--runs-dir", default=str(RUNS_DIR))
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--cohorts", default="natural", help="natural[,constructed]: constructed = catalog-absence no-match cohort (03_constructed_cohort.py), evaluation partitions only")


def cohort_list(args) -> tuple[str, ...]:
    return tuple(c for c in getattr(args, "cohorts", "natural").split(",") if c)


def dataset_list(args) -> list[str] | None:
    return [d for d in args.datasets.split(",") if d] if args.datasets else None


COHORT_FILES = {"natural": "instances.jsonl", "constructed": "instances_constructed.jsonl"}


def benchmark_instances(bench_dir: str | Path, partitions: set[str] | None, datasets: list[str] | None, cohorts: tuple[str, ...] = ("natural",)):
    """Instances of the frozen benchmark. `constructed` adds the catalog-absence no-match cohort (separate file, never
    in train); a dataset without exhaustive gold has no constructed file and contributes natural instances only."""
    bench_dir = Path(bench_dir)
    man = json.load(open(bench_dir / "manifest.json", encoding="utf-8"))
    names = datasets or list(man["datasets"].keys())
    insts = []
    for ds in names:
        screen = _label_screen(bench_dir / ds / "label_screen.jsonl")
        for c in cohorts:
            path = bench_dir / ds / COHORT_FILES[c]
            if c != "natural" and not path.exists():
                continue
            for i in read_instances(path, partitions=partitions):
                if i.instance_id in screen and i.partition != "train" and not i.is_matchable:
                    # review note R1: a natural no-match pool holding a probable unlinked duplicate is not certified;
                    # withdrawn from N-FMR denominators like an unresolved adjudication (training labels unchanged)
                    i.label_provenance, i.acceptable_exhaustive = "unresolved", False
                    i.extra["label_screen"] = screen[i.instance_id]
                insts.append(i)
    return insts, man


def screened_instances_file(path: str | Path, bench_dir: str | Path, partitions: set[str] | None, datasets: list[str] | None):
    """Instances from a cohort file (e.g. the retriever-2 pools, ids `<dataset>|<dir>|<query>|<pool hash>@ret2`). The
    benchmark's label screen applies through the query key (id without pool hash) when a flagged probable duplicate
    is in this pool (same rule as above)."""
    bench_dir = Path(bench_dir)
    insts = read_instances(path, partitions=partitions, datasets=set(datasets) if datasets else None)
    screens: dict[str, dict] = {}
    for ds in {i.dataset for i in insts}:
        screens.update({_query_key(k): r for k, r in _label_screen(bench_dir / ds / "label_screen.jsonl").items()})
    for i in insts:
        r = screens.get(_query_key(i.instance_id))
        if r and i.partition != "train" and not i.is_matchable and set(r["row_ids"]) & {c.row_id for c in i.candidates}:
            i.label_provenance, i.acceptable_exhaustive = "unresolved", False
            i.extra["label_screen"] = r
    return insts


def _query_key(instance_id: str) -> str:
    return instance_id.split("@", 1)[0].rsplit("|", 1)[0]


def _label_screen(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["instance_id"]: r for r in map(json.loads, f)}


def field_maps(man: dict) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, str]]:
    order = {ds: v["fields"] for ds, v in man["datasets"].items()}
    compact = {ds: v.get("compact_fields", v["fields"]) for ds, v in man["datasets"].items()}
    policy = {ds: v["identity_policy"] for ds, v in man["datasets"].items()}
    return order, compact, policy


def dump_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
