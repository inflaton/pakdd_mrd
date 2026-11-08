"""Read frozen pools and render the record strings shared by both CoMEM stages."""
from __future__ import annotations

import json
from pathlib import Path

from _common import REPO
from sigir.data.serialization import serialize_record
from sigir.io import read_instances
from sigir.schema import DecisionInstance

FROZEN_MANIFEST = REPO / "data/sigir/public/_manifests/benchmark/manifest.json"


def load_pool_rows(path: Path) -> dict[str, DecisionInstance]:
    instances = read_instances(path)
    pools = {inst.instance_id: inst for inst in instances}
    if len(pools) != len(instances):
        raise ValueError(f"Duplicate instance IDs in {path}")
    return pools


def record_fields() -> dict[str, list[str]]:
    manifest = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    return {dataset: config["fields"] for dataset, config in manifest["datasets"].items()}


def render_record(inst: DecisionInstance, fields: dict, order: dict[str, list[str]]) -> str:
    return serialize_record(fields, order[inst.dataset], version="rich_v1")
