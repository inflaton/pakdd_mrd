#!/usr/bin/env python
"""Step 1: fetch the four public sources from their maintainers, pin SHA-256 hashes, print an inventory."""
from __future__ import annotations

import argparse
import json

from _common import REPO, dump_json, log, setup_logging
from mrd.config import load_config
from mrd.data.download import download_dataset
from mrd.data.grouping import build_entity_groups, inventory
from mrd.data.sources import load_source

p = argparse.ArgumentParser()
p.add_argument("--config", default="datasets.yaml")
p.add_argument("--only", default=None, help="comma list of dataset names")
p.add_argument("--no-new-pins", action="store_true", help="fail if a file has no recorded pin (use after the freeze)")
a = p.parse_args()
setup_logging()
cfg = load_config(a.config)
raw_dir = REPO / cfg["raw_dir"]
pins_dir = REPO / cfg["pins_dir"]
only = set(a.only.split(",")) if a.only else None
summary = {}
for ds in cfg["datasets"]:
    if only and ds["name"] not in only:
        continue
    log.info("downloading %s", ds["name"])
    if ds["format"] != "synthetic":
        hashes = download_dataset(ds, raw_dir, pins_dir, allow_new_pins=not a.no_new_pins)
    else:
        import mrd.data.synthetic  # noqa: F401  (registers loader)

        hashes = {}
    bundle = load_source(ds, raw_dir)
    ids = list(bundle.table_a["row_id"]) + ([] if bundle.table_b is bundle.table_a else list(bundle.table_b["row_id"]))
    groups = build_entity_groups(ids, bundle.links, ds["name"], bundle.cluster_of or None)
    summary[ds["name"]] = {"files": hashes, **bundle.summary(), "entity_inventory": inventory(groups, bundle.links), "notes": bundle.notes}
    log.info("%s: %s", ds["name"], json.dumps(summary[ds["name"]]["entity_inventory"]))
dump_json(REPO / "data/mrd/public/source_inventory.json", summary)
print(json.dumps(summary, indent=2, default=str))
