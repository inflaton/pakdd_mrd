#!/usr/bin/env python
"""Step 2: build the frozen benchmark: groups -> partitions -> retrieval snapshot -> decision instances -> audits.

Outputs (per dataset under <out_dir>/<dataset>/):
  snapshot_<direction>.jsonl(.meta.json)  frozen candidate pools (pool_size) with snapshot hash
  instances.jsonl                          DecisionInstances for all directions/partitions (K presented)
  aux_pools.jsonl                          verified distractors + missing gold per instance
  exposure_audit.json, candidate_recall.json, inventory.json
and <out_dir>/manifest.json with every hash. Re-running with the same configs reproduces identical files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from _common import REPO, dump_json, log, setup_logging
from sigir.config import load_config
from sigir.data.aux_pools import build_aux_pools, save_aux_pools
from sigir.data.build_instances import build_instances, instance_inventory, load_adjudications, wdc_role
from sigir.data.grouping import build_entity_groups, inventory
from sigir.data.retrieval import build_snapshot, candidate_recall, load_or_build_snapshot, snapshot_inputs_key
from sigir.data.sources import load_source
from sigir.data.splits import LegacyExposureError, assign_partitions, check_exposed_placement, exposure_audit, load_legacy_exposure, partition_counts
from sigir.hashing import sha256_file, sha256_obj
from sigir.io import write_instances

import sigir.data.synthetic  # noqa: F401  (registers the synthetic loader)

p = argparse.ArgumentParser()
p.add_argument("--datasets-config", default="datasets.yaml")
p.add_argument("--benchmark-config", default="benchmark.yaml")
p.add_argument("--only", default=None)
p.add_argument("--out-dir", default=None, help="override benchmark out_dir")
p.add_argument("--override", nargs="*", default=[], help="benchmark config overrides key=value")
p.add_argument("--second-retriever", action="store_true", help="also build the transfer-test snapshot with retrieval.second_config")
a = p.parse_args()
setup_logging()
dcfg = load_config(a.datasets_config)
bcfg = load_config(a.benchmark_config, a.override)
raw_dir = REPO / dcfg["raw_dir"]
out_dir = Path(a.out_dir) if a.out_dir else REPO / bcfg["out_dir"]
out_dir.mkdir(parents=True, exist_ok=True)
only = set(a.only.split(",")) if a.only else None
K = int(bcfg["k"])
manifest = {"version": bcfg["version"], "benchmark_config": bcfg, "benchmark_config_hash": sha256_obj(bcfg), "datasets": {}}

for ds in dcfg["datasets"]:
    name = ds["name"]
    if only and name not in only:
        continue
    log.info("=== %s", name)
    bundle = load_source(ds, raw_dir)
    single_table = bundle.table_b is bundle.table_a
    ids = list(bundle.table_a["row_id"]) + ([] if single_table else list(bundle.table_b["row_id"]))
    groups = build_entity_groups(ids, bundle.links, name, bundle.cluster_of or None)
    inv = inventory(groups, bundle.links)
    # Prior exposure: records previously used as test/dev queries; their groups may only land in train/dev.
    # Configured means required: a missing or unreadable inventory stops the build instead of exposing nothing.
    if ds.get("legacy_exposure_file") and not ds.get("legacy_exposure"):
        raise LegacyExposureError(f"{name}: 'legacy_exposure_file' is no longer read; use the 'legacy_exposure' block (datasets.yaml)")
    exposed_rows: set[str] = set()
    legacy_summary = None
    le = ds.get("legacy_exposure")
    if le:
        inv_path = REPO / le["file"]
        inv_obj = json.load(open(inv_path, encoding="utf-8")) if inv_path.exists() else {}
        want = (inv_obj.get("config") or {}).get("archive_sha256")
        arch = raw_dir / name / le["archive"]
        if not arch.exists():
            raise LegacyExposureError(f"{name}: raw archive {arch} missing; run 01_download_sources.py --only {name}")
        if want and sha256_file(arch) != want:
            raise LegacyExposureError(f"{name}: {arch.name} sha256 differs from the reservation's archive_sha256 {want[:12]}")
        raw_a = pd.read_csv(raw_dir / name / le["raw_table_a"], dtype=str, keep_default_na=False, usecols=["custom_id", "title"])
        exposed_rows, legacy_summary = load_legacy_exposure(
            inv_path, bundle.table_a, raw_a,
            exposed_splits=tuple(le.get("exposed_splits", ["test", "dev"])), id_offset=int(le.get("id_offset", -1)))
    unmapped = sorted(r for r in exposed_rows if r not in groups)
    if unmapped:
        raise LegacyExposureError(f"{name}: {len(unmapped)} exposed records have no entity group, e.g. {unmapped[:5]}")
    exposed_groups = {groups[r] for r in exposed_rows}
    part_of_group = assign_partitions(groups, bcfg["splits"]["fractions"], int(bcfg["splits"]["seed"]), exposed_groups)
    if legacy_summary is not None:
        legacy_summary["n_exposed_groups"] = len(exposed_groups)
        legacy_summary["exposed_group_partitions"] = check_exposed_placement(exposed_groups, part_of_group)
        log.info("legacy exposure: %d records -> %d groups, placed %s", len(exposed_rows), len(exposed_groups), legacy_summary["exposed_group_partitions"])
    log.info("partition counts (groups): %s ; exposed groups forced to train/dev: %d", partition_counts(part_of_group), len(exposed_groups))
    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    adj = load_adjudications(REPO / bcfg["adjudications_dir"] / f"{name}.jsonl")
    all_insts = []
    aux_rows = []
    recall_report = {}
    snap_hashes = {}
    snap_status = {}
    train_groups = {g for g, pt in part_of_group.items() if pt == "train"}
    for direction in ds.get("directions", ["A->B"]):
        if single_table:
            seed = int(ds.get("role_seed", 0))
            roles = {rid: wdc_role(rid, seed) for rid in bundle.table_a["row_id"]}
            q_df = bundle.table_a[bundle.table_a["row_id"].map(roles) == "query"].reset_index(drop=True)
            c_df = bundle.table_a[bundle.table_a["row_id"].map(roles) == "catalog"].reset_index(drop=True)
            fq, fc = bundle.fields_a, bundle.fields_a
            # restrict bundle links to query->catalog pairs for this role assignment
            lk = bundle.links[bundle.links["a_id"].isin(set(q_df["row_id"])) & bundle.links["b_id"].isin(set(c_df["row_id"]))]
            sub = bundle.__class__(**{**bundle.__dict__, "table_a": q_df, "table_b": c_df, "links": lk.reset_index(drop=True)})
        else:
            sub = bundle
            if direction == "A->B":
                q_df, c_df, fq, fc = bundle.table_a, bundle.table_b, bundle.fields_a, bundle.fields_b
            else:
                q_df, c_df, fq, fc = bundle.table_b, bundle.table_a, bundle.fields_b, bundle.fields_a
        snap_path = ds_dir / f"snapshot_{direction.replace('->', '_')}.jsonl"
        rcfg = {**bcfg["retrieval"], "pool_size": bcfg["pool_size"]}
        key = snapshot_inputs_key(name, direction, q_df, c_df, fq, fc, rcfg, exclude_self=single_table)
        log.info("snapshot %s %s: %d queries over %d catalog records", name, direction, len(q_df), len(c_df))
        snap, status = load_or_build_snapshot(snap_path, key, lambda: build_snapshot(name, direction, q_df, c_df, fq, fc, rcfg, exclude_self=single_table))
        (log.warning if "CHANGED" in status else log.info)("snapshot %s: %s (%s)", snap_path.name, status, snap.hash[:12])
        snap_hashes[direction] = snap.hash
        snap_status[direction] = status
        from sigir.data.build_instances import acceptable_map

        acc = acceptable_map(sub, "A->B" if single_table else direction)
        recall_report[direction] = candidate_recall(snap, {q: acc.get(q, set()) for q in snap.pools})
        log.info("candidate recall %s: %s", direction, recall_report[direction])
        w = float(ds.get("direction_weights", {}).get(direction, 1.0))
        insts = list(build_instances(sub, groups, part_of_group, snap, k=K, direction="A->B" if single_table else direction, adjudications=adj, direction_weight=w, entity_disjoint_pools=bool(bcfg.get("entity_disjoint_pools")), train_groups=train_groups))
        if single_table:
            for i in insts:
                i.direction = "offer->offer"
        aux_rows.extend(build_aux_pools(sub, groups, snap, insts, "A->B" if single_table else direction))
        all_insts.extend(insts)
        if a.second_retriever and bcfg["retrieval"].get("second_config"):
            snap2_path = ds_dir / f"snapshot2_{direction.replace('->', '_')}.jsonl"
            rcfg2 = {**bcfg["retrieval"], **bcfg["retrieval"]["second_config"], "pool_size": bcfg["pool_size"]}
            key2 = snapshot_inputs_key(name, direction, q_df, c_df, fq, fc, rcfg2, exclude_self=single_table)
            snap2, status2 = load_or_build_snapshot(snap2_path, key2, lambda: build_snapshot(name, direction, q_df, c_df, fq, fc, rcfg2, exclude_self=single_table))
            (log.warning if "CHANGED" in status2 else log.info)("snapshot %s: %s (%s)", snap2_path.name, status2, snap2.hash[:12])
            snap_status[f"{direction}@ret2"] = status2
            insts2 = list(build_instances(sub, groups, part_of_group, snap2, k=K, direction="A->B" if single_table else direction, adjudications=adj, direction_weight=w))
            for i in insts2:
                i.instance_id += "@ret2"
                i.extra["retriever_config"] = "second"
            write_instances(ds_dir / f"instances_retriever2_{direction.replace('->', '_')}.jsonl", insts2)
    n = write_instances(ds_dir / "instances.jsonl", all_insts)
    save_aux_pools(ds_dir / "aux_pools.jsonl", aux_rows)
    by_part: dict[str, list] = {}
    for i in all_insts:
        by_part.setdefault(i.partition, []).append({"query_row_id": i.query_row_id, "candidate_row_ids": [c.row_id for c in i.candidates], "entity_group": i.entity_group})
    audit = exposure_audit(by_part, groups)
    inv_inst = instance_inventory(all_insts)
    if legacy_summary is not None:
        audit["legacy_exposure"] = legacy_summary
    dump_json(ds_dir / "exposure_audit.json", audit)
    dump_json(ds_dir / "candidate_recall.json", recall_report)
    dump_json(ds_dir / "inventory.json", {"records": inv, "instances": inv_inst, "partition_groups": partition_counts(part_of_group), "notes": bundle.notes})
    manifest["datasets"][name] = {
        "fields": bundle.fields_a if single_table or True else bundle.fields_a,
        "fields_b": bundle.fields_b,
        "compact_fields": ds.get("compact_fields", bundle.fields_a),
        "identity_policy": bundle.identity_policy,
        "link_completeness": bundle.link_completeness,
        "redistribution": bundle.redistribution,
        "n_instances": n,
        "instances_hash": sha256_file(ds_dir / "instances.jsonl"),
        "snapshot_hashes": snap_hashes,
        "snapshot_status": snap_status,
        "split_seed": bcfg["splits"]["seed"],
        "exposed_groups": len(exposed_groups),
        "legacy_exposure": legacy_summary,
        "exposure_audit": audit,
        "candidate_recall": recall_report,
        "adjudicated_pools": len(adj),
    }
    log.info("%s: wrote %d instances; exposure audit %s", name, n, json.dumps(audit))
dump_json(out_dir / "manifest.json", manifest)
log.info("benchmark manifest written to %s", out_dir / "manifest.json")
