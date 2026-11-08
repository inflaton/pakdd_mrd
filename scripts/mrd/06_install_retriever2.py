#!/usr/bin/env python
"""Step 06: install the second-retriever (hybrid BM25 + bge-small RRF) instances for the retriever-transfer test (plan §5, §8).

The hybrid snapshot is built by a full rebuild into a separate directory (`02_build_benchmark.py --second-retriever
--override retrieval.device=cpu --out-dir data/mrd/public/benchmark_rebuild_ret2`), which also re-derives the primary
instances from scratch. This checks that the rebuilt primary `instances.jsonl` and BM25 snapshots are identical to the
frozen ones, copies `snapshot2_*` and `instances_retriever2_*` into the frozen benchmark directory without touching its
manifest, recomputes the R1 label screen on the new pools (DBLP probable unlinked duplicates), and writes
`retriever2_manifest.json` plus the evaluation cohort file (dev and test, natural pools; thresholds stay frozen).

  python scripts/mrd/06_install_retriever2.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from _common import DATA_DIR, RUNS_DIR, dump_json, log, setup_logging
from mrd.data.duplicates import probable_duplicates
from mrd.hashing import sha256_file
from mrd.io import read_instances, write_instances

setup_logging()
bench, rebuilt = DATA_DIR / "public" / "benchmark", DATA_DIR / "public" / "benchmark_rebuild_ret2"
ma, mb = json.load(open(bench / "manifest.json")), json.load(open(rebuilt / "manifest.json"))
out = {"source": str(rebuilt), "rebuild_config_hash": mb["benchmark_config_hash"], "frozen_config_hash": ma["benchmark_config_hash"], "second_config": mb["benchmark_config"]["retrieval"]["second_config"], "datasets": {}}
cohort = []
for ds, a in ma["datasets"].items():
    b = mb["datasets"][ds]
    same = a["instances_hash"] == b["instances_hash"] and a["snapshot_hashes"] == b["snapshot_hashes"]
    if not same:
        raise SystemExit(f"{ds}: rebuilt primary instances or snapshots differ from the frozen benchmark; not installing")
    rec = {"primary_rebuild_identical": True, "files": {}}
    for src in sorted((rebuilt / ds).glob("*2_*.jsonl")) + sorted((rebuilt / ds).glob("snapshot2_*.meta.json")):
        shutil.copy2(src, bench / ds / src.name)
        rec["files"][src.name] = sha256_file(bench / ds / src.name)
    screened = 0
    for f in sorted((bench / ds).glob("instances_retriever2_*.jsonl")):
        for i in read_instances(f, partitions={"dev", "test"}):
            if not i.is_matchable and i.acceptable_exhaustive and probable_duplicates(ds, i.query_fields, i.candidates):
                i.label_provenance, i.acceptable_exhaustive = "unresolved", False
                i.extra["label_screen"] = {"reason": "probable_unlinked_duplicate"}
                screened += 1
            i.extra["cohort"] = "retriever2"
            cohort.append(i)
    rec["label_screened_negatives_dev_test"] = screened
    out["datasets"][ds] = rec
    log.info("%s: primary rebuild identical; installed %d files; %d dev/test negatives screened", ds, len(rec["files"]), screened)
path = RUNS_DIR / "cohorts" / "retriever2_dev_test.jsonl"
path.parent.mkdir(parents=True, exist_ok=True)
write_instances(path, cohort)
out["cohort_file"] = {"path": str(path), "n": len(cohort), "sha256": sha256_file(path)}
dump_json(bench / "retriever2_manifest.json", out)
log.info("retriever-2 cohort: %d dev/test instances -> %s", len(cohort), path)
