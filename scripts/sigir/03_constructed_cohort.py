#!/usr/bin/env python
"""Step 03: the constructed (catalog-absence) no-match cohort for evaluation partitions (plan §5, §12).

Natural certified no-match pools are scarce in the frozen benchmark (dev 39, test 43; none for Walmart-Amazon, whose
negatives stay unresolved until adjudicated), too few for an informative 2 % N-FMR bound. Plan §5 lists
`synthetic_removed` (all acceptable candidates removed from an exhaustive pool) as certifying pool-level NONE in a
separate synthetic cohort. For every matchable query with an exhaustive acceptable set (complete mappings or WDC
clusters) in dev, calib_fit, calib_confirm, test and transfer, this writes the catalog-absence counterfactual:
every acceptable record (in K and beyond K) leaves the frozen pool, the next verified non-matches backfill K, and
ranks are shifted as if the gold records were not in the catalog (`interventions.remove_all_acceptable`).

Natural and constructed cohorts stay separate in every report. Train partitions are excluded (intervention training
builds its own variants from training queries). Instance id = original id + "#catabs2" (v2, Nov 5: probable unlinked duplicates removed too, review note R1); same entity group and
partition as the original, so cluster bootstraps keep them together.

  python scripts/sigir/03_constructed_cohort.py
Writes <benchmark>/<dataset>/instances_constructed.jsonl and <benchmark>/constructed_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from _common import add_common, benchmark_instances, dataset_list, dump_json, log, setup_logging
from sigir.data.aux_pools import load_aux_pools
from sigir.data.duplicates import probable_duplicates
from sigir.data.interventions import remove_all_acceptable
from sigir.data.retrieval import Snapshot
from sigir.hashing import sha256_file
from sigir.io import write_instances
from sigir.schema import Candidate

EVAL_PARTITIONS = {"dev", "calib_fit", "calib_confirm", "test", "transfer"}

p = argparse.ArgumentParser()
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
bench = Path(a.benchmark_dir)
for old in bench.glob("*/label_screen.jsonl"):
    old.unlink()  # regenerated below from the unscreened instances (benchmark_instances applies existing screens)
insts, man = benchmark_instances(bench, EVAL_PARTITIONS, dataset_list(a))
out_man = {"benchmark_manifest_hash": sha256_file(bench / "manifest.json"), "partitions": sorted(EVAL_PARTITIONS), "datasets": {}}
for ds in sorted({i.dataset for i in insts}):
    d_insts = [i for i in insts if i.dataset == ds]
    if man["datasets"][ds].get("link_completeness") not in ("complete", "cluster"):
        log.info("%s: acceptable sets not exhaustive (%s); no constructed cohort", ds, man["datasets"][ds].get("link_completeness"))
        continue
    aux = load_aux_pools(bench / ds / "aux_pools.jsonl")
    snaps = {p.stem.replace("snapshot_", "").replace("_", "->"): Snapshot.load(p) for p in sorted((bench / ds).glob("snapshot_*.jsonl"))}
    rows, stats = [], Counter()
    for inst in d_insts:
        if not (inst.is_matchable and inst.acceptable_exhaustive):
            continue
        snap = snaps.get("A->B" if inst.direction == "offer->offer" else inst.direction)
        ranks = {c["row_id"]: c["rank"] for c in snap.pools.get(inst.query_row_id, [])} if snap else {}
        ax = aux.get(inst.instance_id, {"distractors": [], "missing_gold": []})
        # gold beyond K: snapshot rank known, so the rank shift is exact
        missing = [Candidate(g.row_id, g.fields, ranks.get(g.row_id), g.retrieval_score, g.entity_group) for g in ax["missing_gold"]]
        v = remove_all_acceptable(inst, random.Random(0), distractors=ax["distractors"], missing_gold=missing, suffix="catabs2")
        if v is None:
            continue
        v.extra["cohort"] = "constructed"
        rows.append(v)
        stats[(v.partition, "k_short" if v.extra["k_short"] else "k_full")] += 1
        stats[("removed", v.extra["removed_acceptable"])] += 1
    # R1 label screen: natural certified no-match pools in evaluation partitions holding a probable unlinked duplicate
    screened = []
    for inst in d_insts:
        if inst.is_matchable or not inst.acceptable_exhaustive:
            continue
        hits = probable_duplicates(ds, inst.query_fields, inst.candidates)
        if hits:
            screened.append({"instance_id": inst.instance_id, "partition": inst.partition, "reason": "probable_unlinked_duplicate", "row_ids": hits})
    with open(bench / ds / "label_screen.jsonl", "w", encoding="utf-8") as f:
        for r in screened:
            f.write(json.dumps(r) + "\n")
    path = bench / ds / "instances_constructed.jsonl"
    write_instances(path, rows)
    by_part = Counter(v.partition for v in rows)
    out_man["datasets"][ds] = {"n_instances": len(rows), "by_partition": dict(sorted(by_part.items())), "k_short": sum(1 for v in rows if v.extra["k_short"]), "instances_hash": sha256_file(path), "removed_acceptable_hist": {str(k[1]): n for k, n in sorted(stats.items()) if k[0] == "removed"},
                              "with_probable_duplicates_removed": dict(sorted(Counter(v.partition for v in rows if v.extra["removed_probable_duplicates"]).items())),
                              "natural_negatives_screened": dict(sorted(Counter(r["partition"] for r in screened).items())), "label_screen_hash": sha256_file(bench / ds / "label_screen.jsonl")}
    log.info("%s: %d constructed no-match instances %s (k_short %d)", ds, len(rows), dict(by_part), out_man["datasets"][ds]["k_short"])
dump_json(bench / "constructed_manifest.json", out_man)
log.info("constructed cohort manifest: %s", bench / "constructed_manifest.json")
