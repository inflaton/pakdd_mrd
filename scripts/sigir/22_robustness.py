#!/usr/bin/env python
"""Step 22: candidate-set robustness. Subcommands:

  make    build the fixed diagnostic subset + variants (permutations, counts, hard non-matches, remove/add acceptable,
          duplicates, missing/contradictory fields) -> runs/<run>/variants.jsonl and originals.jsonl
  report  given decisions covering originals + variants, compute flip rates / probability shifts / metrics per family

  python scripts/sigir/22_robustness.py make --partition dev
  python scripts/sigir/11_predict.py --checkpoint ... --instances-file runs/<rob>/all.jsonl --out runs/<rob>/dec_set_joint.jsonl
  python scripts/sigir/22_robustness.py report --variants-dir runs/<rob> --decisions set_joint=runs/<rob>/dec_set_joint.jsonl
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from _common import REPO, add_common, benchmark_instances, dataset_list, dump_json, log, setup_logging
from sigir.config import load_config
from sigir.data.aux_pools import load_aux_pools
from sigir.data.interventions import diagnostic_variants
from sigir.eval.robustness import robustness_report
from sigir.io import read_decisions, read_instances, write_instances
from sigir.ledger import RunLedger
from sigir.report.tables import robustness_table

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)
mk = sub.add_parser("make")
mk.add_argument("--partition", default="dev")
mk.add_argument("--config", default="experiments/robustness.yaml")
add_common(mk)
rp = sub.add_parser("report")
rp.add_argument("--variants-dir", required=True)
rp.add_argument("--decisions", action="append", required=True, help="name=decisions.jsonl (must cover originals and variants)")
rp.add_argument("--repeated", action="append", default=[], help="name=decisions_rep0.jsonl,decisions_rep1.jsonl,... for stochastic variation")
add_common(rp)
a = p.parse_args()
setup_logging(a.log_level)

if a.cmd == "make":
    cfg = load_config(a.config)
    insts, man = benchmark_instances(a.benchmark_dir, {a.partition}, dataset_list(a))
    ledger = RunLedger(a.runs_dir, f"robustness_make_{a.partition}", {"config": cfg, "args": vars(a)})
    aux = {}
    for ds in man["datasets"]:
        ap = Path(a.benchmark_dir) / ds / "aux_pools.jsonl"
        if ap.exists():
            aux.update(load_aux_pools(ap))
    rng = random.Random(int(cfg["seed"]))
    originals = []
    for ds in sorted({i.dataset for i in insts}):
        pool = [i for i in insts if i.dataset == ds and i.variant_of is None]
        # entity-grouped sample: pick groups, keep all their instances, balance matchable / no-match
        by_g: dict[str, list] = {}
        for i in pool:
            by_g.setdefault(i.entity_group, []).append(i)
        groups = sorted(by_g)
        rng.shuffle(groups)
        chosen, n_m, n_n = [], 0, 0
        target = int(cfg["diagnostic_subset_size"])
        for g in groups:
            items = by_g[g]
            m = any(i.is_matchable for i in items)
            if m and n_m >= target // 2:
                continue
            if not m and n_n >= target // 2:
                continue
            chosen.extend(items)
            n_m += int(m)
            n_n += int(not m)
            if len(chosen) >= target:
                break
        originals.extend(chosen)
    variants = []
    for inst in originals:
        ax = aux.get(inst.instance_id, {})
        variants.extend(diagnostic_variants(inst, n_perms=int(cfg["n_permutations"]), counts=cfg["counts"], seed=int(cfg["seed"]), distractors=ax.get("distractors", []), missing_gold=ax.get("missing_gold", [])))
    write_instances(ledger.path("originals.jsonl"), originals)
    write_instances(ledger.path("variants.jsonl"), variants)
    write_instances(ledger.path("all.jsonl"), originals + variants)
    fam_counts: dict[str, int] = {}
    for v in variants:
        fam_counts[v.intervention or "?"] = fam_counts.get(v.intervention or "?", 0) + 1
    ledger.set("outputs", originals=len(originals), variants=len(variants), families=fam_counts, all=str(ledger.path("all.jsonl")))
    ledger.finish("complete")
    log.info("diagnostic subset: %d originals, %d variants (%s) -> %s", len(originals), len(variants), fam_counts, ledger.dir)
else:
    vd = Path(a.variants_dir)
    originals = read_instances(vd / "originals.jsonl")
    variants = read_instances(vd / "variants.jsonl")
    ledger = RunLedger(a.runs_dir, "robustness_report", {"args": vars(a)})
    reps = {}
    for spec in a.repeated:
        name, files = spec.split("=", 1)
        runs = [read_decisions(f) for f in files.split(",")]
        reps[name] = {iid: [r[iid] for r in runs if iid in r] for iid in runs[0]}
    out = {}
    md = []
    for spec in a.decisions:
        name, path = spec.split("=", 1)
        decs = read_decisions(path)
        rep = robustness_report(originals, variants, decs, reps.get(name))
        out[name] = rep
        md.append(f"## {name}\n\nclean: recall {rep['clean']['match_recall']:.3f}, N-FMR {rep['clean']['n_fmr']:.4f}, coverage {rep['clean']['coverage_auto']:.3f}\n\n{robustness_table(rep)}\n")
        log.info("%s: %s", name, {f: round(r["entity_flip_rate"], 3) for f, r in rep["families"].items()})
    dump_json(ledger.path("robustness.json"), out)
    ledger.path("robustness.md").write_text("\n".join(md), encoding="utf-8")
    ledger.set("outputs", report=str(ledger.path("robustness.json")))
    ledger.finish("complete")
    print("\n".join(md))
