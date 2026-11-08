#!/usr/bin/env python
"""Step 13: run an LLM judge arm (rich / minimal_json / symbols / 120B reference / open decision model / Jev).

Resumable: existing decisions in --out are kept and skipped. Records requested and resolved model names.
  python scripts/mrd/13_judge_llm.py --config models/judge_12b_minimal_json.yaml --partitions dev --limit 200
  python scripts/mrd/13_judge_llm.py --config models/judge_120b_reference.yaml --partitions dev,calib_fit,calib_confirm,test
  python scripts/mrd/13_judge_llm.py --config models/judge_12b_symbols.yaml --instances-file runs/<rob>/variants.jsonl --repeat 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import REPO, add_common, benchmark_instances, dataset_list, field_maps, log, setup_logging, cohort_list
from mrd.data.serialization import TEXT_NORMALIZATION
from mrd.config import load_config
from mrd.io import read_decisions, read_instances, write_decisions
from mrd.ledger import RunLedger
from mrd.models.llm_judge import JudgeConfig, LLMJudge

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--override", nargs="*", default=[])
p.add_argument("--partitions", default="dev")
p.add_argument("--instances-file", default=None)
p.add_argument("--limit", type=int, default=None)
p.add_argument("--repeat", type=int, default=1, help="identical repeated requests per instance (stochastic variation); results in decisions_rep<i>.jsonl")
p.add_argument("--out", default=None)
p.add_argument("--name", default=None)
add_common(p)
a = p.parse_args()
setup_logging(a.log_level)
cfg = load_config(a.config, a.override)
name = a.name or cfg["name"]
insts = read_instances(a.instances_file) if a.instances_file else benchmark_instances(a.benchmark_dir, set(a.partitions.split(",")), dataset_list(a), cohort_list(a))[0]
_, man = benchmark_instances(a.benchmark_dir, set(), None)
order, compact, policy = field_maps(man)
if a.limit:
    insts = insts[: a.limit]
ledger = RunLedger(a.runs_dir, f"judge_{name}", {"judge_config": cfg, "args": vars(a)})
try:
    if cfg["kind"] == "jev":
        from mrd.models.jev import JevChoice, JevConfig

        judge = JevChoice(JevConfig(**{k: v for k, v in cfg["jev"].items()}), policy_text="\n".join(f"{d}: {t}" for d, t in policy.items()))
        prompt_hash = None
        requested = judge.cfg.model_version
    else:
        jc = JudgeConfig(**cfg["judge"])
        jc.prompt_file = str(REPO / jc.prompt_file) if not Path(jc.prompt_file).is_absolute() else jc.prompt_file
        # identity policy is per dataset: render with the dataset's policy
        judge = LLMJudge(jc, policy_text="", field_order=None, compact_fields=None)
        prompt_hash = judge.prompt_hash
        requested = jc.model
    ledger.set("model", requested_model=requested, prompt_hash=prompt_hash, mode=cfg.get("judge", {}).get("mode"), serialization=cfg.get("judge", {}).get("serialization"), text_normalization=TEXT_NORMALIZATION)
    ledger.set("inputs", n_instances=len(insts), partitions=a.partitions, instances_file=a.instances_file, instances_hashes={d: v["instances_hash"] for d, v in man["datasets"].items()})
    for rep in range(a.repeat):
        out = Path(a.out) if (a.out and a.repeat == 1) else ledger.path(f"decisions_rep{rep}.jsonl" if a.repeat > 1 else "decisions.jsonl")
        done = read_decisions(out) if out.exists() else {}
        todo = [i for i in insts if i.instance_id not in done]
        log.info("rep %d: %d to judge (%d already done)", rep, len(todo), len(done))
        results = list(done.values())
        # per-dataset policy text: judge_one reads self.policy, so group by dataset
        by_ds: dict[str, list] = {}
        for i in todo:
            by_ds.setdefault(i.dataset, []).append(i)
        n_fail = 0
        for ds, group in by_ds.items():
            judge.policy = policy.get(ds, "")
            if cfg["kind"] != "jev":
                judge.field_order = order.get(ds)
                judge.compact_fields = compact.get(ds)
            batch_out = judge.judge_many(group)
            n_fail += sum(1 for d in batch_out if d.kind.value == "fail")
            results.extend(batch_out)
            write_decisions(out, results)  # checkpoint after each dataset
        write_decisions(out, results)
        ledger.set("outputs", **{f"decisions_rep{rep}": str(out), f"n_rep{rep}": len(results), f"failures_rep{rep}": n_fail})
    ledger.set("model", resolved_model=getattr(judge, "resolved_model", None))
    ledger.finish("complete")
    log.info("done; resolved model %s", getattr(judge, "resolved_model", None))
except Exception as e:
    ledger.failure("judge", repr(e))
    ledger.finish("failed")
    raise
