"""Claims ledger: every numerical claim in the paper links to a complete run (plan §10, §13)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ClaimsLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.claims: list[dict[str, Any]] = []
        if self.path.exists():
            self.claims = json.load(open(self.path, encoding="utf-8"))

    def add(self, claim_id: str, text: str, value: Any, run_ids: list[str], metric: str, partition: str, dataset: str, notes: str = "") -> None:
        self.claims = [c for c in self.claims if c["claim_id"] != claim_id]
        self.claims.append({"claim_id": claim_id, "text": text, "value": value, "run_ids": run_ids, "metric": metric, "partition": partition, "dataset": dataset, "notes": notes})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.claims, open(self.path, "w", encoding="utf-8"), indent=2, default=str)

    def verify(self, runs_dir: str | Path) -> list[str]:
        """Return problems: claims whose runs are missing or not complete."""
        problems = []
        for c in self.claims:
            for rid in c["run_ids"]:
                m = Path(runs_dir) / rid / "manifest.json"
                if not m.exists():
                    problems.append(f"{c['claim_id']}: run {rid} missing")
                    continue
                st = json.load(open(m)).get("status")
                if st != "complete":
                    problems.append(f"{c['claim_id']}: run {rid} status={st}")
        return problems

    def to_markdown(self) -> str:
        lines = ["| claim | value | metric | dataset | partition | runs |", "|---|---|---|---|---|---|"]
        for c in self.claims:
            lines.append(f"| {c['claim_id']}: {c['text']} | {c['value']} | {c['metric']} | {c['dataset']} | {c['partition']} | {', '.join(c['run_ids'])} |")
        return "\n".join(lines)
