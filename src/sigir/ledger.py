"""Run ledger: every run writes a manifest with the provenance the plan requires (§10).

Recorded: code revision, config and its hash, dataset / split / candidate hashes, model and tokenizer
identity, prompt hash, seeds, precision / backend, hardware, threshold provenance, timing, failures,
requested and resolved hosted model versions.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .hashing import sha256_obj


def git_revision(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root or Path(__file__).resolve().parents[2])
    out: dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    if shutil.which("git") is None:
        return out
    try:
        q = dict(cwd=root, text=True, stderr=subprocess.DEVNULL)
        out["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], **q).strip()
        out["branch"] = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], **q).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], **q)
        out["dirty"] = bool(status.strip())
    except Exception:  # not a git repo
        pass
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch  # noqa: WPS433

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["gpu_total_mem_gb"] = round(p.total_memory / 2**30, 2)
            info["gpu_capability"] = f"{p.major}.{p.minor}"
    except Exception:
        info["torch"] = None
    if shutil.which("nvidia-smi"):
        try:
            q = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
            ).strip()
            info["nvidia_smi"] = q
        except Exception:
            pass
    return info


class RunLedger:
    """Creates runs/<run_id>/ and writes manifest.json (+ incremental events)."""

    def __init__(self, runs_dir: str | Path, name: str, config: dict[str, Any] | None = None, run_id: str | None = None):
        self.runs_dir = Path(runs_dir)
        ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{ts}_{name}_{uuid.uuid4().hex[:6]}"
        self.dir = self.runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        self.manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "created_utc": ts,
            "status": "running",
            "code": git_revision(),
            "hardware": hardware_info(),
            "config": config or {},
            "config_hash": sha256_obj(config or {}),
            "inputs": {},  # dataset / split / candidate hashes
            "model": {},  # checkpoint, tokenizer hash, precision, backend, requested/resolved versions
            "seeds": {},
            "thresholds": {},  # provenance of operating points used
            "timing": {},
            "failures": [],
            "outputs": {},
            "notes": [],
        }
        self.flush()

    # ------------------------------------------------------------------
    def set(self, section: str, **kv: Any) -> None:
        self.manifest.setdefault(section, {}).update(kv)
        self.flush()

    def note(self, msg: str) -> None:
        self.manifest["notes"].append({"t": time.time() - self._t0, "msg": msg})
        self.flush()

    def failure(self, where: str, err: str, **extra: Any) -> None:
        self.manifest["failures"].append({"where": where, "error": err, **extra})
        self.flush()

    def finish(self, status: str = "complete") -> None:
        self.manifest["status"] = status
        self.manifest["timing"]["wall_seconds"] = round(time.time() - self._t0, 3)
        self.flush()

    def flush(self) -> None:
        with open(self.dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)

    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    with open(Path(run_dir) / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def list_complete_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Only complete, validated runs may feed generated paper claims."""
    out = []
    for p in sorted(Path(runs_dir).glob("*/manifest.json")):
        m = json.load(open(p, encoding="utf-8"))
        if m.get("status") == "complete":
            out.append(m)
    return out
