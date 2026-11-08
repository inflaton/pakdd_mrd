"""Pinned downloads from the original maintainers (plan §5: pin releases, hashes, reconstruction scripts).

Source records are NOT redistributed by this repository. Each file is fetched from the maintainer URL
listed in configs/mrd/datasets.yaml. The SHA-256 of every file is pinned in
data/mrd/public/_pins/<dataset>.json on first download and verified on every later run. If a pin
exists and the content differs, the download fails loudly so the benchmark cannot silently drift.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from ..hashing import sha256_file


class PinMismatch(RuntimeError):
    pass


def _pins_path(pins_dir: Path, dataset: str) -> Path:
    pins_dir.mkdir(parents=True, exist_ok=True)
    return pins_dir / f"{dataset}.json"


def load_pins(pins_dir: Path, dataset: str) -> dict[str, str]:
    p = _pins_path(pins_dir, dataset)
    return json.load(open(p, encoding="utf-8")) if p.exists() else {}


def save_pins(pins_dir: Path, dataset: str, pins: dict[str, str]) -> None:
    with open(_pins_path(pins_dir, dataset), "w", encoding="utf-8") as f:
        json.dump(pins, f, indent=2, sort_keys=True)


def fetch(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": "mrd-reproducibility/0.1"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False) as bar:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
                bar.update(len(chunk))
    shutil.move(tmp, dest)
    return dest


def download_dataset(ds_cfg: dict[str, Any], raw_dir: Path, pins_dir: Path, *, allow_new_pins: bool = True) -> dict[str, str]:
    """Download every file listed under ds_cfg['files'] and verify/pin hashes.

    ds_cfg['files'] items: {name, url, sha256 (optional)}. Zips are extracted next to the archive.
    Returns {name: sha256}.
    """
    name = ds_cfg["name"]
    pins = load_pins(pins_dir, name)
    out: dict[str, str] = {}
    ddir = raw_dir / name
    for f in ds_cfg["files"]:
        dest = ddir / f["name"]
        fetch(f["url"], dest)
        h = sha256_file(dest)
        expected = f.get("sha256") or pins.get(f["name"])
        if expected and expected != h:
            raise PinMismatch(f"{name}/{f['name']}: sha256 {h} != pinned {expected}. Source changed; do not proceed silently.")
        if not expected:
            if not allow_new_pins:
                raise PinMismatch(f"{name}/{f['name']} has no pin and new pins are disallowed")
            pins[f["name"]] = h
        out[f["name"]] = h
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest) as z:
                z.extractall(ddir / dest.stem)
    save_pins(pins_dir, name, pins)
    return out
