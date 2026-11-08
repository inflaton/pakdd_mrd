"""Source adapters: turn each public release into one uniform SourceBundle.

A SourceBundle has two record tables (A = query side by default, B = catalog side), a link table of
*positive* links, the field list used for serialization, the identity policy text, and the link
completeness tier:

  complete            every true cross-table match is listed (Leipzig perfect mappings)
  positives_complete  positives believed complete, negatives sampled (DeepMatcher Walmart-Amazon)
  cluster             entity clusters (WDC Products); completeness within the released cluster set

Row IDs are exact strings everywhere. Never cast to int/float.
"""
from __future__ import annotations

import glob
import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SourceBundle:
    dataset: str
    table_a: pd.DataFrame  # columns: row_id + fields_a
    table_b: pd.DataFrame  # columns: row_id + fields_b
    links: pd.DataFrame  # columns: a_id, b_id, source
    fields_a: list[str]
    fields_b: list[str]
    identity_policy: str
    link_completeness: str
    redistribution: str
    cluster_of: dict[str, str] = field(default_factory=dict)  # row_id -> cluster id (WDC only)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "n_a": len(self.table_a),
            "n_b": len(self.table_b),
            "n_links": len(self.links),
            "a_with_link": int(self.links["a_id"].nunique()),
            "b_with_link": int(self.links["b_id"].nunique()),
            "link_completeness": self.link_completeness,
            "fields_a": self.fields_a,
            "fields_b": self.fields_b,
        }


def _str_ids(df: pd.DataFrame, col: str) -> pd.Series:
    # read ids as str; strip accidental float formatting like "123.0" only if ALL values end with .0
    s = df[col].astype(str)
    if s.str.endswith(".0").all():
        s = s.str[:-2]
    return s


def _clean(df: pd.DataFrame, id_col: str, fields: list[str], prefix: str) -> pd.DataFrame:
    out = pd.DataFrame({"row_id": prefix + _str_ids(df, id_col)})
    for f in fields:
        out[f] = df[f].astype(str).where(df[f].notna(), "").str.strip() if f in df.columns else ""
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- DeepMatcher-format datasets
def load_deepmatcher(ds: dict[str, Any], raw_dir: Path) -> SourceBundle:
    """tableA.csv / tableB.csv / train|valid|test.csv (ltable_id, rtable_id, label)."""
    d = raw_dir / ds["name"]
    a = pd.read_csv(d / "tableA.csv", dtype=str, keep_default_na=False)
    b = pd.read_csv(d / "tableB.csv", dtype=str, keep_default_na=False)
    fields = ds["fields"]
    ta = _clean(a, "id", fields, "A:")
    tb = _clean(b, "id", fields, "B:")
    pairs = []
    for split in ("train", "valid", "test"):
        p = d / f"{split}.csv"
        if p.exists():
            df = pd.read_csv(p, dtype=str, keep_default_na=False)
            df["source"] = f"deepmatcher_{split}"
            pairs.append(df)
    pr = pd.concat(pairs, ignore_index=True)
    pos = pr[pr["label"].astype(str) == "1"]
    links = pd.DataFrame({"a_id": "A:" + pos["ltable_id"].astype(str), "b_id": "B:" + pos["rtable_id"].astype(str), "source": pos["source"]})
    neg = pr[pr["label"].astype(str) == "0"]
    bundle = SourceBundle(
        dataset=ds["name"],
        table_a=ta,
        table_b=tb,
        links=links.drop_duplicates(["a_id", "b_id"]).reset_index(drop=True),
        fields_a=fields,
        fields_b=fields,
        identity_policy=ds["identity_policy"],
        link_completeness=ds.get("link_completeness", "positives_complete"),
        redistribution=ds.get("redistribution", "see maintainer terms"),
    )
    bundle.notes.append(f"{len(neg)} released negative pairs are *sampled* negatives: an unlisted pair is not a verified negative.")
    return bundle


# ---------------------------------------------------------------- Leipzig releases (complete mappings)
def load_leipzig(ds: dict[str, Any], raw_dir: Path) -> SourceBundle:
    """Abt-Buy.zip -> Abt.csv, Buy.csv, abt_buy_perfectMapping.csv ; DBLP-Scholar.zip -> DBLP1.csv, Scholar.csv, DBLP-Scholar_perfectMapping.csv."""
    d = raw_dir / ds["name"]
    files = ds["leipzig"]
    enc = files.get("encoding", "latin-1")
    # Files in one release can differ: DBLP1.csv is Latin-1, Scholar.csv is UTF-8 with a BOM.
    enc_a = files.get("encoding_a", enc)
    enc_b = files.get("encoding_b", enc)
    enc_m = files.get("encoding_mapping", enc)

    def find(name: str) -> Path:
        hits = glob.glob(str(d / "**" / name), recursive=True)
        if not hits:
            raise FileNotFoundError(f"{name} not found under {d}")
        return Path(hits[0])

    def read(name: str, encoding: str) -> pd.DataFrame:
        df = pd.read_csv(find(name), dtype=str, keep_default_na=False, encoding=encoding)
        # A BOM decoded with the wrong codec ends up in the first header ("\ufeffid" or "ï»¿id"); fail with a clear message.
        bad = [c for c in df.columns if c.startswith(("\ufeff", "\u00ef\u00bb\u00bf"))]
        if bad:
            raise ValueError(f"{name}: header {bad[0]!r} carries a byte-order mark; set its encoding to utf-8-sig in datasets.yaml")
        return df

    a = read(files["table_a"], enc_a)
    b = read(files["table_b"], enc_b)
    m = read(files["mapping"], enc_m)
    fa, fb = ds["fields_a"], ds["fields_b"]
    ta = _clean(a, files["id_a"], fa, "A:")
    tb = _clean(b, files["id_b"], fb, "B:")
    links = pd.DataFrame({"a_id": "A:" + m[files["map_a"]].astype(str).str.strip(), "b_id": "B:" + m[files["map_b"]].astype(str).str.strip(), "source": "perfect_mapping"})
    known_a, known_b = set(ta["row_id"]), set(tb["row_id"])
    bad = links[~links["a_id"].isin(known_a) | ~links["b_id"].isin(known_b)]
    bundle = SourceBundle(
        dataset=ds["name"],
        table_a=ta,
        table_b=tb,
        links=links[links["a_id"].isin(known_a) & links["b_id"].isin(known_b)].drop_duplicates().reset_index(drop=True),
        fields_a=fa,
        fields_b=fb,
        identity_policy=ds["identity_policy"],
        link_completeness="complete",
        redistribution=ds.get("redistribution", "see maintainer terms"),
    )
    if len(bad):
        bundle.notes.append(f"{len(bad)} mapping rows reference unknown ids and were dropped (record this in the construction manifest).")
    return bundle


# ---------------------------------------------------------------- WDC Products
def load_wdc_products(ds: dict[str, Any], raw_dir: Path) -> SourceBundle:
    """WDC Products pairwise JSON(.gz) files with *_left / *_right columns and cluster_id_*.

    Offers are reconstructed by de-duplicating ids; cluster_id supplies entity identity. Query and
    catalog roles are assigned deterministically by the instance builder (not here), self-matches are
    removed there, and multiple acceptable offers are preserved.
    """
    d = raw_dir / ds["name"]
    fields = ds["fields"]
    frames = []
    # `members` are json files inside an extracted archive; fall back to the files list for direct json downloads.
    names = ds.get("members") or [f["name"] for f in ds["files"]]
    for name in names:
        hits = glob.glob(str(d / "**" / name), recursive=True)
        p = Path(hits[0]) if hits else d / name
        if not (p.suffix in (".gz", ".json", ".jsonl")):
            continue
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as fh:
            first = fh.read(1)
            fh.seek(0)
            if first == "[":
                frames.append(pd.DataFrame(json.load(fh)))
            else:
                frames.append(pd.read_json(fh, lines=True))
    if not frames:
        raise FileNotFoundError(f"no WDC json files under {d}")
    pairs = pd.concat(frames, ignore_index=True)
    offers: dict[str, dict[str, Any]] = {}
    cluster_of: dict[str, str] = {}
    for side in ("left", "right"):
        idc, clc = f"id_{side}", f"cluster_id_{side}"
        sub = pairs[[idc, clc] + [f"{f}_{side}" for f in fields if f"{f}_{side}" in pairs.columns]].drop_duplicates(idc)
        for _, r in sub.iterrows():
            rid = "W:" + str(r[idc])
            offers[rid] = {"row_id": rid, **{f: ("" if pd.isna(r.get(f"{f}_{side}")) else str(r.get(f"{f}_{side}", "")).strip()) for f in fields}}
            cluster_of[rid] = str(r[clc])
    table = pd.DataFrame(list(offers.values()))
    # links: all offer pairs sharing a cluster (complete within released clusters)
    by_cl: dict[str, list[str]] = {}
    for rid, cl in cluster_of.items():
        by_cl.setdefault(cl, []).append(rid)
    rows = []
    for cl, members in by_cl.items():
        members = sorted(members)
        for i in range(len(members)):
            for j in range(len(members)):
                if i != j:
                    rows.append((members[i], members[j], "wdc_cluster"))
    links = pd.DataFrame(rows, columns=["a_id", "b_id", "source"])
    bundle = SourceBundle(
        dataset=ds["name"],
        table_a=table,
        table_b=table,  # single multi-source table; roles assigned per split by the builder
        links=links,
        fields_a=fields,
        fields_b=fields,
        identity_policy=ds["identity_policy"],
        link_completeness="cluster",
        redistribution=ds.get("redistribution", "see maintainer terms"),
        cluster_of=cluster_of,
    )
    bundle.notes.append(f"{len(by_cl)} product clusters; published difficulty/unseen variants must be evaluated under their own protocol, not merged.")
    return bundle


LOADERS = {"deepmatcher": load_deepmatcher, "leipzig": load_leipzig, "wdc_products": load_wdc_products}


def load_source(ds: dict[str, Any], raw_dir: Path) -> SourceBundle:
    return LOADERS[ds["format"]](ds, raw_dir)
