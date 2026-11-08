"""Synthetic two-table product source for smoke tests (registered as format `synthetic`)."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd

from .sources import LOADERS, SourceBundle

BRANDS = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Wonka", "Tyrell", "Cyberdyne", "Hooli"]
NOUNS = ["Blender", "Router", "Headphones", "Monitor", "Keyboard", "Camera", "Speaker", "Printer", "Drill", "Kettle"]
ADJ = ["Pro", "Max", "Lite", "Plus", "Mini", "Ultra", "Classic", "Air", "Neo", "Prime"]


def _entity(rng: random.Random, i: int) -> dict[str, str]:
    brand = rng.choice(BRANDS)
    noun = rng.choice(NOUNS)
    adj = rng.choice(ADJ)
    model = f"{brand[:2].upper()}{rng.randint(100, 999)}{rng.choice('ABCDX')}"
    return {"brand": brand, "noun": noun, "adj": adj, "modelno": model, "price": f"{rng.uniform(15, 900):.2f}"}


def _render(e: dict[str, str], rng: random.Random, side: str) -> dict[str, str]:
    title = f"{e['brand']} {e['noun']} {e['adj']} {e['modelno']}" if side == "A" else f"{e['adj']} {e['noun']} by {e['brand']} ({e['modelno']})"
    if rng.random() < 0.3:
        title = title.lower()
    if rng.random() < 0.2:
        title = title.replace(e["modelno"], e["modelno"][:-1])
    price = e["price"] if rng.random() < 0.7 else f"{float(e['price']) * rng.uniform(0.9, 1.1):.2f}"
    return {"title": title, "brand": e["brand"] if rng.random() < 0.85 else "", "modelno": e["modelno"] if rng.random() < 0.7 else "", "price": price}


def load_synthetic(ds: dict[str, Any], raw_dir: Path) -> SourceBundle:
    p = ds.get("synthetic", {})
    rng = random.Random(p.get("seed", 7))
    n_ent = int(p.get("n_entities", 1200))
    n_a = int(p.get("n_a", 1000))
    n_b = int(p.get("n_b", 1500))
    unmatched = float(p.get("unmatched_frac", 0.3))
    ents = [_entity(rng, i) for i in range(n_ent)]
    # hard distractors: same brand/noun, different model
    rows_a, rows_b, links = [], [], []
    a_ents = rng.sample(range(n_ent), n_a)
    for i, ei in enumerate(a_ents):
        rows_a.append({"row_id": f"A:{i}", **_render(ents[ei], rng, "A")})
    b_ents: list[int] = []
    # matched B records for a subset of A entities
    for ei in a_ents:
        if rng.random() > unmatched:
            b_ents.append(ei)
    while len(b_ents) < n_b:
        b_ents.append(rng.randrange(n_ent))
    rng.shuffle(b_ents)
    ent_to_a = {ei: f"A:{i}" for i, ei in enumerate(a_ents)}
    for j, ej in enumerate(b_ents):
        rows_b.append({"row_id": f"B:{j}", **_render(ents[ej], rng, "B")})
        if ej in ent_to_a:
            links.append((ent_to_a[ej], f"B:{j}", "synthetic_truth"))
    fields = ds["fields"]
    return SourceBundle(
        dataset=ds["name"],
        table_a=pd.DataFrame(rows_a)[["row_id"] + fields],
        table_b=pd.DataFrame(rows_b)[["row_id"] + fields],
        links=pd.DataFrame(links, columns=["a_id", "b_id", "source"]),
        fields_a=fields,
        fields_b=fields,
        identity_policy=ds["identity_policy"],
        link_completeness="complete",
        redistribution="synthetic",
    )


LOADERS["synthetic"] = load_synthetic
