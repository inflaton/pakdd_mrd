"""Compact match / reject / defer models over retrieved candidate sets.

Package layout
--------------
schema        decision instances, exact-ID index mapping, permutations, masks, legacy sentinel handling
hashing       content hashes for datasets, candidate pools, configs
ledger        per-run provenance manifests (code revision, model, tokenizer, seeds, hardware, ...)
config        YAML config loading with overrides
data/         downloaders, source adapters, entity grouping, splits, retrieval, instance builder, interventions
models/       pairwise encoder, set model (joint / factorized), LLM judge adapters
train/        torch datasets, collate, trainer
calibration/  temperature scaling, frozen threshold policies, one-sided risk bounds
eval/         CM/WC/MM/SM outcome scorer, metrics, cluster bootstrap, robustness harness
report/       tables and claims ledger
"""

__version__ = "0.1.0"
