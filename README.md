# Match, Reject, or Defer? Selective Entity Resolution over Retrieved Candidate Sets

Reproducibility artifacts for the PAKDD manuscript. The primary result is the seed-0 comparison of pairwise and set-aware scorers, each trained with and without pool interventions. The locked test contains 2,066 scorable pools. The LLM baselines are evaluated at their emitted decisions. The archive distinguishes empirical test rates from calibration confirmation bounds.

## Regenerate the reported numbers

From a fresh checkout with Python 3.12 and Make:

```sh
make -C paper/pakdd check
make -C paper/pakdd numbers
```

The first command compares a fresh generation with the checked-in LaTeX tables and macros. The second updates them. Both read the archived scores, policies, and summaries under `runs/` and require only Python's standard library. No source datasets, model downloads, API calls, or GPU are needed. The notebook in `notebooks/08_pakdd_manuscript.ipynb` presents the same numbers and per-cohort comparisons. To run it, install the `notebooks` extra and open the notebook from the `notebooks` directory.

## Rebuild the benchmark and replay policies

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/sigir/01_download_sources.py --no-new-pins
python scripts/sigir/02_build_benchmark.py
python scripts/sigir/03_constructed_cohort.py
```

The download script obtains the pinned archives from the source maintainers. The benchmark code rebuilds retrieval pools and labels, with released entity links and clusters as the basis for partitioning. The pins and expected pool hashes are under `data/sigir/public/`. The package includes the row-ID membership needed to reproduce the Walmart–Amazon exposure restriction. The scripts rebuild records from the publisher archives. Each dataset remains subject to its original terms.

Run the three policy diagnostics from the repository root after the benchmark build. Use fresh output paths to preserve the reference snapshots.

```sh
mkdir -p reproduced
python paper/pakdd/fit_breakpoints.py --benchmark-dir data/sigir/public/benchmark --output reproduced/fit_breakpoints.json
python paper/pakdd/policy_diagnostics.py --benchmark-dir data/sigir/public/benchmark --output reproduced/policy_diagnostics.json
python paper/pakdd/transfer_confirmation.py --benchmark-dir data/sigir/public/benchmark --output reproduced/transfer_confirmation.json
```

Replay statistics can be compared with the checked-in diagnostic JSON. Transfer provenance hashes describe the archived inputs and may differ when the released predictions omit ancillary metadata. To re-score the main policy from its archived candidate probabilities:

```sh
python scripts/sigir/21_evaluate.py --partition test --cohorts natural,constructed --budget 0.02 \
  --arm set_joint_interv_seed0=runs/preds/set_joint_interv_seed0.jsonl.gz:runs/calib/set_joint_interv_seed0/policies \
  --runs-dir reproduced
```

## Train the compact scorers

The optional `models` extra and a PyTorch installation appropriate for the accelerator are needed for training. The encoder revision, seed, training mixture, and retrieval feature settings are in the model YAMLs. The four seed-0 arms used the following commands after the benchmark build:

```sh
python scripts/sigir/10_train.py --config models/pairwise_encoder.yaml --name pairwise_clean --seed 0
python scripts/sigir/10_train.py --config models/pairwise_encoder.yaml --name pairwise_interv --seed 0 --max-steps 3070 --override train.intervention_training=true
python scripts/sigir/10_train.py --config models/set_model_joint.yaml --name set_joint_clean --seed 0 --max-steps 3070 --override train.intervention_training=false
python scripts/sigir/10_train.py --config models/set_model_joint.yaml --name set_joint_interv --seed 0
```

Use `scripts/sigir/11_predict.py` to score pools from a trained checkpoint and `scripts/sigir/20_calibrate.py` to fit policies on calibration fit. The archive is sufficient for result reproduction without rerunning training. Prompt templates for the direct judges are in `configs/sigir/prompts/`. The model and run settings are in `configs/sigir/models/` and the judge and ComEM runners in `scripts/sigir/`.

## Contents and terms

- `src/`, `scripts/sigir/`, and `configs/` contain implementation, benchmark and evaluation scripts, and configurations.
- `runs/` contains the paper's aggregate results, decision and probability records, and frozen policies. The released records preserve the benchmark IDs and scores needed to reproduce the results and omit verbatim model replies and source-record text.
- `paper/pakdd/` contains the numerical generator, diagnostic replays, and generated tables. `notebooks/` contains the results notebook.
- The source code is released under Apache-2.0 as declared in `pyproject.toml`. Third-party datasets and model weights are distributed by their respective maintainers.
