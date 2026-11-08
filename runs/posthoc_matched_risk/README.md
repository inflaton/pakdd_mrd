# Post hoc matched-target analysis

This analysis pairs the seed-0 intervention-trained set scorer with the five LLM baselines in the PAKDD manuscript. The LLMs retain their emitted decisions.

1. Use each baseline's measured N-FMR as a cap and its accepted-link precision as a floor on the same 1,429 scorable development pools. This includes 811 matchable pools and 618 labeled negatives (37 natural, 581 constructed).
2. On the separate calibration-fit partition, keep the compact scorer's archived pooled temperature and original threshold grid. Fit one pooled policy per target pair because the targets are pooled. Save the policies and target counts in `frozen_fit.json`.
3. Evaluate every frozen policy on calibration confirmation. Report point adherence and per-policy one-sided bounds for the two constraints.
4. Apply every frozen policy to the same 2,066 scorable test pools as the LLM baselines. Report realized risks and recall, false rejection, and deferral. The paired intervals resample query-entity clusters. These comparisons are post hoc and the targets describe the benchmark mixture.

From the repository root, use the reconstructed primary benchmark and a fresh output directory to replay the analysis.

```sh
python scripts/mrd/posthoc_matched_risk.py fit \
  --benchmark-dir /path/to/benchmark --output-dir /path/to/new-run \
  --comem-dev runs/posthoc_matched_risk/comem_dev/decisions.jsonl.gz
python scripts/mrd/posthoc_matched_risk.py score \
  --benchmark-dir /path/to/benchmark --output-dir /path/to/new-run
```

The replay reads archived LLM decisions. The ComEM development archive includes the pool and selected row IDs, action, selected index, and output token count. The fit records the digest of the benchmark pool files and label screens so the score phase can verify the same inputs regardless of their local path.

## Fit and confirmation

| Baseline supplying targets | Dev N-FMR cap | Dev precision floor | Link cutoff | Both point targets on confirmation | Both confirmation bounds |
| --- | ---: | ---: | ---: | :---: | :---: |
| GPT-OSS-120B, rich rubric | 4.37% | 93.74% | 0.90 | Yes | No |
| Gemma 4 12B, minimal JSON | 14.72% | 87.65% | 0.64 | Yes | No |
| Gemma 4 12B, rich rubric | 10.19% | 90.65% | 0.72 | Yes | No |
| Gemma 4 12B, constrained symbols | 14.72% | 87.83% | 0.64 | No | No |
| Adapted ComEM | 43.20% | 70.22% | 0.50 | Yes | Yes |

All five fits met both targets. The rejection cutoff was 0.50 in every fit. The confirmation bounds are query-level and use a one-sided 2.5% error probability per constraint. Only the ComEM-derived pair cleared both bounds. The Gemma symbols fit shared the minimal JSON cutoff and missed its slightly higher precision floor on confirmation.

## Common held-out test

Each paired entry is compact / LLM. The held-out mixture contains 1,095 matchable pools and 971 labeled negatives (42 natural, 929 constructed).

| Baseline | Recall | Precision | N-FMR | Defer | False NONE | Paired recall difference (95% interval), pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS-120B, rich rubric | 60.6% / 68.7% | 96.9% / 93.8% | 1.96% / 4.74% | 39.4% / 0.0% | 5.3% / 30.1% | -8.0 [-11.9, -4.2] |
| Gemma 4 12B, minimal JSON | 78.1% / 95.1% | 90.9% / 87.8% | 8.14% / 13.59% | 27.0% / 0.0% | 5.3% / 3.7% | -17.0 [-20.0, -14.3] |
| Gemma 4 12B, rich rubric | 74.4% / 93.1% | 92.4% / 90.5% | 6.39% / 10.40% | 29.9% / 0.0% | 5.3% / 6.3% | -18.6 [-21.8, -15.9] |
| Gemma 4 12B, constrained symbols | 78.1% / 93.4% | 90.9% / 87.4% | 8.14% / 13.80% | 27.0% / 0.0% | 5.3% / 5.3% | -15.3 [-18.5, -12.4] |
| Adapted ComEM | 84.7% / 88.6% | 86.1% / 66.2% | 14.01% / 45.42% | 20.4% / 0.0% | 5.3% / 6.4% | -3.9 [-6.6, -1.3] |

Every compact policy met both development-derived targets on the test mixture. In all five pairs it had higher realized precision and lower N-FMR than the LLM, with lower recall and more deferral. The target pairs specify a precision floor and N-FMR ceiling, so the realized risks differ. The compact policies made 0–2 false links on the 42 labeled natural negatives. These rates and intervals describe the benchmark pools.
