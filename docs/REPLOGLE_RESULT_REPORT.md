# Inspect completed Replogle experiments on the server

This command only reads existing artifacts and creates a separate inventory.
It does not start training, sampling, evaluation or GPU work, and does not
alter any experiment. It cannot inspect the server from the Mac; run it on the
server where the files exist.

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
git pull --ff-only origin agent/replogle-local-mps

python scripts/server/report_replogle_results.py --verify-hashes
```

The terminal prints separate tables for main/alpha sensitivity, ablations, and
the six-queue follow-up experiments. Existing baseline files are also inventoried.
It follows locally matching `plan.json` files for both `maintext_v1` and
`maintext_train_v1`, so never-started planned jobs remain visible. Smoke,
validation, and earlier report/analysis directories are excluded. It includes
all available alpha settings, without choosing a winner using the test set.

Default reports are written to a new timestamped directory under
`results/replogle/reports/`; the command prints the exact path. Outputs include:

- `report.md`: readable four-metric tables and checks.
- `all_results.csv`: all 14 protocol metrics, available sampler parameters,
  coverage counts, status, and source paths.
- `main.csv`, `ablation.csv`, `additional.csv`: one file per discovered category.
- `inventory.json`: machine-readable status and warnings.
- `evaluate_missing.sh`: optional commands for missing evaluations in the three
  requested experiment categories. **The report never executes this file.**

| Status | Interpretation |
|---|---|
| `EVALUATED` | Full prediction metadata and all test-condition metric rows agree; all 14 metric means are finite. |
| `EVALUATED_UNDEFINED` | Full coverage, but at least one metric contains NaN/Inf. Its mean is withheld, not computed over fewer conditions. |
| `NEEDS_EVALUATION` | Full prediction metadata matches the reference, but no protocol metric files were found. |
| `INCOMPLETE` / `NOT_STARTED` | No full prediction artifact; progress is shown if recorded. A stopped process alone is not completion. |
| `WAITING_FOR_TRAINING` / `FAILED` | Recorded queue dependency/failure, not an evaluated result. |
| `TRAINING_COMPLETE` | The training progress and checkpoint indicate completion; this is not a prediction score. |
| `METRICS_ONLY` / `SUMMARY_ONLY` | Cannot associate a full prediction or verify per-condition coverage. No scores are promoted to complete results. |
| `CHECK_FAILED` | Shape, labels, coverage, summary, or saved hashes disagree. Inspect the note before using scores. |
| `SUPERSEDED` | The legacy Finetuned artifact has been superseded by the corrected category-ID run. |

Means are recomputed from per-perturbation CSVs, with equal weight for each test
perturbation. Each mean needs every condition to have a finite value. Existing
summary CSVs are checked for consistency. Main terminal columns are DEOver,
PDCorr, PDS-cos, and MSE; no simulated values, ranking, or missing-value imputation
is used. Unknown/ambiguous metric-to-run associations are reported, not guessed.

`--verify-hashes` verifies an available `evaluated.json` against its prediction,
reference and metrics. Older manually evaluated runs may have no such record;
they receive `hash_check=no_record`, not `verified`. This report checks metadata
and saved metric identities, **not expression values, scientific validity, or
training provenance from scratch**. No saved hash record means prediction/metric
identity is not independently established. Do not treat the inventory as a
replacement for the evaluation protocol or scientific audit.

If main/ablation prediction files exist but evaluation was never run, inspect
the generated `evaluate_missing.sh`, then explicitly execute its printed full
path with `bash`. It performs CPU evaluation sequentially, not retraining.
Afterwards rerun the inventory command to produce an updated report. Failed,
ambiguous, summary-only, and partial cases are intentionally not automatically
re-evaluated or overwritten.

For non-default results, use `--root /absolute/path/to/results/replogle` and
optionally `--reference /absolute/path/to/real.h5ad`. `--outdir` must be a new
directory. When jobs are still writing, an inventory is only a snapshot; rerun
it after the files settle. Reported concurrent durations are not a GPU benchmark.
