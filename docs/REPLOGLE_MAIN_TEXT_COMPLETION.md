# Finish the Replogle main-text experiments

Main results and ablations retain the planned three datasets. All other
main-text experiments below use Replogle only. This launcher does not start
additional generation seeds, data downloads, main runs, or ablations.

## Run on AutoDL

```bash
cd /root/autodl-tmp/adacell/CellDiffA
git pull --ff-only origin agent/replogle-local-mps
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle

python scripts/server/complete_replogle_maintext.py launch
```

This uses the existing `results/replogle/maintext_train_v1/plan.json`, the
prespecified complete `test_sensitivity/scratch_alpha1` main prediction, and
its actual metrics. Use `--main-run PATH --main-metrics DIRECTORY` if their
locations differ. It refuses ambiguous metrics, other alphas, validation
predictions, changed contracts, or stale evaluation hashes. It does not choose
the alpha with the best test score. `--dry-run` displays commands without writes
or launches.

## What runs

| Main-text section | Work | Execution |
| --- | --- | --- |
| Diagnosing perturbation contrast | Overall-expression R² versus PDCorr; paired response gains stratified by observed effect size | CPU analysis after controls finish |
| Why population steering? | random16, best-of-16, cellwise16, same-prior mean correction, full AdaCell | Resume only unfinished existing lanes 0/1/2; completed results are hash-checked and reused |
| Response accuracy and diversity | Full-population versus eight-particle accuracy; variance, effective rank, projected Wasserstein distances to real populations and random16 | Resume lane 5 if needed, then CPU analysis |
| Computational cost | Random16, best16, cellwise16 and AdaCell with 4/8/16 particles on the same 12 groups; discard the first warm-up group | Wait for controls and an idle GPU 2, then run sequentially |
| Additional backbone | Compare native Squidiff DDIM1000, native DDIM100, and adapter DDIM100 on validation-only small cases | GPU 1, reuse a completed diagnostic for the same weights/config, or wait for an existing diagnostic |

No full Squidiff or Squidiff+AdaCell rerun is launched here. Their previous
output-scale anomaly remains unresolved until the native diagnostic is reviewed.
`needs_scientific_review` is not a successful benchmark result. No retraining or
arbitrary clipping/rescaling is used to hide that anomaly.

Existing screens are not stopped. Each control lane and completion worker uses
a single-writer lock. The mean correction follows random16 generation and
evaluation. Figure generation and timing become runnable once all five controls
are evaluated. The wait expires after 72 hours by default, leaving existing
artifacts intact. Re-running the launcher skips finished or active work.

## Progress and outputs

```bash
screen -ls
python scripts/server/complete_replogle_maintext.py status
python scripts/server/replogle_remaining.py status \
  --config results/replogle/maintext_train_v1/plan.json
tail -n 30 results/replogle/maintext_train_v1/maintext_completion/analysis.screen.log
tail -n 30 results/replogle/maintext_train_v1/maintext_completion/timing.screen.log
tail -n 30 results/replogle/maintext_train_v1/maintext_completion/squidiff.screen.log
```

`maintext_completion/{analysis,timing,squidiff}.status.json` records waiting,
running, complete, failed, or review-required states. Detailed subprocess output
is in `analysis.log`, `timing.log`, and `squidiff.log` in the same directory.

The timestamped `figures_*` directory contains PDF/PNG figures and underlying
CSVs. The timestamped `timing_*` directory contains timing CSV/JSON and a cost
figure. Squidiff diagnostics remain under `results/replogle/diagnostics/`.
No previous figure or timing directory is overwritten.

## Interpretation boundaries

- Figures compare actual full-test predictions. Strict mode verifies recorded
  group partitions, prior settings, and denoised cell-steps for random16,
  best16, cellwise16 and the full main run. Eight particles intentionally have
  half the denoising work; they are not labelled compute-matched.
- Error bars are perturbation bootstrap intervals for one generation seed, not
  variation across independent training or generation seeds.
- Effect-strength strata use observed test responses only for retrospective
  analysis, not for steering or hyperparameter selection.
- Overall R² is expression fit, not proof of cell realism. Variance, rank and
  projected Wasserstein distances are descriptive distribution checks, not a
  guarantee of biological support preservation. The random16 output is an
  unguided reference sample, not the full underlying probability distribution.
- A difference between overall fit and effect accuracy diagnoses an empirical
  pattern; it does not prove that the denoising objective caused the discrepancy.
- Idle-GPU timings measure sampling latency on fixed groups, not end-to-end
  runtime. Do not start another workload on that GPU during timing. The timing
  script rechecks idleness before every method and stops if another job appears.
- Matched predictions/metrics and a complete figure script do not guarantee
  favorable outcomes. Report the observed comparisons, including negative ones.
- Pending PerturbDiff controls and their mean-correction baseline are evaluated
  with their known log1p units explicitly declared, after finite/nonnegative
  and rare-tail checks. Values are unchanged and the audit is saved. This does
  not bypass the unresolved Squidiff scale anomaly.
