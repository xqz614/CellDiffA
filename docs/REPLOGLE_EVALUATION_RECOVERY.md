# Replogle ablation evaluation and Squidiff sampling diagnosis

This recovery does not retrain models, regenerate ablation predictions, change
the experiment plan, or overwrite existing H5AD files.

## 1. Update the server

```bash
cd /root/autodl-tmp/adacell/CellDiffA
git pull --ff-only origin agent/replogle-local-mps
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
mkdir -p results/replogle/logs
```

## 2. Re-evaluate the two ablations (CPU)

```bash
python -u scripts/server/recover_replogle_ablation_metrics.py --num-threads 8
```

Only `scratch_without_anchor` and `scratch_without_direction` are eligible.
The helper checks all input entries. It rejects nonfinite/negative values or
more than one entry per million at/above 15. This conservative guard is not a
statistical proof of scale correctness; the log1p declaration comes from the
audited generation/export pipeline, not from fitting the reference outcomes.

No values are clipped, deleted, divided, or transformed again. The original
errors still count in the metrics. In Cell-Eval 0.6.6, the explicit log1p path
uses `allow_discrete=True` **and** `pdex_kwargs={"is_log1p": True}`. This skips
the heuristic range/counts conversion but preserves the original DE units and
full metric profile. Default evaluation remains unchanged. Record this input
validation override in the experiment audit, rather than describing it as the
unchanged default validator.

Outputs are written to each ablation's new `evaluation/` directory. The original
failed `metrics/` directory is untouched. A hash-checked `evaluated.json` and an
`input_scale_audit.json` record the exact predictions, reference, and metrics.
Complete matching outputs are skipped on rerun. One ablation failing does not
prevent the other from being attempted. Squidiff is not included in recovery.

## 3. Native Squidiff sampling comparison (one GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/server/diagnose_squidiff_sampling.py \
  --config results/replogle/maintext_train_v1/plan.json \
  --device cuda:0 --cells 32 --groups 2
```

This finds the existing checkpoint and its training configuration from the
server plan. It checks completed training, data hashes, split, gene order, and
the pinned upstream revision. It scans training/validation expression ranges.
It never opens test response file `real.h5ad`.

Three validation cases are used: one matched control and two conditions chosen
deterministically from metadata with native training support. Each has 32 cells.
The semantic conditions use encoded validation **controls** plus latent shifts
computed solely from observed training responses. No validation response is used
as a condition. For each case, identical initial noise and semantic conditions
are passed to:

1. The author's native `ddim_sample_loop` with all 1000 training timesteps.
2. The same native loop with the existing `ddim100` schedule.
3. The current Squidiff adapter through the complete unguided one-particle engine
   with the same `ddim100` schedule and a constant zero reward.

All paths use the same frozen weights, `eta=0` and the upstream default
`clip_denoised=False`. No upper clipping or scale repair is added. The native
implementation's own nonnegative clean-sample handling remains unchanged.

Each run creates a timestamped directory under `results/replogle/diagnostics/`.
`report.json` includes input/output ranges, three output distributions, wall time,
the exact timestep maps, and native/adapter maximum absolute difference.
Small NPZ files retain the initial noise, condition vectors and outputs.

Interpretation:

- Native 100 vs adapter 100 disagree: investigate the adapter/engine path first.
- Those agree but native 1000 restores plausible ranges: investigate the reduced
  schedule before changing formal runs. A plausible range alone is not a quality proof.
- Native 1000 is also abnormal: investigate input units, weights and conditioning;
  simply increasing sampling steps is not a demonstrated fix.

These are diagnostics, not new test benchmark results or evidence of AdaCell
performance. Nothing automatically retrains or replaces the current Squidiff runs.

## Optional parallel screen sessions

After the setup above, start each command once:

```bash
screen -L -Logfile "$PWD/results/replogle/logs/ablation_recovery.log" \
  -dmS adacell-eval-repair \
  python -u scripts/server/recover_replogle_ablation_metrics.py --num-threads 8

screen -L -Logfile "$PWD/results/replogle/logs/squidiff_native_check.log" \
  -dmS squidiff-native-check \
  env CUDA_VISIBLE_DEVICES=0 python -u scripts/server/diagnose_squidiff_sampling.py \
  --config results/replogle/maintext_train_v1/plan.json --device cuda:0 --cells 32 --groups 2
```

Use another visible GPU index if GPU 0 is busy. The CPU evaluation can run alongside
the small GPU diagnostic. Monitor both logs with `tail`, or attach using
`screen -r adacell-eval-repair` / `screen -r squidiff-native-check`.

After evaluation completes, refresh the inventory:

```bash
python scripts/server/report_replogle_results.py --verify-hashes
```
