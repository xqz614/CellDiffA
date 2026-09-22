# Replogle main-text-first experiments

For the currently generated results and unresolved Squidiff scale anomaly, use
[the completion workflow](REPLOGLE_MAIN_TEXT_COMPLETION.md). It resumes only
missing Replogle controls and queues figures, isolated timing, and native
Squidiff diagnostics without restarting the anomalous full Squidiff pair.

If the server has no Squidiff checkpoint and uploading is impractical, use
[the train-on-server six-queue plan](REPLOGLE_SERVER_TRAIN_SQUIDIFF.md) instead.
It trains once and automatically releases a matched vanilla/AdaCell pair.
The instructions below assume that completed weights already exist.

This plan supersedes the earlier six-lane *full* launch for the current writing
deadline. It does not change or terminate existing processes. It starts no
training and no extra generation seeds. Full-server results are not implied by
the local tests.

## The six lanes

| Lane | GPU | First experiment | Question |
|---|---|---|---|
| 0 | 0 | Scratch random-of-16 | Does generating more candidates alone explain the gain? |
| 1 | 1 | Scratch terminal best-of-16 | Is selection during denoising needed? |
| 2 | 2 | Scratch with cellwise reward | Does joint population scoring help? |
| 3 | 0 | Squidiff matched vanilla | Matched reference for backbone transfer |
| 4 | 1 | Squidiff + AdaCell | Does steering transfer to another architecture? |
| 5 | 2 | Scratch with 8 particles | Accuracy versus sampling budget; optional |

Lane 0 automatically evaluates its full prediction, then performs and evaluates
the inexpensive same-prior mean correction. Other lanes stop after their one
experiment and evaluation. No lane continues to seed43/44, a new DDPM, or further
Squidiff controls. Each lane has at most one heavy subprocess at a time; data
loader workers can still create additional operating-system processes.

All inference uses the same fixed seed42. Guided methods fix alpha1. Scratch
uses 16 native 32-cell blocks per population and 100 DDIM steps. Lane 5 changes
only the number of candidate populations from 16 to 8, not the number of cells
in each candidate. Compare it with an existing main run only if all other
settings match. It belongs to accuracy/cost analysis or appendix sensitivity,
not the core population-scoring mechanism test. Skip lane 5 if the first five
experiments are the immediate priority.

Squidiff uses 100 respaced DDIM steps in both lanes. It reuses trained weights,
but does not reuse an old score generated with different inference settings.
The vanilla/Squidiff+AdaCell pair alone does not establish a compute-matched
advantage on Squidiff; the budget/selection mechanism controls are on Scratch.

## Update and initialize without starting experiments

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
git pull --ff-only origin agent/replogle-local-mps

python scripts/server/replogle_remaining.py init --preset main-text
python scripts/server/replogle_remaining.py prepare \
  --config results/replogle/maintext_v1/plan.json
```

The new plan/output root is `results/replogle/maintext_v1`, separate from
`remaining_v1` and all existing main/ablation results. Initialization never edits
the earlier plan. Do not run both plans' copies of the same experiment.

## Bind your existing Squidiff weights and check the adapter

The checkpoint must be present on this server; a metric CSV alone is insufficient.
Keep its real training `run_config.json` and existing `prediction_config.json`
alongside it. Do not invent their provenance fields to pass a check. External
checkpoints need the explicit `--model-config` format described in
[the adapter guide](REPLOGLE_MULTIBACKBONE_SERVER.md).

```bash
read -r -p "Squidiff checkpoint full path: " ADACELL_SQUIDIFF_CKPT
python scripts/server/replogle_remaining.py configure-squidiff \
  --config results/replogle/maintext_v1/plan.json \
  --checkpoint "$ADACELL_SQUIDIFF_CKPT"

python scripts/server/replogle_remaining.py smoke \
  --config results/replogle/maintext_v1/plan.json --lane 4
```

The smoke checks one small population, writes no evaluator-ready H5AD, and is
not a parameter-selection run. Do not proceed past a traceback. The adapter
retains the existing explicit unseen-perturbation policy, or stops if unavailable;
it never silently substitutes zero shifts.

## Launch six screen sessions

```bash
python scripts/server/replogle_remaining.py launch-maintext
```

To omit the optional eight-particle experiment, use `--lanes 0 1 2 3 4` instead.
If Squidiff's checkpoint is not yet available, start the four independent
Scratch lanes with `--lanes 0 1 2 5`; after binding the checkpoint and passing
the smoke check, launch `--lanes 3 4`.

The launcher checks required paths, CUDA availability and Cell-Eval 0.6.6 before
dispatch. It logs each screen session, skips already-existing main-text screen
sessions, and refuses to launch alongside known older `adacell-lane-*` screen
sessions. It does **not** kill, pause or modify any experiment. A launch request
is not evidence that inference succeeded: inspect the following logs.

```bash
screen -ls
python scripts/server/replogle_remaining.py status \
  --config results/replogle/maintext_v1/plan.json

tail -n 20 results/replogle/maintext_v1/lane_4.screen.log
tail -n 20 results/replogle/maintext_v1/runs/squidiff_adacell16/run.log
```

Re-run the same `launch-maintext --lanes ...` command to resume a terminated
lane. Existing prediction shards and evaluated artifacts have integrity checks.
No outputs are erased. Each fully generated H5AD is followed by the fixed full
Cell-Eval evaluation and independent population diagnostics. Partial generations
do not receive benchmark scores.

## Figures and timing after predictions finish

Use `scripts/baselines/analyze_adacell_experiments.py` with this plan and the
actual completed Scratch-alpha1 prediction/metric paths. Existing main runs and
the new controls supply contrast-diagnosis, population-control and
accuracy/diversity plots; no new response generation is needed for those plots.
The report now also shows the vanilla/AdaCell Squidiff pair with its unequal
sampling budgets labelled, not as a compute-matched comparison.

Only measure GPU runtime with `benchmark_replogle_steering_cost.py` when that
GPU is idle. Concurrent-job timings are not fair latency measurements. The
eight-particle full-test job supplies an accuracy point; the isolated timing
script supplies its comparable latency measurement.

## Only run Squidiff + AdaCell; keep the old Squidiff result

Do **not** launch lane 3 or the whole plan when only AdaCell is requested:

```bash
python scripts/server/replogle_remaining.py launch-maintext --lanes 4
```

This starts one guided inference job plus its evaluation, with no model training
or unguided Squidiff generation. Copy an existing weight to this server if it is
only on the Mac; Git does not include checkpoints. The Mac's
`results/replogle/squidiff_cpu/best.pt` was verified loadable and its training log
records completion at 100,000 steps, but that directory contains no prediction
configuration. This is evidence of trained weights, not proof that an existing
baseline score used those weights.

Before comparing to an existing baseline, identify its exact checkpoint,
gene order, split, normalization, control sampling, inference step count,
and unseen-perturbation rule. The repository's original
`run_squidiff_replogle.py` uses the native 1,000-step DDIM inference loop. The
new adapter otherwise defaults to 100 respaced steps, which is **not** directly
interchangeable with that baseline. For a confirmed 1,000-step baseline, bind:

```bash
python scripts/server/replogle_remaining.py configure-squidiff \
  --config results/replogle/maintext_v1/plan.json \
  --checkpoint /ACTUAL/SQUIDIFF/DIRECTORY/best.pt \
  --sampling-steps 1000
```

Use another step count only when supported by the actual previous inference
configuration. Do not infer inference steps merely from the training schedule.
If `prediction_config.json` is absent, ask for the original command/configuration
before choosing `--unseen-policy`; never silently choose `zero_shift` or `ridge`.
The existing contract refuses to mix changed settings into old prediction shards.
Changing only the step count does not by itself verify every other comparison
setting. You can generate a guided result without rerunning the baseline, but
cannot claim a controlled gain until the old result's provenance is established.
