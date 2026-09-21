# Six main-text queues when Squidiff weights are absent on the server

This alternative trains Squidiff **once on the server**, then uses that one
checkpoint for a matched vanilla/AdaCell pair. No upload of Mac weights is
needed. It does not stop existing jobs, overwrite previous main/ablation results,
train another DDPM, or run additional generation seeds.

## Queue layout

| Queue | GPU | Work in order | Purpose |
|---|---|---|---|
| 0 | 0 | Scratch random-of-16, evaluate, mean correction, evaluate | Candidate budget and mean-only correction controls |
| 1 | 1 | Scratch terminal best-of-16, evaluate | Terminal selection versus sequential steering |
| 2 | 2 | Scratch cellwise-scoring control, evaluate | Joint population versus independent cell scoring |
| 3 | 0 | Train Squidiff, generate matched vanilla, evaluate | New checkpoint and its baseline |
| 4 | 1 | Wait for queue 3 training, generate Squidiff + AdaCell, evaluate | Transfer to a second diffusion backbone |
| 5 | 2 | Scratch with 8 particles, evaluate | Accuracy versus candidate budget |

Initially there are **five computing queues and one waiting queue**, not six
simultaneous GPU experiments. Queue 4 starts after training completes; it does
not wait for queue 3's vanilla inference/evaluation. Each GPU has at most two
compute queues from this plan. Existing jobs outside this plan are not counted;
check their utilization before launching. Free VRAM alone does not imply free
compute or guarantee higher throughput.

Squidiff training reuses the pinned upstream architecture and diffusion code.
It runs at most **100,000 optimizer steps**, batch size 64, checks the official
validation split every 5,000 steps, and stops after 5 non-improving checks.
These are steps, not epochs. It saves resumable training state every 1,000 steps
and selects the EMA checkpoint by validation denoising loss, not test scores.
The training process reads only training/validation responses. Merely creating
`best.pt` during training does not release inference; successful process exit,
formal completion metadata and matching artifact hashes are required.

Both new Squidiff predictions use **the same selected checkpoint and 100
respaced DDIM inference steps**, with the 1,000-step training schedule unchanged.
Do not compare new guided outputs against old Squidiff scores from a different
checkpoint or inference schedule. This is a backbone-transfer comparison, not
an equal-budget Squidiff comparison: vanilla uses 1 candidate, AdaCell uses 16.
The compute-matched candidate-selection controls are the Scratch experiments.

The commands below explicitly select `zero_shift` for perturbations without a
native training-derived latent shift. This keeps all official test conditions
but is **not** a learned native prediction for an unseen perturbation. The rule
is identical for vanilla and guided inference. For this Replogle artifact,
previous checks identified 8 such test perturbations; the runner recomputes and
records the actual list. Evaluation retains the full test set and additionally
writes `metrics_with_native_support.csv`, `native_support_means.csv`, and
`native_support_counts.csv`. Report this limitation and the subgroup results.
Do not silently replace this policy, drop these conditions, or claim native
Squidiff handles them. `ridge` is another explicit adapter policy, not the
original method; changing policy requires a new output root.

## Update and prepare

Use the already configured server environment; no environment reinstall or
dataset re-download is required when the original setup is complete.

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
git pull --ff-only origin agent/replogle-local-mps

python scripts/server/replogle_remaining.py init \
  --preset main-text-train \
  --squidiff-unseen-policy zero_shift

python scripts/server/replogle_remaining.py prepare \
  --config results/replogle/maintext_train_v1/plan.json
```

The new output directory is `results/replogle/maintext_train_v1`. Initialization
launches nothing. Preparation reuses/verifies the training reference if present,
otherwise exports it without changing existing test/validation files. It clones
the pinned Squidiff source if absent. Do not manually create a fake checkpoint
or training metadata. The plan intentionally points to a future checkpoint.

## Check GPU training, then launch all queues

First run a two-step training smoke check. It writes to a separate `smoke/`
directory; those weights cannot unlock formal inference or become a benchmark
result. Stop if this command prints a traceback.

```bash
python scripts/server/replogle_remaining.py smoke \
  --config results/replogle/maintext_train_v1/plan.json --lane 3

python scripts/server/replogle_remaining.py launch-maintext \
  --config results/replogle/maintext_train_v1/plan.json
```

The launcher creates six detached `screen` sessions automatically. You do not
need six terminals. It checks inputs, CUDA and Cell-Eval 0.6.6, skips an existing
session of the same name, and refuses known older queue sessions to avoid
duplicate experiments. It never kills other processes. If the old main-text
plan is already running, inspect it before starting this alternative.

## Progress and recovery

```bash
screen -ls
python scripts/server/replogle_remaining.py status \
  --config results/replogle/maintext_train_v1/plan.json

tail -n 20 results/replogle/maintext_train_v1/runs/squidiff_training/run.log
tail -n 20 results/replogle/maintext_train_v1/lane_4.screen.log
```

`waiting_for_training` on queue 4 is expected. It checks completion every 15
seconds without doing inference, and stops on a recorded training failure or
after a 72-hour timeout. A killed host/process may not record a failure, so
also inspect queue 3 if progress stops. Resume a failed training queue only after
fixing its cause. Then restart the waiting queue:

```bash
python scripts/server/replogle_remaining.py launch-maintext \
  --config results/replogle/maintext_train_v1/plan.json --lanes 3

# After queue 3 reports running/training again (or completed):
python scripts/server/replogle_remaining.py launch-maintext \
  --config results/replogle/maintext_train_v1/plan.json --lanes 4
```

Training resumes from `last.pt` under its unchanged contract. Completed training
is reused after hash verification. Partial generation resumes from its saved
shards; complete, unchanged evaluated predictions are skipped. Full Cell-Eval
metrics go to `metrics/<experiment>/`; independent population diagnostics go to
`runs/<experiment>/diagnostics/`. Partial predictions are never evaluated as
full results. Generation controls and diagnostics can feed the existing
`analyze_adacell_experiments.py` report. Measure comparable runtime separately
on an idle GPU, not from competing queue timings.

Local unit/synthetic tests validate scheduling and contracts, not full server
CUDA training, elapsed time, or scientific performance. Check server logs after
launch; a successful `screen` launch alone is not evidence of completed work.
