# AdaCell multi-backbone experiments on Replogle

**Current writing priority:** use the narrower
[main-text-first plan](REPLOGLE_MAIN_TEXT_FIRST.md). It omits extra seeds and
conditional-DDPM training. The six-lane table below remains the older full plan;
do not launch it when following the main-text-first instructions.

This release adds runnable adapters and controls, not completed scientific
results. Local verification uses small CPU fixtures and the pinned native
Squidiff kernel. Full CUDA experiments must run on the user's server. Existing
main/ablation outputs, weights and dataset splits are not overwritten.

## What is supported

| Backbone/control | What runs | Status |
|---|---|---|
| PerturbDiff | Existing population-native sampler, Scratch/Fine | Existing adapter retained |
| Squidiff | Frozen semantic encoder + native respaced DDIM, with/without AdaCell | Adapter implemented; server checkpoint path/config required |
| Conditional DDPM | Plain independent-cell epsilon MLP; GenePT + observed control-mean conditioning | New training and inference code; an **in-house reference**, not scDiff |
| Equal-budget controls | 16 unselected populations, terminal best-of-16 | Same denoising budget, no intermediate selection |
| Cellwise reward | Mean of singleton component scores, normalized after averaging | Same model attention and biological inputs; deliberately decomposable reward |
| Mean correction | Post-hoc translation and nonnegative projection toward the same prior mean | Not variance preserving; infeasible negative means reported |
| Cost | Sequential runs on an idle GPU; discard one warm-up population | Separate from timings collected during concurrent jobs |

Scratch and Fine are not two independent architectures. Squidiff and conditional
DDPM provide two additional architectures only after their full paired experiments
pass. Do not claim generality across all diffusion models from these experiments.

Research candidates considered:

- [scDiff](https://github.com/OmicsML/scDiff) has official perturbation and gene
  perturbation experiments, but needs its own graph/data/sampling adapter and
  checkpoint on the exact split. **Not implemented here**.
- [Doloris](https://github.com/ChangxiChi/Doloris) is a perturbation diffusion
  bridge; its boundary conditions and training protocol are not interchangeable
  with an epsilon DDPM. **Not implemented here**.
- [scDiffusion](https://github.com/EperLuo/scDiffusion) emphasizes conditional
  cell generation. This alone does not establish an unseen-gene perturbation
  predictor on our split. Do not relabel conditional DDPM as this published method.

## Six server lanes

| Lane | GPU by default | Sequential work in the lane |
|---|---|---|
| 0 | 0 | Scratch unselected16 -> same-prior mean correction |
| 1 | 1 | Scratch best-of-16 -> cellwise-scoring control |
| 2 | 2 | Squidiff matched vanilla -> Squidiff + AdaCell |
| 3 | 0 | Squidiff unselected16 -> Squidiff best-of-16 |
| 4 | 1 | Train conditional DDPM -> vanilla -> unselected16 -> best-of-16 -> AdaCell |
| 5 | 2 | Scratch 8 particles -> unselected/AdaCell pair at seed43 -> same pair at seed44 |

Each lane executes a single heavy subprocess at a time. A completed prediction
is followed by Cell-Eval 0.6.6 full-profile evaluation and independent population
diagnostics. Incomplete predictions are never evaluated. Undefined metrics are
flagged in `evaluated.json`, not imputed. A lane stops on an error; other lanes
are independent. Same-lane and same-output locks prevent accidental duplicates.

The matched Squidiff pair uses **100 respaced DDIM steps** for both sides.
The trained diffusion schedule remains its native 1,000-step schedule, with
upstream time mapping preserved. This is an inference protocol, not new training.
Do not compare this AdaCell run to an old Squidiff score generated with a different
number of sampling steps, conditioning rule, or preprocessing. Keep that old
result separately. The one-candidate vanilla result is not an equal-16-candidate
budget control; use unselected16 and best-of-16 for that purpose.

## 1. Update and prepare (no training starts)

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
git pull --ff-only origin agent/replogle-local-mps

python scripts/server/replogle_remaining.py init
python scripts/server/replogle_remaining.py prepare
```

`prepare` adds `train.h5ad` if absent and checks an existing training export. It
never overwrites `real.h5ad`, validation or controls. It fetches pinned Squidiff
source if missing; it does not fetch or train weights. The current CUDA environment
is reused; no PyTorch upgrade is required. A plan is local to this server. Do not
copy a Mac-generated plan to Linux.

Before launching, inspect a lane without running it:

```bash
python scripts/server/replogle_remaining.py run --lane 0 --dry-run
```

## 2. Bind the completed Squidiff checkpoint

Replace the path below with the actual completed model checkpoint on the server.

```bash
python scripts/server/replogle_remaining.py configure-squidiff \
  --checkpoint /ACTUAL/SQUIDIFF/DIRECTORY/best.pt

python scripts/server/replogle_remaining.py smoke --lane 2
```

This smoke run checks one tiny population, writes to `remaining_v1/smoke`, and
does not produce an official score. It is an implementation check, not parameter
tuning on test outcomes. Do not tune hyperparameters based on the smoke output.

For weights produced by this repository's `run_squidiff_replogle.py`, keep the
adjacent `run_config.json` and, if available, `prediction_config.json`. The latter
lets `auto` use the **same** explicit unseen-gene policy as the completed baseline.
Unknown conditions without a policy cause an error, not silent zero shifts.

For an external Squidiff checkpoint, pass `--model-config` to `configure-squidiff`.
That JSON must contain `genes` in their actual trained order (or its
`ordered_genes_sha256`), `split_sha256`, `train_sha256`, and `model_kwargs` matching
the original training architecture. These are provenance declarations, not values
to invent merely to pass a check. The loader uses strict state-dict matching.

If the original baseline used `zero_shift`, use that same policy in every paired
run and report the native-unsupported conditions separately. Alternatively,
explicit `--unseen-policy ridge` estimates semantic shifts using observed training
perturbations and GenePT descriptors; **all** Squidiff comparators must then be
regenerated and labelled as this extension. Neither accesses held-out responses.

## 3. Launch with screen

If Squidiff is bound and its smoke check passes:

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
ADACELL_PYTHON="$(command -v python)"
for lane in 0 1 2 3 4 5; do
  screen -L -Logfile "$PWD/results/replogle/remaining_v1/lane_${lane}.screen.log" \
    -dmS "adacell-lane-${lane}" "$ADACELL_PYTHON" -u \
    scripts/server/replogle_remaining.py run --lane "$lane"
done
screen -ls
```

If the checkpoint location/config is still pending, start only lanes `0 1 4 5`.
Bind Squidiff later and launch `2 3` using the same loop. Do not guess a checkpoint
or retrain a model that is already completed. Six processes share three GPUs;
free VRAM is not evidence of free compute. Reduce concurrency if throughput drops.

Progress and logs:

```bash
python scripts/server/replogle_remaining.py status
tail -n 20 results/replogle/remaining_v1/lane_0.screen.log
tail -n 20 results/replogle/remaining_v1/runs/scratch_random16/run.log
tail -n 20 results/replogle/remaining_v1/runs/squidiff_adacell16/run.log
tail -n 20 results/replogle/remaining_v1/runs/conditional_ddpm_training/run.log
screen -r adacell-lane-0
```

Detach screen with Ctrl-A then D. Re-run a stopped lane with the same command;
prediction shards and completed evaluations are reused after integrity checks.
Changing scientific settings requires a new output directory. Never mix partial
outputs from different seeds, population sizes, checkpoints or reward definitions.
The launch preflight requires Cell-Eval 0.6.6 before starting long computations.
If a screen session exits immediately, inspect its `lane_*.screen.log` first.

## 4. Main-text analysis figures (CPU)

Supply the actual complete Scratch alpha1 H5AD and its metric directory:

```bash
python scripts/baselines/analyze_adacell_experiments.py \
  --plan results/replogle/remaining_v1/plan.json \
  --main-pred results/replogle/test_sensitivity/scratch_alpha1/celldiffa_scratch.h5ad \
  --main-metrics /ACTUAL/SCRATCH_ALPHA1/METRICS_DIRECTORY \
  --outdir results/replogle/figures_remaining_v1
```

Outputs include PDF/PNG figures for expression-fit versus response accuracy,
paired gains stratified by observed effect strength, population controls, and
accuracy/diversity tradeoffs, plus numeric CSVs and source hashes. Missing jobs
are listed explicitly; a partial report is not a completed experimental section.
Diagnostic test strata never select alpha. Confidence intervals bootstrap test
perturbations and are conditional on the generation seed, not across-seed errors.
Matched improvement across available backbones is plotted separately. Paired
seed42/43/44 deltas and their across-seed mean/std are exported as separate CSVs;
a missing seed is reported through the count, never filled in.

## 5. Fair computational-cost measurement (wait for an idle GPU)

```bash
python scripts/server/benchmark_replogle_steering_cost.py \
  --plan results/replogle/remaining_v1/plan.json \
  --gpu 0 --groups 12 \
  --outdir results/replogle/isolated_cost_v1
```

This refuses an occupied GPU, runs methods sequentially on identical populations,
and drops the first population as warm-up. It reports sampling time including
rewards/selection, not training, data loading or evaluation. Repeat timing runs if
reporting uncertainty. Do not substitute the concurrent-job timings for these.

## Scientific boundaries and remaining work

- Main alpha stays 1. Test alpha .5/2 runs are sensitivity analysis, not selection.
- Rewards use the same train-derived 20-gene signature, direction, MMD anchor,
  normalization and log-expression/10 units as PerturbDiff steering. Candidate
  population construction is fixed within each matched backbone comparison.
- No paired cell trajectories are invented. Conditional DDPM conditions on a
  context control mean and a descriptor; it is trained with standard epsilon MSE.
- Reward-dependent resampling with deterministic DDIM may duplicate ancestors;
  it does not regenerate independent descendants. Ancestry, variance/rank and
  sliced-W1 diagnostics are recorded. None proves causal mechanisms or support
  preservation. The population sampler is shared, not a newly invented SMC method.
- A direct original-DAS reproduction, native scDiff/Doloris adapters, prior-noise
  robustness grids, PBMC/Tahoe transfer, and biological mechanism case studies
  are **not completed by this release**. They must not appear as completed runs.
- Before asserting improvements, inspect full metrics, paired comparisons,
  diversity diagnostics and seeds. No experimental improvement is assumed here.
