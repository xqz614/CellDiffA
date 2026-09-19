# Replogle local experiment record

Updated 2026-09-19, 17:25 China time. This is an execution record, not a claim
that the full baseline suite or AdaCell experiments have finished. A baseline is
complete only after a full prediction artifact passes the shared contract and
the full published evaluation finishes. Unit tests and smoke runs do not count.

## Scope and controls

- Host: Apple M4 Pro, 48 GB unified memory; Apple MPS or CPU, not CUDA.
- Primary environment: conda `adacell-replogle`, Python 3.10, torch 2.5.1,
  Cell-Eval **0.6.6**. Additional incompatible baselines use separate environments.
- Data: released Replogle-Nadig artifact, not GEARS' Replogle-K562 substitute.
- Split: pinned PerturbDiff `configs/data/perturb_data/replogle.yaml`.
- Features: published ordered 2,000 evaluation genes. The released finetuned
  backbone still runs in its native 12,626-gene space.
- Predictions, references, controls, evaluation metrics and input hashes are
  preserved. Large files remain in ignored local directories.
- Hyperparameters are selected on the official validation split only. Test
  outcome scores must not be used for model selection or repeated tuning.
- Existing example/simulated paper numbers are not experimental evidence.

## Completion checklist

| Component | Current state |
|---|---|
| Source versions | PerturbDiff, GEARS, CPA, STATE, CellFlow, Squidiff, Scouter checked out at registry pins |
| Data and two released checkpoints | Downloaded, SHA256 verified, extracted and inspected |
| Mac runtime | Main and isolated CPA/STATE/CellFlow environments installed; native training/prediction smoke tests passed |
| Reference/validation export | Complete and immutable; official training, validation, test and controls audited |
| Four Mean variants | **Complete**, all 380 test perturbations evaluated |
| Linear | **Complete**, all 380 test perturbations evaluated |
| Scouter | **Complete**, validation-selected checkpoint, all 380 test perturbations evaluated |
| PerturbDiff Scratch test | Full resumable native sampling running; evaluation follows automatically |
| PerturbDiff Finetuned test | Queued behind Scratch in the running job |
| PerturbDiff Scratch validation | **Complete**, all 60 validation perturbations evaluated |
| GEARS | Native training/prediction smoke passed; full training running |
| CPA | Full native CPU training running, 13-epoch cap; not complete |
| STATE / CellFlow / Squidiff | Native training/prediction smoke passed; full runs pending |
| AdaCell Scratch validation | First full validation run running; no complete AdaCell result yet |
| AdaCell Finetuned / test runs | Await validation selection and locked settings |
| Compute-matched controls / ablations | Implemented and unit tested; full experiments pending |
| Independent population diagnostics | Implemented and run for Scouter; remaining experiments pending |

The source has **643,413 cells × 6,642 genes**, with a 22.30 GB uncompressed
matrix. Training contains 611,710 rows. Validation contains 4,825 responses
across 60 conditions, and test contains 26,878 responses across 380 conditions.
Each evaluation reference includes the same 4,976 matched controls. Unused
condition names in YAML do not inflate evaluated condition counts.

Scouter stopped after 11 epochs by the native validation rule, within its
40-epoch cap. Its loss was vectorized with tested CPU/MPS value and gradient
equivalence. Complete means complete prediction artifacts AND all metric columns
covering every test condition, not merely a saved model checkpoint.

## Commands and artifacts

Initial data preparation (resumable; does not fetch PBMC/Tahoe):

```bash
python scripts/data/prepare_replogle_local.py --data-root data --download
```

Baseline environment:

```bash
conda create -n adacell-replogle -c conda-forge python=3.10 pip
conda activate adacell-replogle
python -m pip install -r environments/replogle-macos.txt -e '.[test]'
```

Set `CELLDIFFA_DATA_ROOT` to the absolute local `data` directory and
`CELLDIFFA_DEVICE=mps` for compatible diffusion runs. Existing CUDA launch
defaults are retained for the server. Mac loaders use zero subprocess workers
to avoid unpicklable upstream HDF5 caches and excessive worker memory.

AdaCell validation runs use `CELLDIFFA_EVALUATION_SPLIT=validation`, which selects
`reference/validation.h5ad` and the actual upstream validation loader. Test and
validation shards must live in separate directories. A finite `MAX_GROUPS`
is a smoke run and never produces evaluator-ready full results.

## Checks performed (including historical checkpoints)

- 2026-09-19: 53 tests passed in the pre-existing development test environment,
  including exact CPU agreement of the float32 schedule cast and an MPS test.
- 2026-09-19: actual `adacell-replogle` conda environment installed and checked.
  59 tests pass after adding reward invariance and Scouter coverage tests.
- Anchor reward corrected from adjacent-pair MMD (order dependent, dropping the
  last odd cell) to all-pairs biased squared RBF MMD. The new estimator is
  recorded in the run contract and cannot be mixed with older shards.
- Scouter uses the author's architecture, fixed embedding input, balanced data
  pairing, loss, optimizer, schedule and early-stopping rule. The loop saves
  cloned best checkpoints rather than retaining a mutable state dictionary.
  Held-out expression is never used as the prediction input.
- No full local result was available at the initial checkpoint; six test baselines
  are now fully evaluated, as listed above.
- HF HTTP/2 interrupted a large transfer. Downloader now uses HTTP/1.1 and
  external retries that retain the current partial-file resume offset.
- Published cached test group counts contain 26,878 cells in 13,219 nonempty
  native groups, with at most 22 cells per group. Confirm against the actual
  data loader before relying on these counts. Small native groups are relevant
  to interpretation of population rewards and to total runtime.

## Full-run entry points

Use the project root, set `PYTHONPATH="$PWD"`, and pass
`--split-config external/PerturbDiff/configs/data/perturb_data/replogle.yaml` to
each trainable adapter. Native sources must be checked out at their registry pins.

| Adapter | Conda environment | Full output directory / additional arguments |
|---|---|---|
| `scripts/baselines/run_scouter_replogle.py` | `adacell-replogle` | Completed run is `results/replogle/scouter_vectorized` |
| `scripts/baselines/run_cpa_replogle.py` | `adacell-cpa` | `--output-dir results/replogle/cpa_cpu --device cpu` |
| `scripts/baselines/run_gears_replogle_local.py` | `adacell-replogle` | `--output-dir results/replogle/gears_extended --device mps` |
| `scripts/baselines/run_state_replogle.py` | `adacell-state` | `--output-dir results/replogle/state --device mps` |
| `scripts/baselines/run_cellflow_replogle.py` | `adacell-cellflow` | `--embeddings data/PerturbDiff_data/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl` |
| `scripts/baselines/run_squidiff_replogle.py` | `adacell-replogle` | `--unseen-policy zero_shift` only with the limitation below disclosed |

Each full adapter writes `predictions.h5ad`. Run the shared evaluator from
`adacell-replogle`, **not STATE's newer bundled Cell-Eval**:

```bash
python scripts/baselines/evaluate.py \
  --real results/replogle/reference/real.h5ad \
  --pred results/replogle/cpa_cpu/predictions.h5ad \
  --outdir results/replogle/metrics/cpa \
  --pert-col gene --control-pert non-targeting --num-threads 8
```

Released inference uses `CELLDIFFA_RESUMABLE_SAMPLING=1`,
`CELLDIFFA_MICRO_BATCH_SIZE=1024`, and absolute paths in
`CELLDIFFA_CONTROL_REFERENCE` (`reference/controls.h5ad`) and
`CELLDIFFA_REAL_TEST` (`reference/real.h5ad`). Batching changes no native 32-cell
attention sets, model parameters, 100-step DDIM, eta=0 or CFG=1. The resumable
path matched upstream finetuned predictions bit-for-bit in an actual smoke run.

The first full AdaCell validation run uses:

```bash
export CELLDIFFA_EVALUATION_SPLIT=validation
export CELLDIFFA_REAL_TEST="$PWD/results/replogle/reference/validation.h5ad"
export CELLDIFFA_NUM_PARTICLES=16
export CELLDIFFA_NATIVE_BLOCKS_PER_POPULATION=16
export CELLDIFFA_PARTICLE_BATCH_CELLS=1024
export CELLDIFFA_ALPHA=1
bash scripts/baselines/run_celldiffa_replogle.sh \
  scratch results/replogle/validation/adacell_scratch_alpha1 0 0 1 all
```

`CELLDIFFA_ALIGNMENT_MODE=best_of_n` selects only at the terminal step, while
`random` disables selection with the same denoising budget. Use separate output
directories. Validation-only knobs also include `CELLDIFFA_REWARD_NORMALIZATION`,
`CELLDIFFA_ESS_THRESHOLD`, `CELLDIFFA_SIGNATURE_WEIGHT`,
`CELLDIFFA_DIRECTION_WEIGHT`, `CELLDIFFA_ANCHOR_WEIGHT`, and
`CELLDIFFA_PRIOR_RIDGE`.

Run `scripts/baselines/evaluate_population_diagnostics.py --real <reference>
--pred <prediction> --outdir <metrics directory>` after full outputs exist.
For AdaCell also supply `--base` with matching unsteered predictions. This adds
variance, projected effective rank, unique-cell fractions and fixed-projection
distribution distances. These are descriptive checks, not proofs of realism.

## Scientific qualifications and remaining work

- CPA retains native one-hot embeddings, including untrained unseen IDs.
  GEARS keeps the author's GO edges and adds missing IDs as isolated nodes with
  native self-loops. Report the latter as an extended-vocabulary adaptation,
  not an exact published run. No test condition is silently dropped.
- Squidiff's native latent-shift rule is undefined for CACTIN, CDC20, CMTR2,
  LAMB1, MRM1, NOS1AP, SLC16A5 and SPAG7. The runner errors by default. Explicit
  `zero_shift` retains those eight conditions with a neutral fallback. This must
  be disclosed and is not native unseen-gene generalization.
- Newly adapted baselines use published source, but PerturbDiff does not release
  their exact conversion/hyperparameter files. These are transparent matched-
  protocol reproductions, not promises to reproduce its table exactly.
- AdaCell evaluates and resamples a union of native blocks. Backbone attention
  remains within each original 32-cell block, with padding excluded from rewards
  and outputs. A particle is a population minibatch, not the whole dataset.
- With eta=0, duplicate resampled trajectories do not rejuvenate. Check ancestor
  counts, ESS and independent diversity statistics. Frozen parameters alone do
  not establish preserved diversity or a KL bound. Candidate-wise z-score reward
  normalization is adaptive, not a proof of a fixed reward-tilted density. The
  `none` normalization is exposed for validation comparisons.
- CPA's legacy Torch MPS runtime crashes on this Mac, hence CPU training.
  CellFlow uses CPU JAX; its native 84M-parameter model takes roughly 1.3–2 seconds
  per step. A 500,000-step cap is a multi-day workload before validation, unless
  validation early stopping fires. No short smoke run replaces that training.
- Some native progress files call a full-run *request* `complete_run`. Only final
  prediction artifacts and complete evaluation coverage establish completion.

Remaining: full training/evaluation of pending baselines, both full released
backbone evaluations, validation-only steering selection for both backbones,
locked full test runs, compute-matched controls, reward ablations, independent
diagnostics, multiple seeds and biological case studies. No SOTA or preserved-
diversity claim is established yet.
