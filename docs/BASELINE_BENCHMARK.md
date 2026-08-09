# Baseline benchmark

This benchmark has one non-negotiable rule: every method is evaluated against
the same real test H5AD and the same copied control cells. It reproduces the
PerturbDiff evaluation surface with `cell-eval==0.6.6` plus CellFlow R².

## Scope

The primary (`paper`) suite is the suite named by PerturbDiff:

- Mean and its cell-type, batch, and overall variants;
- Linear;
- CPA;
- STATE;
- CellFlow;
- Squidiff;
- PerturbDiff from scratch and PerturbDiff finetuned.

The PDF adds GEARS, CellOT, scDFM, scLAMBDA, Scouter, and VCWorld. These are
registered as `extended` methods only where the upstream implementation can
actually express the task. In particular:

- GEARS supports Norman and the Replogle 2022 K562 essential dataset through
  official upstream loaders. For Replogle, the benchmark adapter must preserve
  the PerturbDiff gene space and holdout split; GEARS' independently generated
  `simulation` split is not a comparable result. GEARS still does not support
  training across multiple cell types;
- scDFM's released data/configuration covers Norman and ComboSciPlex, not the
  three PerturbDiff datasets;
- scVI is a representation/generative model, not a conditioned perturbation
  response predictor. It is therefore not reported as a response baseline.

The machine-readable matrices and source revisions are in
`configs/benchmark/datasets.yaml` and `configs/benchmark/baselines.yaml`.

## Server setup

Use the repository checkout on the server and keep all data outside Git:

```bash
cd /data/users/jchengak/DiffA/CellDiffA
export CELLDIFFA_DATA_ROOT=/data/users/jchengak/DiffA/CellDiffA/data

conda env create -f environments/benchmark.yaml
conda activate celldiffa-benchmark
python -m pip install -e .
```

The NVIDIA 535 driver can run the CUDA 12.1 PyTorch build. The separate
PerturbDiff environment is:

```bash
conda env create -f environments/perturbdiff.yaml
conda activate celldiffa-perturbdiff
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

GEARS also has a separate environment. Install the pinned official checkout
without allowing it to replace the environment's CUDA-enabled PyTorch:

```bash
conda env create -f environments/gears.yaml
conda activate celldiffa-gears
python -m pip install -e . --no-deps
python -m pip install -e external/GEARS --no-deps
python -c "import torch, gears; print(torch.__version__, torch.cuda.is_available())"
```

Do not try to install every baseline into one environment. CPA, STATE,
CellFlow, Squidiff, and PerturbDiff have incompatible dependency histories.

## Reproducibility status

The registry's `runner` field is intentionally strict:

- `end_to_end`: implemented and tested in this repository;
- `released_checkpoint_end_to_end`: calls the pinned official implementation
  and released weight through a repository script;
- `requires_dataset_adapter`: the official model source is pinned, but its
  native input schema has not yet been converted to the PerturbDiff split;
- `not_applicable`: not a perturbation-response predictor for this protocol.

At this revision, Mean, Linear on Replogle, and both released PerturbDiff
variants are end-to-end. The extended GEARS runner is also end-to-end for the
PerturbDiff-aligned Replogle protocol described below; Norman remains
adapter-only. CPA, STATE, CellFlow, and Squidiff are **not yet end-to-end**:
the PerturbDiff authors state that they used the official implementations, but
do not publish the model-specific conversion/config files in their repository.
Running an upstream demo or letting the model generate a new random split is
not accepted as reproduction of the paper baseline.

## Data

Download only what is needed. Replogle is a sensible first end-to-end test:

```bash
conda activate celldiffa-benchmark
python scripts/data/download_perturbdiff.py --dataset replogle
python scripts/data/download_perturbdiff.py --dataset assets
python scripts/data/download_perturbdiff_checkpoints.py \
  --checkpoint replogle_scratch
python scripts/data/download_perturbdiff_checkpoints.py \
  --checkpoint replogle_finetuned
```

PBMC and Tahoe require explicit acknowledgement because the PerturbDiff README
reports approximately 750 GB and 3 TB after decompression:

```bash
python scripts/data/download_perturbdiff.py \
  --dataset pbmc --confirm-large-download
python scripts/data/download_perturbdiff.py \
  --dataset tahoe100m --confirm-large-download
```

Norman uses the GEARS-distributed preprocessed H5AD:

```bash
python scripts/preprocess_data.py \
  --dataset norman \
  --data_root "$CELLDIFFA_DATA_ROOT"
```

Check a model/dataset combination before submitting a job:

```bash
python scripts/baselines/doctor.py --dataset replogle --baseline state
python scripts/baselines/doctor.py --dataset replogle --baseline gears
python scripts/baselines/doctor.py --dataset norman --baseline gears
```

## Official sources

Clone the pinned official revisions recorded by this repository:

```bash
bash scripts/baselines/clone_official_sources.sh external
```

For each dataset, use the holdout perturbations, batches, and cell types from
the pinned PerturbDiff config under
`external/PerturbDiff/configs/data/perturb_data/`. Do not create an independent
random split inside CPA, STATE, CellFlow, or Squidiff.

PerturbDiff sampling writes a matched pair:

```text
diffusion_predict_<timestamp>.h5ad
diffusion_true_<timestamp>.h5ad
```

Use `diffusion_true_*.h5ad` as the immutable real test file for every method.
Archive it with its SHA-256 checksum. Never use a separate real-test export for
another baseline.

Run each released PerturbDiff variant with its official sampler. This command
uses one visible GPU, samples the complete test loader, and writes both files
to the requested directory:

```bash
conda activate celldiffa-perturbdiff
bash scripts/baselines/run_perturbdiff_released.sh \
  replogle scratch results/replogle/perturbdiff_scratch 0
bash scripts/baselines/run_perturbdiff_released.sh \
  replogle finetuned results/replogle/perturbdiff_finetuned 1
```

Copy one of the resulting `diffusion_true_*.h5ad` files to
`results/replogle/reference/real.h5ad`, then verify that both runs produced the
same ordered real test matrix before evaluating them together.

## Linear on Replogle

The Replogle runner translates the equations in the pinned official
`run_linear_pretrained_model.R`: condition pseudobulk, a 10-dimensional PCA for
gene embeddings, external perturbation embeddings, and two-sided ridge
regression with penalty 0.1. It uses deterministic truncated SVD for the same
rank-10 PCA objective as `prcomp_irlba`. No R or GPU is needed.

Fitting uses the full `X`/`var_names` gene space from the released Replogle
H5AD for gene-side PCA. Perturbation-side vectors use PerturbDiff's released
`replogle_gene_emb_dict_perturbation_emb_dict.pkl` GenePT dictionary because
some CRISPR targets are absent even from the full expression matrix. This is
the external `pert_embedding` branch supported by the official Linear solver.
Only the ordered 2,000 genes in `X_hvg`/`replogle_real_selected_genes.pkl` are
written to the prediction file. Test GenePT coverage is strict; missing test
embeddings are never silently replaced by zero vectors.

The official ridge equations are unconstrained and can extrapolate below zero.
Because Cell-Eval requires valid non-negative log1p expression, the runner
projects only the final expression predictions with `maximum(value, 0)` before
writing the H5AD. The fitted coefficients and saved model remain unchanged.

`pooled` is the primary PerturbDiff-aligned result: every row in the official
training mask is used and context is ignored, matching the context-agnostic
official Linear model. `heldout_only` is an optional sensitivity analysis that
fits only the held-out HepG2 training subset. Both modes remove validation and
test rows before any pseudobulk or PCA calculation.

Run the primary baseline in the benchmark Conda environment:

```bash
conda activate celldiffa-benchmark
export CELLDIFFA_DATA_ROOT=/data/users/jchengak/DiffA/CellDiffA/data

bash scripts/baselines/run_linear_replogle.sh pooled
```

This writes the evaluator-ready prediction to
`results/replogle/predictions/linear.h5ad`, the fitted matrices to
`results/replogle/models/linear.npz`, and a fairness manifest beside the H5AD.
The model predicts one pseudobulk expression vector per perturbation and repeats
it to the exact real-test cell count, as required for deterministic Linear;
real control rows are copied unchanged.

Evaluate it with the exact PerturbDiff metric suite:

```bash
python scripts/baselines/evaluate.py \
  --real results/replogle/reference/real.h5ad \
  --pred results/replogle/predictions/linear.h5ad \
  --outdir results/replogle/metrics/linear \
  --pert-col gene \
  --control-pert non-targeting \
  --num-threads 32
```

The optional sensitivity run is:

```bash
bash scripts/baselines/run_linear_replogle.sh heldout_only
```

## GEARS on Replogle

GEARS officially provides Replogle K562/RPE1 loaders, but the PerturbDiff task
uses the four-context Replogle-Nadig data and holds out HepG2. GEARS is not
context-aware, so it is an extended baseline rather than one of PerturbDiff's
paper baselines. The runner implements two explicitly labelled protocols:

- `pooled` (primary extended result): use every row in PerturbDiff's official
  training mask, merge the contexts for GEARS, and predict one context-agnostic
  response per perturbation;
- `heldout_only` (ablation): use only HepG2 rows in the official training mask.

In both modes, validation and test rows are removed before GEARS preprocessing,
GEARS' random `simulation` split is disabled, predictions use the immutable
`real.h5ad`, and the real control rows are copied into the prediction file.
The runner fails if a test perturbation is absent from the GEARS GO graph.

After `results/replogle/reference/real.h5ad` exists, run the primary result on
physical GPU 2:

```bash
conda activate celldiffa-gears
export CELLDIFFA_DATA_ROOT=/data/users/jchengak/DiffA/CellDiffA/data

bash scripts/baselines/run_gears_replogle.sh \
  pooled \
  results/replogle/gears_pooled \
  2
```

Optionally run the HepG2-only ablation on another free GPU:

```bash
bash scripts/baselines/run_gears_replogle.sh \
  heldout_only \
  results/replogle/gears_heldout_only \
  3
```

Each output has a sibling `*.manifest.json` recording the split policy, row
counts, hyperparameters, paths, and SHA-256 checksums. Do not report a run if
the manifest says `input_hashes_skipped: true`.

## Prediction contract

Every standardized prediction must have:

- exactly the ordered genes in the real test file;
- exactly the same perturbation labels;
- the same number of predicted and real cells per perturbation;
- the exact real control matrix copied into the prediction file.

If an official method writes H5AD, standardize it with:

```bash
python scripts/baselines/standardize_prediction.py \
  --real-test results/replogle/reference/real.h5ad \
  --pred-h5ad results/replogle/state/official_output.h5ad \
  --output results/replogle/predictions/state.h5ad \
  --pert-col gene \
  --control-pert non-targeting
```

For an implementation that emits one matrix per perturbation, put
`<perturbation>.npy` files in one directory and replace `--pred-h5ad` with
`--prediction-dir`.

## Mean baselines

The main Mean gives every training perturbation equal weight, matching the
Cell-Eval 0.6.6 baseline. The other variants are cell-weighted means within
cell type, batch, or the whole training set:

```bash
for variant in perturbation cell_type batch overall; do
  python scripts/baselines/run_mean.py \
    --train results/replogle/reference/train.h5ad \
    --real-test results/replogle/reference/real.h5ad \
    --output "results/replogle/predictions/mean_${variant}.h5ad" \
    --variant "$variant" \
    --pert-col gene \
    --control-pert non-targeting \
    --context-col cell_line \
    --batch-col gem_group
done
```

`train.h5ad` must contain only the official training split plus controls. The
command never reads perturbed expression from the real test file.

For PBMC, Tahoe, and the full Replogle source, avoid materializing another
training H5AD. The streaming implementation reads dense or CSR-backed H5AD in
chunks and computes all four variants in one pass:

```bash
python scripts/baselines/run_streaming_mean.py \
  --source "$CELLDIFFA_DATA_ROOT/PerturbDiff_data/finetune_data/nadig_processed_data/replogle.h5ad" \
  --real-test results/replogle/reference/real.h5ad \
  --upstream-split-config external/PerturbDiff/configs/data/perturb_data/replogle.yaml \
  --output-dir results/replogle/predictions \
  --selected-genes "$CELLDIFFA_DATA_ROOT/PerturbDiff_data/selected_genes/replogle_real_selected_genes.pkl" \
  --split-axis context
```

Use `--split-axis batch` for PBMC and `--split-axis context` for Tahoe100M.
For Tahoe, `--source` is the directory containing its plate-level H5AD files.

## Exact evaluation

Evaluate one method:

```bash
python scripts/baselines/evaluate.py \
  --real results/replogle/reference/real.h5ad \
  --pred results/replogle/predictions/state.h5ad \
  --outdir results/replogle/metrics/state \
  --pert-col gene \
  --control-pert non-targeting \
  --num-threads 32
```

Or evaluate a completed set and create the comparison table:

```bash
python scripts/baselines/evaluate_suite.py \
  --real results/replogle/reference/real.h5ad \
  --prediction-root results/replogle/predictions \
  --output-root results/replogle/metrics \
  --methods mean_perturbation linear cpa state cellflow squidiff \
    perturbdiff_scratch perturbdiff_finetuned \
  --pert-col gene \
  --control-pert non-targeting \
  --num-threads 32
```

The final columns are exactly:

```text
R2, DEOver, DEPrec, ES, DirAgr, LFCSpear, AUPRC, AUROC,
PDCorr, MSE, MAE, PDS_L1, PDS_L2, PDS_cos
```

Cell-Eval's raw per-perturbation and aggregate CSV files are retained beside
the PerturbDiff-labelled tables. Evaluation fails rather than returning a
partial table if any required metric cannot be computed.

## Recommended execution order

1. Replogle: Mean variants, PerturbDiff released checkpoint, STATE, GEARS with
   the PerturbDiff-aligned split, then the remaining formal baselines.
2. Norman: Mean, Linear, CPA, GEARS, scDFM, scLAMBDA, Scouter, and methods whose
   official input adapter has been validated on Norman.
3. PBMC only after checking the roughly 750 GB storage requirement.
4. Tahoe last; the released PerturbDiff representation is roughly 3 TB after
   decompression and should not be duplicated per method.

For each run, record the dataset checksum, split definition, source commit,
environment export, random seed, GPU, checkpoint, and prediction checksum.
