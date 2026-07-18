# CellDiffA

CellDiffA is a research implementation of **test-time, population-level
alignment for single-cell perturbation generators**. A frozen conditional
diffusion model proposes perturbation-response batches; Sequential Monte Carlo
(SMC) reweights and resamples those batches using biological rewards built only
from the training split.

> Research status: the population-level SMC core is tested and the PerturbDiff
> adapter is aligned with the upstream sampling implementation. This repository
> does not include a trained checkpoint or claim benchmark results yet.

## Method contract

One particle is an empirical cell distribution, not one cell:

```text
particles          B_t: [N particles, M cells, G genes]
base-model input       : [N*M cells, G genes]
reward output          : [N particles]
```

At reverse step `k`, CellDiffA evaluates a clean-batch estimate and applies the
incremental potential

```text
log G_k = (beta_k * r_k - beta_{k-1} * r_{k-1}) / alpha.
```

The potential telescopes to the terminal reward. Resampling is disabled at the
last step so MAP and weighted-consensus outputs preserve their terminal weights.

The default reward is a normalized linear combination of:

- `r_DEG`: population-mean agreement on training-derived response genes;
- `r_manifold`: cosine alignment of the population-mean shift;
- `r_anchor`: negative linear-time RBF MMD to a training-derived reference
  distribution (control cells transported by available training shifts).

Held-out perturbation cells are used only for final evaluation.

## Supported integrations

- **PerturbDiff**: generative backbone and step-wise DDIM sampler.
- **GEARS**: deterministic point-estimate baseline.

Earlier experimental adapters for CPA, scDFM, Squidiff, and CellFlow were
removed because they did not implement the corresponding upstream APIs
faithfully. Add a new adapter only with a pinned upstream revision and an
integration test using a real checkpoint.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

PerturbDiff and GEARS are optional upstream projects:

```bash
git clone https://github.com/DeepGraphLearning/PerturbDiff external/PerturbDiff
export PERTURBDIFF_PATH="$PWD/external/PerturbDiff"

pip install cell-gears
```

## Tests

```bash
pytest -q
ruff check .
```

The fast suite checks distribution-valued particle shapes, telescoping
Feynman-Kac potentials, condition batching, terminal weights, reward semantics,
resampling, and deterministic metrics without downloading biological data.

## Data

Download and preprocess Norman or Replogle K562 data:

```bash
python scripts/preprocess_data.py \
  --dataset norman \
  --split additive \
  --fold 0 \
  --compute_priors
```

The GEARS-distributed H5AD files are already log-normalized. The default config
therefore sets `data.already_normalized: true`; set it to `false` only for raw
count matrices.

Splits are deterministic across folds:

- `additive`: hold out combinations while their single components remain in
  training; combination test sets are disjoint across folds;
- `unseen`: hold out disjoint gene sets and every condition containing those
  genes. A cross-gene combination may consequently be a test condition in more
  than one fold.

## Evaluation

The checkpoint and the configured data must use the same ordered gene space.

```bash
# Native PerturbDiff
python scripts/evaluate_model.py \
  --model perturbdiff \
  --checkpoint checkpoints/perturbdiff.ckpt \
  --device cuda:0

# CellDiffA + frozen PerturbDiff
python scripts/evaluate_model.py \
  --model perturbdiff \
  --checkpoint checkpoints/perturbdiff.ckpt \
  --celldiffa \
  --num_particles 100 \
  --device cuda:0
```

Run ablations by changing `cells_per_particle`, reward components,
normalization, particle count, tempering schedule, and `alpha` in
`configs/default.yaml`. Compare methods with identical folds and sampling seeds.

## Important limitations

- A PerturbDiff checkpoint trained for Norman is not supplied here; an upstream
  checkpoint trained on another gene set is not interchangeable.
- SMC changes inference, not the frozen model's learned support. Rewards cannot
  recover a perturbation whose training-derived prior is unavailable.
- Biological reward weights and bandwidths require validation on training or
  validation conditions; never tune them on held-out test outcomes.
- End-to-end benchmark claims require real-checkpoint integration tests and
  multi-seed experiments in addition to the included model-free suite.

## License

MIT. See `LICENSE`.
