# CellDiffA: Test-Time Alignment for Single-Cell Perturbation Prediction

**CellDiffA** is a plug-and-play test-time adaptation framework that enhances any pre-trained diffusion-based single-cell perturbation prediction model through Sequential Monte Carlo (SMC) guided sampling with biologically-informed rewards.

## Key Features

- **Plug-and-Play**: No retraining required. Wrap any diffusion model and immediately improve OOD predictions.
- **Multi-Objective Biological Rewards**: Combines transcriptomic priors (DEG), geometric alignment, and anti-conservative penalties.
- **Population-Level Output**: Generates entire cell distributions, not just point estimates.
- **Theoretically Grounded**: Based on Feynman-Kac SMC framework with convergence guarantees.

## Architecture

```
CellDiffA/
├── celldiffa/              # Core framework
│   ├── smc/                # SMC test-time alignment engine
│   │   ├── engine.py       # Main SMC loop
│   │   └── resampler.py    # Particle resampling strategies
│   ├── rewards/            # Biological reward functions
│   │   ├── transcriptomic.py  # r_DEG: expression fidelity on DE genes
│   │   ├── geometric.py       # r_manifold: shift direction alignment
│   │   └── anchor.py          # r_anchor: anti-conservative penalty
│   └── evaluation/         # Unified evaluation metrics
├── baselines/              # Baseline model adapters
│   ├── adapter_gears.py    # GEARS (GNN, deterministic)
│   ├── adapter_cpa.py      # CPA (VAE, generative)
│   ├── adapter_perturbdiff.py  # PerturbDiff (diffusion, generative)
│   └── adapter_scdfm.py    # scDFM (flow matching, generative)
├── data/                   # Data management
│   └── data_manager.py     # Unified data loading and splitting
├── configs/                # Experiment configurations
├── scripts/                # Entry point scripts
└── requirements.txt
```

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n celldiffa python=3.10 -y
conda activate celldiffa

# Install core dependencies
pip install -r requirements.txt

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Data Preparation

```bash
# Download and preprocess Norman dataset
python scripts/preprocess_data.py --dataset norman --split additive --fold 0 --compute_priors
```

### 3. Run Evaluation

```bash
# Standard baseline evaluation
python scripts/evaluate_model.py --model perturbdiff --checkpoint ./checkpoints/perturbdiff

# CellDiffA test-time alignment (plug-and-play on top of PerturbDiff)
python scripts/evaluate_model.py --model perturbdiff --checkpoint ./checkpoints/perturbdiff \
    --celldiffa --num_particles 100

# Full experiment pipeline
bash scripts/run_experiments.sh norman cuda:0
```

## How CellDiffA Works

CellDiffA operates at **test time only**. Given a pre-trained diffusion model and a novel perturbation condition:

1. **Initialize**: Sample N particles (cells) as pure noise.
2. **Denoise**: Use the frozen base model to denoise one step (black-box call).
3. **Evaluate**: Compute biological rewards on each particle's Tweedie estimate.
4. **Reweight**: Update importance weights with annealed temperature.
5. **Resample**: If ESS drops below threshold, resample particles.
6. **Repeat**: Steps 2-5 for each diffusion timestep.
7. **Aggregate**: Output the aligned cell population.

The reward functions are derived entirely from the **training set** (no test-time ground truth):
- **r_DEG**: Ensures expression changes on expected differentially expressed genes.
- **r_manifold**: Ensures perturbation direction aligns with training-set-derived shift vectors.
- **r_anchor**: Penalizes proximity to control cells, combating conservative bias.

## Baseline Installation

Each baseline has its own dependencies. Install as needed:

```bash
# GEARS
pip install cell-gears

# CPA (requires older scvi-tools)
pip install cpa-tools

# PerturbDiff (install from source)
git clone https://github.com/DeepGraphLearning/PerturbDiff ./external/PerturbDiff
export PERTURBDIFF_PATH=./external/PerturbDiff

# scDFM (install from source)
git clone https://github.com/AI4Science-WestlakeU/scDFM ./external/scDFM
export SCDFM_PATH=./external/scDFM
```

## Citation

```bibtex
@inproceedings{celldiffa2026,
  title={CellDiffA: Test-Time Alignment for Single-Cell Perturbation Prediction via SMC-Guided Diffusion},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

MIT License
