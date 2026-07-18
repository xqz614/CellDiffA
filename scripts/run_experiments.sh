#!/bin/bash
# ============================================================
# CellDiffA: Full Experiment Pipeline
# ============================================================
# This script runs the complete experimental evaluation:
# 1. Data preprocessing
# 2. Baseline model evaluation (standard inference)
# 3. CellDiffA test-time alignment evaluation
#
# Usage:
#   bash scripts/run_experiments.sh [--dataset norman] [--device cuda:0]
# ============================================================

set -e

# --- Configuration ---
DATASET="${1:-norman}"
DEVICE="${2:-cuda:0}"
CONFIG="./configs/default.yaml"
CHECKPOINT_DIR="./checkpoints"
RESULTS_DIR="./results/${DATASET}"
N_SAMPLES=100

echo "============================================================"
echo "  CellDiffA Experiment Pipeline"
echo "  Dataset: ${DATASET}"
echo "  Device: ${DEVICE}"
echo "============================================================"

# --- Step 1: Data Preprocessing ---
echo ""
echo "[Step 1/4] Data Preprocessing..."
python scripts/preprocess_data.py \
    --dataset ${DATASET} \
    --n_top_genes 2000 \
    --split additive \
    --fold 0 \
    --compute_priors

# --- Step 2: Evaluate Baselines (Standard Inference) ---
echo ""
echo "[Step 2/4] Evaluating Baselines (Standard Inference)..."

BASELINES=("gears" "perturbdiff")

for MODEL in "${BASELINES[@]}"; do
    CKPT="${CHECKPOINT_DIR}/${MODEL}"
    if [ -d "${CKPT}" ]; then
        echo "  Evaluating: ${MODEL}"
        python scripts/evaluate_model.py \
            --model ${MODEL} \
            --checkpoint ${CKPT} \
            --config ${CONFIG} \
            --n_samples ${N_SAMPLES} \
            --output_dir ${RESULTS_DIR} \
            --device ${DEVICE}
    else
        echo "  [SKIP] ${MODEL}: checkpoint not found at ${CKPT}"
    fi
done

# --- Step 3: Evaluate CellDiffA (Test-Time Alignment) ---
echo ""
echo "[Step 3/4] Evaluating CellDiffA Test-Time Alignment..."

DIFFUSION_MODELS=("perturbdiff")
PARTICLE_COUNTS=(50 100 200)

for MODEL in "${DIFFUSION_MODELS[@]}"; do
    CKPT="${CHECKPOINT_DIR}/${MODEL}"
    if [ -d "${CKPT}" ]; then
        for N_PARTICLES in "${PARTICLE_COUNTS[@]}"; do
            echo "  CellDiffA + ${MODEL} (N=${N_PARTICLES})"
            python scripts/evaluate_model.py \
                --model ${MODEL} \
                --checkpoint ${CKPT} \
                --config ${CONFIG} \
                --celldiffa \
                --num_particles ${N_PARTICLES} \
                --output_dir ${RESULTS_DIR} \
                --device ${DEVICE}
        done
    else
        echo "  [SKIP] CellDiffA+${MODEL}: checkpoint not found at ${CKPT}"
    fi
done

# --- Step 4: Aggregate Results ---
echo ""
echo "[Step 4/4] Aggregating Results..."
python -c "
import json, os, glob
results_dir = '${RESULTS_DIR}'
files = glob.glob(os.path.join(results_dir, '*_results.json'))
print(f'\n{\"=\"*60}')
print(f'  Aggregated Results ({len(files)} methods)')
print(f'{\"=\"*60}')
print(f'{\"Method\":<30} {\"MSE\":>8} {\"Pearson\":>8} {\"DEG Recall\":>10} {\"E-Dist\":>8}')
print(f'{\"-\"*60}')
for f in sorted(files):
    with open(f) as fh:
        data = json.load(fh)
    m = data['aggregated_metrics']
    name = data['method']
    print(f'{name:<30} {m.get(\"mse_all\", 0):.4f}   {m.get(\"pearson_delta\", 0):.4f}   {m.get(\"deg_recall_top20\", 0):.4f}     {m.get(\"energy_distance\", 0):.4f}')
print(f'{\"=\"*60}')
"

echo ""
echo "Done! Results saved to: ${RESULTS_DIR}/"
