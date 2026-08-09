#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pooled}"
prediction_dir="${2:-results/replogle/predictions}"
repo_root="${3:-$(pwd)}"

if [[ "$mode" != "pooled" && "$mode" != "heldout_only" ]]; then
  echo "mode must be pooled or heldout_only" >&2
  exit 2
fi

data_root="${CELLDIFFA_DATA_ROOT:-$repo_root/data}"
source_h5ad="$data_root/PerturbDiff_data/finetune_data/nadig_processed_data/replogle.h5ad"
selected_genes="$data_root/PerturbDiff_data/selected_genes/replogle_real_selected_genes.pkl"
real_test="$repo_root/results/replogle/reference/real.h5ad"
split_config="$repo_root/external/PerturbDiff/configs/data/perturb_data/replogle.yaml"

for path in "$source_h5ad" "$selected_genes" "$real_test" "$split_config"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

if [[ "$mode" == "pooled" ]]; then
  method_name="linear"
else
  method_name="linear_heldout_only"
fi
model_dir="$repo_root/results/replogle/models"
mkdir -p "$prediction_dir" "$model_dir"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"

python "$repo_root/scripts/baselines/run_linear_replogle.py" \
  --source "$source_h5ad" \
  --real-test "$real_test" \
  --upstream-split-config "$split_config" \
  --selected-genes "$selected_genes" \
  --output "$prediction_dir/$method_name.h5ad" \
  --model-output "$model_dir/$method_name.npz" \
  --mode "$mode"
