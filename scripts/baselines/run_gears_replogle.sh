#!/usr/bin/env bash
set -euo pipefail

mode="${1:-pooled}"
output_dir="${2:-results/replogle/gears_${mode}}"
physical_gpu="${3:-2}"
repo_root="${4:-$(pwd)}"

if [[ "$mode" != "pooled" && "$mode" != "heldout_only" ]]; then
  echo "mode must be pooled or heldout_only" >&2
  exit 2
fi

data_root="${CELLDIFFA_DATA_ROOT:-$repo_root/data}"
source_h5ad="$data_root/PerturbDiff_data/finetune_data/nadig_processed_data/replogle.h5ad"
selected_genes="$data_root/PerturbDiff_data/selected_genes/replogle_real_selected_genes.pkl"
real_test="$repo_root/results/replogle/reference/real.h5ad"
split_config="$repo_root/external/PerturbDiff/configs/data/perturb_data/replogle.yaml"
gears_source="$repo_root/external/GEARS"

for path in "$source_h5ad" "$selected_genes" "$real_test" "$split_config"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done
if [[ ! -d "$gears_source/gears" ]]; then
  echo "Missing pinned GEARS source: $gears_source" >&2
  echo "Run: bash scripts/baselines/clone_official_sources.sh external" >&2
  exit 1
fi

mkdir -p "$output_dir"
export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTHONPATH="$repo_root:$gears_source:${PYTHONPATH:-}"

python "$repo_root/scripts/baselines/run_gears_replogle.py" \
  --source "$source_h5ad" \
  --real-test "$real_test" \
  --upstream-split-config "$split_config" \
  --selected-genes "$selected_genes" \
  --output "$output_dir/gears_${mode}.h5ad" \
  --work-dir "$data_root/gears_cache/replogle/$mode" \
  --model-dir "$output_dir/model" \
  --asset-dir "$data_root/raw" \
  --mode "$mode" \
  --device cuda:0
