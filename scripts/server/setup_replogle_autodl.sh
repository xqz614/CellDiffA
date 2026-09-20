#!/usr/bin/env bash
# Environment installation only. Never starts a scientific experiment.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
storage_root="${ADACELL_SERVER_ROOT:-/root/autodl-tmp/adacell}"
env_prefix="$storage_root/envs/adacell-replogle"

if [[ "$(uname -s)" != Linux ]]; then
  echo 'This installer requires Linux; it does not modify a Mac environment.' >&2
  exit 1
fi
for program in conda git curl nvidia-smi; do
  command -v "$program" >/dev/null || { echo "Missing program: $program" >&2; exit 1; }
done
mkdir -p "$storage_root"
storage_root="$(cd "$storage_root" && pwd)"
case "$repo_root/" in
  "$storage_root/"*) ;;
  *) echo "Clone the repository under $storage_root before installing." >&2; exit 1 ;;
esac
export CONDA_PKGS_DIRS="$storage_root/cache/conda"
export PIP_CACHE_DIR="$storage_root/cache/pip"
export TMPDIR="$storage_root/tmp"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR" "$storage_root/envs"

if [[ ! -d "$env_prefix/conda-meta" ]]; then
  if [[ -e "$env_prefix" ]]; then
    echo "Refusing to overwrite a non-Conda path: $env_prefix" >&2
    exit 1
  fi
  conda create -y --prefix "$env_prefix" python=3.10 pip
fi
conda run --no-capture-output -p "$env_prefix" python -c \
  'import sys; assert sys.version_info[:2] == (3, 10), "Expected Python 3.10"'
# CUDA wheels come from the official PyTorch index. The remaining dependencies
# use pip's existing index configuration (or a user-supplied PIP_INDEX_URL).
conda run --no-capture-output -p "$env_prefix" python -m pip install \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
conda run --no-capture-output -p "$env_prefix" python -m pip install \
  -r "$repo_root/environments/replogle-cuda124.txt"
conda run --no-capture-output -p "$env_prefix" python -m pip install --no-deps -e "$repo_root"
conda run --no-capture-output -p "$env_prefix" python -m pip check
conda run --no-capture-output -p "$env_prefix" python \
  "$script_dir/replogle_autodl.py" check
echo "Environment ready. Activate with: conda activate $env_prefix"
echo 'No dataset or checkpoint was downloaded and no model job was launched.'
