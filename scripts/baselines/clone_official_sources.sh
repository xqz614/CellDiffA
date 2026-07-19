#!/usr/bin/env bash
set -euo pipefail

source_root="${1:-external}"
mkdir -p "$source_root"

clone_one() {
  local url="$1"
  local revision="$2"
  local directory="$3"
  if [[ ! -d "$directory/.git" ]]; then
    git clone "$url" "$directory"
  fi
  git -C "$directory" fetch --tags origin
  git -C "$directory" checkout "$revision"
}

# The repositories are upstream implementations; CellDiffA does not vendor or
# modify them. Use a separate conda environment whenever upstream requires one.
clone_one https://github.com/DeepGraphLearning/PerturbDiff \
  f4e27c155be5325418c4cb3182453d4022754e91 "$source_root/PerturbDiff"
clone_one https://github.com/ArcInstitute/state \
  da4178c930dc917dac6b56faf10a33e21bd8e905 "$source_root/state"
clone_one https://github.com/theislab/cpa \
  fbd7c0250edc23eff003a10c99655579c53afd63 "$source_root/cpa"
clone_one https://github.com/theislab/CellFlow \
  446ed6073c60ac2e8db13c4ea096a43cdec288b2 "$source_root/CellFlow"
clone_one https://github.com/siyuh/Squidiff \
  abdfc27d84947dcccd745d1067c0840a41d32eb8 "$source_root/Squidiff"
clone_one https://github.com/const-ae/linear_perturbation_prediction-Paper \
  bfa6eeea2bd145a1af2ec0127a2e808cc38456a9 \
  "$source_root/linear_perturbation_prediction-Paper"

# Extended methods named in DiffA.pdf. They are cloned for the dataset-method
# combinations declared in configs/benchmark/baselines.yaml.
clone_one https://github.com/snap-stanford/GEARS \
  f374e43e197b295016d80395d7a54ddb81cc6769 "$source_root/GEARS"
clone_one https://github.com/bunnech/cellot \
  522d2b953da8ad244fcf36f64521487fdf763788 "$source_root/cellot"
clone_one https://github.com/AI4Science-WestlakeU/scDFM \
  2cf6bca1f044e74c4e1dc586892c0495880cf125 "$source_root/scDFM"
clone_one https://github.com/gefeiwang/scLAMBDA \
  a5be849797eb26e098c7002b819a5261f51dbead "$source_root/scLAMBDA"
clone_one https://github.com/PancakeZoy/scouter \
  bf763aaf87f162fdf28bec3145d254daaddfef84 "$source_root/scouter"
clone_one https://github.com/GENTEL-lab/VCWorld \
  0e3c44c25e897d3ebf97b582d63c7fcbf1e80f57 "$source_root/VCWorld"

echo "Official PerturbDiff baseline sources are available under $source_root"
