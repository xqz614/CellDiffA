# Replogle on AutoDL, staged migration

This deployment uses the same scientific configuration as the local experiments:
Python 3.10, PyTorch 2.5.1 (CUDA 12.4 instead of MPS), Cell-Eval 0.6.6, official
Replogle splits and ordered 2,000 evaluation genes. It includes the corrected
checkpoint category IDs for Finetuned. It does not change the image's base
Python 3.12 environment, stop local jobs or launch formal tests automatically.

## 1. Checkout and environment

Run in an AutoDL web terminal; SSH is not required. Keep environments, package
caches, data and outputs on the paid data disk, not the 30 GB system disk.

```bash
mkdir -p /root/autodl-tmp/adacell
cd /root/autodl-tmp/adacell
git clone --branch agent/replogle-local-mps https://github.com/xqz614/CellDiffA.git
cd CellDiffA
bash scripts/server/setup_replogle_autodl.sh
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
```

The installer pins the CUDA wheels, checks dependencies and performs a small
matrix multiplication on each visible GPU. This is not a model speed benchmark.
Its timestamped report is under `results/replogle/server/`. If installation fails,
stop and share the error; rerunning the installer can reuse its dedicated
environment and caches. It never deletes or recreates another Conda environment.

For ordinary package downloads, pip uses the image's existing index configuration.
If necessary, set `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` before
the installer. PyTorch CUDA wheels still use the official PyTorch CUDA 12.4 index.
Do not disable TLS verification or install an unverified torch build to get past
a network failure.

## 2. Data and reference preparation

Use a tmux session if available, so closing the web terminal does not end work.
In each new session, activate the dedicated environment again.

```bash
cd /root/autodl-tmp/adacell/CellDiffA
conda activate /root/autodl-tmp/adacell/envs/adacell-replogle
export HF_ENDPOINT=https://hf-mirror.com
set -o pipefail
mkdir -p results/replogle/logs
python scripts/server/replogle_autodl.py prepare 2>&1 | tee -a results/replogle/logs/server_prepare.log
```

The script checks out upstream PerturbDiff at
`f4e27c155be5325418c4cb3182453d4022754e91`, downloads only Replogle and its shared
assets plus both released checkpoints, verifies downloaded sizes and SHA256,
and exports official validation/test references and observed controls. Dataset
revision `10654f2` and checkpoint revision `c33e578` match the local artifacts.
There must be at least 60 GiB free before first-time preparation.

Preparation never fits priors to validation/test outcomes or evaluates test
scores. Existing conflicting references or upstream revisions cause an error,
not an overwrite. Completed preparation is **not** completed inference.

## 3. Two independent small GPU checks

Only after preparation succeeds, run these in two separate tmux windows. They
use GPU 0 and GPU 1 respectively; GPU 2 remains free. Each window needs its own
environment activation. Do not set `CUDA_VISIBLE_DEVICES` outside these scripts.

```bash
# Window 1, after cd and conda activate as above
set -o pipefail
python scripts/server/replogle_autodl.py smoke --variant scratch --gpu 0 2>&1 | tee -a results/replogle/logs/server_smoke_scratch.log
```

```bash
# Window 2, after cd and conda activate as above
set -o pipefail
python scripts/server/replogle_autodl.py smoke --variant finetuned --gpu 1 2>&1 | tee -a results/replogle/logs/server_smoke_finetuned.log
```

These retain 16 candidates, up to 16 native 32-cell blocks, particle batch 1024,
100 DDIM steps, eta 0, guidance 1, alpha 1 and seed 42. Coverage alone is limited
to one validation group, in a separate timestamped smoke directory. A message
`Partial run complete. No evaluator-ready H5AD was written.` is therefore expected.
It must never be reported as a complete baseline or main result. Finetuned
still generates in its native 12,626-gene space, with 2,000 evaluation genes.

## 4. Before formal experiments

Review both smoke logs, GPU memory peaks and group timings first. Complete
backend checks and the validation-only selection protocol before any formal
test launch. One group only detects basic runtime/memory failures; it does not
establish numerical equivalence, complete coverage or a reliable full-run ETA.

Never combine MPS and CUDA shards in one output, retune on test scores, replace
missing perturbations with zeros, or compare wall times across different devices
as if compute were matched. Existing local validation jobs continue independently.
The scripts here do not yet migrate selection evidence, schedule the complete
three-GPU experimental queue, or change the 40-minute local monitor.
