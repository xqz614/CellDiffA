#!/usr/bin/env python
"""Separate CUDA checks, reproducible data preparation and one-group smoke runs.

This is not a formal-test launcher or a parameter-selection bypass. No local
MPS prediction shards are copied into CUDA output directories.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Direct `python scripts/server/...` execution must also find the repository's
# orchestration helpers, which are intentionally not installed as a package.
sys.path.insert(0, str(REPO))
UPSTREAM_URL = "https://github.com/DeepGraphLearning/PerturbDiff.git"
UPSTREAM_REVISION = "f4e27c155be5325418c4cb3182453d4022754e91"


def run(command, *, repo=REPO, env=None):
    print("RUN " + shlex.join([str(x) for x in command]), flush=True)
    return subprocess.run(command, cwd=repo, env=env, check=True)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def check_cuda(*, repo=REPO):
    import torch

    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux":
        raise RuntimeError("Activate the dedicated Linux Python 3.10 environment first.")
    if torch.__version__ != "2.5.1+cu124" or torch.version.cuda != "12.4":
        raise RuntimeError("Expected PyTorch 2.5.1+cu124; do not change the comparison runtime.")
    if importlib.metadata.version("cell-eval") != "0.6.6":
        raise RuntimeError("The published evaluation requires cell-eval==0.6.6.")
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("CUDA is unavailable in the active environment.")
    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        props = torch.cuda.get_device_properties(device)
        # Small functional check, not a speed estimate or a model experiment.
        values = torch.ones((32, 32), device=device)
        if not torch.equal(values @ values, torch.full_like(values, 32)):
            raise RuntimeError(f"GPU {index} failed the matrix multiplication check.")
        torch.cuda.synchronize(device)
        devices.append(dict(index=index, name=props.name, memory_gib=props.total_memory / 1024**3))
        del values
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cell_eval": importlib.metadata.version("cell-eval"),
        "devices": devices,
        "model_benchmark_completed": False,
    }
    output = repo / "results/replogle/server" / f"preflight_{stamp()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return report


def ensure_upstream(*, repo=REPO):
    upstream = repo / "external/PerturbDiff"
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", UPSTREAM_URL, upstream],
            repo=repo,
        )
        run(["git", "-C", upstream, "checkout", "--detach", UPSTREAM_REVISION], repo=repo)
    actual = subprocess.check_output(
        ["git", "-C", upstream, "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != UPSTREAM_REVISION:
        raise RuntimeError("Existing upstream revision differs; it was not overwritten.")
    for arguments in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        run(["git", "-C", upstream, *arguments], repo=repo)
    return upstream


def check_references(reference, *, source, split_config, genes):
    """Permit a completed previous export only when its provenance still agrees."""
    from celldiffa.benchmark.artifacts import sha256_file

    paths = [
        reference / name
        for name in ("validation.h5ad", "real.h5ad", "controls.h5ad", "manifest.json")
    ]
    if not any(path.exists() for path in paths):
        return False
    if not all(path.is_file() for path in paths):
        raise RuntimeError("Incomplete reference export; preserved files, refusing to overwrite.")
    manifest = json.loads((reference / "manifest.json").read_text())
    expected = {
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "split_config_sha256": sha256_file(split_config),
        "selected_genes_sha256": sha256_file(genes),
        "evaluation_genes": 2000,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Existing reference provenance differs; no files were overwritten.")
    return True


def prepare(*, repo=REPO):
    upstream = ensure_upstream(repo=repo)
    run(
        [
            sys.executable,
            "scripts/data/prepare_replogle_local.py",
            "--data-root",
            repo / "data",
            "--download",
            "--workers",
            "2",
        ],
        repo=repo,
    )
    data = repo / "data/PerturbDiff_data"
    source = data / "finetune_data/nadig_processed_data/replogle.h5ad"
    genes = data / "selected_genes/replogle_real_selected_genes.pkl"
    split_config = upstream / "configs/data/perturb_data/replogle.yaml"
    reference = repo / "results/replogle/reference"
    if not check_references(reference, source=source, split_config=split_config, genes=genes):
        run(
            [
                sys.executable,
                "scripts/data/prepare_replogle_reference.py",
                "--source",
                source,
                "--split-config",
                split_config,
                "--selected-genes",
                genes,
                "--output-dir",
                reference,
            ],
            repo=repo,
        )
    # Record file identities once. This does not inspect test response scores.
    from celldiffa.benchmark.artifacts import sha256_file

    report = {
        "upstream_revision": UPSTREAM_REVISION,
        "source_sha256": sha256_file(source),
        "reference_sha256": {
            name: sha256_file(reference / name)
            for name in ("validation.h5ad", "real.h5ad", "controls.h5ad")
        },
    }
    output = repo / "results/replogle/server" / f"prepared_{stamp()}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(f"PREPARED {output}. No model experiment has been launched.", flush=True)


def smoke_command(*, repo, variant, gpu, output):
    """Match the planned scientific settings; limit coverage, not model budget."""
    from scripts.baselines.prepare_replogle_steering_controls import ENVIRONMENT_KEYS

    if variant not in {"scratch", "finetuned"} or gpu < 0:
        raise ValueError("Invalid variant or GPU index")
    repo = Path(repo).resolve()
    spec = json.loads((repo / "configs/benchmark/replogle_steering_controls.json").read_text())
    env = os.environ.copy()
    # Do not let stale user variables silently alter the fixed sampler.
    for key in list(env):
        if key.startswith("CELLDIFFA_"):
            del env[key]
    env.update({ENVIRONMENT_KEYS[key]: str(value) for key, value in spec["shared"].items()})
    env.update(
        CELLDIFFA_DEVICE="cuda:0",
        CELLDIFFA_DATA_ROOT=str(repo / "data"),
        CELLDIFFA_EVALUATION_SPLIT="validation",
        CELLDIFFA_REAL_TEST=str(repo / "results/replogle/reference/validation.h5ad"),
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=str(repo),
    )
    command = [
        "bash",
        str(repo / "scripts/baselines/run_celldiffa_replogle.sh"),
        variant,
        str(output),
        str(gpu),
        "0",
        "1",
        "1",
        str(repo / "external/PerturbDiff"),
    ]
    return command, env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "prepare", "smoke"])
    parser.add_argument("--variant", choices=["scratch", "finetuned"], default="scratch")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10) or platform.system() != "Linux":
        parser.error("Activate the dedicated Linux Python 3.10 environment first.")
    if args.stage == "check":
        check_cuda()
    elif args.stage == "prepare":
        prepare()
    else:
        report = check_cuda()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        identity = ",".join(str(entry["index"]) for entry in report["devices"])
        if visible is not None and visible != identity:
            parser.error("Unset custom CUDA_VISIBLE_DEVICES; select a GPU with --gpu instead.")
        if args.gpu not in {entry["index"] for entry in report["devices"]}:
            parser.error("Requested GPU is not visible; do not pre-set CUDA_VISIBLE_DEVICES.")
        ensure_upstream()
        output = REPO / "results/replogle/server/smoke" / f"{args.variant}_{stamp()}"
        command, env = smoke_command(repo=REPO, variant=args.variant, gpu=args.gpu, output=output)
        run(command, env=env)
        print(f"SMOKE FINISHED: {output}", flush=True)
        print("Only one validation group was requested. This is NOT a complete result.", flush=True)


if __name__ == "__main__":
    main()
