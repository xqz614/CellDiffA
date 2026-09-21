#!/usr/bin/env python
"""Six explicit, resumable server lanes; no background launch on import/init.

Each lane runs at most one compute subprocess at a time, then evaluates the
complete prediction. No current experiment or old result is modified.
"""

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def paths(config):
    repo = Path(config["repo"])
    data = Path(config["data_root"]) / "PerturbDiff_data"
    return dict(
        source=data / "finetune_data/nadig_processed_data/replogle.h5ad",
        genes=data / "selected_genes/replogle_real_selected_genes.pkl",
        embeddings=data / "gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl",
        split=repo / "external/PerturbDiff/configs/data/perturb_data/replogle.yaml",
        reference=repo / "results/replogle/reference",
    )


def jobs(preset="full"):
    if preset == "main-text":
        return [
            [
                dict(id="scratch_random16", kind="perturbdiff", mode="random"),
                dict(id="scratch_mean_correction", kind="mean", parent="scratch_random16"),
            ],
            [dict(id="scratch_best16", kind="perturbdiff", mode="best_of_n")],
            [dict(id="scratch_cellwise16", kind="perturbdiff", mode="smc", unit="cell")],
            [dict(id="squidiff_vanilla", kind="squidiff", mode="random", particles=1)],
            [dict(id="squidiff_adacell16", kind="squidiff", mode="smc")],
            [dict(id="scratch_particles8", kind="perturbdiff", mode="smc", particles=8)],
        ]
    if preset != "full":
        raise ValueError(f"Unknown experiment preset: {preset}")
    return [
        [
            dict(id="scratch_random16", kind="perturbdiff", mode="random"),
            dict(id="scratch_mean_correction", kind="mean", parent="scratch_random16"),
        ],
        [
            dict(id="scratch_best16", kind="perturbdiff", mode="best_of_n"),
            dict(id="scratch_cellwise16", kind="perturbdiff", mode="smc", unit="cell"),
        ],
        [
            dict(id="squidiff_vanilla", kind="squidiff", mode="random", particles=1),
            dict(id="squidiff_adacell16", kind="squidiff", mode="smc"),
        ],
        [
            dict(id="squidiff_random16", kind="squidiff", mode="random"),
            dict(id="squidiff_best16", kind="squidiff", mode="best_of_n"),
        ],
        [
            dict(id="conditional_ddpm_training", kind="train"),
            dict(
                id="conditional_ddpm_vanilla", kind="conditional_ddpm", mode="random", particles=1
            ),
            dict(id="conditional_ddpm_random16", kind="conditional_ddpm", mode="random"),
            dict(id="conditional_ddpm_best16", kind="conditional_ddpm", mode="best_of_n"),
            dict(id="conditional_ddpm_adacell16", kind="conditional_ddpm", mode="smc"),
        ],
        [
            dict(id="scratch_particles8", kind="perturbdiff", mode="smc", particles=8),
            dict(id="scratch_random16_seed43", kind="perturbdiff", mode="random", seed=43),
            dict(id="scratch_seed43", kind="perturbdiff", mode="smc", seed=43),
            dict(id="scratch_random16_seed44", kind="perturbdiff", mode="random", seed=44),
            dict(id="scratch_seed44", kind="perturbdiff", mode="smc", seed=44),
        ],
    ]


def environment(config, lane):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CELLDIFFA_")}
    gpu = config["gpus"][lane % len(config["gpus"])]
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=config["repo"],
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NUMEXPR_NUM_THREADS="4",
        CELLDIFFA_DEVICE="cuda:0",
        CELLDIFFA_DATA_ROOT=config["data_root"],
        CELLDIFFA_EVALUATION_SPLIT="test",
    )
    return env, gpu


def command_for(config, job, gpu, *, smoke=False):
    p = paths(config)
    repo, root = Path(config["repo"]), Path(config["output_root"])
    output = root / ("smoke" if smoke else "runs") / job["id"]
    python = config["python"]
    extra_env = {}
    if job["kind"] == "perturbdiff":
        extra_env = dict(
            CELLDIFFA_ALIGNMENT_MODE=job["mode"],
            CELLDIFFA_REWARD_UNIT=job.get("unit", "population"),
            CELLDIFFA_NUM_PARTICLES=str(job.get("particles", 16)),
            CELLDIFFA_NATIVE_BLOCKS_PER_POPULATION="16",
            CELLDIFFA_PARTICLE_BATCH_CELLS="1024",
            CELLDIFFA_ALPHA="1",
            CELLDIFFA_SEED=str(job.get("seed", 42)),
            CELLDIFFA_REWARD_NORMALIZATION="zscore",
            CELLDIFFA_ESS_THRESHOLD="0.5",
            CELLDIFFA_SIGNATURE_WEIGHT="1",
            CELLDIFFA_DIRECTION_WEIGHT="1",
            CELLDIFFA_ANCHOR_WEIGHT="1",
            CELLDIFFA_ANCHOR_BANDWIDTH="1",
            CELLDIFFA_PRIOR_RIDGE="1",
        )
        command = [
            "bash",
            str(repo / "scripts/baselines/run_celldiffa_replogle.sh"),
            "scratch",
            str(output),
            str(gpu),
            "0",
            "1",
            "1" if smoke else "all",
        ]
        prediction = output / "celldiffa_scratch.h5ad"
    elif job["kind"] == "train":
        command = [
            python,
            "-u",
            str(repo / "scripts/baselines/train_conditional_ddpm_replogle.py"),
            "--reference-dir",
            str(p["reference"]),
            "--split-config",
            str(p["split"]),
            "--embeddings",
            str(p["embeddings"]),
            "--output-dir",
            str(output),
            "--device",
            "cuda:0",
        ]
        if smoke:
            command += ["--smoke", "--width", "64", "--depth", "1", "--batch-size", "16"]
        prediction = None
    elif job["kind"] == "mean":
        parent = root / "runs" / job["parent"] / "celldiffa_scratch.h5ad"
        command = [
            python,
            "-u",
            str(repo / "scripts/baselines/replogle_mean_correction.py"),
            "--real",
            str(p["reference"] / "real.h5ad"),
            "--pred",
            str(parent),
            "--source",
            str(p["source"]),
            "--split-config",
            str(p["split"]),
            "--selected-genes",
            str(p["genes"]),
            "--embeddings",
            str(p["embeddings"]),
            "--outdir",
            str(output),
        ]
        prediction = output / "predictions.h5ad"
    else:
        if job["kind"] == "squidiff" and not config.get("squidiff_checkpoint"):
            raise ValueError("Bind the completed Squidiff checkpoint with configure-squidiff first")
        checkpoint = (
            Path(config["squidiff_checkpoint"])
            if job["kind"] == "squidiff"
            else root / "runs/conditional_ddpm_training/best.pt"
        )
        command = [
            python,
            "-u",
            str(repo / "scripts/baselines/run_adacell_backbone.py"),
            "--backbone",
            job["kind"],
            "--checkpoint",
            str(checkpoint),
            "--reference-dir",
            str(p["reference"]),
            "--source",
            str(p["source"]),
            "--split-config",
            str(p["split"]),
            "--selected-genes",
            str(p["genes"]),
            "--embeddings",
            str(p["embeddings"]),
            "--output-dir",
            str(output),
            "--device",
            "cuda:0",
            "--alignment-mode",
            job["mode"],
            "--num-particles",
            str(job.get("particles", 16)),
            "--population-cells",
            "512",
            "--sampling-steps",
            "100",
            "--alpha",
            "1",
            "--seed",
            "42",
        ]
        if job["kind"] == "squidiff":
            command += ["--unseen-policy", config["squidiff_unseen_policy"]]
            if config.get("squidiff_model_config"):
                command += ["--model-config", config["squidiff_model_config"]]
        if smoke:
            command += ["--max-groups", "1", "--population-cells", "4", "--num-particles", "2"]
        prediction = output / "predictions.h5ad"
    return command, extra_env, output, prediction


def execute(command, config, env, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN " + shlex.join(command), flush=True)
    with log.open("a") as handle:
        handle.write(f"\n[{timestamp()}] {shlex.join(command)}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=config["repo"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def evaluate(config, prediction, output, env):
    from celldiffa.benchmark.artifacts import sha256_file
    from celldiffa.benchmark.backbone_experiments import atomic_json

    real = paths(config)["reference"] / "real.h5ad"
    metrics = Path(config["output_root"]) / "metrics" / output.name
    command = [
        config["python"],
        "-u",
        str(Path(config["repo"]) / "scripts/baselines/evaluate.py"),
        "--real",
        str(real),
        "--pred",
        str(prediction),
        "--outdir",
        str(metrics),
        "--pert-col",
        "gene",
        "--control-pert",
        "non-targeting",
        "--num-threads",
        "4",
    ]
    execute(command, config, env, output / "evaluation.log")
    command = [
        config["python"],
        "-u",
        str(Path(config["repo"]) / "scripts/baselines/evaluate_population_diagnostics.py"),
        "--real",
        str(real),
        "--pred",
        str(prediction),
        "--outdir",
        str(output / "diagnostics"),
    ]
    execute(command, config, env, output / "diagnostics.log")
    import numpy as np
    import pandas as pd

    from celldiffa.benchmark.streaming import read_h5ad_obs

    table = pd.read_csv(metrics / "perturbdiff_metrics_per_perturbation.csv")
    expected = set(read_h5ad_obs(real).gene.astype(str)) - {"non-targeting"}
    if table.perturbation.duplicated().any() or set(table.perturbation) != expected:
        raise ValueError("Evaluation does not cover exactly the full test perturbations")
    finite = np.isfinite(table.drop(columns="perturbation").to_numpy()).all()
    atomic_json(
        output / "evaluated.json",
        dict(
            status="complete" if finite else "evaluated_with_undefined_metrics",
            completed_at=timestamp(),
            prediction_sha256=sha256_file(prediction),
            reference_sha256=sha256_file(real),
            metrics_sha256=sha256_file(metrics / "perturbdiff_metrics_per_perturbation.csv"),
            perturbations=len(expected),
            undefined_counts=table.isna().sum().to_dict(),
        ),
    )


def launch_maintext(config, config_path, lanes, *, dry_run=False):
    """Start only the approved main-text plan; never stop existing screen jobs."""
    if config.get("preset") != "main-text" or config["lanes"] != jobs("main-text"):
        raise ValueError("launch-maintext requires an unchanged --preset main-text plan")
    if len(lanes) != len(set(lanes)):
        raise ValueError("Do not request the same lane twice")
    root = Path(config["output_root"])
    commands = []
    for lane in lanes:
        name = f"adacell-maintext-lane-{lane}"
        command = [
            "screen",
            "-L",
            "-Logfile",
            str(root / f"lane_{lane}.screen.log"),
            "-dmS",
            name,
            config["python"],
            "-u",
            str(Path(config["repo"]) / "scripts/server/replogle_remaining.py"),
            "run",
            "--config",
            str(config_path.resolve()),
            "--lane",
            str(lane),
        ]
        commands.append((name, command))
    if dry_run:
        for _, command in commands:
            print(shlex.join(command))
        return
    if not shutil.which("screen"):
        raise RuntimeError("screen is not installed")
    listing = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
    screen_state = listing.stdout + listing.stderr
    if ".adacell-lane-" in screen_state:
        raise RuntimeError(
            "An older adacell-lane screen session exists. Inspect it before launching "
            "the new plan; no existing process has been stopped."
        )
    from celldiffa.benchmark.metrics import _require_cell_eval_066

    _require_cell_eval_066()
    import torch

    requested_gpus = {config["gpus"][lane % len(config["gpus"])] for lane in lanes}
    if not torch.cuda.is_available() or max(requested_gpus) >= torch.cuda.device_count():
        raise RuntimeError("Requested GPUs are unavailable")
    required = [paths(config)[key] for key in ("source", "genes", "embeddings", "split")]
    required.extend(paths(config)["reference"] / file for file in ("real.h5ad", "controls.h5ad"))
    if any(lane in {0, 1, 2, 5} for lane in lanes):
        required.append(
            Path(config["data_root"])
            / "checkpoints/PerturbDiff_release_ckpt/from_scratch_replogle.ckpt"
        )
    if any(lane in {3, 4} for lane in lanes):
        if not config.get("squidiff_checkpoint"):
            raise ValueError("Bind the completed Squidiff checkpoint before launching lanes 3/4")
        checkpoint = Path(config["squidiff_checkpoint"])
        required.extend(
            [
                checkpoint,
                Path(config["squidiff_model_config"])
                if config.get("squidiff_model_config")
                else checkpoint.parent / "run_config.json",
                paths(config)["reference"] / "train.h5ad",
            ]
        )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required inputs are missing: " + ", ".join(missing))
    for name, command in commands:
        if f".{name}" in screen_state:
            print(f"Already has a screen session; not launching a duplicate: {name}")
            continue
        subprocess.run(command, cwd=config["repo"], check=True)
        print(f"Screen launch requested: {name}; check status and its lane log", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    initialize = sub.add_parser("init")
    initialize.add_argument("--preset", choices=["full", "main-text"], default="full")
    initialize.add_argument("--output-root", type=Path)
    initialize.add_argument("--data-root", type=Path, default=REPO / "data")
    initialize.add_argument("--squidiff-checkpoint", type=Path)
    initialize.add_argument("--squidiff-model-config", type=Path)
    initialize.add_argument(
        "--squidiff-unseen-policy", choices=["auto", "error", "zero_shift", "ridge"], default="auto"
    )
    initialize.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    bind = sub.add_parser("configure-squidiff")
    bind.add_argument(
        "--config", type=Path, default=REPO / "results/replogle/remaining_v1/plan.json"
    )
    bind.add_argument("--checkpoint", type=Path, required=True)
    bind.add_argument("--model-config", type=Path)
    bind.add_argument(
        "--unseen-policy", choices=["auto", "error", "zero_shift", "ridge"], default="auto"
    )
    launch = sub.add_parser("launch-maintext")
    launch.add_argument(
        "--config", type=Path, default=REPO / "results/replogle/maintext_v1/plan.json"
    )
    launch.add_argument("--lanes", type=int, choices=range(6), nargs="+", default=list(range(6)))
    launch.add_argument("--dry-run", action="store_true")
    for action in ("prepare", "run", "smoke", "status"):
        command = sub.add_parser(action)
        command.add_argument(
            "--config", type=Path, default=REPO / "results/replogle/remaining_v1/plan.json"
        )
        if action in {"run", "smoke"}:
            command.add_argument("--lane", type=int, choices=range(6), required=True)
            command.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from celldiffa.benchmark.backbone_experiments import atomic_json

    if args.action == "init":
        if args.output_root is None:
            dirname = "maintext_v1" if args.preset == "main-text" else "remaining_v1"
            args.output_root = REPO / "results/replogle" / dirname
        config = dict(
            repo=str(REPO),
            python=sys.executable,
            data_root=str(args.data_root.resolve()),
            output_root=str(args.output_root.resolve()),
            gpus=args.gpus,
            lanes=jobs(args.preset),
            squidiff_checkpoint=str(args.squidiff_checkpoint.resolve())
            if args.squidiff_checkpoint
            else None,
            squidiff_model_config=str(args.squidiff_model_config.resolve())
            if args.squidiff_model_config
            else None,
            squidiff_unseen_policy=args.squidiff_unseen_policy,
            policy="alpha=1 fixed; other test runs are sensitivity checks, never test-selected",
        )
        if args.preset == "main-text":
            config["preset"] = "main-text"
        if any(g < 0 for g in args.gpus):
            parser.error("GPU indices must be nonnegative")
        path = args.output_root / "plan.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError("Plan already exists with different settings; use a new output root")
        atomic_json(path, config)
        print(f"Plan saved to {path}; nothing has been started")
        for index, lane in enumerate(config["lanes"]):
            print(
                f"lane {index}, GPU {args.gpus[index % len(args.gpus)]}: "
                + " -> ".join(j["id"] for j in lane)
            )
        return
    config = json.loads(args.config.read_text())
    if Path(config["repo"]).resolve() != REPO:
        raise ValueError("Create this plan on the server; do not copy a local plan")
    if args.action == "launch-maintext":
        launch_maintext(config, args.config, args.lanes, dry_run=args.dry_run)
        return
    if args.action == "configure-squidiff":
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        config.update(
            squidiff_checkpoint=str(args.checkpoint.resolve()),
            squidiff_model_config=str(args.model_config.resolve()) if args.model_config else None,
            squidiff_unseen_policy=args.unseen_policy,
        )
        atomic_json(args.config, config)
        squidiff_lanes = [
            index
            for index, lane in enumerate(config["lanes"])
            if any(job["kind"] == "squidiff" for job in lane)
        ]
        print(f"Squidiff checkpoint bound; lanes {squidiff_lanes} have NOT been launched")
        return
    if args.action == "status":
        for lane, items in enumerate(config["lanes"]):
            for job in items:
                output = Path(config["output_root"]) / "runs" / job["id"]
                found = False
                for name in ("evaluated.json", "progress.json", "training_progress.json"):
                    if (output / name).exists():
                        print(lane, job["id"], (output / name).read_text().strip())
                        found = True
                        break
                if not found:
                    progress = output / "shards/worker_000.progress.json"
                    message = (
                        progress.read_text().strip()
                        if progress.exists()
                        else (
                            "output exists; inspect run.log" if output.exists() else "not started"
                        )
                    )
                    print(lane, job["id"], message)
        return
    if args.action == "prepare":
        env, _ = environment(config, 0)
        p = paths(config)
        command = [
            config["python"],
            str(REPO / "scripts/data/prepare_replogle_training_only.py"),
            "--source",
            str(p["source"]),
            "--split-config",
            str(p["split"]),
            "--selected-genes",
            str(p["genes"]),
            "--output",
            str(p["reference"] / "train.h5ad"),
        ]
        execute(command, config, env, Path(config["output_root"]) / "prepare.log")
        upstream = REPO / "external/Squidiff"
        if not upstream.exists():
            from scripts.baselines.run_squidiff_replogle import REVISION

            execute(
                ["git", "clone", "https://github.com/siyuh/Squidiff.git", str(upstream)],
                config,
                env,
                Path(config["output_root"]) / "prepare.log",
            )
            execute(
                ["git", "-C", str(upstream), "checkout", "--detach", REVISION],
                config,
                env,
                Path(config["output_root"]) / "prepare.log",
            )
        print("Training reference prepared; no model training or inference started")
        return
    env, gpu = environment(config, args.lane)
    lane_lock = None
    if not args.dry_run:
        lane_lock = (Path(config["output_root"]) / f".lane_{args.lane}.lock").open("a")
        fcntl.flock(lane_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch

        if not torch.cuda.is_available() or gpu >= torch.cuda.device_count():
            raise RuntimeError("Requested server GPU is unavailable")
        from celldiffa.benchmark.metrics import _require_cell_eval_066

        _require_cell_eval_066()  # Fail before hours of sampling if evaluation is unavailable.
        # Shell wrappers use `python`; ensure it resolves to the recorded environment.
        env["PATH"] = str(Path(config["python"]).parent) + os.pathsep + env.get("PATH", "")
    selected = config["lanes"][args.lane]
    if args.action == "smoke":
        selected = selected[:1]
    for job in selected:
        command, changes, output, prediction = command_for(
            config, job, gpu, smoke=args.action == "smoke"
        )
        job_env = {**env, **changes}
        if args.dry_run:
            print(shlex.join(command))
            continue
        from celldiffa.benchmark.artifacts import sha256_file

        launch_record = dict(job=job, command=command, overrides=changes)
        if job["kind"] == "perturbdiff":
            checkpoint = (
                Path(config["data_root"])
                / "checkpoints/PerturbDiff_release_ckpt/from_scratch_replogle.ckpt"
            )
            launch_record["checkpoint_sha256"] = sha256_file(checkpoint)
        if "--checkpoint" in command:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            launch_record["checkpoint_sha256"] = sha256_file(checkpoint)
            for filename in ("run_config.json", "prediction_config.json"):
                adjacent = checkpoint.parent / filename
                if adjacent.exists():
                    launch_record[filename] = sha256_file(adjacent)
        launch_path = (
            Path(config["output_root"])
            / "launch_contracts"
            / (("smoke_" if args.action == "smoke" else "") + job["id"] + ".json")
        )
        if launch_path.exists() and json.loads(launch_path.read_text()) != launch_record:
            raise ValueError("Job checkpoint, policy or settings changed; use a new output root")
        atomic_json(launch_path, launch_record)
        marker = output / "evaluated.json"
        if marker.exists() and prediction is not None:
            saved = json.loads(marker.read_text())
            real = paths(config)["reference"] / "real.h5ad"
            metrics = (
                Path(config["output_root"])
                / "metrics"
                / job["id"]
                / "perturbdiff_metrics_per_perturbation.csv"
            )
            if (
                saved["prediction_sha256"] != sha256_file(prediction)
                or saved["reference_sha256"] != sha256_file(real)
                or saved["metrics_sha256"] != sha256_file(metrics)
            ):
                raise ValueError("Completed prediction was changed; refusing to reuse evaluation")
            print(f"Already evaluated: {job['id']}", flush=True)
            continue
        # Mean correction refuses overwrites; a successful prediction can resume evaluation.
        if not (job["kind"] == "mean" and prediction.is_file()):
            execute(
                command,
                config,
                job_env,
                output.parent / f"{output.name}.launcher.log"
                if job["kind"] == "mean"
                else output / "run.log",
            )
        if args.action == "smoke":
            print("SMOKE ONLY: partial result, not a benchmark score")
        elif prediction is not None:
            if not prediction.is_file():
                raise RuntimeError("No complete H5AD produced; evaluation will not start")
            evaluate(config, prediction, output, job_env)
    print(f"Lane {args.lane} finished", flush=True)


if __name__ == "__main__":
    main()
