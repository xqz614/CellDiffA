#!/usr/bin/env python
"""Resume the existing Replogle controls; queue figures, idle-GPU timing and diagnosis.

No new seed, dataset, main run, ablation, or full Squidiff generation is launched.
Run on the experiment server, not on the local laptop. Existing results are kept.
"""

import argparse
import fcntl
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from celldiffa.benchmark.artifacts import sha256_file  # noqa: E402
from celldiffa.benchmark.backbone_experiments import atomic_json  # noqa: E402
from scripts.server import replogle_remaining as remaining  # noqa: E402

LANES = (0, 1, 2, 5)
PER_FILE = "perturbdiff_metrics_per_perturbation.csv"


def read(path):
    return json.loads(path.read_text())


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def checked_plan(path):
    config = read(path)
    preset = config.get("preset")
    if preset not in {"main-text", "main-text-train"} or config["lanes"] != remaining.jobs(preset):
        raise ValueError("Use the unchanged existing main-text plan, not the full/extra-seed plan")
    if (
        Path(config["repo"]).resolve() != REPO
        or Path(config["output_root"]).resolve() != path.parent
    ):
        raise ValueError("Plan paths are not bound to this server checkout")
    if not Path(config["python"]).is_file():
        raise FileNotFoundError(config["python"])
    return config


def check_main(run):
    config = read(run / "shards/run_config.json")
    expected = dict(
        variant="scratch",
        evaluation_split="test",
        alpha=1,
        seed=42,
        num_particles=16,
        alignment_mode="smc",
        reward_normalization="zscore",
        reward_weights=[1, 1, 1],
    )
    differences = [k for k, value in expected.items() if config.get(k) != value]
    if config.get("reward_unit", "population") != "population":
        differences.append("reward_unit")
    if differences:
        raise ValueError(f"Not the prespecified full Scratch alpha=1 main run: {differences}")
    prediction = run / "celldiffa_scratch.h5ad"
    if not prediction.is_file():
        raise FileNotFoundError(f"Main prediction is not complete: {prediction}")
    return prediction


def find_main_metrics(run, results, explicit=None):
    if explicit:
        if not (explicit / PER_FILE).is_file():
            raise FileNotFoundError(explicit / PER_FILE)
        return explicit
    candidates = {
        run,
        run / "metrics",
        run / "evaluation",
        run.parent / "metrics" / run.name,
        results / "metrics" / run.relative_to(results),
        results / "metrics" / run.name,
        results / "metrics" / ("adacell_" + run.name),
        results / "metrics" / ("celldiffa_" + run.name),
    }
    found = sorted(p for p in candidates if (p / PER_FILE).is_file())
    if len(found) > 1 and len({sha256_file(p / PER_FILE) for p in found}) > 1:
        raise ValueError("Several different main metric tables found; specify --main-metrics")
    if not found:
        raise FileNotFoundError(
            "Main metrics not found; pass --main-metrics with the actual directory"
        )
    return found[0]


def completed(config, job, *, verify=False):
    root = Path(config["output_root"])
    output = root / "runs" / job["id"]
    marker = output / "evaluated.json"
    prediction = output / (
        "predictions.h5ad" if job["kind"] == "mean" else "celldiffa_scratch.h5ad"
    )
    metrics = root / "metrics" / job["id"] / PER_FILE
    if not all(p.is_file() for p in (marker, prediction, metrics)):
        return False
    saved = read(marker)
    if saved.get("status") != "complete":
        raise ValueError(f"Incomplete or undefined metric results: {job['id']}")
    if verify:
        for key, path in (
            ("prediction_sha256", prediction),
            ("metrics_sha256", metrics),
            ("reference_sha256", remaining.paths(config)["reference"] / "real.h5ad"),
        ):
            if saved.get(key) != sha256_file(path):
                raise ValueError(f"Saved evaluation hash mismatch: {job['id']}/{key}")
    return True


def pending_lanes(config, *, verify=False):
    return [
        lane
        for lane in LANES
        if not all(completed(config, job, verify=verify) for job in config["lanes"][lane])
    ]


def matching_diagnostic(config):
    checkpoint = Path(config["squidiff_checkpoint"])
    model_config = Path(
        config.get("squidiff_model_config") or checkpoint.parent / "run_config.json"
    )
    root = REPO / "results/replogle/diagnostics"
    reports = sorted(root.glob("squidiff_sampling_*/report.json"), reverse=True)
    if not reports:
        return None
    hashes = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_config_sha256": sha256_file(model_config),
    }
    for path in reports:
        data = read(path)
        if data.get("status") == "diagnostic_complete" and all(
            data.get(k) == v for k, v in hashes.items()
        ):
            return path
    return None


def run_analysis(config, args, output):
    for lane in LANES:
        for job in config["lanes"][lane]:
            if not completed(config, job, verify=True):
                raise ValueError(f"Missing completed control: {job['id']}")
    metrics = find_main_metrics(args.main_run, REPO / "results/replogle", args.main_metrics)
    command = [
        config["python"],
        "-u",
        str(REPO / "scripts/baselines/analyze_adacell_experiments.py"),
        "--plan",
        str(args.plan),
        "--main-pred",
        str(check_main(args.main_run)),
        "--main-metrics",
        str(metrics),
        "--outdir",
        str(output),
        "--perturbdiff-only",
        "--require-controls",
    ]
    env, _ = remaining.environment(config, 0)
    remaining.execute(command, config, env, args.root / "analysis.log")


def wait_controls(config, state, deadline):
    previous = None
    while True:
        pending = pending_lanes(config)
        if not pending:
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"Control wait expired, pending lanes: {pending}")
        if pending != previous:
            print(f"Waiting for control evaluation, lanes {pending}", flush=True)
            atomic_json(state, dict(status="waiting_for_controls", pending_lanes=pending))
            previous = pending
        time.sleep(30)


def gpu_idle(gpu):
    # Fail closed if nvidia-smi or its process query is unavailable.
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    return not output


def worker(config, args):
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / f".{args.task}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = args.root / f"{args.task}.status.json"
        deadline = time.monotonic() + args.wait_hours * 3600
        try:
            if args.task == "squidiff":
                report = matching_diagnostic(config)
                if report:
                    print(
                        f"Existing native-sampling diagnostic: {report}; not rerunning", flush=True
                    )
                else:
                    # Do not duplicate the previously supplied standalone diagnostic.
                    while True:
                        active = subprocess.check_output(["ps", "-eo", "args"], text=True)
                        if not any(
                            "diagnose_squidiff_sampling.py" in line for line in active.splitlines()
                        ):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError("Waiting for existing Squidiff diagnostic")
                        time.sleep(30)
                    report = matching_diagnostic(config)
                    if report is None:
                        env, _ = remaining.environment(config, 0)
                        env["CUDA_VISIBLE_DEVICES"] = str(args.diagnostic_gpu)
                        output = (
                            REPO / "results/replogle/diagnostics" / ("squidiff_sampling_" + stamp())
                        )
                        command = [
                            config["python"],
                            "-u",
                            str(REPO / "scripts/server/diagnose_squidiff_sampling.py"),
                            "--config",
                            str(args.plan),
                            "--device",
                            "cuda:0",
                            "--cells",
                            "32",
                            "--groups",
                            "2",
                            "--output-dir",
                            str(output),
                        ]
                        remaining.execute(command, config, env, args.root / "squidiff.log")
                        report = output / "report.json"
                atomic_json(
                    state,
                    dict(
                        status="needs_scientific_review",
                        report=str(report),
                        full_squidiff_rerun_started=False,
                    ),
                )
                return
            wait_controls(config, state, deadline)
            if args.task == "analysis":
                output = args.root / ("figures_" + stamp())
                atomic_json(state, dict(status="running", output=str(output)))
                run_analysis(config, args, output)
            else:
                # Wait for two consecutive idle observations, not simply free VRAM.
                atomic_json(state, dict(status="waiting_for_idle_gpu", gpu=args.timing_gpu))
                idle_observations = 0
                while idle_observations < 2:
                    if time.monotonic() > deadline:
                        raise TimeoutError("No idle GPU available for isolated timing")
                    idle_observations = idle_observations + 1 if gpu_idle(args.timing_gpu) else 0
                    if idle_observations < 2:
                        time.sleep(30)
                output = args.root / ("timing_" + stamp())
                command = [
                    config["python"],
                    "-u",
                    str(REPO / "scripts/server/benchmark_replogle_steering_cost.py"),
                    "--plan",
                    str(args.plan),
                    "--outdir",
                    str(output),
                    "--gpu",
                    str(args.timing_gpu),
                ]
                env, _ = remaining.environment(config, 0)
                env.pop("CUDA_VISIBLE_DEVICES", None)
                atomic_json(state, dict(status="running", output=str(output)))
                remaining.execute(command, config, env, args.root / "timing.log")
            atomic_json(state, dict(status="complete", output=str(output)))
        except BaseException as error:
            atomic_json(state, dict(status="failed", error=f"{type(error).__name__}: {error}"))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["launch", "worker", "status"])
    parser.add_argument(
        "--plan", type=Path, default=REPO / "results/replogle/maintext_train_v1/plan.json"
    )
    parser.add_argument(
        "--main-run", type=Path, default=REPO / "results/replogle/test_sensitivity/scratch_alpha1"
    )
    parser.add_argument("--main-metrics", type=Path)
    parser.add_argument("--timing-gpu", type=int, default=2)
    parser.add_argument("--diagnostic-gpu", type=int, default=1)
    parser.add_argument("--wait-hours", type=float, default=72)
    parser.add_argument("--task", choices=["analysis", "timing", "squidiff"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.wait_hours <= 0 or min(args.timing_gpu, args.diagnostic_gpu) < 0:
        parser.error("Positive wait-hours and nonnegative GPU indices required")
    args.plan, args.main_run = args.plan.resolve(), args.main_run.resolve()
    args.main_metrics = args.main_metrics.resolve() if args.main_metrics else None
    config = checked_plan(args.plan)
    args.root = Path(config["output_root"]) / "maintext_completion"
    if args.action == "status":
        print("Unfinished control lanes:", pending_lanes(config))
        for task in ("analysis", "timing", "squidiff"):
            state = args.root / f"{task}.status.json"
            print(task, json.dumps(read(state) if state.is_file() else {"status": "not_started"}))
        return
    if args.action == "worker":
        if args.task is None:
            parser.error("worker requires --task")
        worker(config, args)
        return
    check_main(args.main_run)
    metrics = find_main_metrics(args.main_run, REPO / "results/replogle", args.main_metrics)
    pending = pending_lanes(config, verify=not args.dry_run)
    print(f"Main: {args.main_run}\nMetrics: {metrics}\nUnfinished control lanes: {pending}")
    commands = []
    for task in ("analysis", "timing", "squidiff"):
        name = f"adacell-finish-{task}"
        command = [
            "screen",
            "-L",
            "-Logfile",
            str(args.root / f"{task}.screen.log"),
            "-dmS",
            name,
            config["python"],
            "-u",
            str(Path(__file__).resolve()),
            "worker",
            "--task",
            task,
            "--plan",
            str(args.plan),
            "--main-run",
            str(args.main_run),
            "--main-metrics",
            str(metrics),
            "--timing-gpu",
            str(args.timing_gpu),
            "--diagnostic-gpu",
            str(args.diagnostic_gpu),
            "--wait-hours",
            str(args.wait_hours),
        ]
        commands.append((task, name, command))
    if args.dry_run:
        if pending:
            remaining.launch_maintext(config, args.plan, pending, dry_run=True)
        for _, _, command in commands:
            print(shlex.join(command))
        return
    args.root.mkdir(parents=True, exist_ok=True)
    binding = dict(
        plan_sha256=sha256_file(args.plan),
        main_run=str(args.main_run),
        main_metrics=str(metrics),
        main_config_sha256=sha256_file(args.main_run / "shards/run_config.json"),
        main_prediction_sha256=sha256_file(check_main(args.main_run)),
        main_metrics_sha256=sha256_file(metrics / PER_FILE),
        timing_gpu=args.timing_gpu,
        diagnostic_gpu=args.diagnostic_gpu,
    )
    with (args.root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bound = args.root / "binding.json"
        if bound.exists() and read(bound) != binding:
            raise ValueError(
                "Existing completion workflow uses different settings; inspect binding.json"
            )
        atomic_json(bound, binding)
        if pending:
            remaining.launch_maintext(config, args.plan, pending)
        listing = subprocess.run(["screen", "-ls"], capture_output=True, text=True)
        sessions = listing.stdout + listing.stderr
        for task, name, command in commands:
            state = args.root / f"{task}.status.json"
            if state.is_file() and read(state).get("status") in {
                "complete",
                "needs_scientific_review",
            }:
                print(f"Already finished: {task}; {state}")
            elif f".{name}" in sessions:
                print(f"Already running: {name}")
            else:
                subprocess.run(command, cwd=REPO, check=True)
                print(f"Queued: {name}; status in {state}", flush=True)
    print("No full Squidiff rerun: inspect native diagnostic before deciding the next step.")


if __name__ == "__main__":
    main()
