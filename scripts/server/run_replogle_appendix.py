#!/usr/bin/env python
"""Parallel, resumable appendix inference and offline analyses; never train a model.

GPU workers produce predictions, while one independent CPU worker evaluates them.
Existing experiments are read-only and may be reused only under matching contracts.
No test-optimal configuration is selected and no result is adjusted to a target mean.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULTS = dict(prior_mode="full", prior_fraction=1.0, prior_seed=42, reward_unit="population")
PARAMS = (
    "alpha",
    "num_particles",
    "top_de",
    "ess_threshold",
    "anchor_bandwidth",
    "prior_ridge",
    "seed",
    "native_blocks_per_population",
    "particle_batch_cells",
    "alignment_mode",
    "reward_normalization",
    "reward_weights",
    *DEFAULTS,
)
FIXED = (
    "variant",
    "evaluation_split",
    "reference",
    "checkpoint",
    "checkpoint_size",
    "checkpoint_mtime_ns",
    "source",
    "selected_genes",
    "selected_genes_sha256",
    "perturbation_embeddings_sha256",
    "split_config_sha256",
    "upstream_revision",
    "normalize_counts",
    "cell_set",
    "start_time",
    "eta",
    "guidance_strength",
)
CODE = (
    "scripts/baselines/run_celldiffa_replogle.py",
    "scripts/baselines/run_celldiffa_replogle.sh",
    "celldiffa/benchmark/replogle_priors.py",
    "scripts/server/run_replogle_appendix.py",
    "scripts/baselines/analyze_adacell_cases.py",
    "scripts/baselines/report_adacell_appendix.py",
    "celldiffa/benchmark/metrics.py",
    "scripts/baselines/evaluate.py",
    "scripts/baselines/evaluate_population_diagnostics.py",
    "celldiffa/benchmark/contracts.py",
    "celldiffa/benchmark/released_sampling.py",
    "celldiffa/benchmark/streaming.py",
    "celldiffa/benchmark/perturbdiff_split.py",
    "celldiffa/benchmark/artifacts.py",
    "celldiffa/benchmark/population_blocks.py",
    "celldiffa/benchmark/replogle_shards.py",
    "celldiffa/smc/engine.py",
    "celldiffa/smc/resampler.py",
    "celldiffa/smc/utils.py",
    "celldiffa/rewards/base.py",
    "celldiffa/rewards/transcriptomic.py",
    "celldiffa/rewards/geometric.py",
    "celldiffa/rewards/anchor.py",
    "celldiffa/rewards/cellwise.py",
    "baselines/adapter_perturbdiff.py",
    "celldiffa/benchmark/perturbdiff_covariates.py",
)
TERMINAL = {"complete", "failed", "failed_generation", "failed_evaluation", "interrupted"}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(path, *, content=True):
    path = Path(path).resolve()
    st = path.stat()
    out = dict(path=str(path), size=st.st_size, mtime_ns=st.st_mtime_ns)
    if content:
        out["sha256"] = sha(path)
    return out


@contextlib.contextmanager
def locked(path, *, blocking=True):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def normal(config):
    return {**DEFAULTS, **config}


def matching(observed, expected):
    observed, expected = normal(observed), normal(expected)
    return all(observed.get(k) == expected.get(k) for k in (*PARAMS, *FIXED))


def make_jobs(base, seeds):
    """Finite one-at-a-time grids frozen before looking at any new score."""
    if len(set(seeds)) != len(seeds) or not seeds or min(seeds) < 0:
        raise ValueError("Provide distinct nonnegative seeds")
    base = normal(base)
    jobs = []

    def add(identifier, section, parameter, value, **updates):
        jobs.append(
            dict(
                id=identifier,
                section=section,
                parameter=parameter,
                value=value,
                settings={**{k: base[k] for k in PARAMS}, **updates},
            )
        )

    # This is both the sensitivity reference and the full-prior paired comparator.
    add("reference", "sensitivity", "reference", "full")
    grids = dict(
        alpha=[0.5, 1.0, 2.0],
        num_particles=[4, 8, 16, 32],
        top_de=[10, 20, 50],
        ess_threshold=[0.25, 0.5, 0.75],
        anchor_bandwidth=[0.5, 1.0, 2.0],
    )
    for parameter, values in grids.items():
        for value in values:
            if value != base[parameter]:
                tag = str(value).replace(".", "p")
                add(f"{parameter}_{tag}", "sensitivity", parameter, value, **{parameter: value})
    for seed in seeds:
        if seed != base["seed"]:
            add(f"full_seed{seed}", "prior", "prior", "full", seed=seed)
        for fraction in (0.5, 0.25):
            tag = str(fraction).replace(".", "p")
            add(
                f"prior_fraction{tag}_seed{seed}",
                "prior",
                "prior",
                fraction,
                seed=seed,
                prior_mode="subsample",
                prior_fraction=fraction,
                prior_seed=seed,
            )
        add(
            f"prior_shuffle_seed{seed}",
            "prior",
            "prior",
            "shuffle",
            seed=seed,
            prior_mode="shuffle",
            prior_fraction=1.0,
            prior_seed=seed,
        )
    if base["seed"] not in seeds:
        raise ValueError("Prior seeds must include the reference generation seed")
    return jobs


def existing_runs(root):
    for path in sorted(Path(root).rglob("run_config.json")):
        if path.parent.name != "shards" or any("smoke" in p for p in path.parts):
            continue
        try:
            config = read(path)
        except (OSError, ValueError):
            continue
        if config.get("evaluation_split") == "test" and config.get("variant") == "scratch":
            output = path.parent.parent
            pred = output / "celldiffa_scratch.h5ad"
            prior = output / "training_priors.npz"
            if pred.is_file() and prior.is_file():
                yield output, config


def reusable_prior(path, source_stat, expected, genes, targets):
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"].item()))
            if list(data["perturbations"].astype(str)) != sorted(targets):
                return False
            if (
                data["shifts"].shape != (len(targets), len(genes))
                or not np.isfinite(data["shifts"]).all()
            ):
                return False
        required = dict(
            source=str(Path(expected["source"]).resolve()),
            source_signature={k: source_stat[k] for k in ("size", "mtime_ns")},
            genes=genes,
            expression_key="X_hvg",
            top_k=max(expected["top_de"], 50),
            target_perturbations=sorted(targets),
            embedding_signature=expected["perturbation_embeddings_sha256"],
            ridge_penalty=expected["prior_ridge"],
            evaluation_split="test",
        )
        if any(meta.get(k) != v for k, v in required.items()):
            return False
        if (
            hashlib.sha256(meta.get("split_text", "").encode()).hexdigest()
            != expected["split_config_sha256"]
        ):
            return False
        for key in ("prior_mode", "prior_fraction", "prior_seed"):
            if meta.get(key, DEFAULTS[key]) != expected.get(key, DEFAULTS[key]):
                return False
        version = (
            3
            if all(
                expected.get(k, DEFAULTS[k]) == DEFAULTS[k]
                for k in ("prior_mode", "prior_fraction", "prior_seed")
            )
            else 4
        )
        return meta.get("format_version") == version and (
            version == 3 or meta.get("prior_robustness_version") == 1
        )
    except (OSError, ValueError, KeyError):
        return False


def evaluation_cache(output, reference_sha):
    """Only hash-attested old metrics can skip reevaluation; older CSVs are not enough."""
    record = output / "evaluated.json"
    if not record.is_file():
        return None
    saved = read(record)
    pred = output / "celldiffa_scratch.h5ad"
    if saved.get("status") != "complete" or saved.get("reference_sha256") != reference_sha:
        return None
    if saved.get("prediction_sha256") != sha(pred):
        return None
    candidates = [
        output / "metrics",
        output / "evaluation",
        output.parent.parent / "metrics" / output.name,
    ]
    for directory in candidates:
        metrics = directory / "perturbdiff_metrics_per_perturbation.csv"
        summary = directory / "perturbdiff_metrics_summary.csv"
        if metrics.is_file() and summary.is_file() and sha(metrics) == saved.get("metrics_sha256"):
            return dict(
                directory=str(directory),
                metrics_sha256=sha(metrics),
                prediction_sha256=saved["prediction_sha256"],
            )
    return None


def prepare(args):
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"Use launch/status for an existing plan; refusing to replace {root}")
    main = args.main_run.resolve()
    base = normal(read(main / "shards/run_config.json"))
    if base["variant"] != "scratch" or base["evaluation_split"] != "test":
        raise ValueError("This appendix suite uses the frozen Replogle Scratch test protocol")
    if (
        base["alignment_mode"] != "smc"
        or base["prior_mode"] != "full"
        or base["prior_fraction"] != 1.0
        or base["reward_normalization"] != "zscore"
    ):
        raise ValueError("Reference must be the complete, uncorrupted-prior AdaCell method")
    if base["reward_unit"] != "population" or base["reward_weights"] != [1.0, 1.0, 1.0]:
        raise ValueError("Reference must use population rewards and all three unit weights")
    for key, value in dict(
        start_time=100, eta=0.0, guidance_strength=1.0, normalize_counts=10.0, cell_set=32
    ).items():
        if base[key] != value:
            raise ValueError(f"Shell runner does not reproduce reference setting {key}={base[key]}")
    source = Path(base["source"]).resolve()
    data_root = source.parents[3]
    expected = data_root / "PerturbDiff_data/finetune_data/nadig_processed_data/replogle.h5ad"
    if expected != source:
        raise ValueError("Unrecognized data layout; do not silently substitute a different dataset")
    checkpoint = data_root / "checkpoints/PerturbDiff_release_ckpt/from_scratch_replogle.ckpt"
    if checkpoint.resolve() != Path(base["checkpoint"]).resolve():
        raise ValueError(
            "Shell runner checkpoint differs from the reference; explicit adapter needed"
        )
    st = checkpoint.stat()
    if st.st_size != base["checkpoint_size"] or st.st_mtime_ns != base["checkpoint_mtime_ns"]:
        raise ValueError("Checkpoint changed since the reference run")
    assets = dict(
        reference=Path(base["reference"]),
        checkpoint=checkpoint,
        source=source,
        genes=Path(base["selected_genes"]),
        embeddings=data_root
        / "PerturbDiff_data/gene_names/replogle_gene_emb_dict_perturbation_emb_dict.pkl",
        split=REPO / "external/PerturbDiff/configs/data/perturb_data/replogle.yaml",
    )
    if (
        assets["genes"].resolve()
        != (
            data_root / "PerturbDiff_data/selected_genes/replogle_real_selected_genes.pkl"
        ).resolve()
    ):
        raise ValueError("Shell runner gene panel differs from the reference")
    inputs = {name: fingerprint(path, content=name != "source") for name, path in assets.items()}
    for key, name in (
        ("selected_genes_sha256", "genes"),
        ("perturbation_embeddings_sha256", "embeddings"),
        ("split_config_sha256", "split"),
    ):
        if base[key] != inputs[name]["sha256"]:
            raise ValueError(f"Reference {name} changed")
    upstream = subprocess.check_output(
        ["git", "-C", str(REPO / "external/PerturbDiff"), "rev-parse", "HEAD"], text=True
    ).strip()
    if upstream != base["upstream_revision"]:
        raise ValueError("Upstream PerturbDiff revision changed")
    jobs = make_jobs(base, args.seeds)
    import h5py

    from celldiffa.benchmark.streaming import read_h5ad_obs

    targets = sorted(set(read_h5ad_obs(assets["reference"]).gene.astype(str)) - {"non-targeting"})
    with h5py.File(assets["reference"], "r") as reference_file:
        index = reference_file["var"].attrs.get("_index", "_index")
        genes = reference_file["var"][index].asstr()[:].tolist()
    candidates = list(existing_runs(REPO / "results/replogle"))
    for job in jobs:
        expected_config = {**base, **job["settings"]}
        found = [
            out
            for out, observed in candidates
            if matching(observed, expected_config)
            and reusable_prior(
                out / "training_priors.npz", inputs["source"], expected_config, genes, targets
            )
        ]
        if main in found:
            found.remove(main)
            found.insert(0, main)
        job["reuse"] = str(found[0]) if found else None
        job["reuse_evaluation"] = (
            evaluation_cache(found[0], inputs["reference"]["sha256"]) if found else None
        )
        if found:
            for name, filename in (
                ("prediction", "celldiffa_scratch.h5ad"),
                ("prior", "training_priors.npz"),
            ):
                inputs[f"reuse_{job['id']}_{name}"] = fingerprint(found[0] / filename)
    baseline = args.base_pred.resolve()
    if not baseline.is_file():
        raise FileNotFoundError(f"Biological case baseline is missing: {baseline}; use --base-pred")
    inputs["case_baseline"] = fingerprint(baseline)
    plan = dict(
        version=1,
        created_at=now(),
        repo=str(REPO),
        python=sys.executable,
        output_root=str(root),
        reference=base["reference"],
        data_root=str(data_root),
        main_run=str(main),
        base_pred=str(baseline),
        base_config=base,
        inputs=inputs,
        code={name: sha(REPO / name) for name in CODE},
        jobs=jobs,
        evaluation_input_scale=args.input_scale,
        num_threads=args.num_threads,
        max_job_hours=args.max_job_hours,
        policy="Fixed descriptive test sweeps; never select test-optimal parameters. "
        "Reuse old predictions only under matching recorded settings, validate and "
        "reevaluate them without writing into their directories. Source H5AD is "
        "bound by path/size/mtime, not a historical full-content hash. No mean adjustment.",
    )
    root.mkdir(parents=True)
    atomic(root / "plan.json", plan)
    for job in jobs:
        atomic(root / "jobs" / job["id"] / "state.json", dict(status="pending", updated=now()))
    print(
        f"Prepared {len(jobs)} jobs: {sum(bool(j['reuse']) for j in jobs)} reusable predictions, "
        f"{sum(not j['reuse'] for j in jobs)} new sampling runs. No jobs launched."
    )
    print(root / "plan.json")


def validate_plan(plan):
    if Path(plan["repo"]).resolve() != REPO:
        raise ValueError("Plan belongs to a different checkout")
    for name, stored in plan["inputs"].items():
        current = fingerprint(stored["path"], content=False)
        if any(current[k] != stored[k] for k in ("path", "size", "mtime_ns")):
            raise ValueError(f"Input changed after planning: {name}; create a new plan")
    for name, digest in plan["code"].items():
        if sha(REPO / name) != digest:
            raise ValueError(f"Code changed after planning: {name}; create a new plan")


def job_root(plan, job):
    return Path(plan["output_root"]) / "jobs" / job["id"]


def state(plan, job):
    return read(job_root(plan, job) / "state.json")


def set_state(plan, job, status, **extra):
    atomic(job_root(plan, job) / "state.json", dict(status=status, updated=now(), **extra))


def command_for(plan, job, gpu, *, smoke=False):
    s = job["settings"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("CELLDIFFA_")}
    env.update(
        PATH=str(Path(plan["python"]).parent) + os.pathsep + env.get("PATH", ""),
        PYTHONPATH=plan["repo"],
        PYTHONUNBUFFERED="1",
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS=str(plan["num_threads"]),
        MKL_NUM_THREADS=str(plan["num_threads"]),
        OPENBLAS_NUM_THREADS=str(plan["num_threads"]),
        NUMEXPR_NUM_THREADS=str(plan["num_threads"]),
        CELLDIFFA_DEVICE="cuda:0",
        CELLDIFFA_DATA_ROOT=plan["data_root"],
        CELLDIFFA_REAL_TEST=plan["reference"],
        CELLDIFFA_EVALUATION_SPLIT="test",
    )
    names = dict(
        alpha="ALPHA",
        num_particles="NUM_PARTICLES",
        top_de="TOP_DE",
        seed="SEED",
        ess_threshold="ESS_THRESHOLD",
        anchor_bandwidth="ANCHOR_BANDWIDTH",
        prior_ridge="PRIOR_RIDGE",
        prior_mode="PRIOR_MODE",
        prior_fraction="PRIOR_FRACTION",
        prior_seed="PRIOR_SEED",
        native_blocks_per_population="NATIVE_BLOCKS_PER_POPULATION",
        particle_batch_cells="PARTICLE_BATCH_CELLS",
        alignment_mode="ALIGNMENT_MODE",
        reward_unit="REWARD_UNIT",
        reward_normalization="REWARD_NORMALIZATION",
    )
    env.update({"CELLDIFFA_" + target: str(s[key]) for key, target in names.items()})
    for name, value in zip(("SIGNATURE", "DIRECTION", "ANCHOR"), s["reward_weights"]):
        env[f"CELLDIFFA_{name}_WEIGHT"] = str(value)
    output = job_root(plan, job) / ("smoke" if smoke else "prediction")
    command = [
        "bash",
        str(REPO / "scripts/baselines/run_celldiffa_replogle.sh"),
        "scratch",
        str(output),
        str(gpu),
        "0",
        "1",
        "1" if smoke else "all",
        str(REPO / "external/PerturbDiff"),
    ]
    return command, env, output


def execute(command, env, log, plan):
    """No shell interpolation/retries; clean up our child group on timeout/interruption."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(dict(start=now(), command=command)) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            command,
            env=env,
            cwd=plan["repo"],
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_record = log.with_suffix(".process.json")
        atomic(process_record, dict(status="running", pid=proc.pid, started=now(), command=command))
        started = time.monotonic()

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f"Worker received signal {signum}")

        previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while proc.poll() is None:
                if time.monotonic() - started > plan["max_job_hours"] * 3600:
                    raise TimeoutError(f"Experiment exceeded {plan['max_job_hours']} hours: {log}")
                time.sleep(2)
        except BaseException as error:
            # Only this subprocess group, created immediately above, is terminated.
            for sig in previous:
                signal.signal(sig, signal.SIG_IGN)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            atomic(
                process_record,
                dict(
                    status="timeout" if isinstance(error, TimeoutError) else "interrupted",
                    pid=proc.pid,
                    ended=now(),
                    exit_code=proc.returncode,
                ),
            )
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        atomic(
            process_record,
            dict(
                status="complete" if not proc.returncode else "failed",
                pid=proc.pid,
                ended=now(),
                exit_code=proc.returncode,
            ),
        )
        if proc.returncode:
            raise RuntimeError(f"Exit {proc.returncode}; inspect {log}")


def claim(plan, gpu):
    with locked(Path(plan["output_root"]) / ".queue.lock"):
        for job in plan["jobs"]:
            if state(plan, job)["status"] == "pending":
                set_state(plan, job, "generating", gpu=gpu, pid=os.getpid())
                return job
    return None


def gpu_worker(plan, gpu):
    validate_plan(plan)
    # Lock is shared across suites in this repo, not just one output directory.
    with locked(REPO / f"results/replogle/.appendix_gpu_{gpu}.lock", blocking=False):
        while (job := claim(plan, gpu)) is not None:
            try:
                validate_plan(plan)
                if job["reuse"]:
                    output = Path(job["reuse"])
                    observed = read(output / "shards/run_config.json")
                    if not matching(observed, {**plan["base_config"], **job["settings"]}):
                        raise ValueError("Reuse contract changed after planning")
                else:
                    command, env, smoke = command_for(plan, job, gpu, smoke=True)
                    execute(command, env, job_root(plan, job) / "smoke.log", plan)
                    if not list((smoke / "shards").glob("group_*.npz")):
                        raise RuntimeError("Smoke run produced no population shard")
                    command, env, output = command_for(plan, job, gpu)
                    execute(command, env, job_root(plan, job) / "sampling.log", plan)
                pred = output / "celldiffa_scratch.h5ad"
                if not pred.is_file():
                    raise FileNotFoundError("Sampling has not assembled a complete prediction")
                set_state(
                    plan,
                    job,
                    "ready_for_evaluation",
                    prediction=str(pred),
                    prior_cache=str(output / "training_priors.npz"),
                    reused=bool(job["reuse"]),
                )
            except Exception as error:
                set_state(plan, job, "failed_generation", error=str(error))


def validate_metrics(metrics, reference):
    import numpy as np
    import pandas as pd

    from celldiffa.benchmark.metrics import PAPER_METRIC_NAMES
    from celldiffa.benchmark.streaming import read_h5ad_obs

    table = pd.read_csv(metrics)
    expected = set(read_h5ad_obs(reference).gene.astype(str)) - {"non-targeting"}
    columns = ["R2", *PAPER_METRIC_NAMES]
    if table.perturbation.duplicated().any() or set(table.perturbation.astype(str)) != expected:
        raise ValueError("Incomplete or duplicate evaluation conditions")
    if not np.isfinite(table[columns].to_numpy(float)).all():
        raise ValueError("Undefined metrics retained; not a completed finite benchmark")
    return len(expected)


def cpu_worker(plan):
    validate_plan(plan)
    root = Path(plan["output_root"])
    _, env, _ = command_for(plan, plan["jobs"][0], "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    with locked(root / ".cpu.lock", blocking=False):
        while True:
            progress = False
            for job in plan["jobs"]:
                saved = state(plan, job)
                if saved["status"] != "ready_for_evaluation":
                    continue
                progress = True
                output = job_root(plan, job)
                set_state(
                    plan,
                    job,
                    "evaluating",
                    **{k: v for k, v in saved.items() if k not in {"status", "updated"}},
                )
                try:
                    validate_plan(plan)
                    # The full evaluator validates ordered genes, controls and every condition.
                    command = [
                        plan["python"],
                        "-u",
                        str(REPO / "scripts/baselines/evaluate.py"),
                        "--real",
                        plan["reference"],
                        "--pred",
                        saved["prediction"],
                        "--outdir",
                        str(output / "metrics"),
                        "--pert-col",
                        "gene",
                        "--control-pert",
                        "non-targeting",
                        "--num-threads",
                        str(plan["num_threads"]),
                        "--input-scale",
                        plan["evaluation_input_scale"],
                    ]
                    cache = job.get("reuse_evaluation")
                    if cache and saved["reused"]:
                        old = Path(cache["directory"])
                        if (
                            sha(saved["prediction"]) != cache["prediction_sha256"]
                            or sha(old / "perturbdiff_metrics_per_perturbation.csv")
                            != cache["metrics_sha256"]
                        ):
                            raise ValueError("Cached evaluation changed after planning")
                        (output / "metrics").mkdir(exist_ok=True)
                        for filename in (
                            "perturbdiff_metrics_per_perturbation.csv",
                            "perturbdiff_metrics_summary.csv",
                        ):
                            shutil.copy2(old / filename, output / "metrics" / filename)
                    else:
                        execute(command, env, output / "evaluation.log", plan)
                    metrics = output / "metrics/perturbdiff_metrics_per_perturbation.csv"
                    count = validate_metrics(metrics, plan["reference"])
                    execute(
                        [
                            plan["python"],
                            "-u",
                            str(REPO / "scripts/baselines/evaluate_population_diagnostics.py"),
                            "--real",
                            plan["reference"],
                            "--pred",
                            saved["prediction"],
                            "--outdir",
                            str(output / "diagnostics"),
                        ],
                        env,
                        output / "diagnostics.log",
                        plan,
                    )
                    set_state(
                        plan,
                        job,
                        "complete",
                        prediction=saved["prediction"],
                        prior_cache=saved["prior_cache"],
                        reused=saved["reused"],
                        prediction_sha256=sha(saved["prediction"]),
                        metrics_sha256=sha(metrics),
                        diagnostics_sha256=sha(output / "diagnostics/population_diagnostics.csv"),
                        reference_sha256=plan["inputs"]["reference"]["sha256"],
                        conditions=count,
                    )
                    if job["id"] == "reference":
                        run_cases(plan, env)
                except Exception as error:
                    set_state(
                        plan,
                        job,
                        "failed_evaluation",
                        error=str(error),
                        prediction=saved["prediction"],
                        prior_cache=saved["prior_cache"],
                        reused=saved["reused"],
                    )
            statuses = [state(plan, job)["status"] for job in plan["jobs"]]
            if all(s in TERMINAL for s in statuses):
                finalize(plan, env)
                return
            if not progress:
                worker_path = root / "workers.json"
                if worker_path.exists() and time.time() - worker_path.stat().st_mtime > 30:
                    workers = read(worker_path).get("workers", [])
                    if not any(
                        live(w["pid"]) for w in workers if w["kind"] == "gpu"
                    ) and not active_processes(root):
                        for job in plan["jobs"]:
                            if state(plan, job)["status"] in {"pending", "generating"}:
                                set_state(
                                    plan,
                                    job,
                                    "interrupted",
                                    error=(
                                        "No live GPU worker; inspect worker logs "
                                        "before explicit retry"
                                    ),
                                )
                time.sleep(10)


def run_cases(plan, env):
    root = Path(plan["output_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    cases_state = read(root / "cases_state.json") if (root / "cases_state.json").exists() else {}
    main = next(j for j in plan["jobs"] if j["id"] == "reference")
    saved = state(plan, main)
    if cases_state and not (
        cases_state.get("status") == "blocked" and saved["status"] == "complete"
    ):
        return  # Failed analyses are never silently retried.
    if saved["status"] == "complete":
        output = root / f"cases_{stamp}"
        try:
            execute(
                [
                    plan["python"],
                    "-u",
                    str(REPO / "scripts/baselines/analyze_adacell_cases.py"),
                    "--real",
                    plan["reference"],
                    "--base",
                    plan["base_pred"],
                    "--pred",
                    saved["prediction"],
                    "--prior-cache",
                    saved["prior_cache"],
                    "--outdir",
                    str(output),
                ],
                env,
                root / "cases.log",
                plan,
            )
            atomic(root / "cases_state.json", dict(status="complete", output=str(output)))
        except Exception as error:
            atomic(root / "cases_state.json", dict(status="failed", error=str(error)))
    elif saved["status"] in TERMINAL:
        atomic(
            root / "cases_state.json", dict(status="blocked", reason="Reference run not evaluated")
        )


def finalize(plan, env, *, include_cases=True):
    root = Path(plan["output_root"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if include_cases:
        run_cases(plan, env)
    report = root / f"report_{stamp}"
    try:
        execute(
            [
                plan["python"],
                "-u",
                str(REPO / "scripts/baselines/report_adacell_appendix.py"),
                "--plan",
                str(root / "plan.json"),
                "--outdir",
                str(report),
            ],
            env,
            root / "report.log",
            plan,
        )
        reported = read(report / "report.json").get("status")
        if reported not in {"complete", "partial"}:
            raise ValueError("Report did not produce a valid completion manifest")
        atomic(root / "report_state.json", dict(status=reported, output=str(report)))
    except Exception as error:
        atomic(root / "report_state.json", dict(status="failed", error=str(error)))
    statuses = [state(plan, j)["status"] for j in plan["jobs"]]
    case_ok = (root / "cases_state.json").exists() and read(root / "cases_state.json").get(
        "status"
    ) == "complete"
    report_ok = read(root / "report_state.json").get("status") == "complete"
    atomic(
        root / "suite_status.json",
        dict(
            status="complete"
            if all(s == "complete" for s in statuses) and case_ok and report_ok
            else "incomplete",
            finished=now(),
            completed=statuses.count("complete"),
            total=len(statuses),
        ),
    )


def live(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def active_processes(root):
    processes = []
    for path in Path(root).rglob("*.process.json"):
        saved = read(path)
        if saved.get("status") == "running" and live(saved.get("pid")):
            processes.append(saved)
    return processes


def gpu_inventory():
    lines = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return {
        int(items[0]): dict(memory=int(items[1]), utilization=int(items[2]))
        for line in lines
        if (items := line.split(","))
    }


def launch(args):
    plan = read(args.plan)
    validate_plan(plan)
    root = Path(plan["output_root"])
    with locked(root / ".launch.lock", blocking=False):
        record = read(root / "workers.json") if (root / "workers.json").exists() else {}
        if any(live(w["pid"]) for w in record.get("workers", [])):
            raise RuntimeError("This suite still has live workers; use status, do not launch twice")
        if active_processes(root):
            raise RuntimeError(
                "A previous experiment subprocess is still live; do not launch duplicate work"
            )
        stale = [
            j["id"]
            for j in plan["jobs"]
            if state(plan, j)["status"] in {"generating", "evaluating"}
        ]
        if stale:
            raise RuntimeError(
                f"Interrupted workers left states {stale}. Inspect logs and use retry explicitly."
            )
        inventory = gpu_inventory()
        gpus = sorted(inventory) if args.gpus == ["auto"] else [int(g) for g in args.gpus]
        if not gpus or len(set(gpus)) != len(gpus) or any(g not in inventory for g in gpus):
            raise ValueError("Choose distinct available GPU indices")
        if any(inventory[g]["memory"] > 1000 or inventory[g]["utilization"] > 5 for g in gpus):
            raise RuntimeError(
                "A requested GPU is busy; select idle GPUs explicitly. "
                "Existing jobs are not stopped."
            )
        if shutil.disk_usage(root).free < 20 * 1024**3:
            raise RuntimeError(
                "Less than 20 GiB free disk; preserve existing data and free space before launch"
            )
        workers = []
        # CPU worker starts first and waits for ready predictions without occupying a GPU.
        for kind, gpu in [("cpu", None)] + [("gpu", g) for g in gpus]:
            cmd = [
                plan["python"],
                str(Path(__file__).resolve()),
                "worker",
                "--plan",
                str(args.plan.resolve()),
                "--kind",
                kind,
            ]
            if gpu is not None:
                cmd += ["--gpu", str(gpu)]
            log = root / f"worker_{kind}{'' if gpu is None else gpu}.log"
            with log.open("a") as handle:
                proc = subprocess.Popen(
                    cmd,
                    cwd=plan["repo"],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    stdin=subprocess.DEVNULL,
                )
            workers.append(dict(kind=kind, gpu=gpu, pid=proc.pid, log=str(log)))
            atomic(root / "workers.json", dict(started=now(), workers=workers))
        print(json.dumps(dict(status="launched", gpus=gpus, workers=workers), indent=2))


def retry(args):
    plan = read(args.plan)
    root = Path(plan["output_root"])
    record = read(root / "workers.json") if (root / "workers.json").exists() else {}
    if any(live(w["pid"]) for w in record.get("workers", [])):
        raise RuntimeError("Workers still live; do not reset active states")
    if active_processes(root):
        raise RuntimeError("Experiment subprocesses remain live; cannot reset their states")
    known = {j["id"]: j for j in plan["jobs"]}
    if not set(args.jobs).issubset(known):
        raise ValueError("Unknown job ID")
    if any(state(plan, known[identifier])["status"] == "complete" for identifier in args.jobs):
        raise ValueError("Do not retry a completed job")
    for identifier in args.jobs:
        job = known[identifier]
        saved = state(plan, job)
        if saved["status"] == "complete":
            raise ValueError("Do not retry a completed job")
        atomic(job_root(plan, job) / f"previous_state_{time.time_ns()}.json", saved)
        if saved["status"] == "failed_evaluation":
            set_state(
                plan,
                job,
                "ready_for_evaluation",
                prediction=saved["prediction"],
                prior_cache=saved["prior_cache"],
                reused=saved["reused"],
            )
        else:
            set_state(plan, job, "pending")
    print("Explicit retry prepared. Inspect settings, then run launch.")


def retry_cases(plan):
    root = Path(plan["output_root"])
    record = read(root / "workers.json") if (root / "workers.json").exists() else {}
    if any(live(w["pid"]) for w in record.get("workers", [])) or active_processes(root):
        raise RuntimeError("Wait for workers/subprocesses before retrying case analyses")
    path = root / "cases_state.json"
    if path.exists():
        if read(path).get("status") == "complete":
            raise ValueError("Case analysis is already complete; no retry needed")
        path.rename(root / f"previous_cases_state_{time.time_ns()}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument(
        "--main-run", type=Path, default=REPO / "results/replogle/test_sensitivity/scratch_alpha1"
    )
    p.add_argument(
        "--base-pred",
        type=Path,
        default=REPO / "results/replogle/perturbdiff_scratch/predictions.h5ad",
    )
    p.add_argument("--output-root", type=Path, default=REPO / "results/replogle/appendix_v1")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument(
        "--input-scale",
        choices=["auto", "log1p"],
        default="auto",
        help="Use log1p only for verified expression units; never rescales values",
    )
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--max-job-hours", type=float, default=72)
    for name in ("launch", "worker", "status", "retry", "report"):
        q = sub.add_parser(name)
        q.add_argument("--plan", type=Path, default=REPO / "results/replogle/appendix_v1/plan.json")
        if name == "launch":
            q.add_argument("--gpus", nargs="+", default=["auto"])
        if name == "worker":
            q.add_argument("--kind", choices=["cpu", "gpu"], required=True)
            q.add_argument("--gpu", type=int, default=0)
        if name == "retry":
            q.add_argument("--jobs", nargs="+", required=True)
        if name == "report":
            q.add_argument("--retry-cases", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.num_threads < 1 or args.max_job_hours <= 0:
            parser.error("threads and timeout must be positive")
        prepare(args)
    elif args.command == "launch":
        launch(args)
    elif args.command == "retry":
        retry(args)
    else:
        plan = read(args.plan)
        if args.command == "worker":
            (cpu_worker(plan) if args.kind == "cpu" else gpu_worker(plan, args.gpu))
        elif args.command == "report":
            validate_plan(plan)
            if args.retry_cases:
                retry_cases(plan)
            _, env, _ = command_for(plan, plan["jobs"][0], "")
            workers_path = Path(plan["output_root"]) / "workers.json"
            workers = read(workers_path).get("workers", []) if workers_path.exists() else []
            finalize(plan, env, include_cases=not any(live(w["pid"]) for w in workers))
        else:
            for job in plan["jobs"]:
                saved = state(plan, job)
                print(f"{job['id']:36s} {saved['status']:24s} {saved.get('error', '')}")
            root = Path(plan["output_root"])
            for name in ("cases_state.json", "report_state.json", "suite_status.json"):
                if (root / name).exists():
                    print(name, json.dumps(read(root / name)))
            if (root / "workers.json").exists():
                for worker in read(root / "workers.json")["workers"]:
                    print(
                        "worker",
                        worker["kind"],
                        worker.get("gpu"),
                        worker["pid"],
                        "alive" if live(worker["pid"]) else "stopped",
                    )


if __name__ == "__main__":
    main()
