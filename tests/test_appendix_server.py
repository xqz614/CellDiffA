"""CPU-only queue-contract tests; no inference, GPU access, or real results."""

import hashlib
import importlib.util
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def runner():
    path = Path(__file__).parents[1] / "scripts/server/run_replogle_appendix.py"
    spec = importlib.util.spec_from_file_location("appendix_server_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def base(runner):
    return {
        **{key: f"fixture_{key}" for key in runner.FIXED},
        "variant": "scratch",
        "evaluation_split": "test",
        "alpha": 1.0,
        "num_particles": 16,
        "top_de": 20,
        "ess_threshold": 0.5,
        "anchor_bandwidth": 1.0,
        "prior_ridge": 1.0,
        "seed": 42,
        "native_blocks_per_population": 1,
        "particle_batch_cells": 128,
        "alignment_mode": "smc",
        "reward_normalization": "zscore",
        "reward_weights": [1.0, 1.0, 1.0],
        "normalize_counts": 10.0,
        "cell_set": 32,
        "start_time": 100,
        "eta": 0.0,
        "guidance_strength": 1.0,
        "checkpoint_size": 1234,
        "checkpoint_mtime_ns": 100,
    }


def test_default_plan_has_23_unique_jobs_and_one_at_a_time_sensitivity(runner, base):
    jobs = runner.make_jobs(base, [42, 43, 44])
    assert len(jobs) == len({job["id"] for job in jobs}) == 23
    reference = jobs[0]
    assert reference["id"] == "reference"
    assert reference["settings"] == {key: runner.normal(base)[key] for key in runner.PARAMS}
    sensitivity = [job for job in jobs if job["section"] == "sensitivity"]
    assert len(sensitivity) == 12
    expected = {
        "alpha": {0.5, 2.0},
        "num_particles": {4, 8, 32},
        "top_de": {10, 50},
        "ess_threshold": {0.25, 0.75},
        "anchor_bandwidth": {0.5, 2.0},
    }
    for parameter, values in expected.items():
        selected = [job for job in sensitivity if job["parameter"] == parameter]
        assert {job["settings"][parameter] for job in selected} == values
        for job in selected:
            changed = {
                key for key in runner.PARAMS if job["settings"][key] != reference["settings"][key]
            }
            assert changed == {parameter}


def test_prior_jobs_have_full_comparator_and_matched_generation_seed(runner, base):
    jobs = runner.make_jobs(base, [42, 43, 44])
    for seed in [42, 43, 44]:
        full = [
            job
            for job in jobs
            if job["settings"]["seed"] == seed
            and job["settings"]["prior_mode"] == "full"
            and (job["id"] == "reference" or job["section"] == "prior")
        ]
        altered = [
            job
            for job in jobs
            if job["settings"]["seed"] == seed and job["settings"]["prior_mode"] != "full"
        ]
        assert len(full) == 1 and len(altered) == 3
        assert {
            (job["settings"]["prior_mode"], job["settings"]["prior_fraction"]) for job in altered
        } == {
            ("subsample", 0.5),
            ("subsample", 0.25),
            ("shuffle", 1.0),
        }
        for job in altered:
            assert job["settings"]["prior_seed"] == seed
            for key in runner.PARAMS:
                if key not in {"prior_mode", "prior_fraction", "prior_seed"}:
                    assert job["settings"][key] == full[0]["settings"][key]


@pytest.mark.parametrize("seeds", [[], [42, 42], [-1, 42], [43, 44]])
def test_invalid_seed_grid_is_rejected(runner, base, seeds):
    with pytest.raises(ValueError):
        runner.make_jobs(base, seeds)


def test_matching_accepts_legacy_default_keys_but_rejects_every_contract_change(runner, base):
    assert runner.matching(base, runner.normal(base))
    assert runner.matching({**base, "device": "cuda:0"}, {**base, "device": "cuda:1"})
    for key in (*runner.PARAMS, *runner.FIXED):
        changed = {**runner.normal(base), key: "different-value"}
        assert not runner.matching(base, changed), key


def test_command_clears_stale_environment_and_preserves_smoke_full_contract(
    runner, base, tmp_path, monkeypatch
):
    monkeypatch.setenv("CELLDIFFA_DEVICE", "cpu")
    monkeypatch.setenv("CELLDIFFA_PRIOR_MODE", "shuffle")
    monkeypatch.setenv("CELLDIFFA_REAL_TEST", "wrong-reference")
    monkeypatch.setenv("CELLDIFFA_UNRECOGNIZED_STALE_SETTING", "wrong")
    job = runner.make_jobs(base, [42, 43, 44])[0]
    plan = dict(
        output_root=str(tmp_path),
        repo=str(runner.REPO),
        python="/fixture/env/bin/python",
        num_threads=3,
        data_root="/fixture/data",
        reference="/fixture/reference.h5ad",
    )
    smoke, smoke_env, smoke_output = runner.command_for(plan, job, 2, smoke=True)
    full, env, output = runner.command_for(plan, job, 2)
    assert smoke[2:4] == ["scratch", str(smoke_output)]
    assert full[2:4] == ["scratch", str(output)]
    assert smoke[4:8] == ["2", "0", "1", "1"]
    assert full[4:8] == ["2", "0", "1", "all"]
    assert smoke_output.name == "smoke" and output.name == "prediction"
    assert smoke_env == env
    assert env["CELLDIFFA_DEVICE"] == "cuda:0" and env["CUDA_VISIBLE_DEVICES"] == "2"
    assert env["CELLDIFFA_PRIOR_MODE"] == "full"
    assert env["CELLDIFFA_REAL_TEST"] == plan["reference"]
    assert env["CELLDIFFA_TOP_DE"] == "20"
    assert env["CELLDIFFA_NUM_PARTICLES"] == "16"
    assert env["OMP_NUM_THREADS"] == "3"
    assert "CELLDIFFA_UNRECOGNIZED_STALE_SETTING" not in env
    assert env["PATH"].startswith("/fixture/env/bin:")


@pytest.fixture
def metric_fixture(tmp_path):
    from celldiffa.benchmark.metrics import PAPER_METRIC_NAMES

    reference = tmp_path / "real.h5ad"
    data = ad.AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"gene": ["non-targeting", "P", "Q"]}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    data.write_h5ad(reference)
    table = pd.DataFrame(
        {"perturbation": ["P", "Q"], **{name: [0.1, 0.2] for name in ["R2", *PAPER_METRIC_NAMES]}}
    )
    return reference, table, tmp_path / "metrics.csv"


def test_validate_metrics_requires_exact_finite_coverage(runner, metric_fixture):
    reference, table, path = metric_fixture
    table.to_csv(path, index=False)
    assert runner.validate_metrics(path, reference) == 2


@pytest.mark.parametrize(
    "bad", ["missing", "extra", "duplicate", "nan", "infinity", "missing_metric"]
)
def test_validate_metrics_rejects_incomplete_or_undefined_scores(runner, metric_fixture, bad):
    reference, table, path = metric_fixture
    if bad == "missing":
        table = table.iloc[:1]
    elif bad == "extra":
        table = pd.concat([table, table.iloc[:1].assign(perturbation="X")], ignore_index=True)
    elif bad == "duplicate":
        table = pd.concat([table, table.iloc[:1]], ignore_index=True)
    elif bad == "missing_metric":
        table = table.drop(columns="PDCorr")
    else:
        table.loc[0, "PDCorr"] = np.nan if bad == "nan" else np.inf
    table.to_csv(path, index=False)
    with pytest.raises((ValueError, KeyError)):
        runner.validate_metrics(path, reference)


def make_retry_plan(runner, tmp_path, states):
    plan = dict(
        repo=str(runner.REPO),
        output_root=str(tmp_path),
        inputs={},
        code={},
        jobs=[{"id": name} for name in states],
    )
    runner.atomic(tmp_path / "plan.json", plan)
    for job in plan["jobs"]:
        runner.atomic(runner.job_root(plan, job) / "state.json", states[job["id"]])
    return plan, SimpleNamespace(plan=tmp_path / "plan.json", jobs=list(states))


def test_retry_rejects_any_completed_job_before_mutating_other_jobs(runner, tmp_path, monkeypatch):
    plan, args = make_retry_plan(
        runner,
        tmp_path,
        {
            "failed": {"status": "failed_generation", "error": "fixture"},
            "done": {"status": "complete"},
        },
    )
    monkeypatch.setattr(runner, "live", lambda pid: False)
    before = {job["id"]: runner.state(plan, job) for job in plan["jobs"]}
    with pytest.raises(ValueError, match="completed"):
        runner.retry(args)
    assert {job["id"]: runner.state(plan, job) for job in plan["jobs"]} == before


@pytest.mark.parametrize("live_kind", ["worker", "child"])
def test_retry_refuses_live_workers_or_experiment_children(
    runner, tmp_path, monkeypatch, live_kind
):
    plan, args = make_retry_plan(runner, tmp_path, {"failed": {"status": "failed_generation"}})
    if live_kind == "worker":
        runner.atomic(tmp_path / "workers.json", {"workers": [{"pid": 123}]})
    else:
        runner.atomic(tmp_path / "sampling.process.json", {"status": "running", "pid": 123})
    monkeypatch.setattr(runner, "live", lambda pid: True)
    with pytest.raises(RuntimeError, match="live"):
        runner.retry(args)
    assert runner.state(plan, plan["jobs"][0])["status"] == "failed_generation"


def test_retry_evaluation_preserves_prediction_and_generation_retry_preserves_history(
    runner, tmp_path, monkeypatch
):
    saved = {
        "status": "failed_evaluation",
        "prediction": "/fixture/pred.h5ad",
        "prior_cache": "/fixture/prior.npz",
        "reused": True,
    }
    plan, args = make_retry_plan(
        runner,
        tmp_path,
        {"eval": saved, "generation": {"status": "failed_generation", "error": "fixture"}},
    )
    monkeypatch.setattr(runner, "live", lambda pid: False)
    runner.retry(args)
    after = runner.state(plan, plan["jobs"][0])
    assert after["status"] == "ready_for_evaluation"
    for key in ["prediction", "prior_cache", "reused"]:
        assert after[key] == saved[key]
    assert runner.state(plan, plan["jobs"][1])["status"] == "pending"
    for job in plan["jobs"]:
        assert len(list(runner.job_root(plan, job).glob("previous_state_*.json"))) == 1


@pytest.mark.parametrize(
    "report_status,expected", [("partial", "incomplete"), ("complete", "complete")]
)
def test_finalize_respects_report_manifest_not_just_subprocess_exit(
    runner, tmp_path, monkeypatch, report_status, expected
):
    plan, _ = make_retry_plan(runner, tmp_path, {"reference": {"status": "complete"}})
    plan["python"] = "/fixture/python"
    monkeypatch.setattr(
        runner,
        "run_cases",
        lambda plan, env: runner.atomic(tmp_path / "cases_state.json", {"status": "complete"}),
    )

    def successful_report(command, env, log, plan):
        output = Path(command[command.index("--outdir") + 1])
        runner.atomic(output / "report.json", {"status": report_status})

    monkeypatch.setattr(runner, "execute", successful_report)
    runner.finalize(plan, {})
    assert runner.read(tmp_path / "suite_status.json")["status"] == expected
    if report_status == "partial":
        assert runner.read(tmp_path / "report_state.json")["status"] != "complete"


@pytest.fixture
def cache_fixture(runner, base, tmp_path):
    split_text = "unit_test_split: true\n"
    source = tmp_path / "source.h5ad"
    expected = {
        **base,
        "source": str(source),
        "split_config_sha256": hashlib.sha256(split_text.encode()).hexdigest(),
    }
    genes, targets = ["g0", "g1"], ["P", "Q"]
    stat = {"size": 123, "mtime_ns": 456}
    metadata = {
        "format_version": 3,
        "source": str(source),
        "source_signature": stat,
        "genes": genes,
        "expression_key": "X_hvg",
        "top_k": 50,
        "target_perturbations": targets,
        "embedding_signature": base["perturbation_embeddings_sha256"],
        "ridge_penalty": 1.0,
        "evaluation_split": "test",
        "split_text": split_text,
    }
    return expected, genes, targets, stat, metadata, tmp_path / "cache.npz"


def save_test_cache(path, metadata, targets, shifts=None):
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(metadata)),
        perturbations=np.asarray(targets),
        shifts=np.ones((2, 2)) if shifts is None else shifts,
    )


def test_reusable_prior_accepts_legacy_default_and_exact_robustness_contract(runner, cache_fixture):
    expected, genes, targets, stat, metadata, path = cache_fixture
    save_test_cache(path, metadata, targets)
    assert runner.reusable_prior(path, stat, expected, genes, targets)
    altered = dict(prior_mode="subsample", prior_fraction=0.25, prior_seed=44)
    metadata.update(format_version=4, prior_robustness_version=1, **altered)
    save_test_cache(path, metadata, targets)
    assert runner.reusable_prior(path, stat, {**expected, **altered}, genes, targets)
    assert not runner.reusable_prior(path, stat, expected, genes, targets)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "/other/source.h5ad"),
        ("source_signature", {"size": 124, "mtime_ns": 456}),
        ("genes", ["g1", "g0"]),
        ("expression_key", "X"),
        ("top_k", 20),
        ("target_perturbations", ["P", "R"]),
        ("embedding_signature", "different"),
        ("ridge_penalty", 2.0),
        ("evaluation_split", "validation"),
        ("split_text", "different split"),
        ("prior_mode", "shuffle"),
        ("prior_fraction", 0.5),
        ("prior_seed", 43),
        ("format_version", 4),
    ],
)
def test_reusable_prior_rejects_metadata_mismatch(runner, cache_fixture, field, value):
    expected, genes, targets, stat, metadata, path = cache_fixture
    metadata[field] = value
    save_test_cache(path, metadata, targets)
    assert not runner.reusable_prior(path, stat, expected, genes, targets)


@pytest.mark.parametrize("bad", ["targets", "order", "shape", "nan"])
def test_reusable_prior_rejects_invalid_target_arrays_or_shifts(runner, cache_fixture, bad):
    expected, genes, targets, stat, metadata, path = cache_fixture
    stored_targets = (
        ["P", "R"] if bad == "targets" else (list(reversed(targets)) if bad == "order" else targets)
    )
    shifts = np.ones((2, 3)) if bad == "shape" else np.ones((2, 2))
    if bad == "nan":
        shifts[0, 0] = np.nan
    save_test_cache(path, metadata, stored_targets, shifts)
    assert not runner.reusable_prior(path, stat, expected, genes, targets)


@pytest.mark.skipif(os.name != "posix", reason="Process-group cleanup is POSIX-specific")
def test_execute_sigterm_reaps_only_its_child_and_restores_handlers(runner, tmp_path, monkeypatch):
    original_sleep = runner.time.sleep
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    interrupted = False

    def terminate_during_poll(seconds):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            signal.raise_signal(signal.SIGTERM)
        else:
            original_sleep(min(seconds, 0.01))

    monkeypatch.setattr(runner.time, "sleep", terminate_during_poll)
    plan = {"repo": str(tmp_path), "max_job_hours": 1}
    log = tmp_path / "child.log"
    with pytest.raises(KeyboardInterrupt, match="signal"):
        runner.execute(
            [sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ), log, plan
        )
    record = runner.read(log.with_suffix(".process.json"))
    assert record["status"] == "interrupted"
    assert record["exit_code"] is not None
    assert not runner.live(record["pid"])
    assert {sig: signal.getsignal(sig) for sig in previous} == previous
