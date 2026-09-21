import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.server import replogle_remaining as server


def settings(tmp_path):
    root = tmp_path / "results"
    return dict(
        repo=str(tmp_path),
        data_root=str(tmp_path / "data"),
        output_root=str(root),
        python=sys.executable,
        gpus=[0, 1, 2],
        preset="main-text-train",
        lanes=server.jobs("main-text-train"),
        squidiff_checkpoint=str(root / "runs/squidiff_training/best.pt"),
        squidiff_model_config=None,
        squidiff_unseen_policy="zero_shift",
        squidiff_sampling_steps=100,
    )


def training_files(config, *, status="training_complete", smoke=False):
    output = Path(config["squidiff_checkpoint"]).parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "best.pt").write_bytes(b"test checkpoint, not real weights")
    (output / "run_config.json").write_text("{}")
    (output / "training_progress.json").write_text(json.dumps(dict(status=status, smoke=smoke)))
    return output


def test_six_lanes_train_once_then_share_checkpoint_and_schedule(tmp_path):
    config = settings(tmp_path)
    lanes = config["lanes"]
    assert len(lanes) == 6
    assert sum(job["kind"] == "train_squidiff" for lane in lanes for job in lane) == 1
    assert all(job["kind"] not in {"train", "conditional_ddpm"} for lane in lanes for job in lane)
    assert all(job.get("seed", 42) == 42 for lane in lanes for job in lane)
    command, _, _, prediction = server.command_for(config, lanes[3][0], 0)
    assert command[command.index("--stage") + 1] == "train"
    assert command[command.index("--iterations") + 1] == "100000"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert prediction is None
    for job in (lanes[3][1], lanes[4][0]):
        assert job["requires"] == "squidiff_training"
        command, _, _, prediction = server.command_for(config, job, 1)
        assert command[command.index("--checkpoint") + 1] == config["squidiff_checkpoint"]
        assert command[command.index("--sampling-steps") + 1] == "100"
        assert command[command.index("--unseen-policy") + 1] == "zero_shift"
        assert prediction.name == "predictions.h5ad"


def test_init_requires_explicit_policy_and_binds_future_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPO", tmp_path)
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected launch"))
    argv = ["plan", "init", "--preset", "main-text-train"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        server.main()
    root = tmp_path / "results/replogle/maintext_train_v1"
    assert not (root / "plan.json").exists()
    monkeypatch.setattr(sys, "argv", argv + ["--squidiff-unseen-policy", "zero_shift"])
    server.main()
    config = json.loads((root / "plan.json").read_text())
    assert config["squidiff_checkpoint"] == str(root / "runs/squidiff_training/best.pt")
    assert not Path(config["squidiff_checkpoint"]).exists()
    assert config["squidiff_sampling_steps"] == 100


def test_early_best_checkpoint_does_not_release_waiting_jobs(tmp_path):
    config = settings(tmp_path)
    output = training_files(config, status="training")
    assert not server.training_ready(config)
    with pytest.raises(ValueError, match="not releasing"):
        server.record_training_ready(config, output)
    output = training_files(config)
    assert not server.training_ready(config)  # successful child exit is still required
    server.record_training_ready(config, output)
    assert server.training_ready(config)
    (output / "best.pt").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="changed"):
        server.training_ready(config)


def test_smoke_cannot_release_formal_prediction(tmp_path):
    config = settings(tmp_path)
    output = training_files(config, smoke=True)
    with pytest.raises(ValueError, match="not releasing"):
        server.record_training_ready(config, output)
    with pytest.raises(RuntimeError, match="training lane first"):
        server.wait_for_training(config, tmp_path / "guided", smoke=True)


def test_wait_releases_only_after_completed_training(tmp_path, monkeypatch):
    config = settings(tmp_path)
    output = training_files(config)
    guided = tmp_path / "guided"
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        assert (
            json.loads((guided / "job_status.json").read_text())["status"] == "waiting_for_training"
        )
        server.record_training_ready(config, output)

    monkeypatch.setattr(server.time, "sleep", fake_sleep)
    server.wait_for_training(config, guided)
    assert calls == [15]
    assert server.training_ready(config)


def test_recorded_training_failure_stops_wait_but_can_resume(tmp_path):
    config = settings(tmp_path)
    output = training_files(config)
    (output / "job_status.json").write_text(json.dumps(dict(status="failed")))
    with pytest.raises(RuntimeError, match="training failed"):
        server.wait_for_training(config, tmp_path / "guided")
    assert not server.training_ready(config, restarting=True)


def test_training_wait_times_out_without_starting_inference(tmp_path):
    with pytest.raises(TimeoutError, match="timed out"):
        server.wait_for_training(settings(tmp_path), tmp_path / "guided", timeout_seconds=0)


def test_train_preset_launches_without_future_weights(tmp_path, monkeypatch):
    import torch

    from celldiffa.benchmark import metrics

    config = settings(tmp_path)
    p = server.paths(config)
    required = [p[key] for key in ("source", "genes", "embeddings", "split")]
    required += [
        p["reference"] / name
        for name in ("real.h5ad", "controls.h5ad", "train.h5ad", "validation.h5ad")
    ]
    required += [
        tmp_path / "external/Squidiff/Squidiff/script_util.py",
        Path(config["data_root"])
        / "checkpoints/PerturbDiff_release_ckpt/from_scratch_replogle.ckpt",
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(server.shutil, "which", lambda _: "/usr/bin/screen")
    monkeypatch.setattr(metrics, "_require_cell_eval_066", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    launched = []

    def run(command, **kwargs):
        if command == ["screen", "-ls"]:
            return SimpleNamespace(stdout="", stderr="", returncode=1)
        launched.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(server.subprocess, "run", run)
    server.launch_maintext(config, tmp_path / "plan.json", list(range(6)))
    assert len(launched) == 6
    assert all(f"adacell-maintext-train-lane-{i}" in command for i, command in enumerate(launched))
    assert not Path(config["squidiff_checkpoint"]).exists()


def test_native_support_metrics_preserve_full_test_set(tmp_path, monkeypatch):
    import anndata as ad
    import numpy as np
    import pandas as pd

    config = settings(tmp_path)
    real = server.paths(config)["reference"] / "real.h5ad"
    real.parent.mkdir(parents=True)
    ad.AnnData(
        np.zeros((3, 2)),
        obs=pd.DataFrame({"gene": ["non-targeting", "A", "B"]}, index=["c", "a", "b"]),
    ).write_h5ad(real)
    output = tmp_path / "results/runs/squidiff_adacell16"
    output.mkdir(parents=True)
    prediction = output / "predictions.h5ad"
    prediction.write_bytes(b"prediction fixture")
    (output / "run_config.json").write_text(
        json.dumps(dict(base=dict(unsupported_native_conditions=["B"])))
    )
    metrics = tmp_path / "results/metrics" / output.name
    metrics.mkdir(parents=True)
    pd.DataFrame({"perturbation": ["A", "B"], "score": [0.2, 0.4]}).to_csv(
        metrics / "perturbdiff_metrics_per_perturbation.csv", index=False
    )
    monkeypatch.setattr(server, "execute", lambda *a, **k: None)
    server.evaluate(config, prediction, output, {})
    table = pd.read_csv(metrics / "metrics_with_native_support.csv")
    assert list(table.native_support) == ["observed_in_training", "native_unseen"]
    assert set(table.perturbation) == {"A", "B"}
    assert json.loads((output / "evaluated.json").read_text())["perturbations"] == 2
    counts = pd.read_csv(metrics / "native_support_counts.csv")
    assert counts.perturbations.sum() == 2
