"""Synthetic launch/contract tests only; no scientific results or GPU workloads."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from celldiffa.benchmark.artifacts import sha256_file
from scripts.server import complete_replogle_maintext as finish
from scripts.server import replogle_remaining as remaining


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "REPO", tmp_path)
    root = tmp_path / "results/replogle/maintext_train_v1"
    config = dict(
        repo=str(tmp_path),
        output_root=str(root),
        python=sys.executable,
        preset="main-text-train",
        lanes=remaining.jobs("main-text-train"),
        gpus=[0, 1, 2],
        data_root=str(tmp_path / "data"),
    )
    plan = root / "plan.json"
    save(plan, config)
    main = tmp_path / "results/replogle/test_sensitivity/scratch_alpha1"
    save(
        main / "shards/run_config.json",
        dict(
            variant="scratch",
            evaluation_split="test",
            alpha=1,
            seed=42,
            num_particles=16,
            alignment_mode="smc",
            reward_normalization="zscore",
            reward_weights=[1, 1, 1],
        ),
    )
    (main / "celldiffa_scratch.h5ad").touch()
    (main / "metrics").mkdir()
    (main / "metrics" / finish.PER_FILE).write_text("synthetic fixture")
    return config, plan, main


def test_plan_rejects_full_preset_and_stale_paths(tmp_path, monkeypatch):
    config, plan, _ = fixture(tmp_path, monkeypatch)
    assert finish.checked_plan(plan) == config
    changed = {**config, "preset": "full", "lanes": remaining.jobs("full")}
    save(plan, changed)
    with pytest.raises(ValueError, match="extra-seed"):
        finish.checked_plan(plan)
    save(plan, {**config, "repo": "/another/machine"})
    with pytest.raises(ValueError, match="server checkout"):
        finish.checked_plan(plan)


def test_status_does_not_launch_or_create_files(tmp_path, monkeypatch, capsys):
    _, plan, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(finish.subprocess, "run", lambda *a, **k: pytest.fail("status launched"))
    monkeypatch.setattr(sys, "argv", ["finish", "status", "--plan", str(plan)])
    finish.main()
    assert "Unfinished control lanes: [0, 1, 2, 5]" in capsys.readouterr().out
    assert not (plan.parent / "maintext_completion").exists()


def test_main_run_cannot_be_replaced_with_test_best_alpha(tmp_path, monkeypatch):
    _, _, main = fixture(tmp_path, monkeypatch)
    assert finish.check_main(main).is_file()
    path = main / "shards/run_config.json"
    config = finish.read(path)
    save(path, {**config, "alpha": 2})
    with pytest.raises(ValueError, match="alpha"):
        finish.check_main(main)


def test_metrics_discovery_requires_explicit_choice_when_ambiguous(tmp_path, monkeypatch):
    _, _, main = fixture(tmp_path, monkeypatch)
    results = tmp_path / "results/replogle"
    assert finish.find_main_metrics(main, results) == main / "metrics"
    (main / "evaluation").mkdir()
    (main / "evaluation" / finish.PER_FILE).write_text("other metrics")
    with pytest.raises(ValueError, match="Several different"):
        finish.find_main_metrics(main, results)
    assert finish.find_main_metrics(main, results, main / "evaluation") == main / "evaluation"


def mark_complete(config, job):
    root = Path(config["output_root"])
    output = root / "runs" / job["id"]
    output.mkdir(parents=True, exist_ok=True)
    pred = output / ("predictions.h5ad" if job["kind"] == "mean" else "celldiffa_scratch.h5ad")
    pred.write_text("synthetic prediction")
    metrics = root / "metrics" / job["id"] / finish.PER_FILE
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("synthetic metrics")
    real = Path(config["repo"]) / "results/replogle/reference/real.h5ad"
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text("synthetic reference")
    save(
        output / "evaluated.json",
        dict(
            status="complete",
            prediction_sha256=sha256_file(pred),
            metrics_sha256=sha256_file(metrics),
            reference_sha256=sha256_file(real),
        ),
    )
    return pred


def test_only_missing_controls_resumed_and_hashes_verified(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    assert finish.pending_lanes(config) == [0, 1, 2, 5]
    for job in config["lanes"][0]:
        mark_complete(config, job)
    assert finish.pending_lanes(config, verify=True) == [1, 2, 5]
    job = config["lanes"][1][0]
    pred = mark_complete(config, job)
    pred.write_text("changed input")
    with pytest.raises(ValueError, match="hash mismatch"):
        finish.pending_lanes(config, verify=True)


def test_dry_run_never_launches_and_excludes_full_squidiff_and_extra_seeds(
    tmp_path, monkeypatch, capsys
):
    _, plan, main = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(finish.subprocess, "run", lambda *a, **k: pytest.fail("dry run launched"))
    monkeypatch.setattr(
        sys, "argv", ["finish", "launch", "--plan", str(plan), "--main-run", str(main), "--dry-run"]
    )
    finish.main()
    text = capsys.readouterr().out
    assert "--lane 0" in text and "--lane 5" in text
    assert "--lane 3" not in text and "--lane 4" not in text
    assert "--task squidiff" in text and "--task analysis" in text and "--task timing" in text
    assert not (plan.parent / "maintext_completion").exists()


def test_launch_skips_finished_workers_and_existing_screen(tmp_path, monkeypatch):
    config, plan, main = fixture(tmp_path, monkeypatch)
    for lane in finish.LANES:
        for job in config["lanes"][lane]:
            mark_complete(config, job)
    root = plan.parent / "maintext_completion"
    save(root / "analysis.status.json", dict(status="complete"))
    monkeypatch.setattr(remaining, "launch_maintext", lambda *a, **k: pytest.fail("done controls"))
    launched = []

    def run(command, **kwargs):
        if command == ["screen", "-ls"]:
            return SimpleNamespace(stdout="123.adacell-finish-timing", stderr="")
        launched.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(finish.subprocess, "run", run)
    monkeypatch.setattr(
        sys, "argv", ["finish", "launch", "--plan", str(plan), "--main-run", str(main)]
    )
    finish.main()
    assert len(launched) == 1 and "adacell-finish-squidiff" in launched[0]
    assert "--lane" not in launched[0]


def test_gpu_idle_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(finish.subprocess, "check_output", lambda *a, **k: "1234\n")
    assert not finish.gpu_idle(2)
    monkeypatch.setattr(finish.subprocess, "check_output", lambda *a, **k: "")
    assert finish.gpu_idle(2)


def test_completed_native_diagnostic_is_not_full_benchmark_approval(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    checkpoint = tmp_path / "squidiff/best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_text("fixture")
    model_config = checkpoint.parent / "run_config.json"
    save(model_config, dict())
    config["squidiff_checkpoint"] = str(checkpoint)
    report = tmp_path / "results/replogle/diagnostics/squidiff_sampling_fixture/report.json"
    save(
        report,
        dict(
            status="diagnostic_complete",
            checkpoint_sha256=sha256_file(checkpoint),
            model_config_sha256=sha256_file(model_config),
            cases=[],
        ),
    )
    assert finish.matching_diagnostic(config) == report
    checkpoint.write_text("different")
    assert finish.matching_diagnostic(config) is None


def test_known_control_log_units_are_audited_but_squidiff_is_not_exempted(tmp_path, monkeypatch):
    from scripts.server import recover_replogle_ablation_metrics as recovery

    config, _, _ = fixture(tmp_path, monkeypatch)
    checked = []
    monkeypatch.setattr(recovery, "check_tail", lambda path: checked.append(path))
    pred = tmp_path / "predictions.h5ad"
    assert remaining.evaluation_scale_arguments(config, "scratch_random16", pred) == [
        "--input-scale",
        "log1p",
    ]
    assert checked == [pred]
    assert remaining.evaluation_scale_arguments(config, "squidiff_vanilla", pred) == []
    assert checked == [pred]

    def reject(path):
        raise ValueError("systemic scale anomaly")

    monkeypatch.setattr(recovery, "check_tail", reject)
    with pytest.raises(ValueError, match="scale anomaly"):
        remaining.evaluation_scale_arguments(config, "scratch_mean_correction", pred)
