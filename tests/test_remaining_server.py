import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.server import replogle_remaining as server


def config(tmp_path):
    return dict(
        repo=str(tmp_path),
        data_root=str(tmp_path / "data"),
        output_root=str(tmp_path / "results"),
        python=sys.executable,
        gpus=[0, 1, 2],
        lanes=server.jobs(),
        squidiff_checkpoint=str(tmp_path / "squidiff/best.pt"),
        squidiff_model_config=None,
        squidiff_unseen_policy="auto",
    )


def test_six_lanes_are_disjoint_and_training_precedes_predictions(tmp_path):
    settings = config(tmp_path)
    assert len(settings["lanes"]) == 6
    names = [job["id"] for lane in settings["lanes"] for job in lane]
    assert len(names) == len(set(names))
    assert settings["lanes"][4][0]["kind"] == "train"
    for seed in (43, 44):
        matching = [job for job in settings["lanes"][5] if job.get("seed") == seed]
        assert {job["mode"] for job in matching} == {"random", "smc"}
    for lane, jobs in enumerate(settings["lanes"]):
        for job in jobs:
            command, env, output, prediction = server.command_for(settings, job, lane % 3)
            assert output.name == job["id"]
            assert "nohup" not in command
            if job["kind"] == "perturbdiff":
                assert command[-1] == "all"
                assert env["CELLDIFFA_ALPHA"] == "1"
                assert env["CELLDIFFA_NATIVE_BLOCKS_PER_POPULATION"] == "16"
            if job["kind"] in {"squidiff", "conditional_ddpm"}:
                assert command[command.index("--sampling-steps") + 1] == "100"
                assert prediction.name == "predictions.h5ad"


def test_unbound_squidiff_never_launches_or_uses_random_weights(tmp_path):
    settings = config(tmp_path)
    settings["squidiff_checkpoint"] = None
    with pytest.raises(ValueError, match="configure-squidiff"):
        server.command_for(settings, settings["lanes"][2][0], 0)
    assert server.command_for(settings, settings["lanes"][0][0], 0)[0]


def test_existing_squidiff_sampling_schedule_can_be_preserved(tmp_path):
    settings = config(tmp_path)
    settings["squidiff_sampling_steps"] = 1000
    for lane in settings["lanes"]:
        for job in lane:
            if job["kind"] not in {"squidiff", "conditional_ddpm"}:
                continue
            command, _, _, _ = server.command_for(settings, job, 0)
            expected = "1000" if job["kind"] == "squidiff" else "100"
            assert command[command.index("--sampling-steps") + 1] == expected


def test_squidiff_schedule_binding_does_not_launch_other_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPO", tmp_path)
    settings = config(tmp_path)
    settings.update(preset="main-text", lanes=server.jobs("main-text"))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(settings))
    checkpoint = Path(settings["squidiff_checkpoint"])
    checkpoint.parent.mkdir()
    checkpoint.touch()
    monkeypatch.setattr(
        server.subprocess, "run", lambda *a, **k: pytest.fail("Launched from binding")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plan",
            "configure-squidiff",
            "--config",
            str(path),
            "--checkpoint",
            str(checkpoint),
            "--sampling-steps",
            "1000",
        ],
    )
    server.main()
    updated = json.loads(path.read_text())
    assert updated["squidiff_sampling_steps"] == 1000
    assert updated["lanes"] == server.jobs("main-text")


def test_environment_cleans_stale_experimental_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CELLDIFFA_ALPHA", "999")
    monkeypatch.setenv("CELLDIFFA_REWARD_UNIT", "cell")
    env, gpu = server.environment(config(tmp_path), 5)
    assert gpu == 2 and env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "CELLDIFFA_ALPHA" not in env
    assert "CELLDIFFA_REWARD_UNIT" not in env
    assert env["CELLDIFFA_EVALUATION_SPLIT"] == "test"


def test_init_is_nonlaunching_and_existing_plan_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPO", tmp_path)
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: pytest.fail("Launched from init"))
    monkeypatch.setattr(sys, "argv", ["plan", "init"])
    server.main()
    path = tmp_path / "results/replogle/remaining_v1/plan.json"
    original = path.read_bytes()
    assert json.loads(original)["squidiff_checkpoint"] is None
    monkeypatch.setattr(sys, "argv", ["plan", "init", "--gpus", "0"])
    with pytest.raises(ValueError, match="different settings"):
        server.main()
    assert path.read_bytes() == original


def test_smoke_uses_separate_outputs_and_partial_coverage(tmp_path):
    settings = config(tmp_path)
    command, _, output, _ = server.command_for(settings, settings["lanes"][2][0], 2, smoke=True)
    assert output.parent.name == "smoke"
    assert command[command.index("--max-groups") + 1] == "1"
    pd_command, _, pd_output, _ = server.command_for(
        settings, settings["lanes"][0][0], 0, smoke=True
    )
    assert pd_output.parent.name == "smoke" and pd_command[-1] == "1"


def test_maintext_preset_has_six_independent_jobs_without_training_or_extra_seeds():
    lanes = server.jobs("main-text")
    assert [lane[0]["id"] for lane in lanes] == [
        "scratch_random16",
        "scratch_best16",
        "scratch_cellwise16",
        "squidiff_vanilla",
        "squidiff_adacell16",
        "scratch_particles8",
    ]
    assert lanes[0][1] == dict(id="scratch_mean_correction", kind="mean", parent="scratch_random16")
    assert all(len(lane) == 1 for lane in lanes[1:])
    assert all(job.get("seed", 42) == 42 for lane in lanes for job in lane)
    assert all(job["kind"] != "train" for lane in lanes for job in lane)
    with pytest.raises(ValueError, match="Unknown"):
        server.jobs("unexpected")


def test_maintext_plan_is_separate_from_old_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "REPO", tmp_path)
    monkeypatch.setattr(server.subprocess, "run", lambda *a, **k: pytest.fail("Launched from init"))
    monkeypatch.setattr(sys, "argv", ["plan", "init"])
    server.main()
    old = tmp_path / "results/replogle/remaining_v1/plan.json"
    previous = old.read_bytes()
    monkeypatch.setattr(sys, "argv", ["plan", "init", "--preset", "main-text"])
    server.main()
    new = json.loads((tmp_path / "results/replogle/maintext_v1/plan.json").read_text())
    assert new["preset"] == "main-text" and new["lanes"] == server.jobs("main-text")
    assert old.read_bytes() == previous


def test_maintext_launch_dry_run_and_old_plan_rejection(tmp_path, monkeypatch, capsys):
    settings = config(tmp_path)
    monkeypatch.setattr(
        server.subprocess, "run", lambda *a, **k: pytest.fail("Launched from dry run")
    )
    with pytest.raises(ValueError, match="requires"):
        server.launch_maintext(settings, tmp_path / "plan.json", list(range(6)), dry_run=True)
    settings.update(preset="main-text", lanes=server.jobs("main-text"))
    server.launch_maintext(settings, tmp_path / "plan.json", list(range(6)), dry_run=True)
    output = capsys.readouterr().out
    assert output.count("screen -L") == 6
    assert "adacell-maintext-lane-5" in output and "--config" in output


def test_maintext_launch_preflights_before_start_and_skips_duplicate_sessions(
    tmp_path, monkeypatch
):
    import torch

    from celldiffa.benchmark import metrics

    settings = config(tmp_path)
    settings.update(preset="main-text", lanes=server.jobs("main-text"))
    monkeypatch.setattr(server.shutil, "which", lambda _: "/usr/bin/screen")
    listing = SimpleNamespace(stdout="", stderr="", returncode=1)
    launched = []

    def run(command, **kwargs):
        if command == ["screen", "-ls"]:
            return listing
        launched.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(server.subprocess, "run", run)
    monkeypatch.setattr(metrics, "_require_cell_eval_066", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    with pytest.raises(FileNotFoundError, match="missing"):
        server.launch_maintext(settings, tmp_path / "plan.json", list(range(6)))
    assert not launched
    p = server.paths(settings)
    required = [p[key] for key in ("source", "genes", "embeddings", "split")]
    required += [p["reference"] / name for name in ("real.h5ad", "controls.h5ad", "train.h5ad")]
    checkpoint = Path(settings["squidiff_checkpoint"])
    required += [checkpoint, checkpoint.parent / "run_config.json"]
    required += [
        Path(settings["data_root"])
        / "checkpoints/PerturbDiff_release_ckpt/from_scratch_replogle.ckpt"
    ]
    for path in required:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    listing.stdout = "123.adacell-maintext-lane-2\t(Detached)"
    server.launch_maintext(settings, tmp_path / "plan.json", list(range(6)))
    assert len(launched) == 5
    assert all("adacell-maintext-lane-2" not in command for command in launched)
    launched.clear()
    listing.stdout = "123.adacell-lane-5\t(Detached)"
    with pytest.raises(RuntimeError, match="older"):
        server.launch_maintext(settings, tmp_path / "plan.json", list(range(6)))
    assert not launched
