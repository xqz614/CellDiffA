import json
import sys

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
