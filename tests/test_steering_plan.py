import copy
import json
from pathlib import Path

import pytest

from scripts.baselines.prepare_replogle_steering_controls import (
    BUDGET_KEYS,
    build_plan,
    write_unchanged_or_new,
)


def spec():
    return json.loads(
        (
            Path(__file__).parents[1] / "configs/benchmark/replogle_steering_controls.json"
        ).read_text()
    )


def plan(config=None):
    return build_plan(
        config or spec(),
        repo="/tmp/example repo",
        variant="scratch",
        device="mps",
        environment="adacell-replogle",
    )


def test_commands_are_validation_only_and_do_not_truncate_groups():
    result = plan()
    assert len(result["cases"]) == 9
    assert len({x["output"] for x in result["cases"]}) == 9
    for case in result["cases"]:
        assert "CELLDIFFA_EVALUATION_SPLIT=validation" in case["command"]
        reference = Path("/tmp/example repo/results/replogle/reference/validation.h5ad").resolve()
        assert f"CELLDIFFA_REAL_TEST={reference}" in case["command"]
        assert case["command"][-2] == "all"
        for key in BUDGET_KEYS:
            assert case["settings"][key] == spec()["shared"][key]
    assert result["cases"][0]["output"].endswith("adacell_scratch_alpha1")


def test_plan_rejects_test_split_and_budget_changes():
    config = spec()
    config["evaluation_split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        plan(config)
    config = spec()
    config["cases"][1]["overrides"]["num_particles"] = 1
    with pytest.raises(ValueError, match="budget"):
        plan(config)


def test_plan_rejects_duplicate_case_names():
    config = spec()
    config["cases"].append(copy.deepcopy(config["cases"][0]))
    with pytest.raises(ValueError, match="unique"):
        plan(config)


def test_prepared_files_are_not_silently_replaced(tmp_path):
    path = tmp_path / "plan.json"
    write_unchanged_or_new(path, "first")
    write_unchanged_or_new(path, "first")
    with pytest.raises(FileExistsError):
        write_unchanged_or_new(path, "changed")
    assert path.read_text() == "first"
