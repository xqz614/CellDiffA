import copy

import pytest

from scripts.baselines.audit_replogle_steering_budget import (
    MATCHED_SETTINGS,
    compare_runs,
    summarize_records,
)


def inputs():
    config = {key: "fixed" for key in MATCHED_SETTINGS}
    config.update(num_particles=4, start_time=2)
    record = dict(
        group=0,
        perturbation="A",
        valid_cells=3,
        padded_population_cells=32,
        ess=[4.0, 2.0],
        resampled=[False, False],
        distinct_initial_ancestors=[4, 4],
        denoised_cell_steps=256,
    )
    return config, record


def test_partial_sampling_is_not_a_full_budget_comparison():
    config, record = inputs()
    summary = summarize_records([record], config, {"A": 6})
    assert not summary["sampling_coverage_complete"]
    assert not compare_runs(config, summary, config, summary)["full_budget_match_verified"]


def test_full_sampling_metadata_and_budget_must_match():
    config, record = inputs()
    summary = summarize_records([record], config, {"A": 3})
    assert compare_runs(config, summary, config, summary)["full_budget_match_verified"]
    other = copy.deepcopy(summary)
    other["groups"][0][3] += 1
    with pytest.raises(ValueError, match="budgets differ"):
        compare_runs(config, summary, config, other)
    with pytest.raises(ValueError, match="settings"):
        compare_runs(config, summary, {**config, "seed": 999}, summary)


def test_bad_or_duplicate_diagnostics_fail():
    config, record = inputs()
    with pytest.raises(ValueError, match="model work"):
        summarize_records([{**record, "denoised_cell_steps": 255}], config, {"A": 3})
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_records([record, record], config, {"A": 6})
