import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from scripts.baselines.analyze_adacell_cases import (
    compute_case_responses,
    gene_table,
    load_prior_cache,
    main,
    response_errors,
    select_cases,
)


def example():
    obs = pd.DataFrame(
        {"gene": ["non-targeting", "non-targeting", "A", "A", "B", "B"], "cell_line": ["ctx"] * 6},
        index=[f"c{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    real = ad.AnnData(
        np.array([[1, 1, 1], [1, 1, 1], [3, 1, 1], [3, 1, 1], [1, 3, 1], [1, 3, 1]], dtype=float),
        obs=obs,
        var=var,
    )
    base = real.copy()
    base.X[2:4] = [1, 2, 1]
    pred = real.copy()
    pred.X[4:6] = [3, 1, 1]
    return real, base, pred


def test_errors_and_undefined_zero_direction():
    result = response_errors(np.array([2, 0]), np.array([0, 1]))
    assert result["angle_error_degrees"] == pytest.approx(90)
    assert result["magnitude_error"] == pytest.approx(1)
    assert result["response_mse"] == pytest.approx(2.5)
    assert np.isnan(response_errors(np.zeros(2), np.ones(2))["angle_error_degrees"])
    with pytest.raises(ValueError, match="finite"):
        response_errors(np.array([np.nan]), np.array([1]))


def test_population_comparison_not_cell_pairing():
    real, base, pred = example()
    table, responses, audit = compute_case_responses(real, base, pred)
    row = table.set_index("case_id").loc["A::ctx"]
    assert row.base_angle_error_degrees == pytest.approx(90)
    assert row.pred_angle_error_degrees == pytest.approx(0)
    assert row.improvement_magnitude_error == pytest.approx(1)
    np.testing.assert_allclose(responses["A::ctx"]["real"], [2, 0, 0])
    assert audit["context_col"] == "cell_line"
    # Reversing treated rows has no effect; cells are never compared one by one.
    permuted = pred[[0, 1, 5, 4, 3, 2]].copy()
    other, _, _ = compute_case_responses(real, base, permuted)
    pd.testing.assert_frame_equal(table, other)


def test_strict_coverage_gene_order_negative_and_controls():
    real, base, pred = example()
    with pytest.raises(ValueError, match="genes or gene order"):
        compute_case_responses(real, base, pred[:, ::-1].copy())
    invalid = pred.copy()
    invalid.X[2, 0] = -0.01
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compute_case_responses(real, base, invalid)
    with pytest.raises(ValueError, match="Cell-count mismatch"):
        compute_case_responses(real, base, pred[:-1].copy())
    invalid = pred.copy()
    invalid.X[0, 0] = 2
    with pytest.raises(ValueError, match="controls differ"):
        compute_case_responses(real, base, invalid)


def test_sparse_reference_dense_prediction_same_controls():
    real, base, pred = example()
    for data in (real, base, pred):
        data.X = (data.X + 0.37).astype(np.float32)
    real.X = sparse.csr_matrix(real.X)
    table, _, _ = compute_case_responses(real, base, pred)
    assert len(table) == 2


def test_missing_prediction_context_safe_only_for_single_context():
    real, base, pred = example()
    del pred.obs["cell_line"]
    compute_case_responses(real, base, pred)
    real.obs.loc[["c1", "c4", "c5"], "cell_line"] = "other"
    base.obs = real.obs.copy()
    with pytest.raises(ValueError, match="multi-context"):
        compute_case_responses(real, base, pred)


def test_missing_matched_control_is_not_silently_pooled():
    real, base, pred = example()
    for data in (real, base, pred):
        data.obs.loc[["c4", "c5"], "cell_line"] = "other"
    with pytest.raises(ValueError, match="No matched controls"):
        compute_case_responses(real, base, pred)


def test_selection_is_deterministic_and_includes_deterioration():
    table = pd.DataFrame(
        {
            "case_id": ["D", "A", "B", "C", "E"],
            "improvement_angle_error_degrees": [-2, 8, 4, 3, np.nan],
            "improvement_response_mse": [1, 2, 3, 4, 5],
        }
    )
    cases, rule = select_cases(table)
    assert [case["case_id"] for case in cases] == ["A", "B", "D"]
    assert rule["excluded_undefined_conditions"] == 1
    assert select_cases(table.sample(frac=1, random_state=2)) == (cases, rule)
    with pytest.raises(ValueError, match="Unknown"):
        select_cases(table, ["missing"])
    cases, rule = select_cases(table, ["E"])
    assert cases[0]["case_id"] == "E"
    assert rule["mode"] == "user_requested"


def test_no_finite_angles_falls_back_to_explicit_response_mse():
    table = pd.DataFrame(
        {
            "case_id": ["A", "B"],
            "improvement_angle_error_degrees": [np.nan, np.nan],
            "improvement_response_mse": [1, -1],
        }
    )
    cases, rule = select_cases(table)
    assert rule["selection_metric"] == "improvement_response_mse"
    assert len(cases) == 2


def test_gene_display_selection_is_common_and_reference_only():
    shifts = {
        "real": np.array([0.5, -2, 1]),
        "base": np.array([99, 0, 0]),
        "pred": np.array([100, 0, 0]),
    }
    table = gene_table(["A", "B", "C"], shifts, top_genes=2)
    assert list(table.loc[table.display_gene, "gene"]) == ["B", "C"]


def test_prior_cache_gene_order_and_provenance(tmp_path):
    path = tmp_path / "prior.npz"
    metadata = {"format_version": 3, "split_text": "split definition", "genes": ["a", "b"]}
    np.savez(
        path,
        metadata=json.dumps(metadata),
        perturbations=["P"],
        shifts=[[1, -1]],
        sources=["direct_training_mean"],
    )
    priors, _ = load_prior_cache(path, ["a", "b"], ["P"])
    np.testing.assert_allclose(priors["P"], [1, -1])
    with pytest.raises(ValueError, match="genes/order"):
        load_prior_cache(path, ["b", "a"], ["P"])


def test_audited_full_v4_prior_accepts_nondefault_seed(tmp_path):
    path = tmp_path / "prior_v4.npz"
    metadata = {
        "format_version": 4,
        "split_text": "split definition",
        "genes": ["a", "b"],
        "prior_robustness_version": 1,
        "prior_mode": "full",
        "prior_fraction": 1.0,
        "prior_seed": 43,
    }
    np.savez(
        path,
        metadata=json.dumps(metadata),
        perturbations=["P"],
        shifts=[[1, -1]],
        sources=["direct_training_mean"],
    )
    priors, observed = load_prior_cache(path, ["a", "b"], ["P"])
    np.testing.assert_allclose(priors["P"], [1, -1])
    assert observed["prior_seed"] == 43


@pytest.mark.parametrize(
    "changed",
    [
        {"prior_mode": "subsample", "prior_fraction": 0.5},
        {"prior_mode": "shuffle"},
        {"prior_fraction": 0.25},
        {"prior_robustness_version": 2},
        {"prior_seed": -1},
        {"prior_seed": 4.5},
        {"prior_seed": True},
    ],
)
def test_v4_prior_rejects_altered_or_invalid_reference(tmp_path, changed):
    path = tmp_path / "prior_v4.npz"
    metadata = {
        "format_version": 4,
        "split_text": "split definition",
        "genes": ["a", "b"],
        "prior_robustness_version": 1,
        "prior_mode": "full",
        "prior_fraction": 1.0,
        "prior_seed": 43,
        **changed,
    }
    np.savez(
        path,
        metadata=json.dumps(metadata),
        perturbations=["P"],
        shifts=[[1, -1]],
        sources=["direct_training_mean"],
    )
    with pytest.raises(ValueError, match="audited full"):
        load_prior_cache(path, ["a", "b"], ["P"])


def test_end_to_end_completion_and_no_overwrite(tmp_path):
    inputs = example()
    argv = []
    for name, data in zip(("real", "base", "pred"), inputs):
        path = tmp_path / f"{name}.h5ad"
        data.write_h5ad(path)
        argv.extend([f"--{name}", str(path)])
    output = tmp_path / "analysis"
    argv.extend(["--outdir", str(output), "--top-genes", "2"])
    main(argv)
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["status"] == "complete"
    assert (output / "case_01.pdf").stat().st_size > 100
    assert (output / "case_01.png").stat().st_size > 100
    assert pd.read_csv(output / "all_condition_errors.csv").shape[0] == 2
    with pytest.raises(SystemExit):
        main(argv)
