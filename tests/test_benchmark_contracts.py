import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from celldiffa.benchmark.contracts import build_prediction_anndata, validate_prediction_pair
from celldiffa.benchmark.metrics import cellflow_r2
from celldiffa.benchmark.simple_baselines import MeanBaselineConfig, predict_mean_baseline


def _adata(values, labels, contexts=None):
    obs = pd.DataFrame({"pert": labels})
    if contexts is not None:
        obs["cell_type"] = contexts
    return ad.AnnData(
        X=np.asarray(values, dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2"]),
    )


def test_prediction_contract_copies_controls_and_preserves_rows():
    real = _adata([[1, 2], [3, 4], [5, 6]], ["ctrl", "drug", "drug"])
    pred = build_prediction_anndata(
        real,
        {"drug": np.array([[7, 8], [9, 10]])},
        pert_col="pert",
        control_pert="ctrl",
    )
    np.testing.assert_array_equal(pred.X[0], real.X[0])
    np.testing.assert_array_equal(pred.X[1:], [[7, 8], [9, 10]])
    validate_prediction_pair(real, pred, pert_col="pert", control_pert="ctrl")


def test_contract_rejects_wrong_cell_count():
    real = _adata([[1, 2], [3, 4], [5, 6]], ["ctrl", "drug", "drug"])
    with pytest.raises(ValueError, match="Prediction shape"):
        build_prediction_anndata(
            real,
            {"drug": np.array([[7, 8]])},
            pert_col="pert",
            control_pert="ctrl",
        )


def test_cellflow_r2_matches_official_formula():
    real = np.array([[1, 2, 4], [3, 6, 8]], dtype=float)
    pred = np.array([[2, 1, 3], [4, 5, 9]], dtype=float)
    expected = r2_score(real.mean(axis=0), pred.mean(axis=0))
    assert cellflow_r2(real, pred) == pytest.approx(expected)


def test_main_mean_is_equal_weighted_across_training_perturbations():
    # Perturbation A has two cells and B has one. Main Mean should be the mean
    # of condition centroids including control:
    # ([0, 0] + [2, 2] + [10, 10]) / 3 = [4, 4], not a cell mean.
    train = _adata(
        [[0, 0], [1, 1], [3, 3], [10, 10]],
        ["ctrl", "A", "A", "B"],
    )
    real = _adata([[0, 0], [7, 7], [8, 8]], ["ctrl", "heldout", "heldout"])
    pred = predict_mean_baseline(
        train,
        real,
        config=MeanBaselineConfig(pert_col="pert", control_pert="ctrl"),
        variant="perturbation",
    )
    np.testing.assert_allclose(pred.X[1:], 4.0)


def test_context_mean_uses_test_context_without_test_expression():
    train = _adata(
        [[0, 0], [2, 4], [10, 20]],
        ["ctrl", "A", "B"],
        ["c1", "c1", "c2"],
    )
    real = _adata(
        [[0, 0], [100, 100], [200, 200]],
        ["ctrl", "heldout", "heldout"],
        ["c1", "c1", "c2"],
    )
    pred = predict_mean_baseline(
        train,
        real,
        config=MeanBaselineConfig(
            pert_col="pert", control_pert="ctrl", context_col="cell_type"
        ),
        variant="cell_type",
    )
    np.testing.assert_array_equal(pred.X[1:], [[2, 4], [10, 20]])
