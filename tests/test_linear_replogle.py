import pickle
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from celldiffa.benchmark.linear_replogle import (
    fit_official_linear,
    pca_scores,
    training_pseudobulk,
)
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from scripts.baselines.run_linear_replogle import main as run_linear


def _write_split(path):
    path.write_text(
        yaml.safe_dump(
            {
                "pert_col": "gene",
                "control_pert": "non-targeting",
                "cell_line_key": "cell_line",
                "perturbseq_batch_col": "gem_group",
                "holdout_celltype": ["hepg2"],
                "holdout_pert": {"validation": ["A"], "test": ["B"]},
            }
        ),
        encoding="utf-8",
    )


def _source(path, test_value):
    labels = ["non-targeting", "non-targeting", "A", "A", "B", "B", "C", "C"]
    contexts = ["k562", "hepg2", "k562", "hepg2", "k562", "hepg2", "hepg2", "hepg2"]
    obs = pd.DataFrame(
        {"gene": labels, "cell_line": contexts, "gem_group": ["b1"] * len(labels)},
        index=[f"cell{i}" for i in range(len(labels))],
    )
    values = np.arange(len(labels) * 6, dtype=np.float32).reshape(len(labels), 6)
    values[3] = test_value + 100  # validation expression must never affect fitting
    values[5] = test_value  # test expression must never affect fitting
    adata = ad.AnnData(
        X=np.zeros((len(labels), 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["unused"]),
    )
    adata.obsm["X_hvg"] = values
    adata.write_h5ad(path)


def test_pseudobulk_excludes_validation_and_test_expression(tmp_path):
    split_path = tmp_path / "split.yaml"
    _write_split(split_path)
    split = PerturbDiffSplit.from_yaml(split_path)
    first_path = tmp_path / "first.h5ad"
    second_path = tmp_path / "second.h5ad"
    _source(first_path, 1_000)
    _source(second_path, 1_000_000)

    first, conditions, stats = training_pseudobulk(
        first_path, split=split, chunk_size=2, mode="pooled"
    )
    second, second_conditions, _ = training_pseudobulk(
        second_path, split=split, chunk_size=3, mode="pooled"
    )

    assert conditions == second_conditions == ["A", "B", "C", "non-targeting"]
    np.testing.assert_array_equal(first, second)
    assert stats["excluded_validation_rows"] == 1
    assert stats["excluded_test_rows"] == 1
    assert stats["training_rows"] == 6


def test_official_two_sided_ridge_equations():
    genes = ["A", "B", "C", "D", "E", "F"]
    conditions = ["A", "C", "D", "non-targeting"]
    pseudobulk = np.asarray(
        [
            [2, 1, 0, 1, 4, 3],
            [0, 3, 1, 2, 2, 1],
            [4, 1, 2, 3, 0, 2],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.float64,
    )
    ridge = 0.1
    fit = fit_official_linear(
        pseudobulk,
        conditions,
        genes,
        control_pert="non-targeting",
        pca_dim=2,
        ridge_penalty=ridge,
    )

    scores = pca_scores(pseudobulk.T, n_components=2)
    pert = np.stack(
        [scores[0], scores[2], scores[3], np.zeros(2, dtype=np.float64)], axis=1
    )
    baseline = pseudobulk[-1]
    response = pseudobulk.T - baseline[:, None]
    center = response.mean(axis=1)
    response = response - center[:, None]
    identity = np.eye(2)
    expected = (
        np.linalg.solve(scores.T @ scores + ridge * identity, scores.T @ response)
        @ pert.T
        @ np.linalg.inv(pert @ pert.T + ridge * identity)
    )
    np.testing.assert_allclose(fit.coefficients, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit.response_center, center)
    predictions = fit.predict_means(["B", "E"], output_genes=["C", "F"])
    assert set(predictions) == {"B", "E"}
    assert predictions["B"].shape == (2,)


def test_test_perturbation_can_be_outside_evaluation_hvgs():
    fit = fit_official_linear(
        np.asarray(
            [[1, 2, 3, 4], [2, 4, 1, 3], [0, 1, 2, 3]],
            dtype=np.float64,
        ),
        ["target", "other", "ctrl"],
        ["target", "other", "hvg1", "hvg2"],
        control_pert="ctrl",
        pca_dim=1,
    )
    prediction = fit.predict_means(["target"], output_genes=["hvg1", "hvg2"])
    assert prediction["target"].shape == (2,)


def test_linear_rejects_non_gene_test_perturbation():
    fit = fit_official_linear(
        np.asarray([[1, 2, 3], [2, 3, 4], [0, 1, 2]], dtype=np.float64),
        ["A", "B", "ctrl"],
        ["A", "B", "C"],
        control_pert="ctrl",
        pca_dim=1,
    )
    with pytest.raises(ValueError, match="missing=.*drug-X"):
        fit.predict_means(["drug-X"])


def test_runner_fits_full_x_but_writes_only_evaluation_hvgs(tmp_path, monkeypatch):
    split_path = tmp_path / "split.yaml"
    _write_split(split_path)
    full_genes = ["A", "B", "C", "g1", "g2", "g3"]
    labels = ["non-targeting", "non-targeting", "A", "A", "B", "B", "C", "C"]
    contexts = ["k562", "hepg2", "k562", "hepg2", "k562", "hepg2", "hepg2", "hepg2"]
    values = np.arange(48, dtype=np.float32).reshape(8, 6)
    source = ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {"gene": labels, "cell_line": contexts, "gem_group": ["b1"] * 8},
            index=[f"source{i}" for i in range(8)],
        ),
        var=pd.DataFrame(index=full_genes),
    )
    source.obsm["X_hvg"] = values[:, [3, 4]]
    source_path = tmp_path / "source.h5ad"
    source.write_h5ad(source_path)

    selected_path = tmp_path / "selected.pkl"
    with selected_path.open("wb") as handle:
        pickle.dump(["g1", "g2"], handle)
    real = ad.AnnData(
        X=np.asarray([[1, 2], [3, 4]], dtype=np.float32),
        obs=pd.DataFrame(
            {"gene": ["non-targeting", "B"], "cell_line": ["hepg2", "hepg2"]},
            index=["real0", "real1"],
        ),
        var=pd.DataFrame(index=["g1", "g2"]),
    )
    real_path = tmp_path / "real.h5ad"
    real.write_h5ad(real_path)
    output_path = tmp_path / "linear.h5ad"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_linear_replogle.py",
            "--source",
            str(source_path),
            "--real-test",
            str(real_path),
            "--upstream-split-config",
            str(split_path),
            "--selected-genes",
            str(selected_path),
            "--output",
            str(output_path),
            "--pca-dim",
            "2",
        ],
    )
    run_linear()

    prediction = ad.read_h5ad(output_path)
    assert list(prediction.var_names) == ["g1", "g2"]
    assert prediction.shape == real.shape
    manifest = prediction.uns["celldiffa_linear_replogle"]
    assert manifest["fitting_genes"] == 6
    assert manifest["evaluation_genes"] == 2
