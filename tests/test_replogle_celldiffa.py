import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml

from celldiffa.benchmark.replogle_priors import compute_replogle_training_priors
from celldiffa.benchmark.replogle_shards import (
    assemble_replogle_shards,
    save_group_shard,
)


@pytest.mark.parametrize("evaluation_split", ["test", "validation"])
def test_replogle_priors_exclude_heldout_expression_and_center_context(tmp_path, evaluation_split):
    obs = pd.DataFrame(
        {
            "gene": ["non-targeting", "P", "non-targeting", "P"],
            "cell_line": ["A", "A", "B", "B"],
        },
        index=["a0", "a1", "b0", "b1"],
    )
    adata = ad.AnnData(X=np.zeros((4, 1)), obs=obs, var=pd.DataFrame(index=["unused"]))
    # Training P in A has shift +2. Held-out P in B is deliberately extreme;
    # it must never affect the prior. B control remains a legal condition input.
    adata.obsm["X_hvg"] = np.asarray([[1, 1], [3, 3], [10, 10], [100, 100]])
    source = tmp_path / "source.h5ad"
    adata.write_h5ad(source)
    split_path = tmp_path / "split.yaml"
    split_path.write_text(
        yaml.safe_dump(
            {
                "pert_col": "gene",
                "control_pert": "non-targeting",
                "cell_line_key": "cell_line",
                "holdout_celltype": ["B"],
                "holdout_pert": {
                    "validation": ["P"] if evaluation_split == "validation" else [],
                    "test": ["P"] if evaluation_split == "test" else [],
                },
            }
        )
    )
    cache = tmp_path / "priors.npz"
    priors = compute_replogle_training_priors(
        source,
        split_path,
        ["g0", "g1"],
        cache_path=cache,
        top_k=1,
        chunk_size=2,
        evaluation_split=evaluation_split,
    )
    np.testing.assert_allclose(priors.shifts["P"], [2, 2])
    assert priors.counts == {"P": 1}
    assert cache.exists()
    cached = compute_replogle_training_priors(
        source,
        split_path,
        ["g0", "g1"],
        cache_path=cache,
        top_k=1,
        evaluation_split=evaluation_split,
    )
    np.testing.assert_array_equal(cached.shifts["P"], priors.shifts["P"])
    other = "validation" if evaluation_split == "test" else "test"
    with pytest.raises(ValueError, match="outside the official"):
        compute_replogle_training_priors(
            source,
            split_path,
            ["g0", "g1"],
            target_perturbations=["P"],
            evaluation_split=other,
        )


def test_unobserved_test_perturbation_uses_training_only_embedding_ridge(tmp_path):
    obs = pd.DataFrame(
        {
            "gene": ["non-targeting", "P", "Q", "non-targeting", "U"],
            "cell_line": ["A", "A", "A", "B", "B"],
        },
        index=["a0", "a1", "a2", "b0", "b1"],
    )
    adata = ad.AnnData(X=np.zeros((5, 1)), obs=obs, var=pd.DataFrame(index=["unused"]))
    # U=100 is held-out truth and must not enter the ridge fit.
    adata.obsm["X_hvg"] = np.asarray([[1, 1], [3, 1], [1, 4], [10, 10], [100, 100]])
    source = tmp_path / "source.h5ad"
    adata.write_h5ad(source)
    split_path = tmp_path / "split.yaml"
    split_path.write_text(
        yaml.safe_dump(
            {
                "pert_col": "gene",
                "control_pert": "non-targeting",
                "cell_line_key": "cell_line",
                "holdout_celltype": ["B"],
                "holdout_pert": {"validation": [], "test": ["P", "U"]},
            }
        )
    )
    embeddings = {
        "P": np.asarray([1.0, 0.0]),
        "Q": np.asarray([0.0, 1.0]),
        "U": np.asarray([0.8, 0.2]),
    }
    priors = compute_replogle_training_priors(
        source,
        split_path,
        ["g0", "g1"],
        perturbation_embeddings=embeddings,
        embedding_signature="fixture-v1",
        top_k=1,
    )
    np.testing.assert_allclose(priors.shifts["P"], [2, 0])
    assert np.isfinite(priors.shifts["U"]).all()
    assert priors.counts["U"] == 0
    assert priors.sources == {
        "P": "direct_training_mean",
        "U": "genept_dual_ridge",
    }


def test_shards_only_assemble_when_cell_counts_are_complete(tmp_path):
    real = ad.AnnData(
        X=np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.float32),
        obs=pd.DataFrame({"gene": ["non-targeting", "P", "P"]}),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    real_path = tmp_path / "real.h5ad"
    output = tmp_path / "pred.h5ad"
    real.write_h5ad(real_path)
    shards = tmp_path / "shards"
    save_group_shard(shards, 0, "P", np.asarray([[3, 3]], dtype=np.float32))
    pred, status = assemble_replogle_shards(
        real_path,
        shards,
        output,
        pert_col="gene",
        control_pert="non-targeting",
        require_complete=False,
    )
    assert pred is None and not status["complete"] and not output.exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        assemble_replogle_shards(
            real_path,
            shards,
            output,
            pert_col="gene",
            control_pert="non-targeting",
        )
    save_group_shard(shards, 1, "P", np.asarray([[4, 4]], dtype=np.float32))
    pred, status = assemble_replogle_shards(
        real_path,
        shards,
        output,
        pert_col="gene",
        control_pert="non-targeting",
    )
    assert status["complete"] and output.exists()
    np.testing.assert_array_equal(pred.X[0], real.X[0])
