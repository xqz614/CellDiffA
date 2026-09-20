import copy
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from celldiffa.benchmark.perturbdiff_covariates import (
    align_checkpoint_covariates,
    validate_checkpoint_covariates,
)
from scripts.baselines.perturbdiff_sampling_entrypoint import patch_covariate_paths


def test_runtime_paths_replace_cluster_paths_without_mutating_checkpoint_config():
    checkpoint_cfg = {
        "celltype_encoding": "onehot",
        "celltype_embedding_path": "/projects/authors/celltype.pkl",
        "gene_embedding_path": ["/projects/authors/gene.pkl"],
        "pert_embedding_path": "/projects/authors/pert.pkl",
        "drug_embedding_path": "/projects/authors/drug.pkl",
        "replogle_gene_embedding_path": "/projects/authors/replogle.pkl",
        "hidden_dim": 128,
    }
    runtime_cfg = SimpleNamespace(
        celltype_encoding="llm",
        get=lambda key, default=None: {
            "celltype_embedding_path": "/data/celltype.pkl",
            "gene_embedding_path": ["/data/gene.pkl"],
            "pert_embedding_path": "/data/pert.pkl",
            "drug_embedding_path": "/data/drug.pkl",
            "replogle_gene_embedding_path": "/data/replogle.pkl",
        }.get(key, default),
    )

    patched = patch_covariate_paths(checkpoint_cfg, runtime_cfg)

    assert patched["celltype_encoding"] == "llm"
    assert patched["celltype_embedding_path"] == "/data/celltype.pkl"
    assert patched["gene_embedding_path"] == ["/data/gene.pkl"]
    assert patched["replogle_gene_embedding_path"] == "/data/replogle.pkl"
    assert patched["hidden_dim"] == 128
    assert checkpoint_cfg["replogle_gene_embedding_path"] == "/projects/authors/replogle.pkl"


def _category_fixture():
    saved = OmegaConf.create(
        {
            "pert_dict": {"A": 0, "B": 1},
            "num_pert": 2,
            "cell_type_dict": {"B cell": 0, "k562": 1},
            "num_celltype": 2,
            "batch_dict": {"PBMC": 0, "replogle_1": 1},
            "num_batch": 2,
        }
    )
    runtime = OmegaConf.create(
        {
            "cov_encoding": {
                "pert_dict": {"B": 0},
                "num_pert": 1,
                "cell_type_dict": {"k562": 0},
                "num_celltype": 1,
                "batch_dict": {"replogle_1": 0},
                "num_batch": 1,
            }
        }
    )
    dm = SimpleNamespace(
        all_split_names=["validation", "test"],
        **{
            key: dict(runtime.cov_encoding[key])
            for key in ("pert_dict", "cell_type_dict", "batch_dict")
        },
    )
    return saved, runtime, dm


def test_subset_categories_use_saved_embedding_rows_and_preserve_checkpoint():
    saved, runtime, dm = _category_fixture()
    original = copy.deepcopy(saved)
    report = align_checkpoint_covariates(runtime, dm, saved)
    assert dm.cell_type_dict["k562"] == 1
    assert dm.batch_dict["replogle_1"] == 1
    assert dm.pert_dict["B"] == 1
    assert runtime.cov_encoding == saved
    assert saved == original
    assert report["cell_type_dict"]["remapped_categories"] == 1
    validate_checkpoint_covariates(dm, saved)


def test_already_aligned_scratch_vocabulary_is_unchanged():
    saved, runtime, dm = _category_fixture()
    align_checkpoint_covariates(runtime, dm, saved)
    before = copy.deepcopy(runtime)
    report = align_checkpoint_covariates(runtime, dm, saved)
    assert runtime == before
    assert all(
        report[key]["remapped_categories"] == 0
        for key in ("pert_dict", "cell_type_dict", "batch_dict")
    )


@pytest.mark.parametrize("key", ["pert_dict", "cell_type_dict", "batch_dict"])
def test_missing_checkpoint_category_errors_without_partial_mutation(key):
    saved, runtime, dm = _category_fixture()
    getattr(dm, key)["unknown"] = 42
    before_cfg, before_dm = copy.deepcopy(runtime), copy.deepcopy(dm.__dict__)
    with pytest.raises(ValueError, match="missing runtime categories"):
        align_checkpoint_covariates(runtime, dm, saved)
    assert runtime == before_cfg
    assert dm.__dict__ == before_dm


@pytest.mark.parametrize(
    "bad_ids",
    [
        {"PBMC": 0, "replogle_1": 0},
        {"PBMC": -1, "replogle_1": 1},
        {"PBMC": 0, "replogle_1": 4},
        {"PBMC": 0, "replogle_1": "1"},
    ],
)
def test_malformed_saved_vocabulary_fails(bad_ids):
    saved, runtime, dm = _category_fixture()
    saved.batch_dict = bad_ids
    with pytest.raises(ValueError, match="unique contiguous integer IDs"):
        align_checkpoint_covariates(runtime, dm, saved)


def test_saved_table_size_must_match_vocabulary():
    saved, runtime, dm = _category_fixture()
    saved.num_celltype = 1
    with pytest.raises(ValueError, match="num_celltype"):
        align_checkpoint_covariates(runtime, dm, saved)


def test_alignment_must_precede_dataset_construction():
    saved, runtime, dm = _category_fixture()
    dm.test_dataset = object()
    with pytest.raises(ValueError, match="before building datasets"):
        align_checkpoint_covariates(runtime, dm, saved)


def test_preflight_rejects_original_misalignment_and_stale_dataset_cache():
    saved, runtime, dm = _category_fixture()
    with pytest.raises(ValueError, match="does not match checkpoint"):
        validate_checkpoint_covariates(dm, saved)
    align_checkpoint_covariates(runtime, dm, saved)
    cache = SimpleNamespace(
        **{key: dict(getattr(dm, key)) for key in ("pert_dict", "cell_type_dict", "batch_dict")}
    )
    dm.test_dataset = SimpleNamespace(meta_cache=cache)
    validate_checkpoint_covariates(dm, saved)
    cache.cell_type_dict = {"k562": 0}
    with pytest.raises(ValueError, match="stale cell_type_dict"):
        validate_checkpoint_covariates(dm, saved)
