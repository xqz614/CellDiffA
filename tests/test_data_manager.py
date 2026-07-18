"""Tests for leakage-resistant perturbation splits and metadata handling."""

import numpy as np
import pandas as pd
from anndata import AnnData

from data.data_manager import PerturbationDataManager


def make_manager(tmp_path, conditions):
    obs = pd.DataFrame(
        {
            "condition": conditions,
            "is_control": ["true" if value == "ctrl" else "false" for value in conditions],
        },
        index=[f"cell_{index}" for index in range(len(conditions))],
    )
    adata = AnnData(
        X=np.zeros((len(conditions), 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["A", "B", "C"]),
    )
    manager = PerturbationDataManager(data_root=str(tmp_path), n_top_genes=3)
    manager.adata = adata
    manager._standardize_metadata()
    manager.ctrl_adata = manager.adata[manager.adata.obs["is_control"]].copy()
    return manager


def test_additive_condition_folds_are_disjoint(tmp_path):
    combinations = [f"g{i}+g{i + 1}" for i in range(10)]
    manager = make_manager(tmp_path, ["ctrl", *combinations])
    folds = [set(manager._generate_split("additive", fold, 5)["test"]) for fold in range(5)]
    assert set.union(*folds) == set(combinations)
    assert sum(len(fold) for fold in folds) == len(set.union(*folds))


def test_unseen_split_supports_single_perturbation_datasets(tmp_path):
    conditions = ["ctrl", "A+ctrl", "B+ctrl", "C+ctrl", "D+ctrl"]
    manager = make_manager(tmp_path, conditions)
    split = manager._generate_split("unseen", fold=0, n_folds=2)
    removed = set(split["removed_genes"])
    assert removed
    for condition in split["test"]:
        genes = set(condition.split("+")) - {"ctrl", "control"}
        assert genes & removed


def test_string_control_flags_are_not_cast_by_truthiness(tmp_path):
    manager = make_manager(tmp_path, ["ctrl", "A+ctrl", "B+ctrl"])
    assert manager.adata.obs["is_control"].tolist() == [True, False, False]


def test_preprocessing_modes_have_distinct_cache_tags(tmp_path):
    log_manager = PerturbationDataManager(data_root=str(tmp_path / "log"), already_normalized=True)
    raw_manager = PerturbationDataManager(data_root=str(tmp_path / "raw"), already_normalized=False)
    assert log_manager._preprocessing_tag != raw_manager._preprocessing_tag
