"""Leakage-safe simple baselines used by PerturbDiff."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .contracts import build_prediction_anndata


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


@dataclass(frozen=True)
class MeanBaselineConfig:
    pert_col: str
    control_pert: str
    context_col: str | None = None
    batch_col: str | None = None


def _group_means(adata: ad.AnnData, columns: list[str]) -> dict[tuple[str, ...], np.ndarray]:
    obs = adata.obs[columns].astype(str)
    result: dict[tuple[str, ...], np.ndarray] = {}
    for raw_key, positions in obs.groupby(columns, observed=True, sort=False).indices.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        result[tuple(str(item) for item in key)] = _dense(adata.X[positions]).mean(axis=0)
    return result


def predict_mean_baseline(
    train: ad.AnnData,
    real_test: ad.AnnData,
    *,
    config: MeanBaselineConfig,
    variant: str = "perturbation",
) -> ad.AnnData:
    """Predict a training mean for every real test population.

    Variants reproduce the PerturbDiff appendix terminology:
    ``perturbation`` (main Mean), ``cell_type``, ``batch``, and ``overall``.
    Main Mean first averages cells within every *training* condition (including
    control) and then gives every condition equal weight. This is the exact
    implementation shipped by Cell-Eval 0.6.6 and remains defined for held-out
    perturbations.
    """
    required = {config.pert_col}
    if variant == "cell_type":
        if not config.context_col:
            raise ValueError("The cell_type variant requires context_col.")
        required.add(config.context_col)
    elif variant == "batch":
        if not config.batch_col:
            raise ValueError("The batch variant requires batch_col.")
        required.add(config.batch_col)
    elif variant not in {"perturbation", "overall"}:
        raise ValueError(f"Unknown mean baseline variant: {variant!r}.")
    for column in required:
        if column not in train.obs or column not in real_test.obs:
            raise ValueError(f"Both train and test data must contain obs[{column!r}].")
    if not train.var_names.equals(real_test.var_names):
        raise ValueError("Train and test genes/order differ.")

    pert_train = train.obs[config.pert_col].astype(str)
    treated_train = train[pert_train != config.control_pert]
    if treated_train.n_obs == 0:
        raise ValueError("Training data contains no perturbed cells.")

    if variant == "cell_type":
        group_columns = [config.context_col]  # type: ignore[list-item]
    elif variant == "batch":
        group_columns = [config.batch_col]  # type: ignore[list-item]
    else:
        group_columns = []

    means = _group_means(treated_train, group_columns) if group_columns else {}
    overall = _dense(treated_train.X).mean(axis=0)
    pert_means = _group_means(train, [config.pert_col])
    perturbation_balanced = np.stack(list(pert_means.values())).mean(axis=0)
    predictions: dict[str, np.ndarray] = {}
    test_labels = real_test.obs[config.pert_col].astype(str)

    for pert in pd.unique(test_labels):
        if pert == config.control_pert:
            continue
        positions = np.flatnonzero(test_labels.to_numpy() == pert)
        if variant == "perturbation":
            values = np.repeat(perturbation_balanced[None, :], len(positions), axis=0)
        elif variant in {"cell_type", "batch"}:
            column = group_columns[0]
            values = np.empty((len(positions), real_test.n_vars), dtype=np.float32)
            contexts = real_test.obs.iloc[positions][column].astype(str).to_numpy()
            for context in pd.unique(contexts):
                key = (str(context),)
                if key not in means:
                    raise ValueError(
                        f"Test {column}={context!r} is unseen in training; "
                        f"{variant} mean is undefined."
                    )
                values[contexts == context] = means[key]
        else:
            values = np.repeat(overall[None, :], len(positions), axis=0)
        predictions[str(pert)] = values.astype(np.float32, copy=False)

    return build_prediction_anndata(
        real_test,
        predictions,
        pert_col=config.pert_col,
        control_pert=config.control_pert,
    )
