"""Input/output contracts shared by every benchmark model.

Cell-Eval compares populations, but it still requires real and predicted AnnData
objects to have identical ordered genes, identical perturbation labels, and a
control population in both files.  Keeping these checks here prevents a model
adapter from silently changing the benchmark population.
"""

from __future__ import annotations

from collections.abc import Mapping

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _as_dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def validate_prediction_pair(
    real: ad.AnnData,
    pred: ad.AnnData,
    *,
    pert_col: str,
    control_pert: str,
    require_identical_controls: bool = True,
) -> None:
    """Validate the strict PerturbDiff/Cell-Eval prediction contract."""
    if pert_col not in real.obs or pert_col not in pred.obs:
        raise ValueError(f"Both files must contain obs[{pert_col!r}].")
    if not real.var_names.equals(pred.var_names):
        raise ValueError("Real and predicted files have different genes or gene order.")

    real_perts = set(real.obs[pert_col].astype(str))
    pred_perts = set(pred.obs[pert_col].astype(str))
    if real_perts != pred_perts:
        missing = sorted(real_perts - pred_perts)
        extra = sorted(pred_perts - real_perts)
        raise ValueError(f"Perturbation mismatch; missing={missing}, extra={extra}.")
    if control_pert not in real_perts:
        raise ValueError(f"Control {control_pert!r} is absent from the real test data.")

    for pert in sorted(real_perts):
        n_real = int((real.obs[pert_col].astype(str) == pert).sum())
        n_pred = int((pred.obs[pert_col].astype(str) == pert).sum())
        if n_real != n_pred:
            raise ValueError(
                f"Cell-count mismatch for {pert!r}: real={n_real}, predicted={n_pred}."
            )

    if require_identical_controls:
        real_ctrl = real[real.obs[pert_col].astype(str) == control_pert]
        pred_ctrl = pred[pred.obs[pert_col].astype(str) == control_pert]
        if not np.array_equal(_as_dense(real_ctrl.X), _as_dense(pred_ctrl.X)):
            raise ValueError(
                "Predicted controls differ from real controls. PerturbDiff evaluation "
                "copies the same control cells into both files."
            )


def build_prediction_anndata(
    real: ad.AnnData,
    predictions: Mapping[str, np.ndarray],
    *,
    pert_col: str,
    control_pert: str,
) -> ad.AnnData:
    """Assemble model arrays into an AnnData aligned exactly to ``real`` rows.

    ``predictions`` must contain one matrix for each non-control perturbation,
    with the same number of rows as its real test population. Control rows are
    copied verbatim, as done in PerturbDiff's unified evaluation.
    """
    if pert_col not in real.obs:
        raise ValueError(f"Real data is missing obs[{pert_col!r}].")
    labels = real.obs[pert_col].astype(str).to_numpy()
    output = np.empty((real.n_obs, real.n_vars), dtype=np.float32)

    for pert in pd.unique(labels):
        positions = np.flatnonzero(labels == pert)
        if pert == control_pert:
            output[positions] = _as_dense(real.X[positions]).astype(np.float32, copy=False)
            continue
        if pert not in predictions:
            raise ValueError(f"No prediction supplied for perturbation {pert!r}.")
        values = np.asarray(predictions[pert], dtype=np.float32)
        expected = (len(positions), real.n_vars)
        if values.shape != expected:
            raise ValueError(
                f"Prediction shape for {pert!r} is {values.shape}; expected {expected}."
            )
        output[positions] = values

    extras = sorted(set(predictions) - (set(labels) - {control_pert}))
    if extras:
        raise ValueError(f"Predictions contain perturbations absent from test data: {extras}.")

    pred = ad.AnnData(
        X=output,
        obs=real.obs[[pert_col]].copy(),
        var=real.var.copy(),
    )
    pred.uns["celldiffa_prediction_contract"] = {
        "version": 1,
        "pert_col": pert_col,
        "control_pert": control_pert,
        "controls_copied_from_real": True,
    }
    validate_prediction_pair(
        real,
        pred,
        pert_col=pert_col,
        control_pert=control_pert,
    )
    return pred
