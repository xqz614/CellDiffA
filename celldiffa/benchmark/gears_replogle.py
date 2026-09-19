"""Fair Replogle-to-GEARS conversion for the PerturbDiff benchmark."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .perturbdiff_split import PerturbDiffSplit
from .streaming import iter_h5ad_expression, read_h5ad_obs


def to_gears_condition(label: str, *, control_pert: str) -> str:
    """Convert a PerturbDiff single-gene label to GEARS' condition syntax."""
    label = str(label)
    if label == control_pert:
        return "ctrl"
    if not label or "+" in label:
        raise ValueError(f"GEARS-Replogle expects a single-gene label, found {label!r}.")
    return f"{label}+ctrl"


def from_gears_condition(condition: str) -> str:
    genes = [value for value in str(condition).split("+") if value != "ctrl"]
    if len(genes) != 1:
        raise ValueError(f"Expected one GEARS perturbation gene, found {condition!r}.")
    return genes[0]


def _fit_mask(
    obs: pd.DataFrame,
    split: PerturbDiffSplit,
    *,
    mode: str,
) -> np.ndarray:
    train = split.masks(obs, split_axis="context")["train"]
    if mode == "pooled":
        return train
    if mode == "heldout_only":
        contexts = obs[split.context_col].astype(str).to_numpy()
        return train & np.isin(contexts, split.holdout_contexts)
    raise ValueError("mode must be 'pooled' or 'heldout_only'.")


def materialize_training_anndata(
    source: str | Path,
    *,
    split: PerturbDiffSplit,
    selected_genes: list[str],
    expression_key: str = "X_hvg",
    mode: str = "pooled",
    chunk_size: int = 8192,
) -> tuple[ad.AnnData, dict[str, int]]:
    """Materialize only official training rows; test expression is never read."""
    source = Path(source)
    obs = read_h5ad_obs(source)
    required = {split.pert_col, split.context_col}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"Replogle source is missing obs columns: {sorted(missing)}")

    fit = _fit_mask(obs, split, mode=mode)
    if not np.any(fit):
        raise ValueError(f"No GEARS training rows remain in mode {mode!r}.")
    labels = obs[split.pert_col].astype(str).to_numpy()
    selected_obs = obs.loc[fit, [split.pert_col, split.context_col]].copy()
    selected_obs["source_condition"] = selected_obs[split.pert_col].astype(str)
    selected_obs["source_context"] = selected_obs[split.context_col].astype(str)
    selected_obs["condition"] = [
        to_gears_condition(value, control_pert=split.control_pert)
        for value in selected_obs[split.pert_col]
    ]
    # GEARS is context agnostic. Pooled mode deliberately learns a single
    # response over all official training contexts; heldout_only uses only the
    # held-out context's released training subset.
    selected_obs["cell_type"] = "pooled" if mode == "pooled" else split.holdout_contexts[0]
    selected_obs.index = selected_obs.index.astype(str)

    chunks: list[sparse.csr_matrix] = []
    n_expression_genes = None
    for start, stop, values in iter_h5ad_expression(
        source,
        expression_key=expression_key,
        chunk_size=chunk_size,
    ):
        n_expression_genes = values.shape[1]
        local = fit[start:stop]
        if np.any(local):
            # CSR substantially reduces the in-memory and GEARS cache size for
            # sparse Replogle expression while preserving exact float values.
            chunks.append(sparse.csr_matrix(values[local], dtype=np.float32))
    if n_expression_genes != len(selected_genes):
        raise ValueError(
            f"Source {expression_key} has {n_expression_genes} columns, but the selected-gene "
            f"file has {len(selected_genes)} genes."
        )
    matrix = sparse.vstack(chunks, format="csr")
    if matrix.shape[0] != len(selected_obs):
        raise AssertionError(
            f"Training expression rows ({matrix.shape[0]}) do not match metadata "
            f"rows ({len(selected_obs)})."
        )

    var = pd.DataFrame(index=pd.Index(selected_genes, name="gene"))
    var["gene_name"] = selected_genes
    training = ad.AnnData(X=matrix, obs=selected_obs, var=var)
    train_labels = labels[fit]
    n_controls = int(np.sum(train_labels == split.control_pert))
    n_treated = int(len(train_labels) - n_controls)
    if not n_controls or not n_treated:
        raise ValueError(
            f"GEARS training data needs controls and treated cells; found "
            f"controls={n_controls}, treated={n_treated}."
        )
    counts = {
        "source_rows": len(obs),
        "training_rows": int(len(train_labels)),
        "training_control_rows": n_controls,
        "training_treated_rows": n_treated,
        "training_conditions": int(len(set(train_labels) - {split.control_pert})),
        "excluded_validation_rows": int(
            split.masks(obs, split_axis="context")["validation"].sum()
        ),
        "excluded_test_rows": int(split.masks(obs, split_axis="context")["test"].sum()),
    }
    return training, counts


def validate_gene_space(
    real: ad.AnnData,
    *,
    selected_genes: list[str],
) -> None:
    if list(real.var_names.astype(str)) != selected_genes:
        raise ValueError("Selected-gene pickle order differs from the real-test H5AD.")
