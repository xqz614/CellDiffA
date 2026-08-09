"""Official Linear baseline equations adapted to PerturbDiff Replogle data.

The reference R implementation is ``run_linear_pretrained_model.R`` from
``const-ae/linear_perturbation_prediction-Paper``.  It pseudobulks training
conditions, obtains both gene and perturbation embeddings from the same PCA,
and solves a two-sided ridge regression.  This module keeps those operations
while reading PerturbDiff's large H5AD in chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from .perturbdiff_split import PerturbDiffSplit
from .streaming import iter_h5ad_expression


@dataclass(frozen=True)
class LinearFit:
    gene_scores: np.ndarray
    coefficients: np.ndarray
    response_center: np.ndarray
    control_baseline: np.ndarray
    genes: tuple[str, ...]
    training_conditions: tuple[str, ...]
    pca_dim: int
    ridge_penalty: float

    def predict_means(
        self,
        perturbations: list[str],
        *,
        output_genes: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        gene_to_position = {gene: position for position, gene in enumerate(self.genes)}
        missing = sorted(set(perturbations) - set(gene_to_position))
        if missing:
            raise ValueError(
                "Linear training-data embeddings require every test perturbation to be an "
                f"expression gene; missing={missing[:20]}."
            )
        pert_scores = np.stack(
            [self.gene_scores[gene_to_position[pert]] for pert in perturbations],
            axis=1,
        )
        values = (
            self.gene_scores @ self.coefficients @ pert_scores
            + self.response_center[:, None]
            + self.control_baseline[:, None]
        )
        if output_genes is not None:
            missing_outputs = sorted(set(output_genes) - set(gene_to_position))
            if missing_outputs:
                raise ValueError(
                    "Linear output genes are absent from the fitting expression space; "
                    f"missing={missing_outputs[:20]}."
                )
            values = values[[gene_to_position[gene] for gene in output_genes]]
        return {pert: values[:, index].copy() for index, pert in enumerate(perturbations)}


def _fit_mask(obs: pd.DataFrame, split: PerturbDiffSplit, *, mode: str) -> np.ndarray:
    training = split.masks(obs, split_axis="context")["train"]
    if mode == "pooled":
        return training
    if mode == "heldout_only":
        contexts = obs[split.context_col].astype(str).to_numpy()
        return training & np.isin(contexts, split.holdout_contexts)
    raise ValueError("mode must be 'pooled' or 'heldout_only'.")


def training_pseudobulk(
    source: str | Path,
    *,
    split: PerturbDiffSplit,
    expression_key: str = "X_hvg",
    mode: str = "pooled",
    chunk_size: int = 8192,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Return condition-balanced pseudobulk using official training rows only."""
    backed = ad.read_h5ad(source, backed="r")
    try:
        obs = backed.obs.copy()
    finally:
        backed.file.close()
    required = {split.pert_col, split.context_col}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"Replogle source is missing obs columns: {sorted(missing)}")

    fit = _fit_mask(obs, split, mode=mode)
    labels = obs[split.pert_col].astype(str).to_numpy()
    conditions = sorted(set(labels[fit]))
    if split.control_pert not in conditions:
        raise ValueError(f"Training rows contain no control {split.control_pert!r}.")
    if len(conditions) < 3:
        raise ValueError("Linear baseline needs control and at least two training perturbations.")

    condition_to_position = {condition: index for index, condition in enumerate(conditions)}
    sums: np.ndarray | None = None
    counts = np.zeros(len(conditions), dtype=np.int64)
    n_expression_genes = None
    for start, stop, values in iter_h5ad_expression(
        source,
        expression_key=expression_key,
        chunk_size=chunk_size,
    ):
        n_expression_genes = values.shape[1]
        local_fit = fit[start:stop]
        if not np.any(local_fit):
            continue
        if sums is None:
            sums = np.zeros((len(conditions), values.shape[1]), dtype=np.float64)
        local_labels = labels[start:stop][local_fit]
        local_values = values[local_fit]
        for condition in np.unique(local_labels):
            selected = local_labels == condition
            position = condition_to_position[str(condition)]
            sums[position] += local_values[selected].sum(axis=0, dtype=np.float64)
            counts[position] += int(selected.sum())
    if sums is None or n_expression_genes is None:
        raise ValueError(f"No training expression was read from {source}.")
    if np.any(counts == 0):
        raise AssertionError("A discovered training condition has zero streamed rows.")

    masks = split.masks(obs, split_axis="context")
    stats = {
        "source_rows": len(obs),
        "training_rows": int(fit.sum()),
        "training_control_rows": int(np.sum(fit & (labels == split.control_pert))),
        "training_treated_rows": int(np.sum(fit & (labels != split.control_pert))),
        "training_conditions": len(conditions) - 1,
        "excluded_validation_rows": int(masks["validation"].sum()),
        "excluded_test_rows": int(masks["test"].sum()),
        "expression_genes": int(n_expression_genes),
    }
    return sums / counts[:, None], conditions, stats


def pca_scores(matrix: np.ndarray, *, n_components: int) -> np.ndarray:
    """Equivalent principal-component scores to R ``prcomp_irlba``.

    Deterministic truncated SVD solves the same leading-component objective as
    ``irlba`` without materializing a full decomposition of the 12,626-gene
    Replogle matrix. PCA signs are arbitrary and cancel in the ridge model.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("PCA input must be a two-dimensional matrix.")
    max_components = min(matrix.shape) - 1
    if n_components < 1 or n_components > max_components:
        raise ValueError(
            f"pca_dim={n_components} is invalid for shape {matrix.shape}; "
            f"choose 1..{max_components}."
        )
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    left, singular_values, _ = svds(
        centered,
        k=n_components,
        which="LM",
        v0=np.ones(min(centered.shape), dtype=np.float64),
        solver="arpack",
    )
    order = np.argsort(singular_values)[::-1]
    return left[:, order] * singular_values[order]


def fit_official_linear(
    pseudobulk: np.ndarray,
    conditions: list[str],
    genes: list[str],
    *,
    control_pert: str,
    pca_dim: int = 10,
    ridge_penalty: float = 0.1,
) -> LinearFit:
    """Fit the equations in the pinned official Linear R implementation."""
    pseudobulk = np.asarray(pseudobulk, dtype=np.float64)
    if pseudobulk.shape != (len(conditions), len(genes)):
        raise ValueError(
            f"Pseudobulk shape {pseudobulk.shape} does not match "
            f"{len(conditions)} conditions x {len(genes)} genes."
        )
    if len(set(conditions)) != len(conditions) or len(set(genes)) != len(genes):
        raise ValueError("Conditions and genes must both be unique.")
    if control_pert not in conditions:
        raise ValueError(f"Control {control_pert!r} is absent from pseudobulk.")
    if ridge_penalty < 0:
        raise ValueError("ridge_penalty must be non-negative.")

    # Official R orientation: genes are rows and training conditions columns.
    expression = pseudobulk.T
    scores = pca_scores(expression, n_components=pca_dim)
    gene_to_position = {gene: position for position, gene in enumerate(genes)}
    usable_conditions = [
        condition
        for condition in conditions
        if condition == control_pert or condition in gene_to_position
    ]
    if len(usable_conditions) <= 1:
        raise ValueError("Too few matches between training conditions and expression genes.")
    condition_to_position = {
        condition: position for position, condition in enumerate(conditions)
    }
    condition_positions = [condition_to_position[value] for value in usable_conditions]
    control_baseline = pseudobulk[condition_to_position[control_pert]]
    response = expression[:, condition_positions] - control_baseline[:, None]
    response_center = response.mean(axis=1)
    centered_response = response - response_center[:, None]

    perturbation_scores = np.stack(
        [
            np.zeros(pca_dim, dtype=np.float64)
            if condition == control_pert
            else scores[gene_to_position[condition]]
            for condition in usable_conditions
        ],
        axis=1,
    )
    identity = np.eye(pca_dim, dtype=np.float64)
    left = np.linalg.solve(
        scores.T @ scores + ridge_penalty * identity,
        scores.T @ centered_response,
    )
    coefficients = (
        left
        @ perturbation_scores.T
        @ np.linalg.solve(
            perturbation_scores @ perturbation_scores.T + ridge_penalty * identity,
            identity,
        )
    )
    coefficients[~np.isfinite(coefficients)] = 0.0
    return LinearFit(
        gene_scores=scores,
        coefficients=coefficients,
        response_center=response_center,
        control_baseline=control_baseline,
        genes=tuple(genes),
        training_conditions=tuple(usable_conditions),
        pca_dim=pca_dim,
        ridge_penalty=ridge_penalty,
    )
