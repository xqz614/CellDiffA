"""
Evaluation metrics for single-cell perturbation prediction.

Implements both point-estimate metrics (MSE, Pearson, R2) and distribution-level
metrics (Energy Distance, MMD, DEG Recall) following the scPerturBench framework
(Wei et al., Nature Methods 2026).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Return a finite correlation for constant or non-finite vectors."""
    x = np.asarray(x)
    y = np.asarray(y)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(pearsonr(x, y).statistic)


def _deterministic_subsample(samples: np.ndarray, max_samples: int = 200) -> np.ndarray:
    if samples.shape[0] <= max_samples:
        return samples
    indices = np.linspace(0, samples.shape[0] - 1, max_samples, dtype=int)
    return samples[indices]


# ============================================================
# Point-Estimate Metrics (Mean-level)
# ============================================================


def mse_all_genes(pred_mean: np.ndarray, true_mean: np.ndarray) -> float:
    """
    Mean Squared Error across all genes between predicted and true mean expression.

    Args:
        pred_mean: Predicted mean expression. Shape: (num_genes,)
        true_mean: True mean expression. Shape: (num_genes,)
    """
    return mean_squared_error(true_mean, pred_mean)


def pearson_all_genes(pred_mean: np.ndarray, true_mean: np.ndarray) -> float:
    """
    Pearson correlation across all genes.

    Args:
        pred_mean: Predicted mean expression. Shape: (num_genes,)
        true_mean: True mean expression. Shape: (num_genes,)
    """
    return _safe_pearson(pred_mean, true_mean)


def pearson_delta(
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    ctrl_mean: np.ndarray,
) -> float:
    """
    Pearson correlation of expression changes (deltas) relative to control.
    This is more informative than raw expression correlation as it measures
    the model's ability to capture perturbation effects specifically.

    Args:
        pred_mean: Predicted mean expression. Shape: (num_genes,)
        true_mean: True mean expression. Shape: (num_genes,)
        ctrl_mean: Control mean expression. Shape: (num_genes,)
    """
    pred_delta = pred_mean - ctrl_mean
    true_delta = true_mean - ctrl_mean
    return _safe_pearson(pred_delta, true_delta)


def mse_deg(
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    deg_indices: np.ndarray,
) -> float:
    """
    MSE computed only on differentially expressed genes.

    Args:
        pred_mean: Predicted mean expression. Shape: (num_genes,)
        true_mean: True mean expression. Shape: (num_genes,)
        deg_indices: Indices of DE genes. Shape: (K,)
    """
    return mean_squared_error(true_mean[deg_indices], pred_mean[deg_indices])


def pearson_deg(
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    deg_indices: np.ndarray,
) -> float:
    """
    Pearson correlation on DE genes only.
    """
    if len(deg_indices) < 3:
        return 0.0
    return _safe_pearson(pred_mean[deg_indices], true_mean[deg_indices])


# ============================================================
# Distribution-Level Metrics
# ============================================================


def energy_distance(
    samples_pred: np.ndarray,
    samples_true: np.ndarray,
) -> float:
    """
    Energy Distance between two empirical distributions.

    E-distance = 2 * E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]

    This is a proper metric on probability distributions that captures
    both location and spread differences.

    Args:
        samples_pred: Predicted samples. Shape: (N, G)
        samples_true: True samples. Shape: (M, G)
    """
    from scipy.spatial.distance import cdist

    samples_pred = _deterministic_subsample(samples_pred)
    samples_true = _deterministic_subsample(samples_true)

    # Cross-distribution distances
    d_xy = cdist(samples_pred, samples_true, metric="euclidean").mean()

    # Within-distribution distances
    d_xx = cdist(samples_pred, samples_pred, metric="euclidean").mean()
    d_yy = cdist(samples_true, samples_true, metric="euclidean").mean()

    return max(0.0, float(2 * d_xy - d_xx - d_yy))


def mmd_rbf(
    samples_pred: np.ndarray,
    samples_true: np.ndarray,
    bandwidth: Optional[float] = None,
) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.

    MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]

    Args:
        samples_pred: Predicted samples. Shape: (N, G)
        samples_true: True samples. Shape: (M, G)
        bandwidth: RBF kernel bandwidth. If None, uses median heuristic.
    """
    from scipy.spatial.distance import cdist

    samples_pred = _deterministic_subsample(samples_pred)
    samples_true = _deterministic_subsample(samples_true)

    # Compute pairwise distances
    d_xx = cdist(samples_pred, samples_pred, metric="sqeuclidean")
    d_yy = cdist(samples_true, samples_true, metric="sqeuclidean")
    d_xy = cdist(samples_pred, samples_true, metric="sqeuclidean")

    # Median heuristic for bandwidth
    if bandwidth is None:
        all_dists = np.concatenate([d_xx[d_xx > 0], d_yy[d_yy > 0], d_xy[d_xy > 0]])
        bandwidth = np.median(all_dists) if all_dists.size else 1.0
        if bandwidth <= 0:
            bandwidth = 1.0

    # RBF kernel
    kernel_xx = np.exp(-d_xx / (2 * bandwidth))
    kernel_yy = np.exp(-d_yy / (2 * bandwidth))
    k_xy = np.exp(-d_xy / (2 * bandwidth)).mean()
    n, m = len(samples_pred), len(samples_true)
    k_xx = (kernel_xx.sum() - np.trace(kernel_xx)) / (n * (n - 1)) if n > 1 else 0.0
    k_yy = (kernel_yy.sum() - np.trace(kernel_yy)) / (m * (m - 1)) if m > 1 else 0.0

    mmd_sq = k_xx + k_yy - 2 * k_xy
    return max(0.0, float(mmd_sq))


def deg_recall(
    pred_mean: np.ndarray,
    true_mean: np.ndarray,
    ctrl_mean: np.ndarray,
    top_k: int = 20,
) -> float:
    """
    DEG Recall: fraction of true top-K DE genes that are also in predicted top-K.

    This measures whether the model correctly identifies WHICH genes are
    most affected by the perturbation.

    Args:
        pred_mean: Predicted mean expression. Shape: (num_genes,)
        true_mean: True mean expression. Shape: (num_genes,)
        ctrl_mean: Control mean expression. Shape: (num_genes,)
        top_k: Number of top DE genes to consider.
    """
    true_delta = np.abs(true_mean - ctrl_mean)
    pred_delta = np.abs(pred_mean - ctrl_mean)

    true_top_k = set(np.argsort(true_delta)[-top_k:])
    pred_top_k = set(np.argsort(pred_delta)[-top_k:])

    if len(true_top_k) == 0:
        return 0.0

    recall = len(true_top_k & pred_top_k) / len(true_top_k)
    return recall


# ============================================================
# Comprehensive Evaluation
# ============================================================


def evaluate_perturbation(
    pred_samples: np.ndarray,
    true_samples: np.ndarray,
    ctrl_mean: np.ndarray,
    deg_indices: Optional[np.ndarray] = None,
    top_k_deg: int = 20,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a single perturbation condition.

    Args:
        pred_samples: Predicted cell expressions. Shape: (N_pred, G)
        true_samples: True cell expressions. Shape: (N_true, G)
        ctrl_mean: Control mean expression. Shape: (G,)
        deg_indices: Pre-computed DE gene indices (optional).
        top_k_deg: Number of top DE genes for recall computation.

    Returns:
        Dictionary of metric_name -> value.
    """
    pred_samples = np.asarray(pred_samples)
    true_samples = np.asarray(true_samples)
    ctrl_mean = np.asarray(ctrl_mean)
    if pred_samples.ndim != 2 or true_samples.ndim != 2:
        raise ValueError("pred_samples and true_samples must have shape (cells, genes).")
    if pred_samples.shape[0] == 0 or true_samples.shape[0] == 0:
        raise ValueError("Prediction and ground-truth populations must be non-empty.")
    if pred_samples.shape[1] != true_samples.shape[1] or ctrl_mean.shape != (
        pred_samples.shape[1],
    ):
        raise ValueError("Prediction, ground truth, and control must share the gene axis.")

    pred_mean = pred_samples.mean(axis=0)
    true_mean = true_samples.mean(axis=0)

    results = {
        "mse_all": mse_all_genes(pred_mean, true_mean),
        "pearson_all": pearson_all_genes(pred_mean, true_mean),
        "pearson_delta": pearson_delta(pred_mean, true_mean, ctrl_mean),
        "deg_recall_top20": deg_recall(pred_mean, true_mean, ctrl_mean, top_k=top_k_deg),
        "energy_distance": energy_distance(pred_samples, true_samples),
        "mmd_rbf": mmd_rbf(pred_samples, true_samples),
    }

    if deg_indices is not None and len(deg_indices) > 0:
        results["mse_deg"] = mse_deg(pred_mean, true_mean, deg_indices)
        results["pearson_deg"] = pearson_deg(pred_mean, true_mean, deg_indices)

    return results


def evaluate_all_conditions(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    ctrl_mean: np.ndarray,
    de_genes_indices: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Evaluate across all test perturbation conditions and compute aggregated metrics.

    Args:
        predictions: Dict mapping condition -> predicted samples (N, G).
        ground_truth: Dict mapping condition -> true samples (M, G).
        ctrl_mean: Control mean expression.
        de_genes_indices: Optional dict mapping condition -> DE gene indices.

    Returns:
        Tuple of (aggregated_metrics, per_condition_metrics).
    """
    missing = sorted(set(ground_truth) - set(predictions))
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} conditions: {missing[:5]}")
    if not predictions:
        raise ValueError("No predictions were provided.")

    per_condition = {}
    all_metrics_lists: Dict[str, List[float]] = {}

    for condition in predictions:
        if condition not in ground_truth:
            continue

        deg_idx = None
        if de_genes_indices and condition in de_genes_indices:
            deg_idx = de_genes_indices[condition]

        metrics = evaluate_perturbation(
            pred_samples=predictions[condition],
            true_samples=ground_truth[condition],
            ctrl_mean=ctrl_mean,
            deg_indices=deg_idx,
        )
        per_condition[condition] = metrics

        for k, v in metrics.items():
            if k not in all_metrics_lists:
                all_metrics_lists[k] = []
            if not np.isnan(v):
                all_metrics_lists[k].append(v)

    # Aggregate: mean across conditions
    aggregated = {k: np.mean(v) for k, v in all_metrics_lists.items() if len(v) > 0}

    return aggregated, per_condition
