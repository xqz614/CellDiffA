"""Utility functions for the SMC engine."""

import torch
import numpy as np
from typing import Dict, List


def compute_ess(log_weights: torch.Tensor) -> float:
    """
    Compute Effective Sample Size (ESS) from log-weights.

    ESS = 1 / sum(w_i^2) where w_i are normalized weights.
    A low ESS indicates particle degeneracy.

    Args:
        log_weights: Unnormalized log importance weights. Shape: (N,)

    Returns:
        ESS value (between 1 and N).
    """
    log_w_norm = log_weights - torch.logsumexp(log_weights, dim=0)
    weights = torch.exp(log_w_norm)
    return (1.0 / (weights ** 2).sum()).item()


def log_mean_exp(log_values: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Numerically stable computation of log(mean(exp(x))).

    Args:
        log_values: Log-space values.
        dim: Dimension to reduce.

    Returns:
        log(mean(exp(log_values))) along dim.
    """
    n = log_values.shape[dim]
    return torch.logsumexp(log_values, dim=dim) - np.log(n)


def build_reward_from_config(
    config: Dict,
    de_genes: Dict,
    shifts: Dict,
    ctrl_mean: np.ndarray,
    gene_names: List[str],
):
    """
    Factory function to build a CompositeReward from a YAML config dict.

    Args:
        config: Reward configuration dictionary with keys like:
            {
                "rewards": [
                    {"type": "transcriptomic", "weight": 1.0, "top_k": 20},
                    {"type": "geometric", "weight": 0.5},
                    {"type": "anchor", "weight": 0.3, "max_distance": 5.0},
                ],
                "aggregation": "linear"
            }
        de_genes: DE genes dict from DataManager.
        shifts: Perturbation shifts dict from DataManager.
        ctrl_mean: Control mean expression.
        gene_names: Gene name list.

    Returns:
        CompositeReward instance.
    """
    from celldiffa.rewards import (
        CompositeReward,
        TranscriptomicReward,
        GeometricReward,
        AnchorReward,
    )

    reward_instances = []
    for r_cfg in config.get("rewards", []):
        r_type = r_cfg["type"]
        weight = r_cfg.get("weight", 1.0)

        if r_type == "transcriptomic":
            reward_instances.append(
                TranscriptomicReward(
                    de_genes=de_genes,
                    perturbation_shifts=shifts,
                    ctrl_mean=ctrl_mean,
                    gene_names=gene_names,
                    weight=weight,
                    top_k=r_cfg.get("top_k", 20),
                    combination_mode=r_cfg.get("combination_mode", "union"),
                )
            )
        elif r_type == "geometric":
            reward_instances.append(
                GeometricReward(
                    perturbation_shifts=shifts,
                    ctrl_mean=ctrl_mean,
                    weight=weight,
                )
            )
        elif r_type == "anchor":
            reward_instances.append(
                AnchorReward(
                    ctrl_mean=ctrl_mean,
                    weight=weight,
                    max_distance=r_cfg.get("max_distance", None),
                )
            )
        else:
            raise ValueError(f"Unknown reward type: {r_type}")

    aggregation = config.get("aggregation", "linear")
    return CompositeReward(rewards=reward_instances, aggregation=aggregation)
