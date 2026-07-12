"""
Base class for all reward functions in CellDiffA.

Each reward function computes a scalar score for a batch of generated cell
expression profiles, guiding the SMC particles toward biologically plausible regions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch


class BaseReward(ABC):
    """
    Abstract base class for reward functions.

    All rewards must implement `compute()` which takes generated cell expressions
    and returns per-particle scalar rewards.
    """

    def __init__(self, weight: float = 1.0, name: str = "base_reward"):
        self.weight = weight
        self.name = name

    @abstractmethod
    def compute(
        self,
        x_pred: torch.Tensor,
        condition: str,
        timestep: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute reward scores for a batch of generated particles.

        Args:
            x_pred: Predicted clean expression profiles (Tweedie estimate).
                    Shape: (num_particles, num_genes)
            condition: The perturbation condition string (e.g., "GeneA+GeneB")
            timestep: Current diffusion timestep (for annealing)
            **kwargs: Additional context (e.g., control mean, gene indices)

        Returns:
            Reward scores. Shape: (num_particles,)
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


class CompositeReward:
    """
    Aggregates multiple reward functions into a single weighted score.

    Supports both linear weighting and Pareto-based multi-objective aggregation.
    """

    def __init__(
        self,
        rewards: list,
        aggregation: str = "linear",
    ):
        """
        Args:
            rewards: List of BaseReward instances (each has its own weight).
            aggregation: Aggregation strategy. One of ["linear", "pareto"].
        """
        self.rewards = rewards
        self.aggregation = aggregation

    def compute(
        self,
        x_pred: torch.Tensor,
        condition: str,
        timestep: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute aggregated reward for all particles.

        Returns:
            Aggregated reward scores. Shape: (num_particles,)
        """
        individual_rewards = []
        for reward_fn in self.rewards:
            r = reward_fn.compute(x_pred, condition, timestep, **kwargs)
            individual_rewards.append(r * reward_fn.weight)

        if self.aggregation == "linear":
            return torch.stack(individual_rewards, dim=0).sum(dim=0)
        elif self.aggregation == "pareto":
            return self._pareto_aggregate(individual_rewards)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def _pareto_aggregate(self, rewards: list) -> torch.Tensor:
        """
        Pareto-based aggregation: dynamically adjust weights to balance
        conflicting objectives, preventing any single reward from dominating.

        Uses min-norm solver to find Pareto-optimal weighting.
        """
        # Stack rewards: (num_objectives, num_particles)
        R = torch.stack(rewards, dim=0)

        # Normalize each objective to [0, 1] range for fair comparison
        R_min = R.min(dim=1, keepdim=True).values
        R_max = R.max(dim=1, keepdim=True).values
        R_norm = (R - R_min) / (R_max - R_min + 1e-8)

        # Compute dynamic weights inversely proportional to mean reward
        # (objectives that are harder to satisfy get higher weight)
        mean_rewards = R_norm.mean(dim=1)  # (num_objectives,)
        inv_weights = 1.0 / (mean_rewards + 1e-8)
        inv_weights = inv_weights / inv_weights.sum()  # normalize

        # Weighted sum with dynamic Pareto weights
        aggregated = (R_norm * inv_weights.unsqueeze(1)).sum(dim=0)
        return aggregated

    def __repr__(self) -> str:
        reward_strs = ", ".join(str(r) for r in self.rewards)
        return f"CompositeReward(aggregation={self.aggregation}, rewards=[{reward_strs}])"
