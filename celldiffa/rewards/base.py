"""
Base class for all reward functions in CellDiffA.

Each reward function computes a scalar score for a batch of generated cell
expression profiles, guiding the SMC particles toward biologically plausible regions.
"""

from abc import ABC, abstractmethod

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
            x_pred: Predicted clean cell batches (Tweedie estimate).
                    Shape: (num_particles, cells_per_particle, num_genes)
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
        normalization: str = "zscore",
    ):
        """
        Args:
            rewards: List of BaseReward instances (each has its own weight).
            aggregation: Aggregation strategy. Currently only "linear".
            normalization: Per-step normalization for each objective. "zscore"
                makes heterogeneous reward scales comparable; "none" preserves
                the raw objective values.
        """
        self.rewards = rewards
        self.aggregation = aggregation
        self.normalization = normalization
        if not rewards:
            raise ValueError("CompositeReward requires at least one reward.")
        if aggregation != "linear":
            raise ValueError(
                "Only linear aggregation is supported. The previous 'pareto' "
                "option was not a valid Pareto/min-norm solver."
            )
        if normalization not in {"zscore", "none"}:
            raise ValueError("normalization must be 'zscore' or 'none'.")

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
            if self.normalization == "zscore" and r.numel() > 1:
                scale = r.std(unbiased=False).clamp_min(1e-6)
                r = (r - r.mean()) / scale
            individual_rewards.append(r * reward_fn.weight)

        return torch.stack(individual_rewards, dim=0).sum(dim=0)

    def __repr__(self) -> str:
        reward_strs = ", ".join(str(r) for r in self.rewards)
        return (
            f"CompositeReward(aggregation={self.aggregation}, "
            f"normalization={self.normalization}, rewards=[{reward_strs}])"
        )


class ProjectedReward:
    """Apply a fixed cell/gene projection before evaluating a reward.

    This lets a model sample in its native feature space while CellDiffA uses
    the ordered evaluation genes. A cell mask also prevents padded members of
    a PerturbDiff cell set from affecting rewards.
    """

    def __init__(self, reward, *, gene_indices=None, cell_mask=None):
        self.reward = reward
        self.gene_indices = gene_indices
        self.cell_mask = cell_mask

    def compute(self, x_pred, condition, timestep, **kwargs):
        ctrl_cells = kwargs.get("ctrl_cells")
        if self.cell_mask is not None:
            mask = torch.as_tensor(self.cell_mask, dtype=torch.bool, device=x_pred.device)
            x_pred = x_pred[:, mask]
            if ctrl_cells is not None:
                ctrl_cells = ctrl_cells[mask]
        if self.gene_indices is not None:
            indices = torch.as_tensor(self.gene_indices, dtype=torch.long, device=x_pred.device)
            x_pred = x_pred.index_select(-1, indices)
            if ctrl_cells is not None:
                ctrl_cells = ctrl_cells.index_select(-1, indices)
        kwargs["ctrl_cells"] = ctrl_cells
        return self.reward.compute(x_pred, condition, timestep, **kwargs)
