"""
Anti-Conservative Penalty Reward (r_anchor).

Penalizes generated cells that remain too close to the control distribution,
directly addressing the well-documented "conservative bias" problem in
perturbation prediction models (Wei et al., Nature Methods 2026).

This reward encourages the diffusion model to produce meaningful perturbation
effects rather than collapsing toward the control mean.
"""

from typing import Optional

import numpy as np
import torch

from .base import BaseReward


class AnchorReward(BaseReward):
    """
    Reward that penalizes proximity to control cells.

    Computation:
        r_anchor = ||x_pred - ctrl_mean||_2  (per particle)

    This is a soft constraint: it does not specify WHERE the cell should go,
    only that it should MOVE AWAY from the control state. The direction is
    governed by the other rewards (r_DEG and r_manifold).

    An optional upper bound prevents the reward from encouraging unrealistic
    extreme expression values.
    """

    def __init__(
        self,
        ctrl_mean: np.ndarray,
        weight: float = 1.0,
        normalize: bool = True,
        max_distance: Optional[float] = None,
    ):
        """
        Args:
            ctrl_mean: Mean expression of control cells. Shape: (num_genes,)
            weight: Reward weight.
            normalize: If True, normalize distance by sqrt(num_genes) for scale invariance.
            max_distance: Optional upper bound on rewarded distance (prevents explosion).
        """
        super().__init__(weight=weight, name="r_anchor")
        self.ctrl_mean = torch.tensor(ctrl_mean, dtype=torch.float32)
        self.normalize = normalize
        self.max_distance = max_distance

    def compute(
        self,
        x_pred: torch.Tensor,
        condition: str,
        timestep: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute anti-conservative reward for each particle.

        Args:
            x_pred: Tweedie estimate of clean expression. Shape: (N, G)
            condition: Perturbation condition string (unused, kept for interface).
            timestep: Current diffusion timestep.

        Returns:
            Distance-based reward scores. Shape: (N,)
        """
        ctrl = self.ctrl_mean.to(x_pred.device)
        diff = x_pred - ctrl.unsqueeze(0)  # (N, G)
        distance = torch.norm(diff, dim=1)  # (N,)

        if self.normalize:
            num_genes = x_pred.shape[1]
            distance = distance / np.sqrt(num_genes)

        if self.max_distance is not None:
            distance = distance.clamp(max=self.max_distance)

        return distance
