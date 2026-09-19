"""Distribution anchor reward for CellDiffA.

The reward anchors a predicted cell batch to a training-derived reference
distribution. For an unseen combination the reference is constructed by
transporting control cells with the sum of available single-perturbation mean
shifts. It never uses held-out perturbation cells.
"""

from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseReward


class AnchorReward(BaseReward):
    """Negative squared RBF MMD between two empirical cell distributions.

    The biased V-statistic includes every cell and is invariant to the ordering
    of either population. Gram matrices use O(particles * cells**2) memory,
    without materializing pairwise differences along the gene dimension.
    """

    def __init__(
        self,
        perturbation_shifts: Dict[str, np.ndarray],
        weight: float = 1.0,
        bandwidth: float = 1.0,
    ):
        super().__init__(weight=weight, name="r_anchor")
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        self.perturbation_shifts = perturbation_shifts
        self.bandwidth = float(bandwidth)
        self._shift_cache: Dict[str, torch.Tensor] = {}

    def compute(
        self,
        x_pred: torch.Tensor,
        condition: str,
        timestep: int,
        ctrl_cells: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if x_pred.ndim != 3:
            raise ValueError(
                f"AnchorReward expects (particles, cells, genes), got {tuple(x_pred.shape)}"
            )

        n_particles, n_cells, n_genes = x_pred.shape
        if n_cells == 0 or n_genes == 0:
            raise ValueError("Candidate populations must contain cells and genes.")
        shift = self._resolve_shift(condition, x_pred.device)
        if shift is None or ctrl_cells is None:
            return torch.zeros(n_particles, device=x_pred.device)

        ctrl_cells = ctrl_cells.to(device=x_pred.device, dtype=x_pred.dtype)
        if ctrl_cells.ndim != 2 or ctrl_cells.shape[1] != n_genes:
            raise ValueError("ctrl_cells must have shape (cells, genes).")

        if ctrl_cells.shape[0] == 0:
            raise ValueError("The control population cannot be empty.")
        reference = (ctrl_cells + shift.to(x_pred.dtype).unsqueeze(0)).unsqueeze(0)

        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            squared_distance = (
                a.square().sum(dim=-1, keepdim=True)
                + b.square().sum(dim=-1).unsqueeze(-2)
                - 2.0 * torch.matmul(a, b.transpose(-1, -2))
            ).clamp_min(0)
            return torch.exp(-squared_distance / (2.0 * n_genes * self.bandwidth**2))

        mmd_squared = (
            kernel(x_pred, x_pred).mean(dim=(-2, -1))
            + kernel(reference, reference).mean(dim=(-2, -1))
            - 2.0 * kernel(x_pred, reference).mean(dim=(-2, -1))
        )
        return -mmd_squared.clamp_min(0)

    def _resolve_shift(self, condition: str, device: torch.device) -> Optional[torch.Tensor]:
        if condition in self._shift_cache:
            return self._shift_cache[condition].to(device)

        if condition in self.perturbation_shifts:
            shift = self.perturbation_shifts[condition]
        else:
            parts = [p for p in condition.split("+") if p not in {"ctrl", "control"}]
            shifts = []
            for part in parts:
                match = next(
                    (
                        self.perturbation_shifts[name]
                        for name in (part, f"{part}+ctrl", f"{part}+control")
                        if name in self.perturbation_shifts
                    ),
                    None,
                )
                if match is None:
                    return None
                shifts.append(match)
            if not shifts:
                return None
            shift = np.stack(shifts).sum(axis=0)

        tensor = torch.as_tensor(shift, dtype=torch.float32)
        self._shift_cache[condition] = tensor
        return tensor.to(device)
