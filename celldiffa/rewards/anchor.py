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
    """Negative linear-time RBF MMD to a training-derived reference batch."""

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
        shift = self._resolve_shift(condition, x_pred.device)
        if shift is None or ctrl_cells is None:
            return torch.zeros(n_particles, device=x_pred.device)

        ctrl_cells = ctrl_cells.to(device=x_pred.device, dtype=x_pred.dtype)
        if ctrl_cells.ndim != 2 or ctrl_cells.shape[1] != n_genes:
            raise ValueError("ctrl_cells must have shape (cells, genes).")

        # Match the particle batch size without introducing test data.
        if ctrl_cells.shape[0] < n_cells:
            repeats = (n_cells + ctrl_cells.shape[0] - 1) // ctrl_cells.shape[0]
            ctrl_cells = ctrl_cells.repeat(repeats, 1)
        reference = ctrl_cells[:n_cells] + shift.unsqueeze(0)

        # A linear-time MMD estimator avoids O(N*M^2*G) memory. Pair adjacent
        # cells; for odd M, omit the last cell.
        paired = n_cells - (n_cells % 2)
        if paired < 2:
            return torch.zeros(n_particles, device=x_pred.device)
        x0, x1 = x_pred[:, :paired:2], x_pred[:, 1:paired:2]
        y0 = reference[:paired:2].unsqueeze(0).expand(n_particles, -1, -1)
        y1 = reference[1:paired:2].unsqueeze(0).expand(n_particles, -1, -1)

        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            mean_sq_distance = (a - b).square().mean(dim=-1)
            return torch.exp(-mean_sq_distance / (2.0 * self.bandwidth**2))

        mmd = (kernel(x0, x1) + kernel(y0, y1) - kernel(x0, y1) - kernel(x1, y0)).mean(dim=1)
        return -mmd

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
