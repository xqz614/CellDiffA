"""
Geometric Prior Reward (r_manifold).

Evaluates whether the perturbation direction of generated cells aligns with
the expected shift direction derived from training set observations.

This reward ensures that generated cells move in the correct direction in
gene expression space relative to control cells, without constraining the
exact magnitude (which allows the diffusion model to capture nonlinear effects).
"""

from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseReward


class GeometricReward(BaseReward):
    """
    Reward based on cosine similarity between predicted and reference shift directions.

    Computation:
        r_manifold = cos(Δ_pred, Δ_ref)

    where:
        Δ_pred = x_pred - ctrl_mean  (predicted perturbation direction)
        Δ_ref = derived from training set (single or combined shifts)
    """

    def __init__(
        self,
        perturbation_shifts: Dict[str, np.ndarray],
        ctrl_mean: np.ndarray,
        weight: float = 1.0,
        min_shift_norm: float = 1e-6,
    ):
        """
        Args:
            perturbation_shifts: Dict mapping condition -> shift vector from training set.
            ctrl_mean: Mean expression of control cells. Shape: (num_genes,)
            weight: Reward weight.
            min_shift_norm: Minimum norm threshold to avoid degenerate cosine similarity.
        """
        super().__init__(weight=weight, name="r_manifold")
        self.perturbation_shifts = perturbation_shifts
        self.ctrl_mean = torch.tensor(ctrl_mean, dtype=torch.float32)
        self.min_shift_norm = min_shift_norm

        # Pre-compute normalized reference directions
        self._ref_cache: Dict[str, torch.Tensor] = {}

    def compute(
        self,
        x_pred: torch.Tensor,
        condition: str,
        timestep: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute geometric alignment reward for each particle.

        Args:
            x_pred: Tweedie estimate for each batch particle. Shape: (N, M, G)
            condition: Perturbation condition string.
            timestep: Current diffusion timestep.

        Returns:
            Cosine similarity scores. Shape: (N,)
        """
        ref_dir = self._get_reference_direction(condition, x_pred.device)
        if ref_dir is None:
            return torch.zeros(x_pred.shape[0], device=x_pred.device)

        if x_pred.ndim != 3:
            raise ValueError(
                f"GeometricReward expects (particles, cells, genes), got {tuple(x_pred.shape)}"
            )

        # Compute predicted population-mean shift direction.
        ctrl = self.ctrl_mean.to(x_pred.device)
        delta_pred = x_pred.mean(dim=1) - ctrl.unsqueeze(0)  # (N, G)

        # Normalize predicted directions
        pred_norm = torch.norm(delta_pred, dim=1, keepdim=True).clamp(min=self.min_shift_norm)
        delta_pred_normalized = delta_pred / pred_norm

        # Cosine similarity with reference direction
        cosine_sim = (delta_pred_normalized * ref_dir.unsqueeze(0)).sum(dim=1)

        return cosine_sim

    def _get_reference_direction(
        self, condition: str, device: torch.device
    ) -> Optional[torch.Tensor]:
        """
        Get normalized reference shift direction for a condition.

        For unseen combinations (A+B): Δ_ref = normalize(shift_A + shift_B)
        """
        if condition in self._ref_cache:
            return self._ref_cache[condition].to(device)

        shift = self._resolve_shift(condition)
        if shift is None:
            return None

        shift_tensor = torch.tensor(shift, dtype=torch.float32)
        norm = torch.norm(shift_tensor).item()
        if norm < self.min_shift_norm:
            return None

        normalized = shift_tensor / norm
        self._ref_cache[condition] = normalized
        return normalized.to(device)

    def _resolve_shift(self, condition: str) -> Optional[np.ndarray]:
        """Resolve shift vector for a condition (direct or combined)."""
        # Direct lookup
        if condition in self.perturbation_shifts:
            return self.perturbation_shifts[condition]

        # Decompose combination
        parts = condition.split("+")
        parts = [p for p in parts if p not in ("ctrl", "control")]

        if len(parts) == 0:
            return None

        combined_shift = None
        for p in parts:
            candidates = [p, f"{p}+ctrl", f"{p}+control"]
            found = False
            for cand in candidates:
                if cand in self.perturbation_shifts:
                    if combined_shift is None:
                        combined_shift = self.perturbation_shifts[cand].copy()
                    else:
                        combined_shift += self.perturbation_shifts[cand]
                    found = True
                    break
            if not found:
                # If any constituent is missing, we cannot build a reliable reference
                return None

        return combined_shift
