"""
Particle resampling strategies for SMC.

Implements standard resampling algorithms used in Sequential Monte Carlo methods.
These are critical for preventing particle degeneracy (weight collapse).
"""

from enum import Enum
from typing import Optional

import torch


class ResamplingStrategy(Enum):
    """Available resampling strategies."""
    MULTINOMIAL = "multinomial"
    SYSTEMATIC = "systematic"
    STRATIFIED = "stratified"


class Resampler:
    """
    Particle resampler implementing multiple standard SMC resampling algorithms.

    Systematic resampling is recommended as default due to lower variance
    compared to multinomial resampling (Douc et al., 2005).
    """

    def __init__(self, strategy: ResamplingStrategy = ResamplingStrategy.SYSTEMATIC):
        self.strategy = strategy

    def resample(self, weights: torch.Tensor, n_samples: int) -> torch.Tensor:
        """
        Resample particle indices according to normalized weights.

        Args:
            weights: Normalized importance weights. Shape: (N,). Must sum to 1.
            n_samples: Number of indices to sample.

        Returns:
            Resampled indices. Shape: (n_samples,)
        """
        if self.strategy == ResamplingStrategy.MULTINOMIAL:
            return self._multinomial(weights, n_samples)
        elif self.strategy == ResamplingStrategy.SYSTEMATIC:
            return self._systematic(weights, n_samples)
        elif self.strategy == ResamplingStrategy.STRATIFIED:
            return self._stratified(weights, n_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _multinomial(self, weights: torch.Tensor, n: int) -> torch.Tensor:
        """Standard multinomial resampling."""
        return torch.multinomial(weights, n, replacement=True)

    def _systematic(self, weights: torch.Tensor, n: int) -> torch.Tensor:
        """
        Systematic resampling (Kitagawa, 1996).

        Uses a single uniform random number to generate all indices,
        producing lower variance than multinomial resampling.
        """
        device = weights.device
        cumsum = torch.cumsum(weights, dim=0)

        # Single random offset
        u = torch.rand(1, device=device) / n
        positions = u + torch.arange(n, device=device, dtype=torch.float32) / n

        # Find indices via searchsorted
        indices = torch.searchsorted(cumsum, positions)
        indices = indices.clamp(max=len(weights) - 1)

        return indices

    def _stratified(self, weights: torch.Tensor, n: int) -> torch.Tensor:
        """
        Stratified resampling.

        Each stratum gets its own independent uniform random number,
        providing a balance between multinomial and systematic approaches.
        """
        device = weights.device
        cumsum = torch.cumsum(weights, dim=0)

        # Independent uniform in each stratum
        u = torch.rand(n, device=device) / n
        offsets = torch.arange(n, device=device, dtype=torch.float32) / n
        positions = u + offsets

        indices = torch.searchsorted(cumsum, positions)
        indices = indices.clamp(max=len(weights) - 1)

        return indices
