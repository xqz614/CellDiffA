"""
Sequential Monte Carlo (SMC) Engine for Test-Time Alignment.

This is the core of CellDiffA: a plug-and-play wrapper that can be applied on top
of any pre-trained diffusion model to perform reward-guided sampling at test time,
without modifying or retraining the base model.

The algorithm follows the theoretical framework of Feynman-Kac models applied to
diffusion processes (Del Moral, 2004; Cardoso et al., ICLR 2025), adapted for
population-level single-cell perturbation prediction.

Key innovations over standard SMC guidance:
    1. Population-level output: returns a distribution (multiple cells), not a single sample.
    2. Multi-objective biological rewards with Pareto aggregation.
    3. Annealed tempering schedule adapted for gene expression space.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .resampler import Resampler, ResamplingStrategy


# ============================================================
# Protocol: Diffusion Model Sampler Interface
# ============================================================

class DiffusionSamplerProtocol(Protocol):
    """
    Protocol that any diffusion model must satisfy to be wrapped by CellDiffA.

    The base model only needs to expose a single-step denoising function.
    This ensures CellDiffA is truly plug-and-play.
    """

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one reverse diffusion step.

        Args:
            x_t: Noisy samples at timestep t. Shape: (batch_size, num_genes)
            t: Current timestep tensor. Shape: (batch_size,)
            condition: Conditioning information (perturbation embedding, etc.)

        Returns:
            Dict containing:
                - "x_prev": Denoised sample at t-1. Shape: (batch_size, num_genes)
                - "x0_pred": Tweedie estimate of clean sample. Shape: (batch_size, num_genes)
                - "noise_pred": Predicted noise (optional). Shape: (batch_size, num_genes)
        """
        ...

    @property
    def num_timesteps(self) -> int:
        """Total number of diffusion timesteps."""
        ...

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample initial noise x_T ~ N(0, I)."""
        ...


# ============================================================
# SMC Configuration
# ============================================================

@dataclass
class SMCConfig:
    """Configuration for the SMC test-time alignment engine."""

    # Particle settings
    num_particles: int = 100
    """Number of SMC particles (cells to generate in parallel)."""

    # Resampling settings
    resampling_strategy: str = "systematic"
    """Resampling method: 'multinomial', 'systematic', 'stratified'."""

    ess_threshold: float = 0.5
    """Effective Sample Size threshold (fraction of N) to trigger resampling."""

    # Tempering / Annealing
    tempering_schedule: str = "linear"
    """How to anneal reward influence: 'linear', 'cosine', 'constant'."""

    initial_temperature: float = 0.1
    """Temperature at t=T (start of reverse process). Low = weak guidance."""

    final_temperature: float = 1.0
    """Temperature at t=0 (end of reverse process). High = strong guidance."""

    # Output aggregation
    output_mode: str = "weighted_mean"
    """How to aggregate final particles: 'weighted_mean', 'top_k', 'all'."""

    top_k: int = 50
    """Number of top particles to keep if output_mode='top_k'."""

    # Computational
    device: str = "cuda"
    batch_size_per_step: int = 100
    """Max particles processed in one forward pass (for memory management)."""


# ============================================================
# SMC Engine
# ============================================================

class SMCEngine:
    """
    Sequential Monte Carlo engine for test-time alignment of diffusion models.

    This engine wraps a pre-trained diffusion model and applies reward-guided
    importance weighting and resampling at each denoising step, steering the
    generated cell population toward biologically plausible distributions.

    Usage:
        >>> engine = SMCEngine(model_sampler, reward_fn, config)
        >>> result = engine.sample_with_alignment(condition, ctrl_cells)
    """

    def __init__(
        self,
        model_sampler: DiffusionSamplerProtocol,
        reward_fn: Callable,
        config: SMCConfig = None,
    ):
        """
        Args:
            model_sampler: Pre-trained diffusion model satisfying DiffusionSamplerProtocol.
            reward_fn: Composite reward function (from celldiffa.rewards).
            config: SMC configuration. Uses defaults if None.
        """
        self.model = model_sampler
        self.reward_fn = reward_fn
        self.config = config or SMCConfig()
        self.resampler = Resampler(
            strategy=ResamplingStrategy(self.config.resampling_strategy)
        )
        self.device = torch.device(self.config.device)

    def sample_with_alignment(
        self,
        condition: str,
        condition_emb: Dict[str, torch.Tensor],
        ctrl_cells: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the full SMC-guided reverse diffusion process.

        This is the main entry point for CellDiffA inference.

        Args:
            condition: Perturbation condition string (e.g., "GeneA+GeneB").
            condition_emb: Conditioning tensors for the diffusion model.
            ctrl_cells: Control cell expressions for reference. Shape: (M, G)
            return_trajectory: If True, also return intermediate states.

        Returns:
            Dict containing:
                - "samples": Final generated cell expressions. Shape depends on output_mode.
                - "weights": Final normalized particle weights. Shape: (N,)
                - "ess_history": ESS values at each timestep.
                - "trajectory": (optional) List of intermediate particle states.
        """
        N = self.config.num_particles
        T = self.model.num_timesteps

        # Step 0: Initialize particles as pure noise
        x_t = self.model.sample_noise(
            shape=(N, condition_emb.get("num_genes", ctrl_cells.shape[1] if ctrl_cells is not None else 2000)),
            device=self.device,
        )

        # Initialize uniform log-weights
        log_weights = torch.zeros(N, device=self.device)
        ess_history = []
        trajectory = [] if return_trajectory else None

        # Reverse diffusion loop: t = T-1, T-2, ..., 0
        timesteps = list(range(T - 1, -1, -1))

        for step_idx, t in enumerate(timesteps):
            t_tensor = torch.full((N,), t, device=self.device, dtype=torch.long)

            # ----------------------------------------------------------
            # Step 1: Parallel denoising (black-box model call)
            # ----------------------------------------------------------
            with torch.no_grad():
                output = self._batched_denoise(x_t, t_tensor, condition_emb)

            x_prev = output["x_prev"]       # (N, G) - denoised one step
            x0_pred = output["x0_pred"]      # (N, G) - Tweedie estimate

            # ----------------------------------------------------------
            # Step 2: Compute rewards on Tweedie estimate
            # ----------------------------------------------------------
            rewards = self.reward_fn.compute(
                x_pred=x0_pred,
                condition=condition,
                timestep=t,
                ctrl_cells=ctrl_cells,
            )  # (N,)

            # ----------------------------------------------------------
            # Step 3: Update importance weights with annealed temperature
            # ----------------------------------------------------------
            temperature = self._get_temperature(step_idx, len(timesteps))
            log_weights = log_weights + temperature * rewards

            # Normalize weights
            log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
            weights = torch.exp(log_weights_normalized)

            # Compute ESS
            ess = 1.0 / (weights ** 2).sum().item()
            ess_history.append(ess)

            # ----------------------------------------------------------
            # Step 3b: Conditional resampling (if ESS drops below threshold)
            # ----------------------------------------------------------
            if ess < self.config.ess_threshold * N:
                indices = self.resampler.resample(weights, N)
                x_prev = x_prev[indices]
                log_weights = torch.zeros(N, device=self.device)  # Reset weights
            
            # Update particles
            x_t = x_prev

            if return_trajectory:
                trajectory.append(x_t.clone().cpu())

        # ----------------------------------------------------------
        # Step 4: Output aggregation
        # ----------------------------------------------------------
        final_weights = torch.exp(
            log_weights - torch.logsumexp(log_weights, dim=0)
        )
        output_samples = self._aggregate_output(x_t, final_weights)

        result = {
            "samples": output_samples,
            "weights": final_weights.cpu(),
            "ess_history": ess_history,
            "all_particles": x_t.cpu(),
        }
        if return_trajectory:
            result["trajectory"] = trajectory

        return result

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _batched_denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_emb: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise particles in batches to manage GPU memory.
        """
        N = x_t.shape[0]
        bs = self.config.batch_size_per_step

        if N <= bs:
            return self.model.denoise_step(x_t, t, condition_emb)

        outputs = {"x_prev": [], "x0_pred": []}
        for i in range(0, N, bs):
            end = min(i + bs, N)
            batch_out = self.model.denoise_step(
                x_t[i:end], t[i:end], condition_emb
            )
            outputs["x_prev"].append(batch_out["x_prev"])
            outputs["x0_pred"].append(batch_out["x0_pred"])

        return {
            "x_prev": torch.cat(outputs["x_prev"], dim=0),
            "x0_pred": torch.cat(outputs["x0_pred"], dim=0),
        }

    def _get_temperature(self, step_idx: int, total_steps: int) -> float:
        """
        Compute annealing temperature for the current step.

        The temperature controls how strongly rewards influence particle weights.
        It increases over the reverse process: weak guidance at high noise levels
        (where Tweedie estimates are unreliable), strong guidance at low noise.
        """
        progress = step_idx / max(total_steps - 1, 1)  # 0 -> 1

        t_init = self.config.initial_temperature
        t_final = self.config.final_temperature

        if self.config.tempering_schedule == "linear":
            return t_init + (t_final - t_init) * progress
        elif self.config.tempering_schedule == "cosine":
            return t_init + (t_final - t_init) * (1 - np.cos(np.pi * progress)) / 2
        elif self.config.tempering_schedule == "constant":
            return t_final
        else:
            raise ValueError(f"Unknown schedule: {self.config.tempering_schedule}")

    def _aggregate_output(
        self, particles: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate final particles into output based on configured mode.

        Args:
            particles: Final particle states. Shape: (N, G)
            weights: Normalized particle weights. Shape: (N,)

        Returns:
            Aggregated output. Shape depends on output_mode.
        """
        mode = self.config.output_mode

        if mode == "weighted_mean":
            # Single consensus cell (weighted average)
            return (particles * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

        elif mode == "top_k":
            # Return top-K highest-weight particles as the generated population
            k = min(self.config.top_k, particles.shape[0])
            _, top_indices = torch.topk(weights, k)
            return particles[top_indices]

        elif mode == "all":
            # Return all particles (for downstream distribution analysis)
            return particles

        else:
            raise ValueError(f"Unknown output_mode: {mode}")
