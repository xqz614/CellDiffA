"""
Sequential Monte Carlo (SMC) Engine for Test-Time Alignment.

Core of CellDiffA: a plug-and-play wrapper that applies reward-guided sampling
on top of any pre-trained diffusion model at test time, without modifying or
retraining the base model.

Implements the Feynman-Kac SMC framework from DAS (Kim et al., ICLR 2025),
adapted for population-level single-cell perturbation prediction.

Key design decisions:
    1. Telescoping potential: log G_k = (β_k r_k - β_{k-1} r_{k-1}) / α.
    2. Each particle is an empirical distribution (a batch of cells).
    3. Reward components are optionally standardized before linear aggregation.
    4. The terminal step retains weights for MAP or weighted output.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

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
        prev_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one reverse diffusion step.

        Args:
            x_t: Noisy cells at timestep t. Cell-wise samplers receive
                ``(batch_size, num_genes)``. Population-native samplers receive
                ``(particle_batch, cells_per_particle, num_genes)``.
            t: Current timestep tensor. Shape: (batch_size,)
            condition: Conditioning information (perturbation embedding, etc.)
            prev_pred: Previous x0 prediction for self-conditioning. Shape: (batch_size, num_genes)

        Returns:
            Dict containing:
                - "x_prev": Denoised sample at t-1. Shape: (batch_size, num_genes)
                - "x0_pred": Tweedie/direct estimate of clean sample. Shape: (batch_size, num_genes)
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
    """Number of SMC particles. Each particle is a batch of cells."""

    cells_per_particle: int = 32
    """Number of cells in each distribution-valued particle."""

    # Resampling settings
    resampling_strategy: str = "systematic"
    """Resampling method: 'multinomial', 'systematic', 'stratified'."""

    ess_threshold: float = 0.5
    """Effective Sample Size threshold (fraction of N) to trigger resampling."""

    # Tempering / Annealing (DAS-style)
    tempering_schedule: str = "linear"
    """How β_t grows from 0 to 1: 'linear' or 'cosine'."""

    alpha: float = 1.0
    """Reward temperature α: controls reward-KL tradeoff in p_tar ∝ p_pre * exp(r/α).
    Smaller α = stronger reward influence. DAS default: 1.0."""

    # Sampling
    start_timestep: Optional[int] = None
    """If set, start reverse process from this timestep instead of T-1.
    PerturbDiff default uses start_time=100 for DDIM."""

    use_ddim: bool = True
    """Whether to use DDIM (deterministic) or DDPM (stochastic) steps."""

    eta: float = 0.0
    """DDIM noise scale. 0 = deterministic."""

    guidance_strength: float = 1.0
    """Classifier-free guidance strength for the base model."""

    # Output aggregation
    output_mode: str = "map"
    """How to aggregate final batch particles: 'map', 'weighted_mean',
    'top_k', or 'all'."""

    top_k: int = 50
    """Number of top particles to keep if output_mode='top_k'."""

    # Computational
    device: str = "cuda"
    batch_size_per_step: int = 256
    """Maximum number of cells processed in one model forward pass."""

    seed: int = 42
    """Sampling seed. Reusing it across conditions enables paired comparisons."""

    alignment_mode: str = "smc"
    """'smc', terminal-only 'best_of_n', or unguided 'random'; same denoising budget."""


# ============================================================
# SMC Engine
# ============================================================


class SMCEngine:
    """
    Sequential Monte Carlo engine for test-time alignment of diffusion models.

    Implements a telescoping Feynman-Kac potential:
        log w_k^(n) += (β_k r_k - β_{k-1} r_{k-1}) / α

    where β_t is the tempering coefficient at step t, r is the composite
    biological reward, and α is the reward temperature.

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
        self.resampler = Resampler(strategy=ResamplingStrategy(self.config.resampling_strategy))
        self.device = torch.device(self.config.device)
        self._validate_config()

    def sample_with_alignment(
        self,
        condition: str,
        condition_emb: Dict[str, torch.Tensor],
        ctrl_cells: Optional[torch.Tensor] = None,
        num_genes: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Execute the full SMC-guided reverse diffusion process.

        This is the main entry point for CellDiffA inference.

        Args:
            condition: Perturbation condition string (e.g., "GeneA+GeneB").
            condition_emb: Conditioning tensors for the diffusion model.
            ctrl_cells: Control cell expressions for reference. Shape: (M, G)
            num_genes: Number of genes (inferred from ctrl_cells if not given).
            return_trajectory: If True, also return intermediate states.

        Returns:
            Dict containing:
                - "samples": Final generated cell expressions. Shape depends on output_mode.
                - "weights": Final normalized particle weights. Shape: (N,)
                - "ess_history": ESS values at each timestep.
                - "resample_history": Boolean list indicating when resampling occurred.
                - "trajectory": (optional) List of intermediate particle states.
        """
        N = self.config.num_particles
        T = self.model.num_timesteps

        # Determine gene dimension
        G = num_genes or (ctrl_cells.shape[1] if ctrl_cells is not None else None)
        if G is None:
            raise ValueError("Must provide either ctrl_cells or num_genes.")

        # A particle represents an empirical distribution B^(n) in R^(M x G),
        # not one cell. This is the central population-level semantics described
        # by CellDiffA.
        if ctrl_cells is not None:
            if ctrl_cells.ndim != 2:
                raise ValueError("ctrl_cells must have shape (cells, genes).")
            if ctrl_cells.shape[0] < 2:
                raise ValueError("At least two control cells are required.")
            if ctrl_cells.shape[1] != G:
                raise ValueError(f"ctrl_cells has {ctrl_cells.shape[1]} genes but num_genes={G}.")
            ctrl_cells = ctrl_cells.to(self.device, dtype=torch.float32)
            M = min(self.config.cells_per_particle, ctrl_cells.shape[0])
            ctrl_cells = ctrl_cells[:M]
        else:
            M = self.config.cells_per_particle

        # Determine start timestep
        start_t = self.config.start_timestep if self.config.start_timestep is not None else (T - 1)
        start_t = max(0, min(start_t, T - 1))

        # Step 0: initialize N distribution-valued particles, each containing M cells.
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        x_t = self.model.sample_noise(shape=(N, M, G), device=self.device)
        if x_t.shape != (N, M, G):
            raise ValueError(
                f"Sampler returned noise shape {tuple(x_t.shape)}, expected {(N, M, G)}."
            )

        # Initialize log-weights to zero (uniform)
        log_weights = torch.zeros(N, device=self.device)

        # Self-conditioning: track previous x0 prediction
        prev_pred = torch.zeros(N, M, G, device=self.device)

        # Stores beta_{k-1} * r_{k-1}. The incremental potential
        # beta_k*r_k - beta_{k-1}*r_{k-1} telescopes to the desired terminal
        # reward when no resampling occurs and is the standard Feynman-Kac form.
        previous_log_potential = torch.zeros(N, device=self.device)
        ancestors = torch.arange(N, device=self.device)
        ancestor_history = []

        # Tracking
        ess_history = []
        resample_history = []
        trajectory = [] if return_trajectory else None

        # Build tempering schedule: β values from 0 to 1
        timesteps = list(range(start_t, -1, -1))
        num_steps = len(timesteps)
        beta_schedule = self._build_tempering_schedule(num_steps)

        # Reverse diffusion loop
        for step_idx, t in enumerate(timesteps):
            t_tensor = torch.full((N,), t, device=self.device, dtype=torch.long)

            # ----------------------------------------------------------
            # Step 1: Parallel denoising (black-box model call)
            # ----------------------------------------------------------
            with torch.no_grad():
                output = self._batched_denoise(x_t, t_tensor, condition_emb, prev_pred)

            x_prev = output["x_prev"]  # (N, M, G) - denoised one step
            x0_pred = output["x0_pred"]  # (N, M, G) - clean-batch prediction

            # Update self-conditioning state
            prev_pred = x0_pred.clone()

            # ----------------------------------------------------------
            # Step 2: Compute rewards on x0 prediction
            # ----------------------------------------------------------
            rewards = self.reward_fn.compute(
                x_pred=x0_pred,
                condition=condition,
                timestep=t,
                ctrl_cells=ctrl_cells,
            )  # (N,) scalar reward per distribution-valued particle

            if rewards.shape != (N,):
                raise ValueError(
                    f"Reward must return one value per particle, expected {(N,)}, "
                    f"got {tuple(rewards.shape)}."
                )
            if not torch.isfinite(rewards).all():
                raise ValueError("Reward returned NaN or infinite values.")

            # ----------------------------------------------------------
            # Step 3: telescoping incremental potential.
            # ----------------------------------------------------------
            beta_t = beta_schedule[step_idx]
            if self.config.alignment_mode == "random":
                beta_t = 0.0
            elif self.config.alignment_mode == "best_of_n":
                beta_t = float(step_idx == num_steps - 1)
            current_log_potential = beta_t * rewards
            log_weights = (
                log_weights + (current_log_potential - previous_log_potential) / self.config.alpha
            )

            # Normalize weights for ESS computation
            log_weights_normalized = log_weights - torch.logsumexp(log_weights, dim=0)
            weights = torch.exp(log_weights_normalized)

            # Compute ESS
            ess = 1.0 / (weights**2).sum().item()
            ess_history.append(ess)

            # ----------------------------------------------------------
            # Step 3b: Conditional resampling (if ESS drops below threshold)
            # ----------------------------------------------------------
            did_resample = False
            # Do not resample at the terminal step: keeping the terminal weights
            # makes MAP and weighted-mean aggregation well-defined.
            if (
                self.config.alignment_mode == "smc"
                and step_idx < num_steps - 1
                and ess < self.config.ess_threshold * N
            ):
                indices = self.resampler.resample(weights, N)
                x_prev = x_prev[indices]
                prev_pred = prev_pred[indices]
                current_log_potential = current_log_potential[indices]
                ancestors = ancestors[indices]
                log_weights = torch.zeros(N, device=self.device)  # Reset weights
                did_resample = True

            resample_history.append(did_resample)
            ancestor_history.append(int(torch.unique(ancestors).numel()))

            # Update particles
            x_t = x_prev
            previous_log_potential = current_log_potential

            if return_trajectory:
                trajectory.append(x_t.clone().cpu())

        # ----------------------------------------------------------
        # Step 4: Output aggregation
        # ----------------------------------------------------------
        final_weights = torch.exp(log_weights - torch.logsumexp(log_weights, dim=0))
        output_samples = self._aggregate_output(x_t, final_weights)

        result = {
            "samples": output_samples,
            "weights": final_weights.cpu(),
            "ess_history": ess_history,
            "resample_history": resample_history,
            "all_particles": x_t.cpu(),
            "cells_per_particle": M,
            "ancestor_history": ancestor_history,
            "denoised_cell_steps": N * M * num_steps,
        }
        if return_trajectory:
            result["trajectory"] = trajectory

        return result

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        if self.config.num_particles == 1 and self.config.alignment_mode == "random":
            pass  # One-candidate frozen-backbone reference; never resampled.
        elif self.config.num_particles < 2:
            raise ValueError("num_particles must be at least 2.")
        if self.config.cells_per_particle < 2:
            raise ValueError("cells_per_particle must be at least 2.")
        if not 0 < self.config.ess_threshold <= 1:
            raise ValueError("ess_threshold must be in (0, 1].")
        if self.config.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if self.config.batch_size_per_step < 1:
            raise ValueError("batch_size_per_step must be positive.")
        if self.config.tempering_schedule not in {"linear", "cosine"}:
            raise ValueError("tempering_schedule must be 'linear' or 'cosine'.")
        if self.config.output_mode not in {"map", "weighted_mean", "top_k", "all"}:
            raise ValueError("Unknown output_mode.")
        if self.config.output_mode == "top_k" and self.config.top_k < 1:
            raise ValueError("top_k must be positive.")
        if self.config.alignment_mode not in {"smc", "best_of_n", "random"}:
            raise ValueError("alignment_mode must be smc, best_of_n, or random.")

    def _build_tempering_schedule(self, num_steps: int) -> List[float]:
        """
        Build the tempering schedule β_0, β_1, ..., β_{T-1}.

        β goes from 0 to 1 over the reverse process.
        At step 0 (t=T-1, high noise), β≈0 → weak reward influence.
        At final step (t=0, clean), β=1 → full reward influence.

        This follows DAS Eq. (7): incremental tempering ensures
        importance weights have bounded variance.
        """
        if num_steps <= 1:
            return [1.0]

        schedule = self.config.tempering_schedule

        if schedule == "linear":
            # β_k = k / (num_steps - 1)
            return [k / (num_steps - 1) for k in range(num_steps)]

        elif schedule == "cosine":
            # Cosine schedule: slower at start, faster at end
            return [0.5 * (1 - np.cos(np.pi * k / (num_steps - 1))) for k in range(num_steps)]

        else:
            raise ValueError(f"Unknown tempering schedule: {schedule}")

    def _batched_denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_emb: Dict[str, torch.Tensor],
        prev_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Denoise particles in batches to manage GPU memory.

        Handles expanding condition tensors to match batch size.
        """
        N, M, G = x_t.shape
        if getattr(self.model, "population_native", False):
            return self._batched_population_denoise(x_t, t, condition_emb, prev_pred)

        bs = self.config.batch_size_per_step

        # Base diffusion models denoise cells independently. Flatten the
        # particle/cell axes for the model, then restore (N, M, G).
        total_cells = N * M
        flat_x = x_t.reshape(total_cells, G)
        flat_prev = prev_pred.reshape(total_cells, G)
        flat_t = t[:, None].expand(N, M).reshape(total_cells)

        outputs = {"x_prev": [], "x0_pred": []}
        for i in range(0, total_cells, bs):
            end = min(i + bs, total_cells)
            batch_condition = self._condition_batch(
                condition_emb, start=i, end=end, total=total_cells, cells_per_particle=M
            )
            batch_out = self.model.denoise_step(
                flat_x[i:end], flat_t[i:end], batch_condition, prev_pred=flat_prev[i:end]
            )
            for key in ("x_prev", "x0_pred"):
                if key not in batch_out or batch_out[key].shape != flat_x[i:end].shape:
                    actual = None if key not in batch_out else tuple(batch_out[key].shape)
                    raise ValueError(
                        f"Sampler output '{key}' has shape {actual}; "
                        f"expected {tuple(flat_x[i:end].shape)}."
                    )
            outputs["x_prev"].append(batch_out["x_prev"])
            outputs["x0_pred"].append(batch_out["x0_pred"])

        return {
            "x_prev": torch.cat(outputs["x_prev"], dim=0).reshape(N, M, G),
            "x0_pred": torch.cat(outputs["x0_pred"], dim=0).reshape(N, M, G),
        }

    def _batched_population_denoise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition_emb: Dict[str, torch.Tensor],
        prev_pred: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Batch particles while preserving the model's cell-set axis.

        PerturbDiff's Cross-DiT attends across the cells in a set. Flattening
        ``(particle, cell)`` into independent rows changes the released model,
        so its adapter opts into this path with ``population_native = True``.
        """
        n_particles, n_cells, n_genes = x_t.shape
        particle_batch = max(1, self.config.batch_size_per_step // n_cells)
        outputs = {"x_prev": [], "x0_pred": []}
        for start in range(0, n_particles, particle_batch):
            end = min(start + particle_batch, n_particles)
            condition = self._population_condition_batch(
                condition_emb,
                start=start,
                end=end,
                total=n_particles,
            )
            batch_out = self.model.denoise_step(
                x_t[start:end],
                t[start:end],
                condition,
                prev_pred=prev_pred[start:end],
            )
            expected = (end - start, n_cells, n_genes)
            for key in ("x_prev", "x0_pred"):
                actual = None if key not in batch_out else tuple(batch_out[key].shape)
                if actual != expected:
                    raise ValueError(
                        f"Population sampler output '{key}' has shape {actual}; "
                        f"expected {expected}."
                    )
                outputs[key].append(batch_out[key])
        return {key: torch.cat(value, dim=0) for key, value in outputs.items()}

    @staticmethod
    def _population_condition_batch(
        condition: Dict,
        *,
        start: int,
        end: int,
        total: int,
    ) -> Dict:
        """Select or broadcast one cell-set condition over particle batches."""
        batch_size = end - start
        selected = {}
        for key, value in condition.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] == total:
                    selected[key] = value[start:end]
                elif value.shape[0] == 1:
                    selected[key] = value.expand(batch_size, *value.shape[1:])
                else:
                    raise ValueError(
                        f"Population condition {key!r} has leading size {value.shape[0]}; "
                        f"expected 1 or {total}."
                    )
            elif isinstance(value, list):
                if len(value) == total:
                    selected[key] = value[start:end]
                elif len(value) == 1:
                    selected[key] = value * batch_size
                else:
                    raise ValueError(
                        f"Population condition {key!r} has length {len(value)}; "
                        f"expected 1 or {total}."
                    )
            else:
                selected[key] = value
        return selected

    def _condition_batch(
        self,
        condition: Dict,
        start: int,
        end: int,
        total: int,
        cells_per_particle: int,
    ) -> Dict:
        """Select/broadcast condition values for a flattened cell mini-batch."""
        batch_size = end - start
        cell_indices = torch.arange(start, end) % cells_per_particle
        selected = {}
        for key, value in condition.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] == total:
                    selected[key] = value[start:end]
                elif value.shape[0] == cells_per_particle:
                    selected[key] = value.index_select(0, cell_indices.to(value.device))
                elif value.shape[0] == 1:
                    selected[key] = value.expand(batch_size, *value.shape[1:])
                else:
                    selected[key] = value
            elif isinstance(value, list):
                if len(value) == total:
                    selected[key] = value[start:end]
                elif len(value) == cells_per_particle:
                    selected[key] = [value[j] for j in cell_indices.tolist()]
                elif len(value) == 1:
                    selected[key] = value * batch_size
                else:
                    selected[key] = value
            else:
                selected[key] = value
        return selected

    def _aggregate_output(self, particles: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Aggregate final particles into output based on configured mode.

        Args:
            particles: Final particle batches. Shape: (N, M, G)
            weights: Normalized particle weights. Shape: (N,)

        Returns:
            Aggregated output. Shape depends on output_mode.
        """
        mode = self.config.output_mode

        if mode == "map":
            # Return the highest-weight empirical distribution (M, G).
            return particles[torch.argmax(weights)]

        if mode == "weighted_mean":
            # Barycentric consensus over aligned cell positions (M, G).
            return (particles * weights[:, None, None]).sum(dim=0)

        elif mode == "top_k":
            # Concatenate cells from the top-K distribution particles.
            k = min(self.config.top_k, particles.shape[0])
            _, top_indices = torch.topk(weights, k)
            return particles[top_indices].flatten(0, 1)

        elif mode == "all":
            # Flatten all particle batches into one cell population.
            return particles.flatten(0, 1)

        else:
            raise ValueError(f"Unknown output_mode: {mode}")
