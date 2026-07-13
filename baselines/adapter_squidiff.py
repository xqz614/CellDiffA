"""
Squidiff Adapter for CellDiffA.

Wraps Squidiff (He et al., 2025) with the unified BaseAdapter interface AND
exposes a DiffusionSampler that satisfies DiffusionSamplerProtocol for
CellDiffA's SMC engine.

Key technical details (from source code analysis):
    - Squidiff uses an MLP-based denoiser (MLPModel) with timestep embedding.
    - Model predicts EPSILON (noise), not x_start.
    - Conditioning via z_mod: a latent vector from an encoder that encodes
      (control_feature, drug_dose) or (x_start, group).
    - For inference (sampling), z_mod is pre-computed and passed directly.
    - Uses SpacedDiffusion (DDIM with configurable timestep spacing).
    - Default: 1000 diffusion steps, DDIM with 100 steps ('ddim100').
    - No self-conditioning mechanism.
    - No classifier-free guidance in the original implementation.

Architecture:
    EncoderMLPModel: control_feature + drug_dose → z_sem (latent, dim=60)
    MLPModel: x_t + timestep_emb + z_sem → predicted_noise (epsilon)

For CellDiffA integration:
    - We expose step-by-step DDIM sampling via SquidiffSampler.
    - The conditioning is encoded as z_mod (latent representation of perturbation).
    - For OOD perturbations, z_mod can be computed from drug structure (SMILES)
      or from a nearest-neighbor approach in the latent space.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_adapter import BaseAdapter


class SquidiffSampler:
    """
    Step-by-step DDIM sampler extracted from a trained Squidiff model.

    Satisfies DiffusionSamplerProtocol for CellDiffA's SMC engine.

    Squidiff predicts epsilon (noise), so we derive x0_pred via:
        x0_pred = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
    """

    def __init__(
        self,
        model,
        diffusion,
        z_mod: torch.Tensor,
        device: str = "cuda",
        eta: float = 0.0,
        clip_denoised: bool = False,
    ):
        """
        Args:
            model: Trained Squidiff MLPModel.
            diffusion: SpacedDiffusion object (with respaced timesteps).
            z_mod: Pre-computed conditioning latent. Shape: (1, latent_dim) or (N, latent_dim).
            device: Computation device.
            eta: DDIM noise scale (0 = deterministic).
            clip_denoised: Whether to clip x0_pred to [0, +inf).
        """
        self.model = model
        self.diffusion = diffusion
        self.z_mod = z_mod
        self.device = torch.device(device)
        self.eta = eta
        self.clip_denoised = clip_denoised

        # Pre-extract schedule parameters
        self.alphas_cumprod = torch.tensor(
            diffusion.alphas_cumprod, dtype=torch.float32, device=self.device
        )
        self.alphas_cumprod_prev = torch.tensor(
            diffusion.alphas_cumprod_prev, dtype=torch.float32, device=self.device
        )

    @property
    def num_timesteps(self) -> int:
        """Number of reverse steps (after respacing, e.g., 100 for ddim100)."""
        return self.diffusion.num_timesteps

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample initial noise x_T ~ N(0, I)."""
        return torch.randn(shape, device=device)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        prev_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one DDIM reverse step using Squidiff's model.

        Squidiff predicts epsilon. We compute:
            x0_pred = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
            x_prev via DDIM formula.

        Args:
            x_t: Noisy samples. Shape: (N, G)
            t: Timestep tensor (in respaced index space). Shape: (N,)
            condition: Ignored (we use self.z_mod set at init).
            prev_pred: Ignored (Squidiff has no self-conditioning).

        Returns:
            Dict with 'x_prev' and 'x0_pred'.
        """
        N, G = x_t.shape
        self.model.eval()

        # Expand z_mod to match batch size
        z_mod = self.z_mod
        if z_mod.shape[0] == 1 and N > 1:
            z_mod = z_mod.expand(N, -1)
        elif z_mod.shape[0] != N:
            z_mod = z_mod[:N] if z_mod.shape[0] > N else z_mod.expand(N, -1)

        # Model forward: predicts epsilon
        model_kwargs = {"z_mod": z_mod.to(self.device)}

        with torch.no_grad():
            # The SpacedDiffusion's _WrappedModel handles timestep remapping
            # But since we're calling the model directly, we need to handle it
            # Use the diffusion's timestep_map for proper remapping
            if hasattr(self.diffusion, 'timestep_map'):
                map_tensor = torch.tensor(
                    self.diffusion.timestep_map, device=self.device, dtype=t.dtype
                )
                actual_t = map_tensor[t]
                # Rescale if needed
                if getattr(self.diffusion, 'rescale_timesteps', False):
                    model_t = actual_t.float() * (
                        1000.0 / self.diffusion.original_num_steps
                    )
                else:
                    model_t = actual_t.float()
            else:
                model_t = t.float()

            eps_pred = self.model(x_t, model_t, **model_kwargs)

        # Compute x0_pred from epsilon prediction
        alpha_bar = self.alphas_cumprod[t].unsqueeze(1)  # (N, 1)
        sqrt_alpha = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_bar)

        pred_xstart = (x_t - sqrt_one_minus_alpha * eps_pred) / sqrt_alpha

        # Squidiff ALWAYS clips x0_pred to [0, +inf) for gene expression
        # (In Squidiff source: process_xstart returns x.clamp(0,) when clip_denoised=False)
        # The clip_denoised flag in Squidiff means clip to [-1,1] (for images),
        # while default behavior is always [0, +inf) for gene expression.
        pred_xstart = pred_xstart.clamp(min=0.0)

        # DDIM step
        alpha_bar_prev = self.alphas_cumprod_prev[t].unsqueeze(1)  # (N, 1)

        # Recompute eps from (potentially clipped) pred_xstart
        eps = (x_t - sqrt_alpha * pred_xstart) / sqrt_one_minus_alpha

        sigma = self.eta * torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)

        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        # Add noise only if t > 0 and eta > 0
        noise = torch.randn_like(x_t) if self.eta > 0 else torch.zeros_like(x_t)
        nonzero_mask = (t != 0).float().unsqueeze(1)  # (N, 1)
        x_prev = mean_pred + nonzero_mask * sigma * noise

        return {
            "x_prev": x_prev,
            "x0_pred": pred_xstart,
        }


class SquidiffAdapter(BaseAdapter):
    """
    Adapter for the Squidiff perturbation prediction model.

    Handles:
        1. Loading Squidiff from checkpoint (model state_dict).
        2. Building proper condition tensors (z_mod from encoder or precomputed).
        3. Standard inference (DDIM loop without CellDiffA).
        4. Exposing DiffusionSampler for CellDiffA's SMC engine.

    Requirements:
        - Squidiff source: set SQUIDIFF_PATH env var to the cloned repo path.
        - Checkpoint: model state_dict file (.pt).
    """

    def __init__(
        self,
        squidiff_path: Optional[str] = None,
        device: str = "cuda",
        gene_size: int = 2000,
        output_dim: int = 2000,
        num_layers: int = 3,
        use_encoder: bool = True,
        use_drug_structure: bool = True,
        drug_dimension: int = 1024,
        comb_num: int = 1,
        diffusion_steps: int = 1000,
        timestep_respacing: str = "ddim100",
        seed: int = 42,
    ):
        super().__init__(model_name="Squidiff", device=device)
        self.squidiff_path = squidiff_path or os.environ.get(
            "SQUIDIFF_PATH", "./external/Squidiff"
        )
        self.seed = seed

        # Model architecture params
        self._gene_size = gene_size
        self._output_dim = output_dim
        self._num_layers = num_layers
        self._use_encoder = use_encoder
        self._use_drug_structure = use_drug_structure
        self._drug_dimension = drug_dimension
        self._comb_num = comb_num
        self._diffusion_steps = diffusion_steps
        self._timestep_respacing = timestep_respacing

        # State
        self._model = None
        self._diffusion = None
        self._encoder = None
        self._gene_names = None
        self._ctrl_mean = None
        self._pert_embeddings = {}  # Cache: perturbation -> z_mod

    def fit(self, adata_train=None, adata_val=None, **kwargs) -> None:
        """Training is handled by Squidiff's own training script."""
        raise NotImplementedError(
            "Squidiff training should be done using the official training script. "
            "Use load_checkpoint() to load a pre-trained model."
        )

    def load_checkpoint(
        self,
        checkpoint_path: str,
        gene_names: Optional[List[str]] = None,
        ctrl_adata=None,
        **kwargs,
    ) -> None:
        """
        Load a pre-trained Squidiff checkpoint.

        Args:
            checkpoint_path: Path to model .pt file or directory containing it.
            gene_names: List of gene names (HVGs).
            ctrl_adata: Control cell AnnData for conditioning.
        """
        # Add Squidiff to path
        if self.squidiff_path not in sys.path:
            sys.path.insert(0, self.squidiff_path)

        from Squidiff.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion,
            args_to_dict,
        )
        from Squidiff import dist_util

        # Build model args
        args = model_and_diffusion_defaults()
        args.update({
            "gene_size": self._gene_size,
            "output_dim": self._output_dim,
            "num_layers": self._num_layers,
            "use_encoder": self._use_encoder,
            "use_drug_structure": self._use_drug_structure,
            "drug_dimension": self._drug_dimension,
            "comb_num": self._comb_num,
            "diffusion_steps": self._diffusion_steps,
            "timestep_respacing": self._timestep_respacing,
            "use_ddim": True,
        })

        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )

        # Load weights
        model_path = checkpoint_path
        if os.path.isdir(checkpoint_path):
            # Look for .pt file in directory
            pt_files = [f for f in os.listdir(checkpoint_path) if f.endswith(".pt")]
            if pt_files:
                model_path = os.path.join(checkpoint_path, sorted(pt_files)[-1])
            else:
                raise FileNotFoundError(f"No .pt file found in {checkpoint_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self._model = model
        self._diffusion = diffusion
        self._gene_names = gene_names or []

        # Store control mean for conditioning
        if ctrl_adata is not None:
            ctrl_X = ctrl_adata.X
            if hasattr(ctrl_X, "toarray"):
                ctrl_X = ctrl_X.toarray()
            self._ctrl_mean = np.mean(ctrl_X, axis=0)

        self.is_trained = True

    def build_condition(
        self,
        perturbation: str,
        ctrl_expr: Optional[torch.Tensor] = None,
        drug_embedding: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the condition dictionary for Squidiff inference.

        For Squidiff, the condition is a latent vector z_mod produced by the
        encoder from (control_feature, drug_dose). For OOD perturbations where
        we don't have SMILES, we can:
            1. Use a pre-computed z_mod if available.
            2. Use the encoder with a zero drug embedding (neutral).
            3. Use a nearest-neighbor z_mod from training set.

        Args:
            perturbation: Perturbation name (e.g., "CBL+CNN1").
            ctrl_expr: Control expression vector. Shape: (G,) or (M, G).
            drug_embedding: Pre-computed drug fingerprint. Shape: (drug_dim,).

        Returns:
            Condition dict with 'z_mod' key.
        """
        device = torch.device(self.device)

        # Check cache first
        if perturbation in self._pert_embeddings:
            z_mod = self._pert_embeddings[perturbation].to(device)
            return {"z_mod": z_mod.unsqueeze(0)}  # (1, latent_dim)

        # If drug_embedding is provided, compute z_mod via encoder
        if drug_embedding is not None and self._use_encoder and self._model is not None:
            encoder = self._model.encoder if hasattr(self._model, "encoder") else None
            if encoder is not None:
                # Prepare control feature
                if ctrl_expr is not None:
                    if ctrl_expr.dim() == 1:
                        ctrl_feat = ctrl_expr.unsqueeze(0).to(device)
                    else:
                        ctrl_feat = ctrl_expr.mean(dim=0, keepdim=True).to(device)
                elif self._ctrl_mean is not None:
                    ctrl_feat = torch.tensor(
                        self._ctrl_mean, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                else:
                    ctrl_feat = torch.zeros(1, self._gene_size, device=device)

                drug_emb = drug_embedding.unsqueeze(0).to(device) if drug_embedding.dim() == 1 else drug_embedding.to(device)

                with torch.no_grad():
                    encoder.eval()
                    z_mod = encoder(
                        x_start=None,
                        label=None,
                        drug_dose=drug_emb,
                        control_feature=ctrl_feat,
                    )
                return {"z_mod": z_mod}  # (1, latent_dim)

        # Fallback: use zero latent (neutral/unconditional)
        latent_dim = 60  # Squidiff default
        z_mod = torch.zeros(1, latent_dim, device=device)
        return {"z_mod": z_mod}

    def get_diffusion_sampler(
        self,
        condition_dict: Optional[Dict[str, torch.Tensor]] = None,
        eta: float = 0.0,
        **kwargs,
    ) -> SquidiffSampler:
        """
        Return a step-by-step sampler for CellDiffA's SMC engine.

        Args:
            condition_dict: Pre-built condition from build_condition().
            eta: DDIM noise (0 = deterministic).

        Returns:
            SquidiffSampler satisfying DiffusionSamplerProtocol.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        z_mod = condition_dict["z_mod"] if condition_dict else torch.zeros(
            1, 60, device=self.device
        )

        return SquidiffSampler(
            model=self._model,
            diffusion=self._diffusion,
            z_mod=z_mod,
            device=self.device,
            eta=eta,
            clip_denoised=True,
        )

    def predict(
        self,
        conditions: List[str],
        n_samples: int = 100,
        ctrl_expr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions using standard DDIM sampling (without CellDiffA).

        Uses Squidiff's built-in sample_fn (ddim_sample_loop) for efficiency.

        Args:
            conditions: List of perturbation condition strings.
            n_samples: Number of cells to generate per condition.
            ctrl_expr: Control expression for conditioning.

        Returns:
            Dict mapping condition string → expression matrix (n_samples, num_genes).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        results = {}
        gene_size = self._output_dim

        for cond_str in conditions:
            # Build condition
            cond_dict = self.build_condition(
                perturbation=cond_str,
                ctrl_expr=ctrl_expr,
            )

            z_mod = cond_dict["z_mod"]
            if z_mod.shape[0] == 1:
                z_mod = z_mod.expand(n_samples, -1)

            # Use Squidiff's built-in DDIM loop
            sample_fn = self._diffusion.ddim_sample_loop
            with torch.no_grad():
                samples = sample_fn(
                    self._model,
                    shape=(n_samples, gene_size),
                    model_kwargs={"z_mod": z_mod.to(self.device)},
                    noise=None,
                    clip_denoised=True,
                    device=self.device,
                )

            results[cond_str] = samples.cpu().numpy()

        return results

    def register_perturbation_embedding(
        self,
        perturbation: str,
        z_mod: torch.Tensor,
    ) -> None:
        """
        Register a pre-computed z_mod for a perturbation condition.

        This is useful for OOD perturbations where we want to use a
        specific latent representation (e.g., from nearest-neighbor lookup).

        Args:
            perturbation: Perturbation name.
            z_mod: Latent vector. Shape: (latent_dim,) or (1, latent_dim).
        """
        if z_mod.dim() == 2:
            z_mod = z_mod.squeeze(0)
        self._pert_embeddings[perturbation] = z_mod.cpu()

    def save_checkpoint(self, path: str) -> None:
        """Save Squidiff checkpoint."""
        os.makedirs(path, exist_ok=True)
        if self._model is not None:
            torch.save(
                self._model.state_dict(),
                os.path.join(path, "model.pt"),
            )

    @property
    def is_generative(self) -> bool:
        """Squidiff is a diffusion-based generative model."""
        return True
