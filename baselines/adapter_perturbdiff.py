"""
PerturbDiff Adapter for CellDiffA.

Wraps PerturbDiff (2026) with the unified BaseAdapter interface AND exposes a
DiffusionSampler that satisfies DiffusionSamplerProtocol for CellDiffA's SMC engine.

Key technical details (from source code analysis):
    - PerturbDiff uses Cross_DiT as backbone.
    - Model predicts x_start (ModelMeanType.START_X), NOT noise.
    - Self-conditioning: prev_pred is concatenated to x_t along last dim before input.
    - Control cells (cont_emb) are also concatenated with zeros for self-cond.
    - CovEncoder takes (pert_idx, celltype_idx, batch_idx) → batch_emb.
    - Gene embeddings are optional 5120-dim vectors per gene.
    - Official DDIM uses start_time=100 (not full 1000 steps), eta=0.0, guidance=1.0.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_adapter import BaseAdapter


class PerturbDiffSampler:
    """
    Step-by-step diffusion sampler extracted from a trained PerturbDiff model.

    Satisfies DiffusionSamplerProtocol for CellDiffA's SMC engine.

    This wraps PerturbDiff's GaussianDiffusion object and Cross_DiT model,
    correctly handling:
        1. x_start prediction (not noise prediction)
        2. Self-conditioning (prev_pred concatenation)
        3. Control cell conditioning (cont_emb)
        4. Classifier-free guidance
        5. DDIM deterministic steps
    """

    def __init__(
        self,
        pl_model,
        diffusion,
        condition_dict: Dict[str, torch.Tensor],
        device: str = "cuda",
        guidance_strength: float = 1.0,
        eta: float = 0.0,
        start_time: int = 100,
        clip_denoised: bool = True,
    ):
        """
        Args:
            pl_model: Loaded PerturbDiff PlModel (Lightning module).
            diffusion: GaussianDiffusion object from PerturbDiff.
            condition_dict: Pre-built self_condition dict containing:
                - "batch_emb": (1, cov_dim) from CovEncoder
                - "cont_emb": (1, 1, G) control cell expression
                - "gene_emb": (1, G, 5120) or None
                - "ds_name": list of dataset name strings
            device: Computation device.
            guidance_strength: Classifier-free guidance scale.
            eta: DDIM stochasticity (0 = deterministic).
            start_time: Number of reverse steps (PerturbDiff default: 100).
            clip_denoised: Whether to clip predictions by model cutoff.
        """
        self.pl_model = pl_model
        self.model = pl_model.model  # Cross_DiT
        self.diffusion = diffusion
        self.condition_dict = condition_dict
        self.device = torch.device(device)
        self.guidance_strength = guidance_strength
        self.eta = eta
        self._start_time = start_time
        self.clip_denoised = clip_denoised

        # Pre-extract schedule parameters as tensors
        self.alphas_cumprod = torch.tensor(
            diffusion.alphas_cumprod, dtype=torch.float32, device=self.device
        )
        self.alphas_cumprod_prev = torch.tensor(
            diffusion.alphas_cumprod_prev, dtype=torch.float32, device=self.device
        )

    @property
    def num_timesteps(self) -> int:
        """Return the actual number of reverse steps (start_time, not full T)."""
        return self._start_time

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
        Perform one DDIM reverse step using PerturbDiff's actual logic.

        This replicates the logic from:
            diffusion_sampling.py → p_mean_variance() → ddim_sample()

        Args:
            x_t: Noisy samples. Shape: (N, G)
            t: Timestep tensor. Shape: (N,)
            condition: Ignored (we use self.condition_dict set at init).
            prev_pred: Previous x0 prediction for self-conditioning. Shape: (N, G)

        Returns:
            Dict with 'x_prev' and 'x0_pred'.
        """
        N, G = x_t.shape
        self.model.eval()

        # Use stored condition_dict (expanded to N particles)
        cond = self.condition_dict

        # --- Build model input with self-conditioning ---
        # PerturbDiff concatenates prev_pred along last dim: x_in = [x_t, prev_pred]
        if prev_pred is None:
            prev_pred = torch.zeros_like(x_t)

        # Add sequence dimension: (N, G) → (N, 1, G)
        x_3d = x_t.unsqueeze(1)
        prev_pred_3d = prev_pred.unsqueeze(1)

        # x_input = [x_t, prev_pred] → (N, 1, 2G)
        x_input = torch.cat([x_3d, prev_pred_3d], dim=-1)

        # Control input: cont_emb with zeros for self-cond
        # cont_emb shape from condition_dict: (1, 1, G) → expand to (N, 1, G)
        cont_emb = cond["cont_emb"].expand(N, -1, -1)  # (N, 1, G)
        control_zeros = torch.zeros_like(cont_emb)
        control_input = torch.cat([cont_emb, control_zeros], dim=-1)  # (N, 1, 2G)

        # Expand batch_emb: (1, D) → (N, D)
        batch_emb = cond["batch_emb"].expand(N, -1)

        # Expand gene_emb if present: (1, G, 5120) → (N, G, 5120)
        gene_emb = cond.get("gene_emb")
        if gene_emb is not None:
            gene_emb = gene_emb.expand(N, -1, -1)

        # Build self_condition for Cross_DiT
        model_cond = {
            "batch_emb": batch_emb,
            "cont_emb": cont_emb,
            "gene_emb": gene_emb,
            "ds_name": cond["ds_name"],
        }

        # Timestep: Cross_DiT expects (N, 1) after _scale_timesteps
        # PerturbDiff uses rescale_timesteps: t_scaled = t * 1000/T
        rescale = getattr(self.diffusion, 'rescale_timesteps', False)
        if rescale:
            t_float = t.float() * (1000.0 / self.diffusion.num_timesteps)
        else:
            t_float = t.float()
        t_input = t_float.unsqueeze(1)  # (N, 1)

        # --- Forward pass through Cross_DiT ---
        with torch.no_grad():
            output = self.model(x_input, control_input, t_input, self_condition=model_cond)

        # Model output is x_start prediction
        model_output = output["x"]  # (N, 1, G)
        model_output = model_output.squeeze(1)  # (N, G)

        # Apply clipping (PerturbDiff clips values below cutoff to 0)
        if self.clip_denoised:
            cutoff = getattr(self.model.model_cfg, 'cutoff', 0.0)
            model_output = model_output.masked_fill(model_output < cutoff, 0.0)

        pred_xstart = model_output

        # --- Classifier-free guidance ---
        if self.guidance_strength != 0.0:
            # Unconditional forward: only gene_emb + ds_name (drop batch_emb)
            uncond_cond = {
                "gene_emb": gene_emb,
                "ds_name": cond["ds_name"],
            }
            with torch.no_grad():
                uncond_output = self.model(
                    x_input, control_input, t_input, self_condition=uncond_cond
                )
            uncond_xstart = uncond_output["x"].squeeze(1)
            if self.clip_denoised:
                uncond_xstart = uncond_xstart.masked_fill(uncond_xstart < cutoff, 0.0)

            # Guided prediction in epsilon space
            # eps_cond = (x_t - sqrt(alpha_t) * pred_xstart) / sqrt(1-alpha_t)
            alpha_t = self.alphas_cumprod[t].unsqueeze(1)  # (N, 1)
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)

            eps_cond = (x_t - sqrt_alpha * pred_xstart) / sqrt_one_minus_alpha
            eps_uncond = (x_t - sqrt_alpha * uncond_xstart) / sqrt_one_minus_alpha
            eps_guided = (1 + self.guidance_strength) * eps_cond - self.guidance_strength * eps_uncond

            # Convert back to x_start
            pred_xstart = (x_t - sqrt_one_minus_alpha * eps_guided) / sqrt_alpha
            if self.clip_denoised:
                pred_xstart = pred_xstart.masked_fill(pred_xstart < cutoff, 0.0)

        # --- DDIM step ---
        alpha_bar = self.alphas_cumprod[t].unsqueeze(1)  # (N, 1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].unsqueeze(1)  # (N, 1)

        # Compute eps from x_start: eps = (x_t - sqrt(alpha_t) * x0) / sqrt(1-alpha_t)
        eps = (x_t - torch.sqrt(alpha_bar) * pred_xstart) / torch.sqrt(1.0 - alpha_bar)

        # DDIM formula
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


class PerturbDiffAdapter(BaseAdapter):
    """
    Adapter for the PerturbDiff perturbation prediction model.

    Handles:
        1. Loading PerturbDiff from official Lightning checkpoint.
        2. Building proper condition tensors (CovEncoder, gene_emb, cont_emb).
        3. Standard inference (without CellDiffA).
        4. Exposing DiffusionSampler for CellDiffA's SMC engine.

    Requirements:
        - PerturbDiff source: https://github.com/DeepGraphLearning/PerturbDiff
        - Set PERTURBDIFF_PATH env var to the cloned repo path.
    """

    def __init__(
        self,
        perturbdiff_path: Optional[str] = None,
        device: str = "cuda",
        seed: int = 42,
    ):
        super().__init__(model_name="PerturbDiff", device=device)
        self.perturbdiff_path = perturbdiff_path or os.environ.get(
            "PERTURBDIFF_PATH", "./external/PerturbDiff"
        )
        self.seed = seed
        self._pl_model = None
        self._diffusion = None
        self._gene_names = None
        self._pert_dict = None
        self._cell_type_dict = None
        self._batch_dict = None

        # Add PerturbDiff to path
        if self.perturbdiff_path not in sys.path:
            sys.path.insert(0, self.perturbdiff_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        gene_names: Optional[List[str]] = None,
        ctrl_adata=None,
    ) -> None:
        """
        Load a pre-trained PerturbDiff checkpoint.

        Args:
            checkpoint_path: Path to .ckpt file (PyTorch Lightning format).
            gene_names: List of gene names (2000 HVGs). If None, loaded from ckpt.
            ctrl_adata: Control AnnData for building cont_emb. Optional.
        """
        from src.models.lightning.lightning_module import PlModel

        # Load Lightning checkpoint
        self._pl_model = PlModel.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device,
        )
        self._pl_model.eval()
        self._pl_model.to(self.device)

        # Extract key components
        self._diffusion = self._pl_model.diffusion
        self.model = self._pl_model.model

        # Extract dictionaries from checkpoint hparams
        hparams = self._pl_model.hparams
        cov_cfg = hparams.get("cov_encoding_cfg", None)
        if cov_cfg is not None:
            self._pert_dict = getattr(cov_cfg, 'pert_dict', {})
            self._cell_type_dict = getattr(cov_cfg, 'cell_type_dict', {})
            self._batch_dict = getattr(cov_cfg, 'batch_dict', {})

        # Gene names
        self._gene_names = gene_names

        # Store control data
        self._ctrl_adata = ctrl_adata

        # Build gene_name_embedding_cache if model uses gene embeddings
        # PerturbDiff's gene_embedding module requires a pre-built cache
        # mapping ds_name -> (G, 5120) tensor of gene name embeddings.
        if (
            self._pl_model.gene_embedding is not None
            and gene_names is not None
            and not self._pl_model.gene_name_embedding_cache
        ):
            try:
                from src.apps.sampling.sampling_generation_helpers import (
                    build_gene_embedding_cache,
                )
                # build_gene_embedding_cache populates the model's internal cache
                build_gene_embedding_cache(
                    self._pl_model,
                    gene_names=gene_names,
                    ds_name="norman",  # default; can be overridden per-call
                )
                print("[PerturbDiffAdapter] Gene embedding cache built successfully.")
            except (ImportError, Exception) as e:
                # If the helper is unavailable, gene_emb will be None (model handles gracefully)
                print(f"[PerturbDiffAdapter] Warning: Could not build gene embedding cache: {e}")
                print("  → gene_emb will be None (model uses null_emb fallback).")

        self.is_trained = True

    def build_condition(
        self,
        perturbation: str,
        cell_type: str = "K562",
        batch_name: str = "default",
        ctrl_expr: Optional[torch.Tensor] = None,
        ds_name: str = "norman",
    ) -> Dict[str, torch.Tensor]:
        """
        Build the complete condition dictionary for PerturbDiff inference.

        This correctly constructs all tensors needed by Cross_DiT:
            - batch_emb: from CovEncoder(pert_idx, celltype_idx, batch_idx)
            - cont_emb: control cell expression (mean of control population)
            - gene_emb: optional gene name embeddings
            - ds_name: dataset identifier

        Args:
            perturbation: Perturbation name (e.g., "CBL+CNN1").
            cell_type: Cell type name.
            batch_name: Batch/experiment identifier.
            ctrl_expr: Control expression vector. Shape: (G,) or (M, G).
            ds_name: Dataset name for gene embedding lookup.

        Returns:
            Condition dict ready for PerturbDiffSampler.
        """
        device = torch.device(self.device)

        # --- Encode covariates ---
        # Look up indices in dictionaries
        # For OOD perturbations not in pert_dict, use -1 (NOT 0).
        # PerturbDiff's CovEncoder does `pert_input + 1` before embedding lookup,
        # so -1 + 1 = 0 → index 0 is the reserved neutral/control slot.
        # This is the official behavior from dataset_core.py line 166-171.
        pert_idx = self._pert_dict.get(perturbation, -1) if self._pert_dict else -1
        ct_idx = self._cell_type_dict.get(cell_type, 0) if self._cell_type_dict else 0
        batch_key = f"{ds_name}_{batch_name}" if self._batch_dict else batch_name
        batch_idx = self._batch_dict.get(batch_key, 0) if self._batch_dict else 0

        pert_tensor = torch.tensor([pert_idx], dtype=torch.long, device=device)
        ct_tensor = torch.tensor([ct_idx], dtype=torch.long, device=device)
        batch_tensor = torch.tensor([batch_idx], dtype=torch.long, device=device)

        # Get batch_emb from CovEncoder
        with torch.no_grad():
            batch_emb = self._pl_model.cov_encoder(pert_tensor, ct_tensor, batch_tensor)
        # batch_emb shape: (1, cov_output_dim)

        # --- Control expression ---
        if ctrl_expr is None:
            # Use zeros as fallback (not ideal but won't crash)
            G = len(self._gene_names) if self._gene_names else 2000
            cont_emb = torch.zeros(1, 1, G, device=device)
        else:
            if ctrl_expr.dim() == 1:
                cont_emb = ctrl_expr.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, G)
            elif ctrl_expr.dim() == 2:
                # Take mean of control population
                cont_emb = ctrl_expr.mean(dim=0, keepdim=True).unsqueeze(0).to(device)  # (1, 1, G)
            else:
                cont_emb = ctrl_expr.to(device)

        # --- Gene embeddings ---
        gene_emb = None
        if self._pl_model.gene_embedding is not None and ds_name:
            # Check cache first
            if ds_name in self._pl_model.gene_name_embedding_cache:
                gene_emb = self._pl_model.gene_name_embedding_cache[ds_name].unsqueeze(0).to(device)
            # Otherwise gene_emb stays None (model handles this gracefully)

        condition_dict = {
            "batch_emb": batch_emb,           # (1, D)
            "cont_emb": cont_emb,             # (1, 1, G)
            "gene_emb": gene_emb,             # (1, G, 5120) or None
            "ds_name": [[ds_name]],           # nested list matching PerturbDiff format
        }

        return condition_dict

    def get_diffusion_sampler(
        self,
        condition_dict: Dict[str, torch.Tensor],
        guidance_strength: float = 1.0,
        eta: float = 0.0,
        start_time: int = 100,
    ) -> PerturbDiffSampler:
        """
        Return a step-by-step sampler for CellDiffA's SMC engine.

        Args:
            condition_dict: Pre-built condition from build_condition().
            guidance_strength: CFG strength (PerturbDiff default: 1.0).
            eta: DDIM noise (0 = deterministic).
            start_time: Number of reverse steps (PerturbDiff default: 100).

        Returns:
            PerturbDiffSampler satisfying DiffusionSamplerProtocol.
        """
        if self._pl_model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        return PerturbDiffSampler(
            pl_model=self._pl_model,
            diffusion=self._diffusion,
            condition_dict=condition_dict,
            device=self.device,
            guidance_strength=guidance_strength,
            eta=eta,
            start_time=start_time,
        )

    def predict(
        self,
        conditions: List[str],
        n_samples: int = 100,
        cell_type: str = "K562",
        ctrl_expr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions using standard DDIM sampling (without CellDiffA).

        Args:
            conditions: List of perturbation condition strings.
            n_samples: Number of cells to generate per condition.
            cell_type: Cell type for condition encoding.
            ctrl_expr: Control expression for conditioning.

        Returns:
            Dict mapping condition string → expression matrix (n_samples, num_genes).
        """
        if self._pl_model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        results = {}
        for cond_str in conditions:
            # Build condition
            cond_dict = self.build_condition(
                perturbation=cond_str,
                cell_type=cell_type,
                ctrl_expr=ctrl_expr,
            )

            # Create sampler
            sampler = self.get_diffusion_sampler(cond_dict, **kwargs)

            # Standard DDIM loop (no SMC)
            G = cond_dict["cont_emb"].shape[-1]
            x_t = sampler.sample_noise(shape=(n_samples, G), device=torch.device(self.device))
            prev_pred = torch.zeros_like(x_t)

            for t in range(sampler.num_timesteps - 1, -1, -1):
                t_tensor = torch.full(
                    (n_samples,), t, device=torch.device(self.device), dtype=torch.long
                )
                output = sampler.denoise_step(x_t, t_tensor, cond_dict, prev_pred=prev_pred)
                x_t = output["x_prev"]
                prev_pred = output["x0_pred"]

            results[cond_str] = x_t.cpu().numpy()

        return results

    def fit(self, *args, **kwargs) -> None:
        """
        Training is handled by PerturbDiff's own training script.
        This adapter is designed for checkpoint-based usage.
        """
        raise NotImplementedError(
            "PerturbDiff training should be done via its own training script. "
            "Use load_checkpoint() to load a pre-trained model."
        )

    def save_checkpoint(self, path: str) -> None:
        """Save is not needed - use original PerturbDiff checkpoints."""
        raise NotImplementedError("Use PerturbDiff's native checkpoint format.")

    @property
    def is_generative(self) -> bool:
        return True
