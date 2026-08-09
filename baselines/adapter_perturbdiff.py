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
        if start_time < 1:
            raise ValueError("start_time must be positive.")
        self._start_time = min(start_time, diffusion.num_timesteps)
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

        # The engine supplies a mini-batch-aligned condition. Fall back to the
        # condition stored at construction for native sampling.
        cond = condition or self.condition_dict

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
        cont_emb = self._match_batch(cond["cont_emb"], N)  # (N, 1, G)
        control_zeros = torch.zeros_like(cont_emb)
        control_input = torch.cat([cont_emb, control_zeros], dim=-1)  # (N, 1, 2G)

        # Expand batch_emb: (1, D) → (N, D)
        batch_emb = self._match_batch(cond["batch_emb"], N)

        # Expand gene_emb if present: (1, G, 5120) → (N, G, 5120)
        gene_emb = cond.get("gene_emb")
        if gene_emb is not None:
            gene_emb = self._match_batch(gene_emb, N)

        ds_name = cond["ds_name"]
        if len(ds_name) == 1:
            ds_name = ds_name * N
        elif len(ds_name) != N:
            raise ValueError(f"ds_name batch has length {len(ds_name)}, expected 1 or {N}.")

        # Build self_condition for Cross_DiT
        model_cond = {
            "batch_emb": batch_emb,
            "cont_emb": cont_emb,
            "gene_emb": gene_emb,
            "ds_name": ds_name,
        }

        # Timestep: Cross_DiT expects (N, 1) after _scale_timesteps
        # PerturbDiff uses rescale_timesteps: t_scaled = t * 1000/T
        rescale = getattr(self.diffusion, "rescale_timesteps", False)
        if rescale:
            t_float = t.float() * (1000.0 / self.diffusion.num_timesteps)
        else:
            t_float = t.float()
        t_input = t_float.unsqueeze(1)  # (N, 1)

        # --- Forward pass through Cross_DiT ---
        with torch.no_grad():
            output = self.model(x_input, control_input, t_input, self_condition=model_cond)

        # Model output is x_start prediction (raw, NOT clipped yet)
        model_output = output["x"]  # (N, 1, G)
        model_output = model_output.squeeze(1)  # (N, G)

        # NOTE: In PerturbDiff source, process_xstart (clipping) is applied AFTER CFG.
        # We must NOT clip before CFG computation.
        cutoff = getattr(self.model.model_cfg, "cutoff", 0.0) if self.clip_denoised else 0.0

        pred_xstart = model_output  # raw, unclipped

        # --- Classifier-free guidance ---
        if self.guidance_strength != 0.0:
            # Unconditional forward: only gene_emb + ds_name (drop batch_emb)
            # This matches PerturbDiff source: self_condition={"gene_emb": ..., "ds_name": ...}
            uncond_cond = {
                "gene_emb": gene_emb,
                "ds_name": ds_name,
            }
            with torch.no_grad():
                uncond_output = self.model(
                    x_input, control_input, t_input, self_condition=uncond_cond
                )
            uncond_xstart = uncond_output["x"].squeeze(1)  # raw, unclipped

            # Guided prediction in epsilon space
            # eps_cond = (x_t - sqrt(alpha_t) * pred_xstart) / sqrt(1-alpha_t)
            alpha_t = self.alphas_cumprod[t].unsqueeze(1)  # (N, 1)
            sqrt_alpha = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha = torch.sqrt(1.0 - alpha_t)

            eps_cond = (x_t - sqrt_alpha * pred_xstart) / sqrt_one_minus_alpha
            eps_uncond = (x_t - sqrt_alpha * uncond_xstart) / sqrt_one_minus_alpha
            eps_guided = (
                1 + self.guidance_strength
            ) * eps_cond - self.guidance_strength * eps_uncond

            # Convert back to x_start
            pred_xstart = (x_t - sqrt_one_minus_alpha * eps_guided) / sqrt_alpha

        # Apply clipping AFTER CFG (matches PerturbDiff source: process_xstart after guidance)
        if self.clip_denoised:
            pred_xstart = pred_xstart.masked_fill(pred_xstart < cutoff, 0.0)

        # --- DDIM step ---
        alpha_bar = self.alphas_cumprod[t].unsqueeze(1)  # (N, 1)
        alpha_bar_prev = self.alphas_cumprod_prev[t].unsqueeze(1)  # (N, 1)

        # Compute eps from x_start: eps = (x_t - sqrt(alpha_t) * x0) / sqrt(1-alpha_t)
        eps = (x_t - torch.sqrt(alpha_bar) * pred_xstart) / torch.sqrt(1.0 - alpha_bar)

        # DDIM formula
        sigma = (
            self.eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        mean_pred = (
            pred_xstart * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )

        # Add noise only if t > 0 and eta > 0
        noise = torch.randn_like(x_t) if self.eta > 0 else torch.zeros_like(x_t)
        nonzero_mask = (t != 0).float().unsqueeze(1)  # (N, 1)
        x_prev = mean_pred + nonzero_mask * sigma * noise

        return {
            "x_prev": x_prev,
            "x0_pred": pred_xstart,
        }

    @staticmethod
    def _match_batch(value: torch.Tensor, batch_size: int) -> torch.Tensor:
        if value.shape[0] == batch_size:
            return value
        if value.shape[0] == 1:
            return value.expand(batch_size, *value.shape[1:])
        raise ValueError(f"Condition batch has size {value.shape[0]}, expected 1 or {batch_size}.")


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
        self._cov_cfg = None
        self._gene_embedding_matrix = None

        # Add PerturbDiff to path
        if self.perturbdiff_path not in sys.path:
            sys.path.insert(0, self.perturbdiff_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        gene_names: Optional[List[str]] = None,
        ctrl_adata=None,
        **kwargs,
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
        cov_cfg = getattr(self._pl_model, "cov_encoding_cfg", None)
        if cov_cfg is None:
            cov_cfg = hparams.get("cov_encoding_cfg", None)
        if cov_cfg is not None:
            self._cov_cfg = cov_cfg
            self._pert_dict = dict(self._cfg_get(cov_cfg, "pert_dict", {}) or {})
            self._cell_type_dict = dict(self._cfg_get(cov_cfg, "cell_type_dict", {}) or {})
            self._batch_dict = dict(self._cfg_get(cov_cfg, "batch_dict", {}) or {})

        # Gene names
        self._gene_names = gene_names

        # Store control data
        self._ctrl_adata = ctrl_adata

        # Build gene_name_embedding_cache if model uses gene embeddings
        # PerturbDiff's gene_embedding module requires a pre-built cache
        # mapping ds_name -> (G, 5120) tensor of gene name embeddings.
        if self._pl_model.gene_embedding is not None and gene_names is None:
            raise ValueError("gene_names are required by this PerturbDiff checkpoint.")
        if self._pl_model.gene_embedding is not None and gene_names is not None:
            embedding_dict = self._pl_model.gene_embedding
            if not embedding_dict:
                raise ValueError(
                    "Checkpoint requires gene embeddings, but no embedding table was loaded."
                )
            example = next(iter(embedding_dict.values()))
            missing = [name for name in gene_names if name not in embedding_dict]
            vectors = [embedding_dict.get(name, torch.zeros_like(example)) for name in gene_names]
            self._gene_embedding_matrix = torch.stack(vectors)
            if missing:
                print(
                    f"[PerturbDiffAdapter] Warning: {len(missing)}/{len(gene_names)} "
                    "genes have no name embedding and use zeros."
                )

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
        pert_idx = self._lookup_perturbation(perturbation)
        if self._cell_type_dict:
            ct_idx = self._lookup_case_insensitive(self._cell_type_dict, cell_type, default=None)
            if ct_idx is None:
                raise ValueError(
                    f"Cell type '{cell_type}' is absent from the checkpoint vocabulary."
                )
        else:
            ct_idx = 0
        batch_key = f"{ds_name}_{batch_name}" if self._batch_dict else batch_name
        if self._batch_dict:
            batch_idx = self._lookup_case_insensitive(self._batch_dict, batch_key, default=None)
            if batch_idx is None and batch_name == "default":
                candidates = [
                    value
                    for key, value in self._batch_dict.items()
                    if str(key).lower().startswith(f"{ds_name.lower()}_")
                ]
                if len(candidates) == 1:
                    batch_idx = candidates[0]
            if batch_idx is None:
                raise ValueError(f"Batch '{batch_key}' is absent from the checkpoint vocabulary.")
        else:
            batch_idx = 0

        pert_tensor = torch.tensor([pert_idx], dtype=torch.long, device=device)
        ct_tensor = torch.tensor([ct_idx], dtype=torch.long, device=device)
        batch_tensor = torch.tensor([batch_idx], dtype=torch.long, device=device)

        # Get batch_emb from CovEncoder
        with torch.no_grad():
            batch_emb = self._pl_model.cov_encoder(pert_tensor, ct_tensor, batch_tensor)
        # batch_emb shape: (1, cov_output_dim)

        # --- Control expression ---
        if ctrl_expr is None:
            raise ValueError("ctrl_expr is required for PerturbDiff conditioning.")
        else:
            if ctrl_expr.dim() == 1:
                cont_emb = ctrl_expr.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, G)
            elif ctrl_expr.dim() == 2:
                # Preserve the control population: one control cell conditions
                # each generated cell in the distribution particle.
                cont_emb = ctrl_expr.unsqueeze(1).to(device)  # (M, 1, G)
            else:
                cont_emb = ctrl_expr.to(device)

        # --- Gene embeddings ---
        gene_emb = None
        if self._pl_model.gene_embedding is not None and ds_name:
            if (
                ds_name not in self._pl_model.gene_name_embedding_cache
                and self._gene_embedding_matrix is not None
            ):
                self._pl_model.gene_name_embedding_cache[ds_name] = self._gene_embedding_matrix
            # Check cache first
            if ds_name in self._pl_model.gene_name_embedding_cache:
                gene_emb = self._pl_model.gene_name_embedding_cache[ds_name].unsqueeze(0).to(device)
            # Otherwise gene_emb stays None (model handles this gracefully)

        condition_dict = {
            "batch_emb": batch_emb,  # (1, D)
            "cont_emb": cont_emb,  # (M, 1, G)
            "gene_emb": gene_emb,  # (1, G, embedding_dim) or None
            "ds_name": [[ds_name]],
        }

        return condition_dict

    @staticmethod
    def _lookup_case_insensitive(mapping: Dict, key: str, default):
        if key in mapping:
            return mapping[key]
        lowered = {str(k).lower(): v for k, v in mapping.items()}
        return lowered.get(key.lower(), default)

    @staticmethod
    def _cfg_get(config, key: str, default=None):
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _lookup_perturbation(self, perturbation: str) -> int:
        if not self._pert_dict:
            return -1
        if perturbation in self._pert_dict:
            return self._pert_dict[perturbation]

        # One-hot PerturbDiff reserves input -1 (embedding index 0 after +1)
        # as the neutral perturbation. Other encoders index directly and must
        # use an explicit control entry instead of Python's accidental -1 index.
        pert_encoding = self._cfg_get(self._cov_cfg, "pert_encoding", "onehot")
        drug_encoding = self._cfg_get(self._cov_cfg, "drug_encoding", "onehot")
        gene_encoding = self._cfg_get(self._cov_cfg, "replogle_gene_encoding", "onehot")
        if (
            pert_encoding != "non"
            and drug_encoding != "chemberta_cls"
            and gene_encoding != "genept"
        ):
            return -1

        for control_name in ("ctrl", "control", "non-targeting", "vehicle"):
            if control_name in self._pert_dict:
                return self._pert_dict[control_name]
        raise ValueError(
            f"Perturbation '{perturbation}' is absent from the checkpoint vocabulary, "
            "and this covariate encoder has no explicit control token."
        )

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
        batch_name = kwargs.get("batch_name", "default")
        ds_name = kwargs.get("ds_name", "norman")
        for cond_str in conditions:
            condition_ctrl = ctrl_expr
            if condition_ctrl is not None and condition_ctrl.dim() == 2:
                if condition_ctrl.shape[0] < n_samples:
                    repeats = (n_samples + condition_ctrl.shape[0] - 1) // condition_ctrl.shape[0]
                    condition_ctrl = condition_ctrl.repeat(repeats, 1)
                condition_ctrl = condition_ctrl[:n_samples]
            # Build condition
            cond_dict = self.build_condition(
                perturbation=cond_str,
                cell_type=cell_type,
                batch_name=batch_name,
                ctrl_expr=condition_ctrl,
                ds_name=ds_name,
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
