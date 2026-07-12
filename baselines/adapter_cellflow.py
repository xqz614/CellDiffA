"""
CellFlow Adapter for CellDiffA.

Wraps CellFlow (theislab, 2025) with the unified BaseAdapter interface AND
exposes a DiffusionSampler that satisfies DiffusionSamplerProtocol for
CellDiffA's SMC engine.

Key technical details (from source code analysis):
    - CellFlow uses Optimal Transport Flow Matching (OTFM) with a conditional
      velocity field (ConditionalVelocityField) implemented in JAX/Flax.
    - The velocity field takes (t, x_t, condition, encoder_noise) and outputs
      the velocity v(x_t, t | condition).
    - ODE is solved from t=0 (source/noise) to t=1 (target/perturbed) using
      diffrax with adaptive step-size control (Tsit5 + PID controller).
    - Conditioning is done via perturbation covariates (drug name, dose, etc.)
      encoded through a learned condition encoder.
    - CellFlow's predict() takes control cells as source and pushes them forward
      under the learned flow conditioned on the perturbation.

For CellDiffA integration:
    - We expose step-by-step Euler integration via CellFlowSampler.
    - The flow goes from t=0 (control/source) to t=1 (perturbed/target).
    - We discretize into N steps and expose each step for SMC guidance.
    - x0_pred at each step is the endpoint estimate: x_t + (1-t)*v(x_t, t).

Note: CellFlow is JAX-based. The sampler converts between JAX arrays and
PyTorch tensors at the boundary for compatibility with CellDiffA's SMC engine.

Requirements:
    pip install cellflow-tools
    (JAX must be installed; CPU-only is sufficient for inference)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base_adapter import BaseAdapter


class CellFlowSampler:
    """
    Step-by-step Euler sampler extracted from a trained CellFlow model.

    Satisfies DiffusionSamplerProtocol for CellDiffA's SMC engine.

    CellFlow uses flow matching: the ODE goes from t=0 (source) to t=1 (target).
    We discretize into num_steps Euler steps. For compatibility with the SMC engine
    (which counts timesteps in reverse), we map:
        SMC timestep T-1 → flow time 0 (start)
        SMC timestep 0   → flow time 1 (end)

    At each step, we compute:
        v = velocity_field(x_t, flow_t, condition)
        x_{t+dt} = x_t + dt * v
        x1_pred = x_t + (1 - flow_t) * v  (endpoint estimate for reward)
    """

    def __init__(
        self,
        cellflow_model,
        condition: Dict[str, np.ndarray],
        source_cells: np.ndarray,
        num_steps: int = 100,
        device: str = "cuda",
    ):
        """
        Args:
            cellflow_model: Trained CellFlow model instance.
            condition: Condition dict for the velocity field (perturbation encoding).
            source_cells: Control/source cells to transport. Shape: (M, G).
            num_steps: Number of Euler discretization steps.
            device: PyTorch device for output tensors.
        """
        self.cf_model = cellflow_model
        self.condition = condition
        self.source_cells = source_cells
        self._num_steps = num_steps
        self.device = torch.device(device)
        self.dt = 1.0 / num_steps

        # Cache the solver and velocity field state
        self._solver = cellflow_model.solver
        self._vf_state = self._solver.vf_state_inference

    @property
    def num_timesteps(self) -> int:
        """Number of discrete steps in the flow."""
        return self._num_steps

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        For flow matching, the 'noise' is actually the source distribution.
        We sample from the control cells (with replacement if needed).

        Args:
            shape: (N, G) where N is num_particles, G is num_genes.

        Returns:
            Source samples as PyTorch tensor.
        """
        N, G = shape
        M = self.source_cells.shape[0]

        if N <= M:
            # Subsample without replacement
            indices = np.random.choice(M, size=N, replace=False)
        else:
            # Sample with replacement
            indices = np.random.choice(M, size=N, replace=True)

        source_samples = self.source_cells[indices]

        # Add small noise for diversity (optional, helps SMC exploration)
        noise_scale = 0.01
        source_samples = source_samples + noise_scale * np.random.randn(*source_samples.shape)

        return torch.tensor(source_samples, dtype=torch.float32, device=device)

    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        prev_pred: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one Euler step of the flow ODE.

        Maps SMC reverse timestep to flow time:
            flow_t = 1.0 - (t / num_steps)
            So t=num_steps-1 → flow_t ≈ 0 (start), t=0 → flow_t ≈ 1 (end)

        Actually for flow matching going forward:
            flow_t = (num_steps - 1 - t) / num_steps * 1.0
            Simpler: flow_t = 1.0 - (t+1)/num_steps

        We use the convention that SMC counts down from T-1 to 0,
        and flow goes from 0 to 1.

        Args:
            x_t: Current state. Shape: (N, G)
            t: Timestep tensor (counting down). Shape: (N,)
            condition: Ignored (we use self.condition set at init).
            prev_pred: Ignored (CellFlow has no self-conditioning).

        Returns:
            Dict with 'x_prev' (next state) and 'x0_pred' (endpoint estimate).
        """
        import jax
        import jax.numpy as jnp

        N, G = x_t.shape

        # Convert SMC timestep to flow time
        # SMC: t goes from num_steps-1 down to 0
        # Flow: we want to go from 0 to 1
        # At SMC t=num_steps-1, flow_t should be near 0 (start)
        # At SMC t=0, flow_t should be near 1 (end)
        t_val = t[0].item()  # All particles have same timestep
        flow_t = 1.0 - (t_val + 1) / self._num_steps

        # Convert to JAX arrays
        x_np = x_t.detach().cpu().numpy()
        x_jax = jnp.array(x_np)
        t_jax = jnp.array(flow_t, dtype=jnp.float32)

        # Get velocity from the trained model
        # The velocity field expects: (params, t, x, condition, encoder_noise)
        params = self._vf_state.params
        encoder_noise = jnp.zeros((1, self._solver.vf.condition_embedding_dim))

        # Vectorized velocity computation
        def compute_velocity(x_single):
            v, _, _ = self._vf_state.apply_fn(
                {"params": params},
                t_jax,
                x_single,
                self.condition,
                encoder_noise,
                train=False,
            )
            return v

        velocity = jax.vmap(compute_velocity)(x_jax)  # (N, G)

        # Euler step: x_{t+dt} = x_t + dt * v
        x_next_jax = x_jax + self.dt * velocity

        # Endpoint estimate: x_1 ≈ x_t + (1 - flow_t) * v
        remaining_time = 1.0 - flow_t
        x1_pred_jax = x_jax + remaining_time * velocity

        # Convert back to PyTorch
        x_next = torch.tensor(np.array(x_next_jax), dtype=torch.float32, device=self.device)
        x1_pred = torch.tensor(np.array(x1_pred_jax), dtype=torch.float32, device=self.device)

        return {
            "x_prev": x_next,
            "x0_pred": x1_pred,
        }


class CellFlowAdapter(BaseAdapter):
    """
    Adapter for the CellFlow perturbation prediction model.

    Handles:
        1. Loading CellFlow from saved checkpoint directory.
        2. Building proper condition dicts (perturbation covariates).
        3. Standard inference using CellFlow's built-in predict().
        4. Exposing DiffusionSampler for CellDiffA's SMC engine.

    Requirements:
        pip install cellflow-tools
        (JAX backend required)
    """

    def __init__(
        self,
        device: str = "cuda",
        num_integration_steps: int = 100,
        seed: int = 42,
    ):
        super().__init__(model_name="CellFlow", device=device)
        self.num_integration_steps = num_integration_steps
        self.seed = seed
        self._cf_model = None
        self._gene_names = None
        self._ctrl_cells = None
        self._ctrl_adata = None
        self._data_manager = None

    def fit(
        self,
        adata_train=None,
        adata_val=None,
        num_iterations: int = 50000,
        batch_size: int = 1024,
        perturbation_key: str = "condition",
        control_key: str = "is_control",
        sample_rep: str = "X",
        **kwargs,
    ) -> None:
        """
        Train CellFlow model.

        Args:
            adata_train: Training AnnData with perturbation annotations.
            adata_val: Optional validation AnnData.
            num_iterations: Number of training iterations.
            batch_size: Training batch size.
            perturbation_key: Key in obs for perturbation condition.
            control_key: Key in obs for control indicator.
            sample_rep: Key in obsm for sample representation (or 'X').
        """
        from cellflow.model import CellFlow

        # Initialize CellFlow
        cf = CellFlow(adata_train, solver="otfm")

        # Prepare data
        cf.prepare_data(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbation_covariates={perturbation_key: adata_train.obs[perturbation_key].unique().tolist()},
        )

        # Prepare model with defaults
        cf.prepare_model(seed=self.seed)

        # Train
        cf.train(
            num_iterations=num_iterations,
            batch_size=batch_size,
        )

        self._cf_model = cf
        self._gene_names = list(adata_train.var_names)
        self.is_trained = True

    def load_checkpoint(
        self,
        checkpoint_path: str,
        gene_names: Optional[List[str]] = None,
        ctrl_adata=None,
        **kwargs,
    ) -> None:
        """
        Load a pre-trained CellFlow checkpoint.

        CellFlow saves models as directories with model state and config.

        Args:
            checkpoint_path: Path to saved CellFlow model directory or file.
            gene_names: List of gene names (HVGs).
            ctrl_adata: Control cell AnnData for source distribution.
        """
        from cellflow.model import CellFlow

        # Load the saved model
        self._cf_model = CellFlow.load(checkpoint_path)
        self._gene_names = gene_names or []

        # Store control cells for flow source
        if ctrl_adata is not None:
            self._ctrl_adata = ctrl_adata
            ctrl_X = ctrl_adata.X
            if hasattr(ctrl_X, "toarray"):
                ctrl_X = ctrl_X.toarray()
            self._ctrl_cells = ctrl_X

        self._data_manager = self._cf_model.data_manager
        self.is_trained = True

    def build_condition(
        self,
        perturbation: str,
        ctrl_expr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Build the condition dictionary for CellFlow inference.

        CellFlow conditions on perturbation covariates (drug name, dose, etc.)
        encoded through its learned condition encoder.

        Args:
            perturbation: Perturbation name (e.g., "CBL+CNN1").
            ctrl_expr: Control expression (unused, CellFlow uses full ctrl cells).

        Returns:
            Condition dict compatible with CellFlow's velocity field.
        """
        import pandas as pd

        if self._cf_model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        # Build covariate DataFrame matching CellFlow's expected format
        # This depends on how the model was trained (perturbation_covariates config)
        dm = self._data_manager
        if dm is not None:
            # Use the data manager to encode the condition
            # Build a minimal DataFrame with the perturbation info
            perturbation_key = list(dm.perturbation_covariates.keys())[0] if dm.perturbation_covariates else "condition"
            covariate_df = pd.DataFrame({perturbation_key: [perturbation]})

            # Get condition embedding from the solver
            try:
                cond_dict = self._cf_model.solver.get_condition_embedding(
                    {perturbation_key: np.array([perturbation])}
                )
                return {"condition_embedding": cond_dict, "perturbation": perturbation}
            except Exception:
                pass

        # Fallback: return perturbation string for later processing
        return {"perturbation": perturbation}

    def get_diffusion_sampler(
        self,
        condition_dict: Optional[Dict] = None,
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> CellFlowSampler:
        """
        Return a step-by-step sampler for CellDiffA's SMC engine.

        Args:
            condition_dict: Pre-built condition from build_condition().
            num_steps: Override number of Euler steps.

        Returns:
            CellFlowSampler satisfying DiffusionSamplerProtocol.
        """
        if self._cf_model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        if self._ctrl_cells is None:
            raise RuntimeError(
                "Control cells not available. Pass ctrl_adata to load_checkpoint()."
            )

        # Get the condition for the velocity field
        condition = condition_dict.get("condition_embedding") if condition_dict else None
        if condition is None:
            # Build a neutral condition
            import jax.numpy as jnp
            cond_dim = self._cf_model.solver.vf.condition_embedding_dim
            condition = jnp.zeros((1, cond_dim))

        steps = num_steps or self.num_integration_steps

        return CellFlowSampler(
            cellflow_model=self._cf_model,
            condition=condition,
            source_cells=self._ctrl_cells,
            num_steps=steps,
            device=self.device,
        )

    def predict(
        self,
        conditions: List[str],
        n_samples: int = 100,
        ctrl_expr: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions using CellFlow's built-in ODE solver.

        This uses CellFlow's native predict() with adaptive step-size control
        (Tsit5 + PID controller) for high-quality predictions.

        Args:
            conditions: List of perturbation condition strings.
            n_samples: Number of cells to generate per condition.
            ctrl_expr: Control expression (unused, uses stored ctrl_adata).

        Returns:
            Dict mapping condition string → expression matrix (n_samples, num_genes).
        """
        if self._cf_model is None:
            raise RuntimeError("Model not loaded. Call load_checkpoint() first.")

        import pandas as pd

        results = {}

        for cond_str in conditions:
            # Prepare source cells (subsample control cells)
            if self._ctrl_cells is not None:
                M = self._ctrl_cells.shape[0]
                if n_samples <= M:
                    indices = np.random.choice(M, size=n_samples, replace=False)
                else:
                    indices = np.random.choice(M, size=n_samples, replace=True)
                source = self._ctrl_cells[indices]
            else:
                G = len(self._gene_names) if self._gene_names else 2000
                source = np.zeros((n_samples, G))

            # Build condition for this perturbation
            cond_dict = self.build_condition(cond_str)

            try:
                # Use CellFlow's native predict
                # This requires proper condition format matching training setup
                condition_emb = cond_dict.get("condition_embedding")
                if condition_emb is not None:
                    pred = self._cf_model.solver.predict(
                        x=source,
                        condition=condition_emb,
                    )
                    results[cond_str] = np.array(pred)
                else:
                    # Fallback: use the high-level predict API
                    # This requires ctrl_adata and covariate_data
                    dm = self._data_manager
                    perturbation_key = list(dm.perturbation_covariates.keys())[0] if dm else "condition"
                    covariate_df = pd.DataFrame({perturbation_key: [cond_str]})

                    pred_dict = self._cf_model.predict(
                        adata=self._ctrl_adata[:n_samples] if self._ctrl_adata is not None else None,
                        covariate_data=covariate_df,
                    )
                    if pred_dict:
                        # Get the first (and likely only) prediction
                        first_key = list(pred_dict.keys())[0]
                        results[cond_str] = pred_dict[first_key]
                    else:
                        results[cond_str] = source  # Fallback
            except Exception as e:
                # If prediction fails, return source as fallback
                import warnings
                warnings.warn(
                    f"CellFlow prediction failed for '{cond_str}': {e}. "
                    f"Returning source cells as fallback."
                )
                results[cond_str] = source

        return results

    def save_checkpoint(self, path: str) -> None:
        """Save CellFlow checkpoint."""
        if self._cf_model is not None:
            self._cf_model.save(dir_path=path, overwrite=True)

    @property
    def is_generative(self) -> bool:
        """CellFlow is a flow-based generative model."""
        return True
