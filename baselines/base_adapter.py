"""
Base adapter class for all baseline models.

All baselines are wrapped with a unified interface so that CellDiffA's
evaluation pipeline can treat them interchangeably.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
from anndata import AnnData


class BaseAdapter(ABC):
    """
    Abstract base class for baseline model adapters.

    Each adapter encapsulates a specific baseline model (GEARS, CPA, PerturbDiff, etc.)
    and exposes a unified interface for training, prediction, and (optionally)
    providing a diffusion sampler for CellDiffA's SMC engine.
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.is_trained = False

    @abstractmethod
    def fit(
        self,
        adata_train: AnnData,
        adata_val: Optional[AnnData] = None,
        **kwargs,
    ) -> None:
        """
        Train the baseline model.

        Args:
            adata_train: Training AnnData (includes control cells).
            adata_val: Optional validation AnnData.
            **kwargs: Model-specific training hyperparameters.
        """
        pass

    @abstractmethod
    def predict(
        self,
        conditions: list,
        n_samples: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions for given perturbation conditions.

        Args:
            conditions: List of perturbation condition strings to predict.
            n_samples: Number of samples (cells) to generate per condition.
                       For deterministic models (e.g., GEARS), this may be 1.

        Returns:
            Dict mapping condition -> predicted expression matrix (n_samples, num_genes).
        """
        pass

    def load_checkpoint(self, path: str) -> None:
        """Load a pre-trained model checkpoint."""
        raise NotImplementedError(f"{self.model_name} does not support checkpoint loading.")

    def save_checkpoint(self, path: str) -> None:
        """Save the current model state."""
        raise NotImplementedError(f"{self.model_name} does not support checkpoint saving.")

    def get_diffusion_sampler(self):
        """
        Return a diffusion sampler object compatible with CellDiffA's SMC engine.

        Only diffusion-based models (PerturbDiff, Squidiff, scDFM) need to implement this.
        Non-diffusion models should raise NotImplementedError.

        Returns:
            Object satisfying DiffusionSamplerProtocol.
        """
        raise NotImplementedError(
            f"{self.model_name} is not a diffusion model and does not provide a sampler."
        )

    @property
    def is_generative(self) -> bool:
        """Whether this model generates multiple samples (distribution-level)."""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, trained={self.is_trained})"
