"""
CPA Adapter for CellDiffA.

Wraps the Compositional Perturbation Autoencoder (Lotfollahi et al., 2023)
with the unified BaseAdapter interface. CPA is a VAE-based model that learns
disentangled perturbation embeddings.

CPA is generative (VAE sampling) and can produce multiple samples per condition.
"""

import os
from typing import Dict, List, Optional

import numpy as np
from anndata import AnnData

from .base_adapter import BaseAdapter


class CPAAdapter(BaseAdapter):
    """
    Adapter for the CPA perturbation prediction model.

    CPA uses a conditional VAE with compositional perturbation embeddings
    to predict and disentangle perturbation effects.

    Requirements:
        pip install cpa-tools (or install from source: theislab/CPA)
        Requires: scvi-tools < 1.0.0, torch <= 2.0.1
    """

    def __init__(self, device: str = "cuda", seed: int = 42):
        super().__init__(model_name="CPA", device=device)
        self.seed = seed
        self._model = None
        self._adata = None

    def fit(
        self,
        adata_train: AnnData,
        adata_val: Optional[AnnData] = None,
        max_epochs: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        n_latent: int = 128,
        **kwargs,
    ) -> None:
        """
        Train CPA model.

        Args:
            adata_train: Training AnnData.
            max_epochs: Maximum training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            n_latent: Latent space dimension.
        """
        try:
            import cpa
        except ImportError:
            raise ImportError(
                "CPA not installed. Install via: pip install cpa-tools "
                "or clone https://github.com/theislab/CPA"
            )

        self._adata = adata_train.copy()

        # Setup CPA
        cpa.CPA.setup_anndata(
            self._adata,
            perturbation_key="condition",
            control_group="ctrl",
            batch_key=None,
        )

        self._model = cpa.CPA(
            self._adata,
            n_latent=n_latent,
            recon_loss="gauss",
        )

        self._model.train(
            max_epochs=max_epochs,
            batch_size=batch_size,
            plan_kwargs={"lr": lr},
            use_gpu=(self.device == "cuda"),
        )
        self.is_trained = True

    def predict(
        self,
        conditions: List[str],
        n_samples: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Predict perturbation responses using CPA.

        CPA generates samples via VAE decoding, naturally producing a distribution.

        Args:
            conditions: List of perturbation conditions.
            n_samples: Number of cells to generate per condition.

        Returns:
            Dict mapping condition -> expression matrix (n_samples, num_genes).
        """
        if not self.is_trained and self._model is None:
            raise RuntimeError("Model not trained. Call fit() or load_checkpoint() first.")

        results = {}
        for cond in conditions:
            # CPA prediction interface
            pred = self._model.predict(
                adata=self._adata,
                perturbation=cond,
                n_samples=n_samples,
            )
            results[cond] = pred  # (n_samples, num_genes)

        return results

    def load_checkpoint(self, path: str) -> None:
        """Load pre-trained CPA model."""
        try:
            import cpa
        except ImportError:
            raise ImportError("CPA not installed.")
        self._model = cpa.CPA.load(path)
        self.is_trained = True

    def save_checkpoint(self, path: str) -> None:
        """Save CPA model."""
        if self._model is not None:
            self._model.save(path, overwrite=True)

    @property
    def is_generative(self) -> bool:
        return True
