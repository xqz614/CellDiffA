"""
GEARS Adapter for CellDiffA.

Wraps the GEARS model (Roohani et al., Nature Biotechnology 2023) with the
unified BaseAdapter interface. GEARS is a GNN-based deterministic model that
predicts mean perturbation responses using gene regulatory knowledge graphs.

Note: GEARS produces a single point estimate per condition (not a distribution).
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

from .base_adapter import BaseAdapter


class GEARSAdapter(BaseAdapter):
    """
    Adapter for the GEARS perturbation prediction model.

    GEARS uses a graph neural network operating on a gene interaction graph
    to predict transcriptional responses to genetic perturbations.

    Requirements:
        pip install cell-gears (or install from source: snap-stanford/GEARS)
    """

    def __init__(
        self,
        data_path: str = "./data/gears_cache",
        device: str = "cuda",
        seed: int = 42,
    ):
        super().__init__(model_name="GEARS", device=device)
        self.data_path = data_path
        self.seed = seed
        self._model = None
        self._pert_data = None

    def fit(
        self,
        adata_train: "AnnData",
        adata_val: Optional["AnnData"] = None,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        hidden_size: int = 64,
        **kwargs,
    ) -> None:
        """
        Train GEARS model.

        Args:
            adata_train: Training AnnData with 'condition' in obs.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            hidden_size: GNN hidden dimension.
        """
        try:
            from gears import GEARS, PertData
        except ImportError:
            raise ImportError(
                "GEARS not installed. Install via: pip install cell-gears "
                "or clone https://github.com/snap-stanford/GEARS"
            )

        os.makedirs(self.data_path, exist_ok=True)

        # Setup PertData from AnnData
        self._pert_data = PertData(self.data_path)
        self._pert_data.new_data_process(
            dataset_name="custom",
            adata=adata_train,
        )
        self._pert_data.prepare_split(split="simulation", seed=self.seed)
        self._pert_data.get_dataloader(batch_size=batch_size, test_batch_size=batch_size)

        # Initialize and train model
        self._model = GEARS(
            self._pert_data,
            device=self.device,
            weight_bias_track=False,
        )
        self._model.model_initialize(hidden_size=hidden_size)
        self._model.train(epochs=epochs, lr=lr)
        self.is_trained = True

    def predict(
        self,
        conditions: List[str],
        n_samples: int = 1,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Predict perturbation responses using GEARS.

        Note: GEARS is deterministic, so n_samples > 1 will return repeated predictions.

        Args:
            conditions: List of perturbation conditions.
            n_samples: Number of samples per condition (repeated for GEARS).

        Returns:
            Dict mapping condition -> expression matrix (n_samples, num_genes).
        """
        if not self.is_trained and self._model is None:
            raise RuntimeError("Model not trained. Call fit() or load_checkpoint() first.")

        results = {}
        for cond in conditions:
            genes = [g for g in cond.split("+") if g not in {"ctrl", "control"}]
            if not genes:
                raise ValueError(f"Invalid perturbation condition: {cond!r}")
            pred = self._model.predict([genes])
            pred_expr = np.asarray(pred["_".join(genes)]).reshape(-1)

            # Repeat for n_samples (deterministic model)
            results[cond] = np.tile(pred_expr, (n_samples, 1))

        return results

    def load_checkpoint(
        self,
        checkpoint_path: str,
        gene_names=None,
        ctrl_adata=None,
        adata_train=None,
        dataset_name: str = "custom",
        **kwargs,
    ) -> None:
        """Load a GEARS checkpoint with the matching processed AnnData."""
        try:
            from gears import GEARS, PertData
        except ImportError:
            raise ImportError("GEARS not installed.")

        if adata_train is None:
            raise ValueError(
                "GEARS checkpoint loading requires adata_train to reconstruct "
                "the gene and perturbation graphs."
            )
        self._pert_data = PertData(self.data_path)
        self._pert_data.new_data_process(dataset_name=dataset_name, adata=adata_train)
        self._pert_data.prepare_split(split="simulation", seed=self.seed)
        self._pert_data.get_dataloader(batch_size=32, test_batch_size=32)
        self._model = GEARS(self._pert_data, device=self.device)
        self._model.load_pretrained(checkpoint_path)
        self.is_trained = True

    def save_checkpoint(self, path: str) -> None:
        """Save GEARS model checkpoint."""
        if self._model is not None:
            self._model.save_model(path)

    @property
    def is_generative(self) -> bool:
        return False
