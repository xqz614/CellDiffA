"""
Unified data manager for CellDiffA.

Handles downloading, preprocessing, splitting, and loading of single-cell
perturbation datasets in a standardized AnnData format.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


class PerturbationDataManager:
    """
    Centralized data manager that provides a unified interface for all
    perturbation datasets used in CellDiffA experiments.

    Supported datasets:
        - norman: K562 CRISPRa combinatorial perturbations (Norman et al., 2019)
        - replogle_k562: K562 CRISPRi single perturbations (Replogle et al., 2022)

    Supported split strategies:
        - additive: Disjoint folds of held-out combinatorial perturbations
        - unseen: Disjoint folds of held-out genes and all related conditions
    """

    SUPPORTED_DATASETS = ["norman", "replogle_k562"]
    SUPPORTED_SPLITS = ["additive", "unseen"]
    SPLIT_VERSION = "v2"

    def __init__(
        self,
        data_root: str = "./data",
        dataset_name: str = "norman",
        n_top_genes: int = 2000,
        seed: int = 42,
        already_normalized: bool = True,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.n_top_genes = n_top_genes
        self.seed = seed
        self.already_normalized = already_normalized

        self.raw_dir = os.path.join(data_root, "raw")
        self.processed_dir = os.path.join(data_root, "processed")
        self.splits_dir = os.path.join(data_root, "splits")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

        self.adata: Optional[AnnData] = None
        self.adata_train: Optional[AnnData] = None
        self.adata_test: Optional[AnnData] = None
        self.ctrl_adata: Optional[AnnData] = None
        self.de_genes: Optional[Dict[str, List[str]]] = None

        # Track split state for cache key construction
        self._split_strategy: Optional[str] = None
        self._fold: Optional[int] = None
        self._n_folds: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_preprocess(self) -> AnnData:
        """
        Load raw data and apply standard preprocessing pipeline:
        1. Normalize total counts to 10,000
        2. Log1p transform
        3. Select highly variable genes (HVGs)
        4. Force perturbation target genes into HVG set
        """
        processed_path = os.path.join(
            self.processed_dir,
            f"{self.dataset_name}_{self._preprocessing_tag}_hvg{self.n_top_genes}.h5ad",
        )

        if os.path.exists(processed_path):
            print(f"[DataManager] Loading preprocessed data from {processed_path}")
            self.adata = sc.read_h5ad(processed_path)
        else:
            raw_path = os.path.join(self.raw_dir, f"{self.dataset_name}.h5ad")
            if not os.path.exists(raw_path):
                raise FileNotFoundError(
                    f"Raw data not found at {raw_path}. "
                    f"Please download it first using `scripts/preprocess_data.py`."
                )
            print(f"[DataManager] Loading raw data from {raw_path}")
            self.adata = sc.read_h5ad(raw_path)
            self._preprocess()
            self.adata.write_h5ad(processed_path)
            print(f"[DataManager] Preprocessed data saved to {processed_path}")

        self._standardize_metadata()
        # Extract control cells
        self.ctrl_adata = self.adata[self.adata.obs["is_control"]].copy()
        if self.ctrl_adata.n_obs == 0:
            raise ValueError("Dataset contains no control cells.")
        return self.adata

    @property
    def _preprocessing_tag(self) -> str:
        return "log_input" if self.already_normalized else "raw_counts"

    def create_split(
        self,
        split_strategy: str = "additive",
        fold: int = 0,
        n_folds: int = 5,
    ) -> Tuple[AnnData, AnnData]:
        """
        Create train/test split based on specified strategy.

        Returns:
            Tuple of (adata_train, adata_test)
        """
        if self.adata is None:
            self.load_and_preprocess()

        if split_strategy not in self.SUPPORTED_SPLITS:
            raise ValueError(
                f"Unsupported split: {split_strategy}. Choose from {self.SUPPORTED_SPLITS}"
            )

        # Store split state for cache key construction
        self._split_strategy = split_strategy
        self._fold = fold
        self._n_folds = n_folds

        split_file = os.path.join(
            self.splits_dir,
            f"{self.dataset_name}_{split_strategy}_{self.SPLIT_VERSION}_"
            f"fold{fold}_of{n_folds}_seed{self.seed}.pkl",
        )

        if os.path.exists(split_file):
            with open(split_file, "rb") as f:
                split_info = pickle.load(f)
        else:
            split_info = self._generate_split(split_strategy, fold, n_folds)
            with open(split_file, "wb") as f:
                pickle.dump(split_info, f)
            print(f"[DataManager] Split saved to {split_file}")

        # Apply split
        test_conditions = set(split_info["test"])
        self.adata.obs["split"] = "train"
        mask_test = self.adata.obs["condition"].isin(test_conditions)
        self.adata.obs.loc[mask_test, "split"] = "test"

        self.adata_train = self.adata[
            (self.adata.obs["split"] == "train") | (self.adata.obs["is_control"])
        ].copy()
        self.adata_test = self.adata[
            (self.adata.obs["split"] == "test") | (self.adata.obs["is_control"])
        ].copy()

        print(
            f"[DataManager] Split '{split_strategy}' fold {fold}: "
            f"train={self.adata_train.n_obs} cells, test={self.adata_test.n_obs} cells"
        )
        return self.adata_train, self.adata_test

    def _get_prior_cache_key(self) -> str:
        """
        Construct a unique cache key that includes dataset, split strategy,
        fold, and seed to prevent cross-fold/cross-split cache contamination.
        """
        if self._split_strategy is None or self._fold is None:
            raise RuntimeError(
                "Must call create_split() before computing priors. "
                "Cache keys depend on split state."
            )
        return (
            f"{self.dataset_name}_{self._split_strategy}_fold{self._fold}"
            f"_of{self._n_folds}_seed{self.seed}_{self._preprocessing_tag}"
        )

    def compute_de_genes(self, top_k: int = 20) -> Dict[str, List[str]]:
        """
        Compute differentially expressed genes for each perturbation condition
        in the training set. These serve as the transcriptomic prior for rewards.

        Returns:
            Dictionary mapping condition -> list of top DE gene names
        """
        if self.adata_train is None:
            raise RuntimeError("Must call create_split() before compute_de_genes()")
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        cache_key = self._get_prior_cache_key()
        de_cache = os.path.join(
            self.processed_dir,
            f"{cache_key}_de_top{top_k}.pkl",
        )
        if os.path.exists(de_cache):
            with open(de_cache, "rb") as f:
                self.de_genes = pickle.load(f)
            return self.de_genes

        print("[DataManager] Computing DE genes from training set...")
        ctrl_expr = self.ctrl_adata.X
        if hasattr(ctrl_expr, "toarray"):
            ctrl_expr = ctrl_expr.toarray()
        ctrl_mean = ctrl_expr.mean(axis=0)

        gene_names = list(self.adata_train.var_names)
        conditions = [
            c for c in self.adata_train.obs["condition"].unique() if c != "ctrl" and c != "control"
        ]

        self.de_genes = {}
        for cond in conditions:
            cond_cells = self.adata_train[self.adata_train.obs["condition"] == cond]
            cond_expr = cond_cells.X
            if hasattr(cond_expr, "toarray"):
                cond_expr = cond_expr.toarray()
            cond_mean = cond_expr.mean(axis=0)

            # Compute absolute fold change
            diff = np.abs(cond_mean - ctrl_mean).flatten()
            top_indices = np.argsort(diff)[-top_k:][::-1]
            self.de_genes[cond] = [gene_names[i] for i in top_indices]

        with open(de_cache, "wb") as f:
            pickle.dump(self.de_genes, f)
        print(f"[DataManager] DE genes computed for {len(self.de_genes)} conditions")
        return self.de_genes

    def compute_perturbation_shifts(self) -> Dict[str, np.ndarray]:
        """
        Compute mean expression shift vectors for each perturbation in the training set.
        These serve as the geometric prior (manifold direction) for rewards.

        Returns:
            Dictionary mapping condition -> shift vector (num_genes,)
        """
        if self.adata_train is None:
            raise RuntimeError("Must call create_split() before compute_perturbation_shifts()")

        cache_key = self._get_prior_cache_key()
        shift_cache = os.path.join(
            self.processed_dir,
            f"{cache_key}_shifts.pkl",
        )
        if os.path.exists(shift_cache):
            with open(shift_cache, "rb") as f:
                return pickle.load(f)

        print("[DataManager] Computing perturbation shift vectors from training set...")
        ctrl_expr = self.ctrl_adata.X
        if hasattr(ctrl_expr, "toarray"):
            ctrl_expr = ctrl_expr.toarray()
        ctrl_mean = ctrl_expr.mean(axis=0).flatten()

        conditions = [
            c for c in self.adata_train.obs["condition"].unique() if c != "ctrl" and c != "control"
        ]

        shifts = {}
        for cond in conditions:
            cond_cells = self.adata_train[self.adata_train.obs["condition"] == cond]
            cond_expr = cond_cells.X
            if hasattr(cond_expr, "toarray"):
                cond_expr = cond_expr.toarray()
            shifts[cond] = cond_expr.mean(axis=0).flatten() - ctrl_mean

        with open(shift_cache, "wb") as f:
            pickle.dump(shifts, f)
        print(f"[DataManager] Shift vectors computed for {len(shifts)} conditions")
        return shifts

    def get_control_mean(self) -> np.ndarray:
        """Return mean expression vector of control cells."""
        ctrl_expr = self.ctrl_adata.X
        if hasattr(ctrl_expr, "toarray"):
            ctrl_expr = ctrl_expr.toarray()
        return ctrl_expr.mean(axis=0).flatten()

    def get_control_cells(self, n_cells: Optional[int] = None) -> np.ndarray:
        """Return expression matrix of control cells (optionally subsampled)."""
        ctrl_expr = self.ctrl_adata.X
        if hasattr(ctrl_expr, "toarray"):
            ctrl_expr = ctrl_expr.toarray()
        if n_cells is not None and n_cells < ctrl_expr.shape[0]:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(ctrl_expr.shape[0], n_cells, replace=False)
            return ctrl_expr[idx]
        return ctrl_expr

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _preprocess(self):
        """Standard preprocessing pipeline."""
        adata = self.adata

        # GEARS-distributed H5AD files are already log-normalized. Repeating
        # normalize_total/log1p changes the checkpoint input space.
        if not self.already_normalized:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

        # Select HVGs
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes, inplace=True)

        # Force perturbation target genes into the fixed-size HVG set.
        conditions = adata.obs["condition"].unique()
        pert_genes = set()
        for cond in conditions:
            for g in cond.split("+"):
                if g not in ("ctrl", "control"):
                    pert_genes.add(g)

        forced = {gene for gene in pert_genes if gene in adata.var_names}
        if len(forced) > self.n_top_genes:
            raise ValueError(
                f"Found {len(forced)} perturbation genes but n_top_genes={self.n_top_genes}."
            )
        selected = set(adata.var_names[adata.var["highly_variable"]]) | forced
        if len(selected) > self.n_top_genes:
            score_column = (
                "dispersions_norm" if "dispersions_norm" in adata.var.columns else "dispersions"
            )
            candidates = sorted(
                selected - forced,
                key=lambda gene: float(adata.var.loc[gene, score_column]),
                reverse=True,
            )
            selected = forced | set(candidates[: self.n_top_genes - len(forced)])
        mask = adata.var_names.isin(selected)
        adata = adata[:, mask].copy()
        if adata.n_vars != self.n_top_genes:
            raise ValueError(
                f"Expected exactly {self.n_top_genes} genes after selection, got {adata.n_vars}."
            )

        self.adata = adata
        self._standardize_metadata()

    def _standardize_metadata(self) -> None:
        if "condition" not in self.adata.obs.columns:
            raise ValueError("AnnData.obs must contain a 'condition' column.")
        if "is_control" not in self.adata.obs.columns:
            self.adata.obs["is_control"] = self.adata.obs["condition"].isin(["ctrl", "control"])
        values = self.adata.obs["is_control"]
        if pd.api.types.is_bool_dtype(values.dtype):
            normalized = values
        elif pd.api.types.is_numeric_dtype(values.dtype):
            normalized = values.astype(bool)
        else:
            normalized = values.astype(str).str.lower().isin({"true", "1", "yes"})
        self.adata.obs["is_control"] = normalized

    def _generate_split(self, strategy: str, fold: int, n_folds: int) -> Dict[str, list]:
        """Generate train/test split indices."""
        conditions = self.adata.obs["condition"].unique()
        combo_conditions = [
            c
            for c in conditions
            if c not in ("ctrl", "control") and "+" in c and "ctrl" not in c and "control" not in c
        ]
        combo_conditions = np.array(sorted(combo_conditions))

        if not 0 <= fold < n_folds:
            raise ValueError(f"fold must be in [0, {n_folds}), got {fold}.")
        rng = np.random.default_rng(self.seed)

        if strategy == "additive":
            shuffled = combo_conditions.copy()
            rng.shuffle(shuffled)
            if len(shuffled) < n_folds:
                raise ValueError(
                    f"Split 'additive' requires at least {n_folds} combination conditions; "
                    f"found {len(shuffled)}."
                )
            test_conditions = np.array_split(shuffled, n_folds)[fold].tolist()
            return {
                "test": test_conditions,
                "strategy": strategy,
                "fold": fold,
                "n_folds": n_folds,
            }

        elif strategy == "unseen":
            # Extract all perturbation genes, including single-only datasets.
            all_singles = set()
            for condition in conditions:
                for gene in condition.split("+"):
                    if gene not in {"ctrl", "control"}:
                        all_singles.add(gene)
            all_singles = sorted(all_singles)
            rng.shuffle(all_singles)

            # Partition genes into disjoint folds; all conditions involving a
            # held-out gene become test conditions.
            if len(all_singles) < n_folds:
                raise ValueError(
                    f"Split 'unseen' requires at least {n_folds} perturbation genes; "
                    f"found {len(all_singles)}."
                )
            remove_genes = set(np.array_split(np.asarray(all_singles), n_folds)[fold].tolist())

            # All conditions involving removed genes become test
            test_conditions = []
            for cond in conditions:
                if cond in ("ctrl", "control"):
                    continue
                genes_in_cond = set(cond.split("+")) - {"ctrl", "control"}
                if genes_in_cond & remove_genes:
                    test_conditions.append(cond)

            return {
                "test": test_conditions,
                "removed_genes": list(remove_genes),
                "strategy": strategy,
                "fold": fold,
                "n_folds": n_folds,
            }

        raise ValueError(f"Unknown strategy: {strategy}")
