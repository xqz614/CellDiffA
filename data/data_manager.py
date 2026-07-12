"""
Unified data manager for CellDiffA.

Handles downloading, preprocessing, splitting, and loading of single-cell
perturbation datasets in a standardized AnnData format.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import scanpy as sc
from anndata import AnnData


class PerturbationDataManager:
    """
    Centralized data manager that provides a unified interface for all
    perturbation datasets used in CellDiffA experiments.

    Supported datasets:
        - norman: K562 CRISPRa combinatorial perturbations (Norman et al., 2019)
        - replogle_k562: K562 CRISPRi single perturbations (Replogle et al., 2022)
        - adamson: CRISPRi perturbations (Adamson et al., 2016)

    Supported split strategies:
        - additive: Random 70/30 train/test split of combinatorial perturbations
        - combinations: Hold out 15 combos + their constituent single perturbations
        - unseen: Remove 12 genes entirely (all related perturbations become test)
    """

    SUPPORTED_DATASETS = ["norman", "replogle_k562", "adamson"]
    SUPPORTED_SPLITS = ["additive", "combinations", "unseen"]

    def __init__(
        self,
        data_root: str = "./data",
        dataset_name: str = "norman",
        n_top_genes: int = 2000,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.n_top_genes = n_top_genes
        self.seed = seed

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
            self.processed_dir, f"{self.dataset_name}_hvg{self.n_top_genes}.h5ad"
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

        # Extract control cells
        self.ctrl_adata = self.adata[self.adata.obs["is_control"]].copy()
        return self.adata

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

        assert split_strategy in self.SUPPORTED_SPLITS, (
            f"Unsupported split: {split_strategy}. Choose from {self.SUPPORTED_SPLITS}"
        )

        # Store split state for cache key construction
        self._split_strategy = split_strategy
        self._fold = fold

        split_file = os.path.join(
            self.splits_dir,
            f"{self.dataset_name}_{split_strategy}_fold{fold}_of{n_folds}_seed{self.seed}.pkl",
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
        return f"{self.dataset_name}_{self._split_strategy}_fold{self._fold}_seed{self.seed}"

    def compute_de_genes(self, top_k: int = 20) -> Dict[str, List[str]]:
        """
        Compute differentially expressed genes for each perturbation condition
        in the training set. These serve as the transcriptomic prior for rewards.

        Returns:
            Dictionary mapping condition -> list of top DE gene names
        """
        if self.adata_train is None:
            raise RuntimeError("Must call create_split() before compute_de_genes()")

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
            c for c in self.adata_train.obs["condition"].unique()
            if c != "ctrl" and c != "control"
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
            c for c in self.adata_train.obs["condition"].unique()
            if c != "ctrl" and c != "control"
        ]

        shifts = {}
        for cond in conditions:
            cond_cells = self.adata_train[self.adata_train.obs["condition"] == cond]
            cond_expr = cond_cells.X
            if hasattr(cond_expr, "toarray"):
                cond_expr = cond_expr.toarray()
            shifts[cond] = (cond_expr.mean(axis=0).flatten() - ctrl_mean)

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

        # Normalize and log-transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Select HVGs
        sc.pp.highly_variable_genes(adata, n_top_genes=self.n_top_genes, inplace=True)

        # Force perturbation target genes into HVG set
        conditions = adata.obs["condition"].unique()
        pert_genes = set()
        for cond in conditions:
            for g in cond.split("+"):
                if g not in ("ctrl", "control"):
                    pert_genes.add(g)

        for gene in pert_genes:
            if gene in adata.var_names:
                adata.var.loc[gene, "highly_variable"] = True

        adata = adata[:, adata.var["highly_variable"]].copy()

        # Standardize metadata columns
        if "is_control" not in adata.obs.columns:
            adata.obs["is_control"] = adata.obs["condition"].isin(["ctrl", "control"])

        self.adata = adata

    def _generate_split(
        self, strategy: str, fold: int, n_folds: int
    ) -> Dict[str, list]:
        """Generate train/test split indices."""
        conditions = self.adata.obs["condition"].unique()
        combo_conditions = [
            c for c in conditions
            if c not in ("ctrl", "control") and "+" in c
            and "ctrl" not in c and "control" not in c
        ]
        combo_conditions = np.array(sorted(combo_conditions))

        rng = np.random.default_rng(self.seed + fold)

        if strategy == "additive":
            shuffled = combo_conditions.copy()
            rng.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.3)
            test_conditions = shuffled[:split_idx].tolist()
            return {"test": test_conditions, "strategy": strategy, "fold": fold}

        elif strategy == "combinations":
            shuffled = combo_conditions.copy()
            rng.shuffle(shuffled)
            test_combos = shuffled[:15].tolist()

            # Also hold out constituent single perturbations
            single_genes = set()
            for combo in test_combos:
                for g in combo.split("+"):
                    single_genes.add(g)

            single_conditions = [f"{g}+ctrl" for g in single_genes]
            single_conditions += [f"{g}+control" for g in single_genes]
            # Filter to only those that actually exist
            existing = set(conditions)
            single_conditions = [c for c in single_conditions if c in existing]

            test_conditions = test_combos + single_conditions
            return {"test": test_conditions, "strategy": strategy, "fold": fold}

        elif strategy == "unseen":
            # Extract all unique single genes from combo perturbations
            all_singles = set()
            for combo in combo_conditions:
                for g in combo.split("+"):
                    all_singles.add(g)
            all_singles = sorted(all_singles)
            rng.shuffle(all_singles)

            # Remove 12 genes entirely
            remove_genes = set(all_singles[:12])

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
            }

        raise ValueError(f"Unknown strategy: {strategy}")
