"""
Transcriptomic Prior Reward (r_DEG).

Evaluates whether generated cells exhibit expected expression changes on
differentially expressed genes (DEGs). The DEG reference is derived entirely
from the training set, ensuring no data leakage.

For combinatorial perturbations (A+B) where only single perturbations A and B
are observed in training, the reference is constructed by combining the DEG
sets and expression shifts of A and B.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import torch

from .base import BaseReward


class TranscriptomicReward(BaseReward):
    """
    Reward based on expression fidelity on reference DEG set.

    The reward measures how well the generated cells' expression on key genes
    matches the expected perturbation direction derived from training data.

    Computation:
        r_DEG = -MSE(x_pred[G_ref], mu_ref[G_ref])

    where G_ref is the reference gene set and mu_ref is the expected expression
    level, both derived from training set priors.
    """

    def __init__(
        self,
        de_genes: Dict[str, List[str]],
        perturbation_shifts: Dict[str, np.ndarray],
        ctrl_mean: np.ndarray,
        gene_names: List[str],
        weight: float = 1.0,
        top_k: int = 20,
        combination_mode: str = "union",
    ):
        """
        Args:
            de_genes: Dict mapping condition -> list of DE gene names (from training set).
            perturbation_shifts: Dict mapping condition -> shift vector (from training set).
            ctrl_mean: Mean expression of control cells. Shape: (num_genes,)
            gene_names: Ordered list of gene names matching expression matrix columns.
            weight: Reward weight in composite scoring.
            top_k: Number of top DE genes to use per condition.
            combination_mode: How to combine DEGs for unseen combos. "union" or "intersection".
        """
        super().__init__(weight=weight, name="r_DEG")
        self.de_genes = de_genes
        self.perturbation_shifts = perturbation_shifts
        self.ctrl_mean = torch.tensor(ctrl_mean, dtype=torch.float32)
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.top_k = top_k
        self.combination_mode = combination_mode

        # Pre-compute reference for known conditions
        self._cache: Dict[str, dict] = {}

    def compute(
        self,
        x_pred: torch.Tensor,
        condition: str,
        timestep: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute DEG-based reward for each particle.

        Args:
            x_pred: Tweedie estimate of clean expression. Shape: (N, G)
            condition: Perturbation condition string.
            timestep: Current diffusion timestep (unused here, reserved for annealing).

        Returns:
            Reward scores. Shape: (N,)
        """
        ref = self._get_reference(condition, x_pred.device)
        if ref is None:
            # No reference available: return zero reward (neutral)
            return torch.zeros(x_pred.shape[0], device=x_pred.device)

        gene_indices = ref["gene_indices"]
        target_expr = ref["target_expr"]

        # Extract predicted expression on reference genes
        pred_on_ref = x_pred[:, gene_indices]  # (N, K)

        # Negative MSE as reward (higher is better)
        mse = ((pred_on_ref - target_expr.unsqueeze(0)) ** 2).mean(dim=1)
        reward = -mse

        return reward

    def _get_reference(self, condition: str, device: torch.device) -> Optional[dict]:
        """
        Get or compute the reference DEG set and target expression for a condition.

        For known conditions: directly use training set DE genes and shift.
        For unseen combinations (A+B): combine references from A and B.
        """
        if condition in self._cache:
            cached = self._cache[condition]
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in cached.items()}

        # Case 1: Condition exists directly in training set
        if condition in self.de_genes:
            ref = self._build_reference_direct(condition)
        else:
            # Case 2: Unseen combination - decompose and combine
            ref = self._build_reference_combination(condition)

        if ref is not None:
            self._cache[condition] = ref
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in ref.items()}
        return None

    def _build_reference_direct(self, condition: str) -> dict:
        """Build reference from a directly observed condition."""
        de_gene_names = self.de_genes[condition][: self.top_k]
        gene_indices = [self.gene_to_idx[g] for g in de_gene_names if g in self.gene_to_idx]

        if len(gene_indices) == 0:
            return None

        gene_indices = torch.tensor(gene_indices, dtype=torch.long)

        # Target expression = ctrl_mean + shift on those genes
        shift = self.perturbation_shifts[condition]
        target_full = self.ctrl_mean + torch.tensor(shift, dtype=torch.float32)
        target_expr = target_full[gene_indices]

        return {"gene_indices": gene_indices, "target_expr": target_expr}

    def _build_reference_combination(self, condition: str) -> Optional[dict]:
        """
        Build reference for an unseen combination by combining constituent
        single-perturbation references from the training set.

        For condition "A+B":
            - G_ref = DEG(A) ∪ DEG(B)  (or intersection)
            - target = ctrl_mean + shift(A) + shift(B)
        """
        parts = condition.split("+")
        parts = [p for p in parts if p not in ("ctrl", "control")]

        if len(parts) == 0:
            return None

        # Find available single-perturbation references
        available_parts = []
        for p in parts:
            # Try different naming conventions
            candidates = [p, f"{p}+ctrl", f"{p}+control"]
            for cand in candidates:
                if cand in self.de_genes and cand in self.perturbation_shifts:
                    available_parts.append(cand)
                    break

        if len(available_parts) == 0:
            return None

        # Combine DE gene sets
        if self.combination_mode == "union":
            combined_genes: Set[str] = set()
            for part in available_parts:
                combined_genes.update(self.de_genes[part][: self.top_k])
        else:  # intersection
            gene_sets = [set(self.de_genes[p][: self.top_k]) for p in available_parts]
            combined_genes = gene_sets[0].intersection(*gene_sets[1:])

        gene_indices = [self.gene_to_idx[g] for g in combined_genes if g in self.gene_to_idx]
        if len(gene_indices) == 0:
            return None

        gene_indices = torch.tensor(sorted(gene_indices), dtype=torch.long)

        # Combine shift vectors (additive assumption as soft prior)
        combined_shift = np.zeros_like(self.ctrl_mean.numpy())
        for part in available_parts:
            combined_shift += self.perturbation_shifts[part]

        target_full = self.ctrl_mean + torch.tensor(combined_shift, dtype=torch.float32)
        target_expr = target_full[gene_indices]

        return {"gene_indices": gene_indices, "target_expr": target_expr}
