"""Leakage-safe CellDiffA priors for PerturbDiff's Replogle split."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from .perturbdiff_split import PerturbDiffSplit
from .streaming import iter_h5ad_expression


@dataclass(frozen=True)
class ReplogleTrainingPriors:
    """Context-corrected mean shifts computed from official training rows."""

    genes: tuple[str, ...]
    shifts: dict[str, np.ndarray]
    de_genes: dict[str, list[str]]
    counts: dict[str, int]


def _source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _metadata(
    source: Path,
    split_path: Path,
    genes: list[str],
    expression_key: str,
    top_k: int,
    target_perturbations: list[str],
) -> dict:
    return {
        "format_version": 1,
        "source": str(source),
        "source_signature": _source_signature(source),
        "split_path": str(split_path),
        "split_text": split_path.read_text(encoding="utf-8"),
        "genes": genes,
        "expression_key": expression_key,
        "top_k": top_k,
        "target_perturbations": target_perturbations,
    }


def _load_cache(path: Path, expected: dict) -> ReplogleTrainingPriors | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        observed = json.loads(str(values["metadata"].item()))
        if observed != expected:
            return None
        perts = [str(value) for value in values["perturbations"]]
        shifts_matrix = values["shifts"].astype(np.float32, copy=False)
        counts_array = values["counts"]
        de_indices = values["de_indices"]
    genes = tuple(expected["genes"])
    return ReplogleTrainingPriors(
        genes=genes,
        shifts={pert: shifts_matrix[i] for i, pert in enumerate(perts)},
        de_genes={
            pert: [genes[int(index)] for index in de_indices[i] if int(index) >= 0]
            for i, pert in enumerate(perts)
        },
        counts={pert: int(counts_array[i]) for i, pert in enumerate(perts)},
    )


def compute_replogle_training_priors(
    source: str | Path,
    split_path: str | Path,
    genes: list[str],
    *,
    cache_path: str | Path | None = None,
    expression_key: str = "X_hvg",
    top_k: int = 50,
    chunk_size: int = 8192,
    target_perturbations: Iterable[str] | None = None,
) -> ReplogleTrainingPriors:
    """Compute test-perturbation priors without reading held-out responses.

    Each training response is centered by the control mean from its own cell
    line before pooling. Controls in the held-out context are allowed because
    the prediction task explicitly conditions on control cells; validation and
    test perturbation rows are excluded by the released split mask.
    """
    source = Path(source).resolve()
    split_path = Path(split_path).resolve()
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    split = PerturbDiffSplit.from_yaml(split_path, split_axis="context")
    target_perts = sorted(
        split.test_perts
        if target_perturbations is None
        else {str(value) for value in target_perturbations}
    )
    unexpected = sorted(set(target_perts) - set(split.test_perts))
    if unexpected:
        raise ValueError(f"Requested priors outside the official test split: {unexpected}.")
    if not target_perts:
        raise ValueError("At least one test perturbation is required.")
    expected = _metadata(source, split_path, list(genes), expression_key, top_k, target_perts)
    cache = Path(cache_path).resolve() if cache_path is not None else None
    if cache is not None:
        loaded = _load_cache(cache, expected)
        if loaded is not None:
            return loaded

    backed = ad.read_h5ad(source, backed="r")
    try:
        obs = backed.obs[[split.pert_col, split.context_col]].copy()
        if expression_key == "X" and list(backed.var_names.astype(str)) != genes:
            raise ValueError("X gene order does not match the requested prior genes.")
        if expression_key != "X" and backed.obsm[expression_key].shape[1] != len(genes):
            raise ValueError(
                f"{expression_key} has {backed.obsm[expression_key].shape[1]} columns; "
                f"expected {len(genes)}."
            )
    finally:
        backed.file.close()

    masks = split.masks(obs, split_axis="context")
    labels = obs[split.pert_col].astype(str).to_numpy()
    contexts = obs[split.context_col].astype(str).to_numpy()
    train = masks["train"]
    control = train & (labels == split.control_pert)
    context_names = sorted(set(contexts[control]))
    control_sums = {name: np.zeros(len(genes), dtype=np.float64) for name in context_names}
    control_counts = {name: 0 for name in context_names}
    for start, stop, values in iter_h5ad_expression(
        source, expression_key=expression_key, chunk_size=chunk_size
    ):
        for context in context_names:
            local = control[start:stop] & (contexts[start:stop] == context)
            if np.any(local):
                control_sums[context] += values[local].sum(axis=0, dtype=np.float64)
                control_counts[context] += int(local.sum())
    missing_controls = [name for name, count in control_counts.items() if count == 0]
    if missing_controls:
        raise ValueError(f"No training controls for contexts {missing_controls}.")
    control_means = {name: control_sums[name] / control_counts[name] for name in context_names}

    shift_sums = {pert: np.zeros(len(genes), dtype=np.float64) for pert in target_perts}
    counts = {pert: 0 for pert in target_perts}
    target_mask = train & np.isin(labels, target_perts)
    for start, stop, values in iter_h5ad_expression(
        source, expression_key=expression_key, chunk_size=chunk_size
    ):
        local_labels = labels[start:stop]
        local_contexts = contexts[start:stop]
        active = target_mask[start:stop]
        for pert in np.unique(local_labels[active]):
            pert_mask = active & (local_labels == pert)
            rows = values[pert_mask].astype(np.float64, copy=False)
            row_contexts = local_contexts[pert_mask]
            missing_contexts = sorted(set(row_contexts) - set(control_means))
            if missing_contexts:
                raise ValueError(
                    f"Training perturbation {pert!r} has contexts without controls: "
                    f"{missing_contexts}."
                )
            residual_sum = np.zeros(len(genes), dtype=np.float64)
            for context in np.unique(row_contexts):
                context_rows = rows[row_contexts == context]
                residual_sum += (context_rows - control_means[str(context)][None, :]).sum(axis=0)
            shift_sums[str(pert)] += residual_sum
            counts[str(pert)] += len(rows)

    missing = [pert for pert, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"Official training rows provide no prior for test perturbations: {missing[:20]}"
        )
    shifts = {pert: (shift_sums[pert] / counts[pert]).astype(np.float32) for pert in target_perts}
    de_indices = np.stack(
        [np.argsort(-np.abs(shifts[pert]))[: min(top_k, len(genes))] for pert in target_perts]
    )
    priors = ReplogleTrainingPriors(
        genes=tuple(genes),
        shifts=shifts,
        de_genes={
            pert: [genes[int(index)] for index in de_indices[i]]
            for i, pert in enumerate(target_perts)
        },
        counts=counts,
    )
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f"{cache.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            metadata=np.asarray(json.dumps(expected, sort_keys=True)),
            perturbations=np.asarray(target_perts),
            shifts=np.stack([shifts[pert] for pert in target_perts]),
            counts=np.asarray([counts[pert] for pert in target_perts]),
            de_indices=de_indices,
        )
        temporary.replace(cache)
    return priors
