"""Leakage-safe CellDiffA priors for PerturbDiff's Replogle split."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
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
    sources: dict[str, str]


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
    embedding_signature: str | None,
    ridge_penalty: float,
) -> dict:
    return {
        "format_version": 2,
        "source": str(source),
        "source_signature": _source_signature(source),
        "split_path": str(split_path),
        "split_text": split_path.read_text(encoding="utf-8"),
        "genes": genes,
        "expression_key": expression_key,
        "top_k": top_k,
        "target_perturbations": target_perturbations,
        "embedding_signature": embedding_signature,
        "ridge_penalty": ridge_penalty,
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
        sources_array = values["sources"]
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
        sources={pert: str(sources_array[i]) for i, pert in enumerate(perts)},
    )


def _impute_shifts_with_embeddings(
    direct_shifts: Mapping[str, np.ndarray],
    embeddings: Mapping[str, np.ndarray],
    targets: list[str],
    *,
    ridge_penalty: float,
) -> dict[str, np.ndarray]:
    """Predict unseen shifts with centered dual ridge on training GenePT vectors."""
    training_names = sorted(set(direct_shifts) & set(embeddings))
    missing_embeddings = sorted(set(targets) - set(embeddings))
    if missing_embeddings:
        raise ValueError(
            f"GenePT embeddings are missing unseen test perturbations: {missing_embeddings}."
        )
    if len(training_names) < 2:
        raise ValueError(
            "At least two embedded training perturbations are required for ridge priors."
        )

    x_train = np.stack([embeddings[name] for name in training_names]).astype(np.float64)
    norms = np.linalg.norm(x_train, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Training perturbation embeddings contain zero vectors.")
    x_train /= norms
    y_train = np.stack([direct_shifts[name] for name in training_names]).astype(np.float64)
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    gram = x_centered @ x_centered.T
    dual = np.linalg.solve(
        gram + ridge_penalty * np.eye(len(training_names), dtype=np.float64),
        y_centered,
    )

    result = {}
    for target in targets:
        vector = np.asarray(embeddings[target], dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vector)
        if norm <= 0 or vector.shape[0] != x_train.shape[1]:
            raise ValueError(f"Invalid GenePT embedding for unseen perturbation {target!r}.")
        centered = vector / norm - x_mean[0]
        prediction = y_mean[0] + (centered @ x_centered.T) @ dual
        result[target] = prediction.astype(np.float32)
    return result


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
    perturbation_embeddings: Mapping[str, np.ndarray] | None = None,
    embedding_signature: str | None = None,
    ridge_penalty: float = 1.0,
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
    if ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be positive.")
    if perturbation_embeddings is not None and not embedding_signature:
        raise ValueError("embedding_signature is required when embeddings are supplied.")
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
    expected = _metadata(
        source,
        split_path,
        list(genes),
        expression_key,
        top_k,
        target_perts,
        embedding_signature,
        ridge_penalty,
    )
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

    training_perts = sorted(set(labels[train]) - {split.control_pert})
    shift_sums = {pert: np.zeros(len(genes), dtype=np.float64) for pert in training_perts}
    training_counts = {pert: 0 for pert in training_perts}
    target_mask = train & np.isin(labels, training_perts)
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
            training_counts[str(pert)] += len(rows)

    direct_shifts = {
        pert: (shift_sums[pert] / training_counts[pert]).astype(np.float32)
        for pert in training_perts
        if training_counts[pert] > 0
    }
    shifts = {pert: direct_shifts[pert] for pert in target_perts if pert in direct_shifts}
    missing = [pert for pert in target_perts if pert not in shifts]
    if missing:
        if perturbation_embeddings is None:
            raise ValueError(
                "Official training rows provide no direct prior and no embedding fallback "
                f"was supplied for: {missing[:20]}"
            )
        shifts.update(
            _impute_shifts_with_embeddings(
                direct_shifts,
                perturbation_embeddings,
                missing,
                ridge_penalty=ridge_penalty,
            )
        )
    counts = {pert: int(training_counts.get(pert, 0)) for pert in target_perts}
    sources = {
        pert: "direct_training_mean" if counts[pert] > 0 else "genept_dual_ridge"
        for pert in target_perts
    }
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
        sources=sources,
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
            sources=np.asarray([sources[pert] for pert in target_perts]),
            de_indices=de_indices,
        )
        temporary.replace(cache)
    return priors
