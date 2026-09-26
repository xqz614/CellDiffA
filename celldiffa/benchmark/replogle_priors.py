"""Leakage-safe CellDiffA priors for PerturbDiff's Replogle split."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .perturbdiff_split import PerturbDiffSplit
from .streaming import (
    h5ad_expression_shape,
    iter_h5ad_expression,
    read_h5ad_obs,
    read_h5ad_var,
)


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


def validate_prior_robustness(mode: str, fraction: float, seed: int) -> None:
    """Validate prior-only interventions before reading expression data."""
    if mode not in {"full", "subsample", "shuffle"}:
        raise ValueError("prior_mode must be full, subsample, or shuffle.")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("prior_fraction must be finite and in (0, 1].")
    if mode != "subsample" and fraction != 1.0:
        raise ValueError("prior_fraction must be 1.0 unless prior_mode is subsample.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("prior_seed must be a nonnegative integer.")


def _subsample_training_mask(
    train: np.ndarray,
    labels: np.ndarray,
    contexts: np.ndarray,
    control_pert: str,
    *,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Keep all training controls and sample treated rows within each stratum.

    A stratum is a (perturbation, context) pair. Sampling uses only row metadata
    and a local RNG; it never inspects held-out or training expression values.
    Each nonempty stratum retains max(1, floor(fraction * n)) treated rows.
    """
    selected = train.copy()
    if fraction == 1.0:
        return selected
    treated = train & (labels != control_pert)
    selected[treated] = False
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[int]] = {}
    for index in np.flatnonzero(treated):
        strata.setdefault((labels[index], contexts[index]), []).append(int(index))
    for key in sorted(strata):
        indices = np.asarray(strata[key], dtype=np.int64)
        size = max(1, int(np.floor(fraction * len(indices))))
        selected[rng.choice(indices, size=size, replace=False)] = True
    return selected


def _shuffled_prior_donors(targets: Iterable[str], seed: int) -> dict[str, str]:
    """Make a reproducible single-cycle derangement of eligible target priors."""
    names = sorted(set(targets))
    if len(names) < 2:
        raise ValueError("Shuffled priors require at least two eligible target perturbations.")
    order = np.random.default_rng(seed).permutation(names).tolist()
    return {target: order[(index + 1) % len(order)] for index, target in enumerate(order)}


def _metadata(
    source: Path,
    split_path: Path,
    genes: list[str],
    expression_key: str,
    top_k: int,
    target_perturbations: list[str],
    embedding_signature: str | None,
    ridge_penalty: float,
    evaluation_split: str,
    prior_mode: str = "full",
    prior_fraction: float = 1.0,
    prior_seed: int = 42,
) -> dict:
    metadata = {
        "format_version": 3,
        "evaluation_split": evaluation_split,
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
    # Exact legacy metadata remains valid for the unmodified default. Every
    # robustness variant has a distinct cache contract, preventing false hits.
    if (prior_mode, prior_fraction, prior_seed) != ("full", 1.0, 42):
        metadata.update(
            format_version=4,
            prior_robustness_version=1,
            prior_mode=prior_mode,
            prior_fraction=prior_fraction,
            prior_seed=prior_seed,
        )
    return metadata


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
    evaluation_split: str = "test",
    prior_mode: str = "full",
    prior_fraction: float = 1.0,
    prior_seed: int = 42,
) -> ReplogleTrainingPriors:
    """Compute validation/test priors without using any held-out responses.

    Each training response is centered by the control mean from its own cell
    line before pooling. Controls in the held-out context are allowed because
    the prediction task explicitly conditions on control cells; validation and
    test perturbation rows are excluded by the released split mask.

    Robustness options affect only these reward priors. Subsampling retains
    all training controls and refits direct shifts and descriptor regression
    from selected treated rows. Shuffling reassigns the resulting target
    shifts without changing any denoiser condition or reading test responses.
    """
    validate_prior_robustness(prior_mode, prior_fraction, prior_seed)
    source = Path(source).resolve()
    split_path = Path(split_path).resolve()
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if ridge_penalty <= 0:
        raise ValueError("ridge_penalty must be positive.")
    if perturbation_embeddings is not None and not embedding_signature:
        raise ValueError("embedding_signature is required when embeddings are supplied.")
    split = PerturbDiffSplit.from_yaml(split_path, split_axis="context")
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test.")
    allowed = split.validation_perts if evaluation_split == "validation" else split.test_perts
    target_perts = sorted(
        allowed
        if target_perturbations is None
        else {str(value) for value in target_perturbations}
    )
    unexpected = sorted(set(target_perts) - set(allowed))
    if unexpected:
        raise ValueError(
            f"Requested priors outside the official {evaluation_split} split: {unexpected}."
        )
    if not target_perts:
        raise ValueError("At least one target perturbation is required.")
    if prior_mode == "shuffle" and len(target_perts) < 2:
        raise ValueError("Shuffled priors require at least two eligible target perturbations.")
    expected = _metadata(
        source,
        split_path,
        list(genes),
        expression_key,
        top_k,
        target_perts,
        embedding_signature,
        ridge_penalty,
        evaluation_split,
        prior_mode,
        prior_fraction,
        prior_seed,
    )
    cache = Path(cache_path).resolve() if cache_path is not None else None
    if cache is not None:
        loaded = _load_cache(cache, expected)
        if loaded is not None:
            return loaded

    obs = read_h5ad_obs(source)[[split.pert_col, split.context_col]].copy()
    if expression_key == "X" and list(read_h5ad_var(source).index.astype(str)) != genes:
        raise ValueError("X gene order does not match the requested prior genes.")
    if h5ad_expression_shape(source, expression_key)[1] != len(genes):
        raise ValueError(f"{expression_key} must have {len(genes)} columns.")

    masks = split.masks(obs, split_axis="context")
    labels = obs[split.pert_col].astype(str).to_numpy()
    contexts = obs[split.context_col].astype(str).to_numpy()
    train = masks["train"]
    if prior_mode == "subsample":
        train = _subsample_training_mask(
            train,
            labels,
            contexts,
            split.control_pert,
            fraction=prior_fraction,
            seed=prior_seed,
        )
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
    if prior_mode == "shuffle":
        donors = _shuffled_prior_donors(target_perts, prior_seed)
        original_shifts, original_counts, original_sources = shifts, counts, sources
        shifts = {pert: original_shifts[donors[pert]].copy() for pert in target_perts}
        counts = {pert: original_counts[donors[pert]] for pert in target_perts}
        sources = {
            pert: f"shuffled:{donors[pert]}:{original_sources[donors[pert]]}"
            for pert in target_perts
        }
    # Re-select signature genes from the intervened shift, never from the
    # original target's shift or held-out differential-expression statistics.
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
