#!/usr/bin/env python
"""Same-prior post-hoc mean calibration, not AdaCell or a causal mechanism.

Per-gene translation followed by projection to nonnegative expression with the
specified feasible mean. This explicitly discloses how clipping changes residuals.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np

from celldiffa.benchmark.artifacts import load_embedding_dict, load_selected_genes, sha256_file
from celldiffa.benchmark.backbone_experiments import atomic_json, dense
from celldiffa.benchmark.contracts import validate_prediction_pair
from celldiffa.benchmark.replogle_priors import compute_replogle_training_priors


def correct_mean(values, target):
    """Nonnegative shifted distribution with target mean max(target, 0)."""
    target = np.maximum(np.asarray(target, dtype=np.float64), 0)
    values = np.asarray(values, dtype=np.float64)
    low = -values.max(0) - 1
    high = target + np.abs(values).max(0) + 1
    for _ in range(55):
        offset = (low + high) / 2
        below = np.maximum(values + offset, 0).mean(0) < target
        low = np.where(below, offset, low)
        high = np.where(below, high, offset)
    result = np.maximum(values + (low + high) / 2, 0).astype(np.float32)
    result[:, target == 0] = 0
    return result


def prediction_contexts(real, prediction):
    """Recover missing metadata by verified row IDs, never by assumed row order."""
    if "cell_line" in prediction.obs:
        return prediction.obs.cell_line.astype(str).to_numpy()
    if (
        not real.obs_names.is_unique
        or not prediction.obs_names.is_unique
        or not prediction.obs_names.isin(real.obs_names).all()
    ):
        raise ValueError("Prediction lacks cell_line and uniquely alignable reference row IDs")
    metadata = real.obs.loc[prediction.obs_names]
    if not np.array_equal(metadata.gene.astype(str), prediction.obs.gene.astype(str)):
        raise ValueError("Prediction/reference perturbations disagree at the same row IDs")
    return metadata.cell_line.astype(str).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "real",
        "pred",
        "source",
        "split-config",
        "selected-genes",
        "embeddings",
        "outdir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError("Use a new mean-correction output directory")
    real, prediction = ad.read_h5ad(args.real), ad.read_h5ad(args.pred)
    validate_prediction_pair(real, prediction, pert_col="gene", control_pert="non-targeting")
    genes = load_selected_genes(args.selected_genes)
    if list(prediction.var_names) != genes:
        raise ValueError("Prediction gene order differs from prior genes")
    labels = prediction.obs.gene.astype(str).to_numpy()
    names = sorted(set(labels) - {"non-targeting"})
    priors = compute_replogle_training_priors(
        args.source,
        args.split_config,
        genes,
        top_k=20,
        target_perturbations=names,
        perturbation_embeddings=load_embedding_dict(args.embeddings),
        embedding_signature=sha256_file(args.embeddings),
    )
    values = dense(prediction.X).copy()
    # Prediction rows may be ordered differently from the reference. Metadata
    # must index the matrix it belongs to, even when population counts match.
    contexts = prediction_contexts(real, prediction)
    prediction.obs["cell_line"] = contexts
    records = []
    for name in names:
        for context in sorted(set(contexts[labels == name])):
            rows = (labels == name) & (contexts == context)
            control = (labels == "non-targeting") & (contexts == context)
            if not control.any():
                raise ValueError("No matched observed controls")
            target = values[control].mean(0) + priors.shifts[name]
            corrected = correct_mean(values[rows], target)
            records.append(
                dict(
                    perturbation=name,
                    context=context,
                    infeasible_negative_target_fraction=float((target < 0).mean()),
                    output_zero_fraction=float((corrected == 0).mean()),
                )
            )
            values[rows] = corrected
    prediction.X = values
    prediction.uns["method"] = "Same-prior nonnegative mean correction (post-hoc)"
    args.outdir.mkdir(parents=True)
    prediction.write_h5ad(args.outdir / "predictions.h5ad", compression="gzip")
    atomic_json(
        args.outdir / "provenance.json",
        dict(
            source_prediction_sha256=sha256_file(args.pred),
            records=records,
            reference_sha256=sha256_file(args.real),
            selected_genes_sha256=sha256_file(args.selected_genes),
            split_config_sha256=sha256_file(args.split_config),
            perturbation_embeddings_sha256=sha256_file(args.embeddings),
            prior_ridge=1.0,
            prior_sources=priors.sources,
            target_policy="max(observed-control mean + training prior, 0)",
            residual_policy=(
                "per-gene translation and nonnegative projection; not variance preserving"
            ),
            test_response_values_used_for_correction=False,
        ),
    )


if __name__ == "__main__":
    main()
