#!/usr/bin/env python
"""Run the official Linear equations on the PerturbDiff Replogle split."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from celldiffa.benchmark.artifacts import load_selected_genes, sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.linear_replogle import fit_official_linear, training_pseudobulk
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit

OFFICIAL_REPOSITORY = "https://github.com/const-ae/linear_perturbation_prediction-Paper"
OFFICIAL_REVISION = "bfa6eeea2bd145a1af2ec0127a2e808cc38456a9"
OFFICIAL_SCRIPT = "benchmark/src/run_linear_pretrained_model.R"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PCA plus two-sided ridge Linear baseline for the PerturbDiff "
            "Replogle-Nadig test split."
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--real-test", required=True)
    parser.add_argument("--upstream-split-config", required=True)
    parser.add_argument("--selected-genes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-output", help="Optional fitted .npz path")
    parser.add_argument(
        "--expression-key",
        default="X",
        help=(
            "Expression space used to learn PCA and perturbation embeddings. Replogle must "
            "use full X because many CRISPR targets are outside the 2,000 evaluation HVGs."
        ),
    )
    parser.add_argument("--mode", choices=["pooled", "heldout_only"], default="pooled")
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument("--ridge-penalty", type=float, default=0.1)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--skip-input-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = Path(args.source).resolve()
    real_path = Path(args.real_test).resolve()
    split_path = Path(args.upstream_split_config).resolve()
    genes_path = Path(args.selected_genes).resolve()
    output_path = Path(args.output).resolve()
    model_path = (
        Path(args.model_output).resolve()
        if args.model_output
        else output_path.with_suffix(".model.npz")
    )
    for path in (source, real_path, split_path, genes_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    split = PerturbDiffSplit.from_yaml(split_path, split_axis="context")
    if len(split.holdout_contexts) != 1:
        raise ValueError(
            "Linear-Replogle expects one held-out context; "
            f"found {split.holdout_contexts}."
        )
    selected_genes = load_selected_genes(genes_path)
    real = ad.read_h5ad(real_path)
    split.validate_real_test(real)
    if list(real.var_names.astype(str)) != selected_genes:
        raise ValueError("Selected-gene pickle order differs from the real-test H5AD.")

    source_backed = ad.read_h5ad(source, backed="r")
    try:
        if args.expression_key == "X":
            fit_genes = list(source_backed.var_names.astype(str))
        elif args.expression_key == "X_hvg":
            fit_genes = selected_genes
        else:
            raise ValueError(
                "Linear currently supports expression-key X (full fitting space) or "
                "X_hvg (evaluation space)."
            )
    finally:
        source_backed.file.close()

    pseudobulk, conditions, counts = training_pseudobulk(
        source,
        split=split,
        expression_key=args.expression_key,
        mode=args.mode,
        chunk_size=args.chunk_size,
    )
    if pseudobulk.shape[1] != len(fit_genes):
        raise ValueError(
            f"Source {args.expression_key} has {pseudobulk.shape[1]} genes, but "
            f"the inferred fitting gene space has {len(fit_genes)}."
        )
    fit = fit_official_linear(
        pseudobulk,
        conditions,
        fit_genes,
        control_pert=split.control_pert,
        pca_dim=args.pca_dim,
        ridge_penalty=args.ridge_penalty,
    )

    labels = real.obs[split.pert_col].astype(str).to_numpy()
    test_perts = sorted(set(labels) - {split.control_pert})
    predicted_means = fit.predict_means(test_perts, output_genes=selected_genes)
    predictions = {
        pert: np.repeat(
            mean[None, :],
            int(np.sum(labels == pert)),
            axis=0,
        )
        for pert, mean in predicted_means.items()
    }
    pred = build_prediction_anndata(
        real,
        predictions,
        pert_col=split.pert_col,
        control_pert=split.control_pert,
    )

    manifest = {
        "baseline": "Linear",
        "benchmark_tier": "paper",
        "dataset": "replogle",
        "mode": args.mode,
        "context_handling": (
            "context_agnostic_pooled_training"
            if args.mode == "pooled"
            else "heldout_context_training_only"
        ),
        "split_policy": "PerturbDiff official masks; validation/test expression excluded",
        "prediction_population": "official deterministic pseudobulk mean repeated per test cell",
        "implementation": (
            "Python translation of official PCA and two-sided ridge equations; "
            "deterministic full SVD replaces prcomp_irlba"
        ),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": OFFICIAL_REVISION,
        "official_script": OFFICIAL_SCRIPT,
        "source": str(source),
        "real_test": str(real_path),
        "split_config": str(split_path),
        "selected_genes": str(genes_path),
        "output": str(output_path),
        "model_output": str(model_path),
        "expression_key": args.expression_key,
        "fitting_genes": len(fit_genes),
        "evaluation_genes": len(selected_genes),
        "pca_dim": args.pca_dim,
        "ridge_penalty": args.ridge_penalty,
        "holdout_contexts": list(split.holdout_contexts),
        "test_perturbations": len(test_perts),
        "matched_training_conditions": len(fit.training_conditions) - 1,
        "counts": counts,
        "input_hashes_skipped": args.skip_input_hashes,
    }
    if not args.skip_input_hashes:
        manifest["sha256"] = {
            "source": sha256_file(source),
            "real_test": sha256_file(real_path),
            "split_config": sha256_file(split_path),
            "selected_genes": sha256_file(genes_path),
        }
    pred.uns["celldiffa_linear_replogle"] = manifest
    pred.write_h5ad(output_path, compression="gzip")
    np.savez_compressed(
        model_path,
        gene_scores=fit.gene_scores,
        coefficients=fit.coefficients,
        response_center=fit.response_center,
        control_baseline=fit.control_baseline,
        genes=np.asarray(fit.genes),
        training_conditions=np.asarray(fit.training_conditions),
        pca_dim=np.asarray(fit.pca_dim),
        ridge_penalty=np.asarray(fit.ridge_penalty),
    )
    manifest["sha256_output"] = sha256_file(output_path)
    manifest["sha256_model"] = sha256_file(model_path)
    write_manifest(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    print(f"Wrote Linear predictions: {output_path}")
    print(f"Wrote Linear model: {model_path}")


if __name__ == "__main__":
    main()
