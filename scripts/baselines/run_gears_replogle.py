#!/usr/bin/env python
"""Train GEARS on the exact PerturbDiff Replogle training rows and predict its test set."""

from __future__ import annotations

import argparse
import pickle
import random
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from baselines.adapter_gears import GEARSAdapter
from celldiffa.benchmark.artifacts import load_selected_genes, sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.gears_replogle import (
    materialize_training_anndata,
    validate_gene_space,
)
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit

GEARS_ASSETS = ("gene2go_all.pkl", "essential_all_data_pert_genes.pkl")


def _copy_cached_assets(asset_dir: Path | None, work_dir: Path) -> list[str]:
    copied = []
    if asset_dir is None:
        return copied
    for name in GEARS_ASSETS:
        source = asset_dir / name
        target = work_dir / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def _preflight_default_graph(work_dir: Path, test_perts: list[str]) -> None:
    gene2go_path = work_dir / "gene2go_all.pkl"
    essential_path = work_dir / "essential_all_data_pert_genes.pkl"
    if not gene2go_path.exists() or not essential_path.exists():
        return
    with gene2go_path.open("rb") as handle:
        gene2go = pickle.load(handle)
    with essential_path.open("rb") as handle:
        essential = pickle.load(handle)
    available = set(gene2go) & {str(value) for value in essential}
    missing = sorted(set(test_perts) - available)
    if missing:
        raise ValueError(
            f"GEARS' official perturbation graph is missing {len(missing)} test genes: "
            f"{missing[:20]}. This was detected before preprocessing."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GEARS extended baseline for the PerturbDiff Replogle-Nadig split. "
            "Test expression is excluded before GEARS preprocessing."
        )
    )
    parser.add_argument("--source", required=True, help="PerturbDiff Replogle source H5AD")
    parser.add_argument("--real-test", required=True, help="Immutable diffusion_true H5AD")
    parser.add_argument("--upstream-split-config", required=True)
    parser.add_argument("--selected-genes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True, help="GEARS data/cache directory")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--asset-dir",
        help="Optional directory containing cached GEARS gene2go/essential-gene assets",
    )
    parser.add_argument(
        "--mode",
        choices=["pooled", "heldout_only"],
        default="pooled",
        help=(
            "pooled uses every official training row but ignores context; heldout_only uses "
            "only the held-out cell line's released training subset"
        ),
    )
    parser.add_argument("--expression-key", default="X_hvg")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument(
        "--dynamic-perturbation-graph",
        action="store_true",
        help="Use GEARS' smaller data-derived perturbation graph instead of its official default",
    )
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Skip SHA-256 calculation for large inputs (recorded in the manifest)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.hidden_size < 1:
        raise ValueError("epochs, batch-size, and hidden-size must be positive.")

    source = Path(args.source).resolve()
    real_path = Path(args.real_test).resolve()
    split_path = Path(args.upstream_split_config).resolve()
    genes_path = Path(args.selected_genes).resolve()
    output_path = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    for path in (source, real_path, split_path, genes_path):
        if not path.exists():
            raise FileNotFoundError(path)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    split = PerturbDiffSplit.from_yaml(split_path, split_axis="context")
    if len(split.holdout_contexts) != 1:
        raise ValueError(
            "GEARS-Replogle requires exactly one PerturbDiff holdout context; "
            f"found {split.holdout_contexts}."
        )
    selected_genes = load_selected_genes(genes_path)
    real = ad.read_h5ad(real_path)
    split.validate_real_test(real)
    validate_gene_space(real, selected_genes=selected_genes)

    real_labels = real.obs[split.pert_col].astype(str).to_numpy()
    test_perts = sorted(set(real_labels) - {split.control_pert})
    copied_assets = _copy_cached_assets(
        Path(args.asset_dir).resolve() if args.asset_dir else None,
        work_dir,
    )
    if not args.dynamic_perturbation_graph:
        _preflight_default_graph(work_dir, test_perts)

    training, counts = materialize_training_anndata(
        source,
        split=split,
        selected_genes=selected_genes,
        expression_key=args.expression_key,
        mode=args.mode,
        chunk_size=args.chunk_size,
    )
    split_digest = sha256_file(split_path)[:12]
    dataset_name = f"replogle_pd_{args.mode}_{split_digest}"
    adapter = GEARSAdapter(
        data_path=str(work_dir),
        device=args.device,
        seed=args.seed,
        default_pert_graph=not args.dynamic_perturbation_graph,
    )
    adapter.fit(
        training,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        split_strategy="all_train",
        dataset_name=dataset_name,
    )

    available = set(str(value) for value in adapter._pert_data.pert_names)
    missing_from_graph = sorted(set(test_perts) - available)
    if missing_from_graph:
        raise ValueError(
            f"GEARS perturbation graph is missing {len(missing_from_graph)} test genes: "
            f"{missing_from_graph[:20]}. No perturbations were silently dropped."
        )
    predictions = {}
    for pert in test_perts:
        n_cells = int(np.sum(real_labels == pert))
        predictions[pert] = adapter.predict([pert], n_samples=n_cells)[pert]

    pred = build_prediction_anndata(
        real,
        predictions,
        pert_col=split.pert_col,
        control_pert=split.control_pert,
    )
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest = {
        "baseline": "GEARS",
        "benchmark_tier": "extended",
        "dataset": "replogle",
        "mode": args.mode,
        "context_handling": (
            "context_agnostic_pooled_training"
            if args.mode == "pooled"
            else "heldout_context_training_only"
        ),
        "holdout_contexts": list(split.holdout_contexts),
        "split_policy": "PerturbDiff official masks; GEARS random splitting disabled",
        "checkpoint_selection": "fixed epochs; training rows reused for validation",
        "source": str(source),
        "real_test": str(real_path),
        "split_config": str(split_path),
        "selected_genes": str(genes_path),
        "output": str(output_path),
        "model_dir": str(model_dir),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "learning_rate": args.lr,
        "device": args.device,
        "default_perturbation_graph": not args.dynamic_perturbation_graph,
        "copied_cached_assets": copied_assets,
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
    pred.uns["celldiffa_gears_replogle"] = manifest
    pred.write_h5ad(output_path, compression="gzip")
    adapter.save_checkpoint(str(model_dir))
    manifest["sha256_output"] = sha256_file(output_path)
    write_manifest(manifest_path, manifest)
    print(f"Wrote GEARS predictions: {output_path}")
    print(f"Wrote fairness manifest: {manifest_path}")
    print(f"Saved GEARS model: {model_dir}")


if __name__ == "__main__":
    main()
