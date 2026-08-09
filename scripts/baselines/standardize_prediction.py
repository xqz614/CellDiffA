#!/usr/bin/env python
"""Convert official baseline arrays into the common Cell-Eval H5AD contract."""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from celldiffa.benchmark.contracts import build_prediction_anndata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-test", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prediction-dir",
        help="Directory containing one <perturbation>.npy matrix per test perturbation",
    )
    source.add_argument(
        "--pred-h5ad",
        help="Official H5AD output; rows are regrouped and genes are aligned to real-test",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-pert", required=True)
    args = parser.parse_args()

    real = ad.read_h5ad(args.real_test)
    labels = set(real.obs[args.pert_col].astype(str)) - {args.control_pert}
    arrays = {}
    if args.prediction_dir:
        source_dir = Path(args.prediction_dir)
        for pert in labels:
            path = source_dir / f"{pert}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Missing official prediction array: {path}")
            arrays[pert] = np.load(path, allow_pickle=False)
    else:
        official = ad.read_h5ad(args.pred_h5ad)
        if args.pert_col not in official.obs:
            raise ValueError(f"Official output is missing obs[{args.pert_col!r}].")
        missing_genes = real.var_names.difference(official.var_names)
        if len(missing_genes):
            raise ValueError(f"Official output is missing {len(missing_genes)} real genes.")
        official = official[:, real.var_names]
        official_labels = official.obs[args.pert_col].astype(str).to_numpy()
        for pert in labels:
            arrays[pert] = official.X[official_labels == pert]
            if hasattr(arrays[pert], "toarray"):
                arrays[pert] = arrays[pert].toarray()
    pred = build_prediction_anndata(
        real,
        arrays,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
    )
    pred.write_h5ad(args.output, compression="gzip")


if __name__ == "__main__":
    main()
