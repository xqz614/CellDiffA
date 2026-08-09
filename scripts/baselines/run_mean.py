#!/usr/bin/env python
"""Generate one of the Mean baselines from train/test H5AD files."""

import argparse

import anndata as ad

from celldiffa.benchmark.simple_baselines import MeanBaselineConfig, predict_mean_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--real-test", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant",
        choices=["perturbation", "cell_type", "batch", "overall"],
        default="perturbation",
    )
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-pert", required=True)
    parser.add_argument("--context-col")
    parser.add_argument("--batch-col")
    args = parser.parse_args()

    train = ad.read_h5ad(args.train)
    real = ad.read_h5ad(args.real_test)
    pred = predict_mean_baseline(
        train,
        real,
        config=MeanBaselineConfig(
            pert_col=args.pert_col,
            control_pert=args.control_pert,
            context_col=args.context_col,
            batch_col=args.batch_col,
        ),
        variant=args.variant,
    )
    pred.write_h5ad(args.output, compression="gzip")
    print(f"Wrote {pred.n_obs} cells x {pred.n_vars} genes to {args.output}")


if __name__ == "__main__":
    main()
