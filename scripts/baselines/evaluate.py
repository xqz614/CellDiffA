#!/usr/bin/env python
"""Evaluate one standardized prediction with the PerturbDiff protocol."""

import argparse

from celldiffa.benchmark.metrics import evaluate_perturbdiff_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Real test H5AD including controls")
    parser.add_argument("--pred", required=True, help="Standardized predicted H5AD")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-pert", required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument(
        "--continue-on-metric-error",
        action="store_true",
        help="Keep partial Cell-Eval results; strict failure is the default.",
    )
    args = parser.parse_args()
    _, summary = evaluate_perturbdiff_protocol(
        real_path=args.real,
        pred_path=args.pred,
        outdir=args.outdir,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        num_threads=args.num_threads,
        break_on_error=not args.continue_on_metric_error,
    )
    print(summary.loc["mean"].to_string())


if __name__ == "__main__":
    main()
