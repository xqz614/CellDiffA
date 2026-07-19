#!/usr/bin/env python
"""Evaluate every standardized baseline prediction and build one comparison table."""

import argparse
from pathlib import Path

import pandas as pd

from celldiffa.benchmark.metrics import evaluate_perturbdiff_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--pert-col", required=True)
    parser.add_argument("--control-pert", required=True)
    parser.add_argument("--num-threads", type=int, default=16)
    args = parser.parse_args()

    prediction_root = Path(args.prediction_root)
    output_root = Path(args.output_root)
    rows = []
    for method in args.methods:
        prediction = prediction_root / f"{method}.h5ad"
        if not prediction.exists():
            raise FileNotFoundError(f"Missing standardized prediction: {prediction}")
        _, summary = evaluate_perturbdiff_protocol(
            real_path=args.real,
            pred_path=prediction,
            outdir=output_root / method,
            pert_col=args.pert_col,
            control_pert=args.control_pert,
            num_threads=args.num_threads,
        )
        row = summary.loc["mean"].to_dict()
        row["method"] = method
        rows.append(row)

    table = pd.DataFrame(rows).set_index("method")
    output_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_root / "comparison_mean.csv")
    table.to_markdown(output_root / "comparison_mean.md")
    print(table.to_string())


if __name__ == "__main__":
    main()
