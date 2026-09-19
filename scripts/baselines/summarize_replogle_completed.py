#!/usr/bin/env python
"""Summarize only full, contract-checked test results; never select parameters."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.contracts import validate_prediction_pair
from celldiffa.benchmark.metrics import PAPER_METRIC_NAMES

PREDICTIONS = {
    "mean_perturbation": "predictions/mean_perturbation.h5ad",
    "mean_cell_type": "predictions/mean_cell_type.h5ad",
    "mean_batch": "predictions/mean_batch.h5ad",
    "mean_overall": "predictions/mean_overall.h5ad",
    "linear": "predictions/linear.h5ad",
    "scouter": "scouter_vectorized/predictions.h5ad",
    "cpa": "cpa_cpu/predictions.h5ad",
}


def check_metrics(table, expected):
    metrics = ["R2", *PAPER_METRIC_NAMES]
    if set(table.columns) != {"perturbation", *metrics}:
        raise ValueError("Expected exactly the frozen 14 metrics and condition labels")
    if table.perturbation.duplicated().any() or set(table.perturbation) != expected:
        raise ValueError("Metric coverage does not match all expected test conditions")
    if not np.isfinite(table[metrics].to_numpy()).all():
        raise ValueError("Partial or non-finite metric results are not complete")
    return table[metrics].mean().to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/replogle"))
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    reference = args.root / "reference/real.h5ad"
    real = ad.read_h5ad(reference)
    expected = set(real.obs.gene.astype(str)) - {"non-targeting"}
    reference_hash = sha256_file(reference)
    rows, diversity, records = [], [], {}
    for name, relative in PREDICTIONS.items():
        prediction = args.root / relative
        directory = args.root / "metrics" / name
        per_condition = directory / "perturbdiff_metrics_per_perturbation.csv"
        if not prediction.exists() or not per_condition.exists():
            records[name] = {"complete": False, "reason": "Missing full prediction or metrics"}
            continue
        pred = ad.read_h5ad(prediction)
        validate_prediction_pair(real, pred, pert_col="gene", control_pert="non-targeting")
        del pred
        values = check_metrics(pd.read_csv(per_condition), expected)
        summary = pd.read_csv(directory / "perturbdiff_metrics_summary.csv", index_col=0)
        if not (summary.loc["count"] == len(expected)).all():
            raise ValueError(f"Incomplete summary for {name}")
        if not np.allclose(summary.loc["mean", list(values)], list(values.values())):
            raise ValueError(f"Stale summary for {name}")
        rows.append({"Model": name, **values})
        prediction_hash = sha256_file(prediction)
        records[name] = {
            "complete": True,
            "conditions": len(expected),
            "prediction_sha256": prediction_hash,
            "metrics_sha256": sha256_file(per_condition),
        }
        manifest = directory / "diagnostics_manifest.json"
        if manifest.exists():
            provenance = json.loads(manifest.read_text())
            if provenance["real_sha256"] != reference_hash or (
                provenance["prediction_sha256"] != prediction_hash
            ):
                raise ValueError(f"Stale population diagnostics for {name}")
            diagnostic = pd.read_csv(directory / "population_diagnostics.csv")
            if diagnostic.perturbation.duplicated().any() or (
                set(diagnostic.perturbation) != expected
            ):
                raise ValueError(f"Incomplete population diagnostics for {name}")
            keys = [
                "predicted_to_real_variance",
                "predicted_to_real_rank",
                "predicted_unique_fraction",
                "sliced_w1_to_real",
            ]
            diversity.append({"Model": name, **diagnostic[keys].mean().to_dict()})
            records[name]["diagnostics_sha256"] = sha256_file(manifest)
    args.outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    text = [
        "# Replogle completed test results",
        "",
        f"Generated {timestamp}.",
        "",
        f"{len(rows)} complete baselines; each covers all {len(expected)} test conditions "
        f"and the same {real.n_vars} ordered genes. Values are macro-averages across conditions.",
        "",
        "## Frozen PerturbDiff evaluation",
        "",
        pd.DataFrame(rows).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Independent descriptive population checks",
        "",
        pd.DataFrame(diversity).to_markdown(index=False, floatfmt=".6f"),
        "",
        "Variance and rank ratios compare predicted to real populations, condition by "
        "condition; 1 means matching that statistic, not proven biological realism. "
        "Unique fraction counts exact distinct vectors. Sliced W1 uses 64 fixed random "
        "projections (seed 1729). Controls are excluded. These are not steering rewards.",
        "",
        "Mean and the current Linear adapter produce population means, so zero "
        "within-condition variance can be an intentional method property. Scouter and CPA "
        "have nonzero but attenuated variance; this is a diagnostic observation, not evidence "
        "that AdaCell is better. No complete AdaCell test result is included here.",
        "",
        "Test scores in this report are not used for tuning. The separate validation "
        "plan fixes temperature candidates and its selection rule before AdaCell completion.",
    ]
    (args.outdir / "completed_results.md").write_text("\n".join(text) + "\n")
    write_manifest(
        args.outdir / "manifest.json",
        {
            "generated_at": timestamp,
            "reference_sha256": reference_hash,
            "split": "test",
            "models": records,
        },
    )
    print(f"WROTE {len(rows)} verified complete baselines: {args.outdir}")


if __name__ == "__main__":
    main()
