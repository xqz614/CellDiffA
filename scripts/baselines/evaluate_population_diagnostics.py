#!/usr/bin/env python
"""Independent descriptive diversity/distribution checks, never steering rewards.

These supplement, and do not replace, the frozen Cell-Eval protocol. Random
projections are fixed before reading outcomes; controls are excluded from scores.
Distances are descriptive, not likelihoods or guarantees of biological validity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.contracts import validate_prediction_pair


def dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def effective_rank(values):
    centered = np.asarray(values, dtype=np.float64) - values.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered)
    eigenvalues = np.maximum(eigenvalues, 0)
    if eigenvalues.sum() < 1e-12:
        return 0.0
    probability = eigenvalues / eigenvalues.sum()
    probability = probability[probability > 1e-12]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def sliced_wasserstein(first, second):
    # Matching cell counts are required by the experiment contract.
    if first.shape != second.shape:
        raise ValueError("Sliced distances require matched population sizes")
    return float(np.abs(np.sort(first, axis=0) - np.sort(second, axis=0)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--projections", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.projections < 2:
        parser.error("At least two fixed projections are required")
    real, prediction = ad.read_h5ad(args.real), ad.read_h5ad(args.pred)
    base = ad.read_h5ad(args.base) if args.base else None
    for data in [prediction] + ([base] if base is not None else []):
        validate_prediction_pair(real, data, pert_col="gene", control_pert="non-targeting")
    matrix = np.random.default_rng(args.seed).normal(size=(real.n_vars, args.projections))
    matrix = (matrix / np.linalg.norm(matrix, axis=0)).astype(np.float32)
    r_labels = real.obs.gene.astype(str).to_numpy()
    p_labels = prediction.obs.gene.astype(str).to_numpy()
    b_labels = base.obs.gene.astype(str).to_numpy() if base is not None else None
    rows = []
    for name in sorted(set(r_labels) - {"non-targeting"}):
        r, p = dense(real.X[r_labels == name]), dense(prediction.X[p_labels == name])
        if not np.isfinite(r).all() or not np.isfinite(p).all():
            raise ValueError(f"Non-finite expression in {name}")
        rp, pp = r @ matrix, p @ matrix
        rv = float(np.var(r, axis=0, dtype=np.float64).sum())
        pv = float(np.var(p, axis=0, dtype=np.float64).sum())
        rr, pr = effective_rank(rp), effective_rank(pp)
        row = dict(
            perturbation=name,
            cells=len(r),
            predicted_to_real_variance=pv / rv if rv > 0 else np.nan,
            predicted_unique_fraction=len(np.unique(p, axis=0)) / len(p),
            real_unique_fraction=len(np.unique(r, axis=0)) / len(r),
            projected_effective_rank_real=rr,
            projected_effective_rank_pred=pr,
            predicted_to_real_rank=pr / rr if rr > 0 else np.nan,
            sliced_w1_to_real=sliced_wasserstein(rp, pp),
        )
        if base is not None:
            b = dense(base.X[b_labels == name])
            bp = b @ matrix
            bv = float(np.var(b, axis=0, dtype=np.float64).sum())
            row["sliced_w1_to_base"] = sliced_wasserstein(pp, bp)
            row["base_sliced_w1_to_real"] = sliced_wasserstein(rp, bp)
            row["predicted_to_base_variance"] = pv / bv if bv > 0 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("perturbation")
    args.outdir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.outdir / "population_diagnostics.csv")
    # pandas converts unavailable ratios to JSON null rather than invalid NaN.
    stats = json.loads(table.agg(["mean", "median", "count"]).to_json())
    write_manifest(args.outdir / "population_diagnostics_summary.json", stats)
    write_manifest(
        args.outdir / "diagnostics_manifest.json",
        dict(
            real_sha256=sha256_file(args.real),
            prediction_sha256=sha256_file(args.pred),
            base_sha256=sha256_file(args.base) if args.base else None,
            projections=args.projections,
            seed=args.seed,
            interpretation="descriptive checks, not proof of realism or support preservation",
        ),
    )
    print(f"WROTE independent diagnostics for {len(table)} perturbations: {args.outdir}")


if __name__ == "__main__":
    main()
