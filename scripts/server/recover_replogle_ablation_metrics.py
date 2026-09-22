#!/usr/bin/env python
"""Re-evaluate only two already-generated, audited log1p-tail ablations.

Never regenerate cells, modify an H5AD, or change an existing metric directory.
The tail-fraction check is an operational guard, not a proof of correct units.
"""

# ruff: noqa: E402
import argparse
import fcntl
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import anndata as ad
import numpy as np
import pandas as pd

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.metrics import evaluate_perturbdiff_protocol, expression_scale_summary
from celldiffa.benchmark.streaming import read_h5ad_obs

RUNS = ("scratch_without_anchor", "scratch_without_direction")


def check_tail(path):
    data = ad.read_h5ad(path, backed="r")
    try:
        summary = expression_scale_summary(data)
    finally:
        data.file.close()
    if summary["nonfinite"] or summary["negative"] or summary["fraction_ge_15"] > 1e-6:
        raise ValueError(f"Not a finite/nonnegative rare-tail input; diagnose instead: {summary}")
    return summary


def record_evaluated(run, output, real, prediction):
    path = output / "perturbdiff_metrics_per_perturbation.csv"
    table = pd.read_csv(path)
    expected = set(read_h5ad_obs(real).gene.astype(str)) - {"non-targeting"}
    if table.perturbation.duplicated().any() or set(table.perturbation) != expected:
        raise ValueError("Metrics do not cover the exact complete test condition set")
    audit = json.loads((output / "input_scale_audit.json").read_text())
    actual_hash = sha256_file(path)
    if audit.get("metrics_sha256") != actual_hash:
        raise ValueError("Evaluation metric file no longer matches its recorded hash")
    marker = dict(
        reference_sha256=sha256_file(real),
        prediction_sha256=sha256_file(prediction),
        metrics_sha256=actual_hash,
        perturbations=len(table),
        all_metrics_finite=bool(np.isfinite(table.drop(columns="perturbation")).all().all()),
        input_scale="log1p",
        input_values_changed=False,
        metrics_directory=str(output),
    )
    marker_path = run / "evaluated.json"
    if marker_path.exists() and json.loads(marker_path.read_text()) != marker:
        raise ValueError("Refusing to overwrite a different existing evaluation record")
    write_manifest(marker_path, marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO / "results/replogle")
    parser.add_argument("--runs", choices=RUNS, nargs="+", default=list(RUNS))
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("num-threads must be positive")
    real = args.results_root / "reference/real.h5ad"
    failures = []
    for name in args.runs:
        try:
            run = args.results_root / "ablations" / name
            prediction = run / "celldiffa_scratch.h5ad"
            if not prediction.is_file():
                raise FileNotFoundError(prediction)
            with (run / ".evaluation_recovery.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                output = run / "evaluation"
                audit_file = output / "input_scale_audit.json"
                hashes = dict(
                    real_sha256=sha256_file(real), prediction_sha256=sha256_file(prediction)
                )
                if output.exists() and any(output.iterdir()):
                    if not audit_file.is_file():
                        raise ValueError(f"Refusing to overwrite an unrelated evaluation: {output}")
                    saved = json.loads(audit_file.read_text())
                    if any(saved.get(k) != v for k, v in hashes.items()):
                        raise ValueError("Input changed since the saved evaluation")
                    if saved.get("status") == "complete":
                        record_evaluated(run, output, real, prediction)
                        print(f"ALREADY EVALUATED: {name}\n{output}", flush=True)
                        continue
                print(f"AUDIT {name}: {check_tail(prediction)}", flush=True)
                check_tail(real)
                print("Keep all values; explicit log1p, DE is_log1p=True", flush=True)
                table, summary = evaluate_perturbdiff_protocol(
                    real_path=real,
                    pred_path=prediction,
                    outdir=output,
                    pert_col="gene",
                    control_pert="non-targeting",
                    num_threads=args.num_threads,
                    input_scale="log1p",
                )
                record_evaluated(run, output, real, prediction)
                # Report undefined metrics explicitly; do not promote nan-skipping means.
                columns = ["DEOver", "PDCorr", "PDS_cos", "MSE"]
                means = {
                    c: float(summary.loc["mean", c])
                    if np.isfinite(table[c]).all()
                    else float("nan")
                    for c in columns
                }
                print(f"EVALUATED: {name}\n{pd.Series(means).to_string()}\n{output}", flush=True)
        except Exception as error:
            failures.append(name)
            print(f"FAILED {name}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f"Incomplete evaluations: {', '.join(failures)}")


if __name__ == "__main__":
    main()
