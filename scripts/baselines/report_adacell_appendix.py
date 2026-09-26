#!/usr/bin/env python
"""Report completed appendix runs without modifying predictions or metric values.

Incomplete condition sets and non-finite values are reported, never silently
dropped from an aggregate. Figures show actual completed runs, not interpolated
or benchmark-calibrated values. No GPU/model imports are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("DEOver", "PDCorr", "PDS_cos", "MSE")
METRIC_LABELS = {"DEOver": "DEOver", "PDCorr": "PDCorr", "PDS_cos": "PDS-cos", "MSE": "MSE"}
PARAMETERS = {
    "alpha": r"Temperature $\tau$",
    "num_particles": r"Population particles $K$",
    "top_de": r"Signature genes $L$",
    "ess_threshold": r"ESS threshold $\rho$",
    "anchor_bandwidth": r"Anchor bandwidth $\sigma$",
}
PER_FILE = "perturbdiff_metrics_per_perturbation.csv"
DIAGNOSTIC_FILE = "population_diagnostics.csv"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value, parent):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def expected_conditions(reference):
    """Read reference metadata only; avoid materializing expression matrices."""
    import h5py

    try:
        from anndata.io import read_elem
    except ImportError:
        from anndata._io.specs import read_elem
    with h5py.File(reference, "r") as handle:
        obs = read_elem(handle["obs"])
    if "gene" not in obs:
        raise ValueError("Reference obs must contain the gene label column")
    if obs["gene"].isna().any():
        raise ValueError("Reference gene labels contain missing values")
    conditions = set(obs["gene"].astype(str)) - {"non-targeting"}
    if not conditions:
        raise ValueError("Reference has no non-control perturbations")
    return conditions


def summarize_table(table, expected, metrics):
    """Return raw long rows and strict aggregate rows, including invalid counts."""
    table = table.copy()
    label = "perturbation" if "perturbation" in table else "gene"
    if label not in table:
        raise ValueError("Metric table is missing perturbation/gene labels")
    if table[label].isna().any():
        raise ValueError("Metric table contains missing perturbation labels")
    table[label] = table[label].astype(str)
    table = table.loc[table[label] != "non-targeting"]
    labels = table[label]
    present = set(labels)
    coverage = present == expected and not labels.duplicated().any() and len(table) == len(expected)
    long, summary = [], []
    for metric in metrics:
        available = metric in table
        values = (
            pd.to_numeric(table[metric], errors="coerce").to_numpy(dtype=float)
            if available
            else np.full(len(table), np.nan)
        )
        finite = np.isfinite(values)
        valid = bool(coverage and available and finite.all())
        summary.append(
            dict(
                metric=metric,
                expected_count=len(expected),
                observed_count=len(table),
                matched_count=int(labels.isin(expected).sum()),
                finite_count=int(finite.sum()),
                missing_conditions=len(expected - present),
                unexpected_conditions=len(present - expected),
                duplicate_rows=int(labels.duplicated().sum()),
                missing_metric=not available,
                complete=valid,
                mean=float(values.mean()) if valid else None,
            )
        )
        for condition, value, ok in zip(labels, values, finite):
            long.append(
                dict(
                    perturbation=condition,
                    metric=metric,
                    value=float(value),
                    finite=bool(ok),
                    expected_condition=condition in expected,
                )
            )
    return long, summary


def expected_hash(state, kind, path):
    """Support direct digests and queue hash maps, without accepting mismatches."""
    direct = state.get(f"{kind}_sha256")
    if direct:
        return direct
    for field in ("hashes", "artifacts", "artifact_hashes", "outputs"):
        mapping = state.get(field, {})
        if not isinstance(mapping, dict):
            continue
        for key in (kind, f"{kind}_sha256", path.name, str(path), f"{kind}/{path.name}"):
            value = mapping.get(key)
            if isinstance(value, dict):
                value = value.get("sha256")
            if value:
                return value
    return None


def read_run(root, job, expected, *, reference_hash=None):
    job_id = job["id"]
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError(f"Unsafe job id: {job_id!r}")
    directory = root / "jobs" / job_id
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"status": "pending"}
    record = dict(
        id=job_id,
        section=job.get("section"),
        parameter=job.get("parameter"),
        value=job.get("value"),
        seed=job.get("settings", {}).get("seed"),
        status=state.get("status", "unknown"),
        report_status="not_complete",
        issues=[],
    )
    long, aggregates, tables = [], [], {}
    if record["status"] != "complete":
        return record, long, aggregates, tables
    if reference_hash is not None and state.get("reference_sha256") not in {None, reference_hash}:
        record["issues"].append("SHA-256 mismatch for plan reference")
        record["report_status"] = "incomplete_or_invalid"
        return record, long, aggregates, tables
    for kind, relative, required in (
        ("metrics", f"metrics/{PER_FILE}", METRICS),
        ("diagnostics", f"diagnostics/{DIAGNOSTIC_FILE}", None),
    ):
        path = directory / relative
        try:
            if not path.is_file():
                raise ValueError(f"Missing {relative}")
            digest = sha256(path)
            recorded = expected_hash(state, kind, path)
            if recorded is not None and recorded != digest:
                raise ValueError(f"SHA-256 mismatch for {relative}")
            record[f"{kind}_sha256"] = digest
            record[f"{kind}_hash_verified"] = recorded is not None
            table = pd.read_csv(path)
            metrics = (
                required
                if required is not None
                else [
                    name for name in table.columns if name not in {"perturbation", "gene", "cells"}
                ]
            )
            if not metrics:
                raise ValueError(f"No diagnostic metrics in {relative}")
            rows, stats = summarize_table(table, expected, metrics)
            common = {key: record[key] for key in ("id", "section", "parameter", "seed")}
            common["parameter_value"] = record["value"]
            long.extend({**common, "kind": kind, **row} for row in rows)
            aggregates.extend({**common, "kind": kind, **row} for row in stats)
            tables[kind] = table
            if not all(row["complete"] for row in stats):
                record["issues"].append(f"{kind}: incomplete coverage or non-finite/missing values")
        except (ValueError, OSError, pd.errors.ParserError) as error:
            record["issues"].append(str(error))
    record["report_status"] = "validated" if not record["issues"] else "incomplete_or_invalid"
    return record, long, aggregates, tables


def full_prior(settings):
    return (
        settings.get("prior_mode", "full") in {"full", "original", "none"}
        and float(settings.get("prior_fraction", 1.0)) == 1.0
    )


def prior_label(job):
    settings = job.get("settings", {})
    mode = settings.get("prior_mode", "full")
    if mode in {"shuffle", "shuffled", "permuted"}:
        return "Shuffled"
    fraction = float(settings.get("prior_fraction", 1.0))
    if fraction == 1 and full_prior(settings):
        return "Full"
    return f"{100 * fraction:g}% training cells"


def settings_match(first, second, excluded=()):
    excluded = set(excluded) | {"prior_seed"}
    keys = (set(first) | set(second)) - excluded
    return all(first.get(key) == second.get(key) for key in keys)


def paired_prior_rows(jobs, tables, expected):
    """Pair each altered prior only with the full prior at the same sampler seed."""
    full = {}
    for job in jobs:
        if full_prior(job.get("settings", {})) and (
            job.get("section") == "prior" or job.get("parameter") == "reference"
        ):
            seed = job.get("settings", {}).get("seed")
            if seed in full:
                raise ValueError(f"Ambiguous full-prior reference for seed {seed}")
            full[seed] = job
    rows, summaries = [], []
    for job in jobs:
        if job.get("section") != "prior" or full_prior(job.get("settings", {})):
            continue
        seed = job.get("settings", {}).get("seed")
        base = full.get(seed)
        if not base or not settings_match(
            job["settings"], base["settings"], {"prior_mode", "prior_fraction"}
        ):
            continue
        left, right = (
            tables.get(base["id"], {}).get("metrics"),
            tables.get(job["id"], {}).get("metrics"),
        )
        if left is None or right is None:
            continue
        _, ls = summarize_table(left, expected, METRICS)
        _, rs = summarize_table(right, expected, METRICS)
        lkey = "perturbation" if "perturbation" in left else "gene"
        rkey = "perturbation" if "perturbation" in right else "gene"
        for li, ri in zip(ls, rs):
            metric = li["metric"]
            valid = li["complete"] and ri["complete"]
            delta = None
            if valid:
                lvalues = left.set_index(lkey).loc[sorted(expected), metric].to_numpy(dtype=float)
                rvalues = right.set_index(rkey).loc[sorted(expected), metric].to_numpy(dtype=float)
                differences = rvalues - lvalues
                delta = float(differences.mean())
                for condition, before, after, difference in zip(
                    sorted(expected), lvalues, rvalues, differences
                ):
                    rows.append(
                        dict(
                            id=job["id"],
                            reference_id=base["id"],
                            seed=seed,
                            prior=prior_label(job),
                            perturbation=condition,
                            metric=metric,
                            full_value=before,
                            altered_value=after,
                            delta=difference,
                        )
                    )
            summaries.append(
                dict(
                    id=job["id"],
                    reference_id=base["id"],
                    seed=seed,
                    prior=prior_label(job),
                    metric=metric,
                    complete=bool(valid),
                    expected_count=len(expected),
                    paired_count=len(expected) if valid else 0,
                    mean_delta=delta,
                )
            )
    return rows, summaries


def configure_plots():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    return plt


def style_axis(axis, metric, index):
    axis.set_title(f"({chr(97 + index)}) {METRIC_LABELS[metric]}", loc="center", pad=10)
    axis.set_ylabel(METRIC_LABELS[metric])
    axis.set_box_aspect(1)
    axis.tick_params(direction="in", top=True, right=True)
    for spine in axis.spines.values():
        spine.set_visible(True)
    axis.grid(axis="y", alpha=0.18, linewidth=0.6)


def make_figures(outdir, jobs, aggregates, expected):
    plt = configure_plots()
    means = {
        (row["id"], row["metric"]): row["mean"] for row in aggregates if row["kind"] == "metrics"
    }
    captions, files = [], []
    references = [job for job in jobs if job.get("parameter") == "reference"]
    for parameter, xlabel in PARAMETERS.items():
        selected = [
            job
            for job in jobs
            if job.get("section") == "sensitivity" and job.get("parameter") == parameter
        ]
        if not selected:
            continue
        configurations = [job["settings"] for job in selected]
        if any(
            not settings_match(configurations[0], settings, {parameter, "seed"})
            for settings in configurations[1:]
        ):
            captions.append(
                f"Sensitivity ({parameter}): no figure generated because additional settings "
                "differ between planned runs; compare configurations before interpreting effects."
            )
            continue
        for reference in references:
            matches = [
                job
                for job in selected
                if settings_match(job["settings"], reference["settings"], {parameter})
            ]
            if matches and not any(
                job["settings"].get(parameter) == reference["settings"].get(parameter)
                and job["settings"].get("seed") == reference["settings"].get("seed")
                for job in selected
            ):
                selected.append(reference)
        if not any(
            means.get((job["id"], metric)) is not None for job in selected for metric in METRICS
        ):
            captions.append(
                f"Sensitivity ({parameter}): no complete finite measurements; no figure generated."
            )
            continue
        fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), layout="constrained")
        seeds = sorted({job["settings"].get("seed", 42) for job in selected})
        for index, (axis, metric) in enumerate(zip(axes, METRICS)):
            for seed in seeds:
                group = sorted(
                    [job for job in selected if job["settings"].get("seed", 42) == seed],
                    key=lambda job: float(job["settings"][parameter]),
                )
                x = [float(job["settings"][parameter]) for job in group]
                y = [means.get((job["id"], metric)) for job in group]
                axis.plot(
                    x,
                    [np.nan if value is None else value for value in y],
                    marker="o",
                    markersize=5,
                    linewidth=1.5,
                    label=f"Seed {seed}",
                )
                axis.set_xticks(sorted(set(x)))
            style_axis(axis, metric, index)
            axis.set_xlabel(xlabel)
        if len(seeds) > 1:
            axes[-1].legend(frameon=False, fontsize=9)
        stem = f"sensitivity_{parameter}"
        for suffix in ("pdf", "png"):
            fig.savefig(outdir / f"{stem}.{suffix}", bbox_inches="tight")
            files.append(f"{stem}.{suffix}")
        plt.close(fig)
        counts = {
            metric: sum(means.get((job["id"], metric)) is not None for job in selected)
            for metric in METRICS
        }
        captions.append(
            f"{stem}: {xlabel} sensitivity. Each point is an unmodified arithmetic mean "
            f"over all {len(expected)} non-control test perturbations for one completed run. "
            f"Seeds: {seeds}. Complete points / {len(selected)} planned or matched-reference "
            f"points: {counts}. Missing or non-finite aggregates are omitted and not connected "
            "across; no confidence intervals are estimated. Other settings must be fixed "
            "within each comparison. MSE is lower-is-better; "
            "the other metrics are higher-is-better."
        )
    prior_jobs = [job for job in jobs if job.get("section") == "prior"]
    if not prior_jobs:
        return captions, files, []
    for reference in references:
        if full_prior(reference.get("settings", {})) and not any(
            full_prior(job.get("settings", {}))
            and job["settings"].get("seed") == reference["settings"].get("seed")
            for job in prior_jobs
        ):
            prior_jobs.append(reference)
    seeds = sorted({job["settings"].get("seed", 42) for job in prior_jobs})
    labels = sorted(
        {prior_label(job) for job in prior_jobs},
        key=lambda label: (
            {"Full": 0, "50% training cells": 1, "25% training cells": 2, "Shuffled": 3}.get(
                label, 4
            ),
            label,
        ),
    )
    grouped = {}
    for job in prior_jobs:
        key = prior_label(job), job["settings"].get("seed", 42)
        if key in grouped:
            raise ValueError(f"Duplicate prior/seed jobs: {key}")
        grouped[key] = job
    eligible = {}
    for (label, seed), job in grouped.items():
        full = grouped.get(("Full", seed))
        eligible[(label, seed)] = bool(
            full
            and settings_match(job["settings"], full["settings"], {"prior_mode", "prior_fraction"})
        )
    across = []
    for metric in METRICS:
        for label in labels:
            values = [
                means.get((grouped.get((label, seed), {}).get("id"), metric))
                if eligible.get((label, seed))
                else None
                for seed in seeds
            ]
            full_values = [
                means.get((grouped.get(("Full", seed), {}).get("id"), metric)) for seed in seeds
            ]
            complete = all(value is not None for value in values + full_values)
            across.append(
                dict(
                    prior=label,
                    metric=metric,
                    expected_seeds=len(seeds),
                    completed_seeds=sum(value is not None for value in values),
                    full_reference_seeds=sum(value is not None for value in full_values),
                    seed_ids=",".join(map(str, seeds)),
                    complete=complete,
                    mean=float(np.mean(values)) if complete else None,
                )
            )
    if any(means.get((job["id"], metric)) is not None for job in prior_jobs for metric in METRICS):
        fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.0), layout="constrained")
        for index, (axis, metric) in enumerate(zip(axes, METRICS)):
            for position, label in enumerate(labels):
                for offset, seed in zip(
                    np.linspace(-0.16, 0.16, max(len(seeds), 2))[: len(seeds)], seeds
                ):
                    job = grouped.get((label, seed), {})
                    value = means.get((job.get("id"), metric))
                    if value is not None and eligible.get((label, seed)):
                        axis.scatter(position + offset, value, color="#559578", alpha=0.8, s=27)
                row = next(
                    row for row in across if row["metric"] == metric and row["prior"] == label
                )
                if row["mean"] is not None:
                    axis.plot(
                        [position - 0.22, position + 0.22],
                        [row["mean"]] * 2,
                        color="#203F37",
                        linewidth=2.2,
                    )
            style_axis(axis, metric, index)
            axis.set_xticks(
                range(len(labels)),
                [label.replace(" training cells", " cells") for label in labels],
                rotation=28,
                ha="right",
            )
        for suffix in ("pdf", "png"):
            fig.savefig(outdir / f"prior_robustness.{suffix}", bbox_inches="tight")
            files.append(f"prior_robustness.{suffix}")
        plt.close(fig)
        counts = {
            label: {
                metric: next(
                    row["completed_seeds"]
                    for row in across
                    if row["prior"] == label and row["metric"] == metric
                )
                for metric in METRICS
            }
            for label in labels
        }
        captions.append(
            "prior_robustness: Each dot is a complete-run mean across the same "
            f"{len(expected)} non-control test perturbations. Planned sampler seeds: {seeds}. "
            f"Complete seed counts by prior and metric: {counts}. Horizontal marks show "
            "across-seed means only when every planned seed and its full-prior reference "
            "have complete finite metrics. Missing runs are not treated as zero, no seeds "
            "are silently dropped, and no confidence intervals are fabricated. The exported "
            "paired tables use altered minus full-prior values at the same sampler seed; "
            "negative MSE differences indicate improvement."
        )
    else:
        captions.append("Prior robustness: no complete finite measurements; no figure generated.")
    return captions, files, across


def save_csv(path, rows):
    # Default floating-point serialization preserves numerical precision; do not round.
    empty_columns = {
        "job_status.csv": [
            "id",
            "section",
            "parameter",
            "value",
            "seed",
            "status",
            "report_status",
            "issues",
        ],
        "per_condition.csv": [
            "id",
            "section",
            "parameter",
            "parameter_value",
            "seed",
            "kind",
            "perturbation",
            "metric",
            "value",
            "finite",
            "expected_condition",
        ],
        "aggregate_summary.csv": [
            "id",
            "section",
            "parameter",
            "parameter_value",
            "seed",
            "kind",
            "metric",
            "expected_count",
            "observed_count",
            "finite_count",
            "complete",
            "mean",
        ],
        "prior_paired_per_condition.csv": [
            "id",
            "reference_id",
            "seed",
            "prior",
            "perturbation",
            "metric",
            "full_value",
            "altered_value",
            "delta",
        ],
        "prior_paired_per_seed.csv": [
            "id",
            "reference_id",
            "seed",
            "prior",
            "metric",
            "complete",
            "expected_count",
            "paired_count",
            "mean_delta",
        ],
        "prior_across_seed_summary.csv": [
            "prior",
            "metric",
            "expected_seeds",
            "completed_seeds",
            "full_reference_seeds",
            "seed_ids",
            "complete",
            "mean",
        ],
    }
    pd.DataFrame(rows, columns=None if rows else empty_columns.get(path.name)).to_csv(
        path, index=False
    )


def build_report(plan_path, outdir, *, figures=True):
    plan_path, outdir = Path(plan_path).resolve(), Path(outdir).resolve()
    plan = json.loads(plan_path.read_text())
    root = resolve_path(plan["output_root"], plan_path.parent)
    reference_value = plan["reference"]
    if isinstance(reference_value, dict):
        reference_value = reference_value.get("real") or reference_value.get("path")
    reference = resolve_path(reference_value, plan_path.parent)
    expected = expected_conditions(reference)
    reference_digest = sha256(reference)
    jobs = plan["jobs"]
    if len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job ids in plan")
    if outdir.exists():
        raise FileExistsError(f"Use a fresh report directory; refusing to overwrite {outdir}")
    records, long, aggregates, tables = [], [], [], {}
    for job in jobs:
        try:
            record, rows, stats, raw = read_run(
                root, job, expected, reference_hash=reference_digest
            )
        except (ValueError, OSError) as error:
            record, rows, stats, raw = (
                dict(
                    id=job["id"],
                    status="unreadable",
                    report_status="incomplete_or_invalid",
                    issues=[str(error)],
                ),
                [],
                [],
                {},
            )
        records.append(record)
        long.extend(rows)
        aggregates.extend(stats)
        tables[job["id"]] = raw
    paired, paired_summary = paired_prior_rows(jobs, tables, expected)
    outdir.mkdir(parents=True)
    for filename, rows in (
        ("job_status.csv", records),
        ("per_condition.csv", long),
        ("aggregate_summary.csv", aggregates),
        ("prior_paired_per_condition.csv", paired),
        ("prior_paired_per_seed.csv", paired_summary),
    ):
        save_csv(outdir / filename, rows)
    captions, files, across = (
        make_figures(outdir, jobs, aggregates, expected) if figures else ([], [], [])
    )
    save_csv(outdir / "prior_across_seed_summary.csv", across)
    complete = bool(jobs) and all(record["report_status"] == "validated" for record in records)
    manifest = dict(
        status="complete" if complete else "partial",
        plan=str(plan_path),
        plan_sha256=sha256(plan_path),
        reference=str(reference),
        reference_sha256=reference_digest,
        expected_conditions=sorted(expected),
        jobs=records,
        figures=files,
        value_policy=(
            "Raw values only; no scaling, shifting, clipping, imputation, or NaN-skipping means."
        ),
    )
    (outdir / "report.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (outdir / "captions.txt").write_text("\n\n".join(captions) + "\n")
    counts = Counter(record["status"] for record in records)
    markdown = [
        "# AdaCell appendix experiment report",
        "",
        "Status: **"
        f"{'complete' if complete else 'partial — not a complete experimental result'}**.",
        "",
        f"Expected non-control perturbations: {len(expected)}. Job states: {dict(counts)}.",
        "",
        "Only jobs explicitly marked complete are read. Each mean requires the exact "
        "reference condition set, no duplicates, and finite values for every condition. "
        "Missing values remain missing. Raw per-condition metrics and descriptive "
        "population diagnostics are exported separately; diagnostic ratios are not proof "
        "of preserved biological diversity.",
        "",
        "The sensitivity figures use actual measurements and include a matching "
        "main-reference configuration where available. Prior comparisons are seed-matched; "
        "across-seed means require all planned seeds and corresponding full-prior results. "
        "No confidence intervals or biological conclusions are inferred from missing measurements.",
        "",
        "## Run coverage",
        "",
        "| Job | Queue state | Report validation | Issues |",
        "|---|---|---|---|",
    ]
    markdown.extend(
        f"| {record['id']} | {record['status']} | {record['report_status']} | "
        f"{'; '.join(record['issues']) or 'none'} |"
        for record in records
    )
    markdown += [
        "",
        "## Figure captions",
        "",
        *captions,
        "",
        "Figures and tables use the measured outputs from this plan, not paper-table "
        "target means. Both improvements and degradations are retained. No baseline "
        "is selected using test-set performance.",
    ]
    (outdir / "report.md").write_text("\n".join(markdown) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Fresh output directory; existing paths are never overwritten",
    )
    args = parser.parse_args()
    manifest = build_report(args.plan, args.outdir)
    print(f"Report {manifest['status']}: {args.outdir / 'report.md'}")


if __name__ == "__main__":
    main()
