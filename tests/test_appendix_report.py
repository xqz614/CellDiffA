"""Synthetic fixture tests for reporting contracts; not biological results."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scripts.baselines import report_adacell_appendix as report


def metric_table():
    return pd.DataFrame(
        {
            "perturbation": ["A", "B"],
            **{metric: [0.123456789123, 0.423456789123] for metric in report.METRICS},
        }
    )


def test_means_require_all_conditions_and_finite_values():
    source = metric_table()
    _, rows = report.summarize_table(source, {"A", "B"}, report.METRICS)
    assert all(row["complete"] for row in rows)
    assert rows[0]["mean"] == pytest.approx(0.273456789123)
    source.loc[1, "MSE"] = np.nan
    _, rows = report.summarize_table(source, {"A", "B"}, report.METRICS)
    mse = next(row for row in rows if row["metric"] == "MSE")
    assert mse["mean"] is None
    assert mse["finite_count"] == 1
    assert not mse["complete"]
    assert rows[0]["mean"] is not None
    _, partial = report.summarize_table(source.iloc[:1], {"A", "B"}, report.METRICS)
    assert all(row["mean"] is None for row in partial)
    assert partial[0]["missing_conditions"] == 1


@pytest.mark.parametrize("labels", [["A", "A"], ["A", "C"]])
def test_duplicate_or_changed_condition_sets_are_not_aggregated(labels):
    source = metric_table()
    source["perturbation"] = labels
    _, rows = report.summarize_table(source, {"A", "B"}, report.METRICS)
    assert all(not row["complete"] and row["mean"] is None for row in rows)


def test_non_targeting_is_excluded_and_missing_metric_remains_missing():
    source = pd.concat(
        [metric_table(), pd.DataFrame({"perturbation": ["non-targeting"], "MSE": [9999]})],
        ignore_index=True,
    )
    source = source.drop(columns="PDS_cos")
    long, rows = report.summarize_table(source, {"A", "B"}, report.METRICS)
    assert not any(row["perturbation"] == "non-targeting" for row in long)
    assert next(row for row in rows if row["metric"] == "MSE")["mean"] < 1
    missing = next(row for row in rows if row["metric"] == "PDS_cos")
    assert missing["mean"] is None and missing["missing_metric"]


def make_job(
    job_id, *, seed=42, mode="full", fraction=1.0, section="prior", parameter="prior", value="full"
):
    return dict(
        id=job_id,
        section=section,
        parameter=parameter,
        value=value,
        settings=dict(
            alpha=1.0,
            num_particles=16,
            top_de=20,
            ess_threshold=0.5,
            anchor_bandwidth=1.0,
            seed=seed,
            prior_mode=mode,
            prior_fraction=fraction,
            prior_seed=seed,
        ),
    )


def make_plan(tmp_path, jobs):
    reference = tmp_path / "real.h5ad"
    ad.AnnData(
        np.ones((3, 2)),
        obs=pd.DataFrame({"gene": ["non-targeting", "A", "B"]}, index=["c", "a", "b"]),
    ).write_h5ad(reference)
    root = tmp_path / "experiment"
    root.mkdir()
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(dict(output_root=str(root), reference=str(reference), jobs=jobs))
    )
    return plan_path, root


def complete_job(root, job, *, state="complete", values=None):
    directory = root / "jobs" / job["id"]
    (directory / "metrics").mkdir(parents=True)
    (directory / "diagnostics").mkdir()
    metrics = directory / "metrics" / report.PER_FILE
    (metric_table() if values is None else values).to_csv(metrics, index=False)
    diagnostics = directory / "diagnostics" / report.DIAGNOSTIC_FILE
    pd.DataFrame(
        {"perturbation": ["A", "B"], "cells": [1, 1], "predicted_to_real_variance": [1.0, 1.2]}
    ).to_csv(diagnostics, index=False)
    (directory / "state.json").write_text(
        json.dumps(
            dict(
                status=state,
                metrics_sha256=report.sha256(metrics),
                diagnostics_sha256=report.sha256(diagnostics),
            )
        )
    )
    return directory


def test_only_complete_jobs_read_hash_mismatches_rejected(tmp_path):
    full = make_job("full", parameter="reference", section="sensitivity")
    failed = make_job("failed", mode="subsample", fraction=0.5)
    pending = make_job("pending", mode="shuffle")
    plan_path, root = make_plan(tmp_path, [full, failed, pending])
    directory = complete_job(root, full)
    complete_job(root, failed, state="failed")
    manifest = report.build_report(plan_path, tmp_path / "report", figures=False)
    assert manifest["status"] == "partial"
    long = pd.read_csv(tmp_path / "report" / "per_condition.csv")
    assert set(long.id) == {"full"}
    source = directory / "metrics" / report.PER_FILE
    source.write_text(source.read_text() + "\n")
    record, long, _, _ = report.read_run(root, full, {"A", "B"})
    assert "SHA-256 mismatch" in record["issues"][0]
    assert not any(row["kind"] == "metrics" for row in long)
    with pytest.raises(FileExistsError, match="fresh"):
        report.build_report(plan_path, tmp_path / "report", figures=False)


def test_reference_hash_mismatch_excludes_all_job_measurements(tmp_path):
    job = make_job("full", parameter="reference")
    _, root = make_plan(tmp_path, [job])
    directory = complete_job(root, job)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text())
    state["reference_sha256"] = "wrong_reference"
    state_path.write_text(json.dumps(state))
    record, long, stats, tables = report.read_run(
        root, job, {"A", "B"}, reference_hash="expected_reference"
    )
    assert not long and not stats and not tables
    assert record["report_status"] == "incomplete_or_invalid"


def test_all_pending_report_exports_readable_empty_tables(tmp_path):
    job = make_job("full", section="sensitivity", parameter="reference")
    plan, _ = make_plan(tmp_path, [job])
    destination = tmp_path / "report"
    result = report.build_report(plan, destination, figures=False)
    assert result["status"] == "partial"
    assert pd.read_csv(destination / "aggregate_summary.csv").empty
    assert "mean" in pd.read_csv(destination / "aggregate_summary.csv")
    assert pd.read_csv(destination / "per_condition.csv").empty


def test_prior_differences_are_paired_not_shifted_or_cross_seed():
    full = make_job("full42", parameter="reference", section="sensitivity")
    half = make_job("half42", mode="subsample", fraction=0.5)
    wrong_seed = make_job("half43", seed=43, mode="subsample", fraction=0.5)
    reduced = metric_table()
    reduced["DEOver"] -= 0.04
    reduced["MSE"] += 0.02
    tables = {
        "full42": {"metrics": metric_table()},
        "half42": {"metrics": reduced},
        "half43": {"metrics": reduced},
    }
    raw, aggregates = report.paired_prior_rows([full, half, wrong_seed], tables, {"A", "B"})
    assert {row["id"] for row in raw} == {"half42"}
    assert next(row for row in aggregates if row["metric"] == "DEOver")[
        "mean_delta"
    ] == pytest.approx(-0.04)
    assert next(row for row in aggregates if row["metric"] == "MSE")["mean_delta"] == pytest.approx(
        0.02
    )
    assert raw[0]["full_value"] == pytest.approx(0.123456789123)


def test_figures_require_complete_across_seed_coverage(tmp_path):
    full42 = make_job("reference42", section="sensitivity", parameter="reference")
    full43 = make_job("full43", seed=43)
    half42 = make_job("half42", mode="subsample", fraction=0.5)
    half43 = make_job("half43", seed=43, mode="subsample", fraction=0.5)
    temperature = make_job("alpha05", section="sensitivity", parameter="alpha", value=0.5)
    temperature["settings"]["alpha"] = 0.5
    plan_path, root = make_plan(tmp_path, [full42, full43, half42, half43, temperature])
    for job in [full42, full43, half42, temperature]:
        complete_job(root, job)
    destination = tmp_path / "report"
    manifest = report.build_report(plan_path, destination)
    across = pd.read_csv(destination / "prior_across_seed_summary.csv")
    assert across.loc[across.prior == "50% training cells", "mean"].isna().all()
    assert across.loc[across.prior == "Full", "mean"].notna().all()
    assert (destination / "sensitivity_alpha.pdf").is_file()
    assert (destination / "prior_robustness.png").is_file()
    assert manifest["status"] == "partial"
    assert "2 planned or matched-reference points" in (destination / "captions.txt").read_text()


def test_prior_pairing_rejects_changed_other_hyperparameter():
    full = make_job("full", parameter="reference")
    changed = make_job("changed", mode="subsample", fraction=0.5)
    changed["settings"]["num_particles"] = 8
    rows, stats = report.paired_prior_rows(
        [full, changed],
        {job["id"]: {"metrics": metric_table()} for job in [full, changed]},
        {"A", "B"},
    )
    assert not rows and not stats
