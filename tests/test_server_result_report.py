import json

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from celldiffa.benchmark.artifacts import sha256_file
from scripts.server import report_replogle_results as report


def make_reference(root):
    reference = root / "reference/real.h5ad"
    reference.parent.mkdir(parents=True)
    ad.AnnData(
        np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"gene": ["non-targeting", "A", "A", "B"], "cell_line": ["cell"] * 4},
            index=["c", "a0", "a1", "b"],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    ).write_h5ad(reference)
    return reference


def make_run(root, name, reference, *, with_metrics=True):
    output = root / name
    output.mkdir(parents=True)
    (output / "run_config.json").write_text(
        json.dumps(dict(evaluation_split="test", alpha=1, variant="scratch"))
    )
    prediction = output / "celldiffa_scratch.h5ad"
    prediction.write_bytes(reference.read_bytes())
    metrics = output / "metrics"
    if with_metrics:
        metrics.mkdir()
        pd.DataFrame(
            {"perturbation": ["A", "B"], **{metric: [0.2, 0.4] for metric in report.METRICS}}
        ).to_csv(metrics / report.PER_FILE, index=False)
    return output, prediction, metrics


def inspect(root, reference, output, **kwargs):
    runs, _ = report.discover(root)
    run = next(run for run in runs if run.output == output)
    return report.inspect_run(run, root, reference, report.reference_metadata(reference), **kwargs)


def test_discover_main_ablation_and_new_plan_without_validation_or_smoke(tmp_path):
    reference = make_reference(tmp_path)
    main, _, _ = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    ablation, _, _ = make_run(tmp_path, "test_ablations/scratch_without_anchor", reference)
    additional, _, _ = make_run(tmp_path, "maintext_train_v1/runs/scratch_best16", reference)
    (additional.parent.parent / "plan.json").write_text(
        json.dumps(
            dict(
                output_root=str(additional.parent.parent),
                lanes=[
                    [dict(id="scratch_best16", kind="perturbdiff")],
                    [dict(id="squidiff_training", kind="train_squidiff")],
                ],
            )
        )
    )
    make_run(tmp_path, "validation/adacell_alpha1", reference)
    make_run(tmp_path, "server/smoke/scratch", reference)
    outside_validation, _, _ = make_run(tmp_path, "hidden_validation", reference)
    (outside_validation / "run_config.json").write_text(
        json.dumps(dict(evaluation_split="validation"))
    )
    make_run(tmp_path, "reports/old", reference)
    runs, warnings = report.discover(tmp_path)
    assert not warnings
    mapping = {run.output: run for run in runs}
    assert mapping[main].category == "main"
    assert mapping[ablation].category == "ablation"
    assert mapping[additional].category == "additional"
    assert mapping[additional.parent / "squidiff_training"].kind == "train_squidiff"
    assert len([run for run in runs if run.category == "main"]) == 6
    assert outside_validation not in mapping
    assert not any("hidden_validation" in str(run.output) for run in runs)
    assert not any(report.excluded(run.output.relative_to(tmp_path).parts) for run in runs)


def test_complete_coverage_and_saved_hashes(tmp_path):
    reference = make_reference(tmp_path)
    output, prediction, metrics = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    marker = dict(
        reference_sha256=sha256_file(reference),
        prediction_sha256=sha256_file(prediction),
        metrics_sha256=sha256_file(metrics / report.PER_FILE),
    )
    (output / "evaluated.json").write_text(json.dumps(marker))
    row = inspect(tmp_path, reference, output, verify_hashes=True)
    assert row["status"] == "EVALUATED" and row["hash_check"] == "verified"
    assert row["DEOver"] == pytest.approx(0.3) and row["finite_MSE"] == 2
    table = pd.read_csv(metrics / report.PER_FILE)
    table.MSE = 0.9
    table.to_csv(metrics / report.PER_FILE, index=False)
    row = inspect(tmp_path, reference, output, verify_hashes=True)
    assert row["status"] == "CHECK_FAILED" and row["DEOver"] is None


def test_no_hash_record_not_claimed_verified(tmp_path):
    reference = make_reference(tmp_path)
    output, _, _ = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    row = inspect(tmp_path, reference, output, verify_hashes=True)
    assert row["status"] == "EVALUATED"
    assert row["hash_check"] == "no_record" and "identity not proven" in row["note"]


@pytest.mark.parametrize("change", ["drop", "duplicate", "extra", "missing_metric"])
def test_partial_and_wrong_metrics_are_not_results(tmp_path, change):
    reference = make_reference(tmp_path)
    output, _, metrics = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    table = pd.read_csv(metrics / report.PER_FILE)
    if change == "drop":
        table = table.iloc[:1]
    elif change == "duplicate":
        table.loc[1, "perturbation"] = "A"
    elif change == "extra":
        table.loc[1, "perturbation"] = "C"
    else:
        table = table.drop(columns="MSE")
    table.to_csv(metrics / report.PER_FILE, index=False)
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "CHECK_FAILED" and row["MSE"] is None


def test_undefined_metric_is_not_nanmean(tmp_path):
    reference = make_reference(tmp_path)
    output, _, metrics = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    table = pd.read_csv(metrics / report.PER_FILE)
    table.loc[0, "PDCorr"] = np.nan
    table.to_csv(metrics / report.PER_FILE, index=False)
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "EVALUATED_UNDEFINED"
    assert row["PDCorr"] is None and row["finite_PDCorr"] == 1
    assert row["MSE"] == pytest.approx(0.3)


def test_stale_summary_caught(tmp_path):
    reference = make_reference(tmp_path)
    output, _, metrics = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    table = pd.read_csv(metrics / report.PER_FILE).drop(columns="perturbation")
    summary = table.agg(["mean", "count"])
    summary.loc["mean", "MSE"] = 9
    summary.to_csv(metrics / report.SUMMARY_FILE)
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "CHECK_FAILED" and "stale" in row["note"]


def test_partial_prediction_and_swapped_genes_fail(tmp_path):
    reference = make_reference(tmp_path)
    output, prediction, _ = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    data = ad.read_h5ad(prediction)
    data[:-1].copy().write_h5ad(prediction)
    assert inspect(tmp_path, reference, output)["status"] == "CHECK_FAILED"
    data[:, ::-1].copy().write_h5ad(prediction)
    assert inspect(tmp_path, reference, output)["status"] == "CHECK_FAILED"


def test_no_prediction_or_only_summary_is_not_complete(tmp_path):
    reference = make_reference(tmp_path)
    output, prediction, metrics = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    prediction.unlink()
    assert inspect(tmp_path, reference, output)["status"] == "METRICS_ONLY"
    prediction.write_bytes(reference.read_bytes())
    (metrics / report.PER_FILE).rename(metrics / report.SUMMARY_FILE)
    assert inspect(tmp_path, reference, output)["status"] == "SUMMARY_ONLY"


def test_ambiguous_metrics_are_not_chosen(tmp_path):
    reference = make_reference(tmp_path)
    first, _, _ = make_run(tmp_path, "first/scratch_alpha1", reference, with_metrics=False)
    second, _, _ = make_run(tmp_path, "second/scratch_alpha1", reference, with_metrics=False)
    metrics = tmp_path / "metrics/scratch_alpha1"
    metrics.mkdir(parents=True)
    (metrics / report.PER_FILE).write_text("ambiguous")
    runs, warnings = report.discover(tmp_path)
    assert any("Ambiguous" in item for item in warnings)
    assert all(not run.metric_dirs for run in runs if run.output in {first, second})
    assert any(run.kind == "metrics_only" for run in runs)


def test_report_writes_commands_not_predictions_or_metrics(tmp_path):
    reference = make_reference(tmp_path)
    output, prediction, metrics = make_run(
        tmp_path, "test_sensitivity/scratch_alpha1", reference, with_metrics=False
    )
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "NEEDS_EVALUATION"
    before = prediction.read_bytes()
    outdir = tmp_path / "reports/new"
    report.write_report([row], [], tmp_path, reference, outdir)
    assert not metrics.exists() and prediction.read_bytes() == before
    assert "evaluate.py" in (outdir / "evaluate_missing.sh").read_text()
    assert "train_squidiff" not in (outdir / "evaluate_missing.sh").read_text()
    assert (outdir / "main.csv").exists() and (outdir / "all_results.csv").exists()
    assert json.loads((outdir / "inventory.json").read_text())["rows"][0]["MSE"] is None
    with pytest.raises(FileExistsError):
        report.write_report([row], [], tmp_path, reference, outdir)


def test_training_and_waiting_queue_status(tmp_path):
    reference = make_reference(tmp_path)
    output = tmp_path / "maintext_train_v1/runs/squidiff_training"
    output.mkdir(parents=True)
    (output / "best.pt").touch()
    (output / "training_progress.json").write_text(
        json.dumps(dict(status="training_complete", completed_steps=100000, total_steps=100000))
    )
    row = report.inspect_run(
        report.Run(output, "additional", kind="train_squidiff"),
        tmp_path,
        reference,
        report.reference_metadata(reference),
    )
    assert row["status"] == "TRAINING_COMPLETE" and row["MSE"] is None
    guided = output.parent / "squidiff_adacell16"
    guided.mkdir()
    (guided / "job_status.json").write_text(json.dumps(dict(status="waiting_for_training")))
    row = report.inspect_run(
        report.Run(guided, "additional"), tmp_path, reference, report.reference_metadata(reference)
    )
    assert row["status"] == "WAITING_FOR_TRAINING"


def test_prefixed_manual_metrics_attach_to_unique_main_run(tmp_path):
    reference = make_reference(tmp_path)
    output, _, metrics = make_run(tmp_path, "test_sensitivity/scratch_alpha1", reference)
    destination = tmp_path / "metrics/adacell_scratch_alpha1"
    destination.parent.mkdir()
    metrics.rename(destination)
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "EVALUATED"
    assert row["metrics_dir"] == str(destination)


def test_scouter_alias_uses_completed_vectorized_prediction(tmp_path):
    reference = make_reference(tmp_path)
    output, prediction, metrics = make_run(tmp_path, "scouter_vectorized", reference)
    prediction.rename(output / "predictions.h5ad")
    old = tmp_path / "scouter"
    old.mkdir()
    (old / "run_config.json").write_text("{}")
    destination = tmp_path / "metrics/scouter"
    destination.parent.mkdir()
    metrics.rename(destination)
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "EVALUATED" and row["metrics_dir"] == str(destination)
    assert inspect(tmp_path, reference, old)["status"] == "INCOMPLETE"


def test_old_finetuned_is_not_promoted_when_corrected_results_exist(tmp_path):
    reference = make_reference(tmp_path)
    old, _, _ = make_run(tmp_path, "perturbdiff_finetuned", reference)
    corrected, prediction, _ = make_run(tmp_path, "perturbdiff_finetuned_fixed_ids", reference)
    prediction.rename(corrected / "predictions.h5ad")
    assert inspect(tmp_path, reference, old)["status"] == "SUPERSEDED"
    assert inspect(tmp_path, reference, corrected)["status"] == "EVALUATED"


def test_failed_evaluation_is_flagged_without_silent_retry(tmp_path):
    reference = make_reference(tmp_path)
    output, _, _ = make_run(
        tmp_path, "test_sensitivity/scratch_alpha1", reference, with_metrics=False
    )
    (output / "job_status.json").write_text(
        json.dumps(dict(status="failed", error="evaluation error"))
    )
    row = inspect(tmp_path, reference, output)
    assert row["status"] == "FAILED" and "evaluation error" in row["note"]
    outdir = tmp_path / "reports/failed"
    report.write_report([row], [], tmp_path, reference, outdir)
    assert "evaluate.py" not in (outdir / "evaluate_missing.sh").read_text()
