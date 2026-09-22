import importlib.util
import json
import sys
import types

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from celldiffa.benchmark import metrics
from scripts.server.recover_replogle_ablation_metrics import check_tail


def data(values):
    return ad.AnnData(np.asarray(values, dtype=np.float32))


@pytest.mark.parametrize("matrix", [np.array, sparse.csr_matrix, sparse.csc_matrix])
def test_scale_summary_counts_all_values_and_implicit_zeros(matrix):
    values = np.array([[0.0, 0.5], [15.2377, 0.0]], dtype=np.float32)
    x = ad.AnnData(matrix(values))
    before = x.X.copy()
    summary = metrics.expression_scale_summary(x, block_rows=1)
    assert summary["minimum"] == 0
    assert summary["maximum"] == pytest.approx(15.2377)
    assert summary["entries_ge_15"] == 1
    assert summary["fraction_ge_15"] == 0.25
    assert summary["mean"] == pytest.approx(values.mean())
    np.testing.assert_array_equal(
        x.X.toarray() if sparse.issparse(x.X) else x.X,
        before.toarray() if sparse.issparse(before) else before,
    )


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf, -np.inf])
def test_explicit_scale_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        metrics.explicit_log1p_options(data([[0.1, 0.2]]), data([[0.1, bad]]))


def test_explicit_scale_does_not_change_log1p_de_or_raw_values():
    real, pred = data([[0.2, 0.3]]), data([[15.2377, 0.3]])
    old = pred.X.copy()
    kwargs, audit = metrics.explicit_log1p_options(real, pred)
    assert kwargs == {"allow_discrete": True, "pdex_kwargs": {"is_log1p": True}}
    assert audit["pred"]["entries_ge_15"] == 1
    np.testing.assert_array_equal(pred.X, old)


def test_recovery_guard_rejects_systemic_anomaly(tmp_path):
    path = tmp_path / "squidiff.h5ad"
    data([[16.8, 40], [0.0, 200.0]]).write_h5ad(path)
    with pytest.raises(ValueError, match="rare-tail"):
        check_tail(path)


def test_recovery_guard_accepts_single_tail_without_clipping(tmp_path):
    values = np.zeros((1000, 1000), dtype=np.float32)
    values[0, 0] = 15.2377
    path = tmp_path / "tail.h5ad"
    data(values).write_h5ad(path)
    summary = check_tail(path)
    assert summary["entries_ge_15"] == 1
    assert summary["maximum"] > 15


@pytest.mark.parametrize("scale", ["auto", "log1p"])
def test_metrics_options_and_audit_preserve_source_files(tmp_path, monkeypatch, scale):
    real = data([[0.2, 0.3], [0.7, 0.6], [0.5, 0.8]])
    real.obs["gene"] = ["ctrl", "A", "A"]
    pred = real.copy()
    pred.X[1, 0] = 15.2377
    real_path, pred_path = tmp_path / "real.h5ad", tmp_path / "pred.h5ad"
    real.write_h5ad(real_path)
    pred.write_h5ad(pred_path)
    previous = (real_path.read_bytes(), pred_path.read_bytes())
    calls = []

    class Evaluator:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def compute(self, **kwargs):
            assert kwargs["profile"] == "full" and kwargs["break_on_error"]
            frame = pd.DataFrame(
                {"perturbation": ["A"], **{k: [0.5] for k in metrics.PAPER_METRIC_NAMES.values()}}
            )
            return types.SimpleNamespace(to_pandas=lambda: frame), None

    monkeypatch.setattr(metrics, "_require_cell_eval_066", lambda: None)
    monkeypatch.setitem(sys.modules, "cell_eval", types.SimpleNamespace(MetricsEvaluator=Evaluator))
    table, _ = metrics.evaluate_perturbdiff_protocol(
        real_path=real_path,
        pred_path=pred_path,
        outdir=tmp_path / "metrics",
        pert_col="gene",
        control_pert="ctrl",
        input_scale=scale,
    )
    assert list(table.perturbation) == ["A"]
    assert (real_path.read_bytes(), pred_path.read_bytes()) == previous
    if scale == "auto":
        assert "allow_discrete" not in calls[0] and "pdex_kwargs" not in calls[0]
        assert not (tmp_path / "metrics/input_scale_audit.json").exists()
    else:
        assert calls[0]["allow_discrete"] is True
        assert calls[0]["pdex_kwargs"] == {"is_log1p": True}
        audit = json.loads((tmp_path / "metrics/input_scale_audit.json").read_text())
        assert audit["status"] == "complete" and not audit["input_values_changed"]


@pytest.mark.parametrize("fail_first", [False, True])
def test_recovery_independent_jobs_resume_and_report_discovery(tmp_path, monkeypatch, fail_first):
    from scripts.server import recover_replogle_ablation_metrics as recovery
    from scripts.server.report_replogle_results import discover

    root = tmp_path / "results"
    (root / "reference").mkdir(parents=True)
    real = data([[0.0, 0.2], [0.7, 0.6], [0.4, 0.9]])
    real.obs["gene"] = ["non-targeting", "A", "A"]
    real.write_h5ad(root / "reference/real.h5ad")
    for name in recovery.RUNS:
        run = root / "ablations" / name
        run.mkdir(parents=True)
        real.write_h5ad(run / "celldiffa_scratch.h5ad")
        (run / "metrics").mkdir()
        (run / "metrics/old_error.log").write_text("keep this failure record")
    calls = []

    def evaluate(**kwargs):
        name = kwargs["pred_path"].parent.name
        calls.append(name)
        assert kwargs["input_scale"] == "log1p"
        if fail_first and name == recovery.RUNS[0]:
            raise RuntimeError("test first-job failure")
        output = kwargs["outdir"]
        output.mkdir()
        frame = pd.DataFrame(
            {"perturbation": ["A"], "R2": [0.5], **{k: [0.5] for k in metrics.PAPER_METRIC_NAMES}}
        )
        csv = output / "perturbdiff_metrics_per_perturbation.csv"
        frame.to_csv(csv, index=False)
        (output / "input_scale_audit.json").write_text(
            json.dumps(
                dict(
                    status="complete",
                    real_sha256=metrics.sha256_file(kwargs["real_path"]),
                    prediction_sha256=metrics.sha256_file(kwargs["pred_path"]),
                    metrics_sha256=metrics.sha256_file(csv),
                )
            )
        )
        summary = frame.drop(columns="perturbation").agg(["mean"])
        summary.to_csv(output / "perturbdiff_metrics_summary.csv")
        return frame, summary

    monkeypatch.setattr(recovery, "evaluate_perturbdiff_protocol", evaluate)
    monkeypatch.setattr(sys, "argv", ["recovery", "--results-root", str(root)])
    if fail_first:
        with pytest.raises(SystemExit, match="scratch_without_anchor"):
            recovery.main()
    else:
        recovery.main()
        recovery.main()  # hash-verified completed results are not recomputed
    assert calls == list(recovery.RUNS)
    assert (root / "ablations/scratch_without_direction/evaluated.json").is_file()
    runs, _ = discover(root)
    direction = next(r for r in runs if r.output.name == "scratch_without_direction")
    assert direction.metric_dirs == {direction.output / "evaluation"}
    for name in recovery.RUNS:
        assert (
            root / "ablations" / name / "metrics/old_error.log"
        ).read_text() == "keep this failure record"


@pytest.mark.skipif(
    importlib.util.find_spec("cell_eval") is None,
    reason="Cell-Eval is an optional benchmark dependency",
)
def test_real_cell_eval_keeps_values_and_log1p_de_semantics(monkeypatch, tmp_path):
    # Execute the pinned evaluator's actual input conversion and DE-kwargs path.
    # Stub only the expensive differential-expression worker; do not stub those paths.
    import cell_eval._evaluator as upstream

    metrics._require_cell_eval_066()
    observed = []
    actual_builder = upstream._build_pdex_kwargs

    def de_stub(**kwargs):
        observed.append(
            actual_builder(
                reference=kwargs["anndata_pair"].control_pert,
                groupby_key="gene",
                num_workers=1,
                batch_size=100,
                metric="wilcoxon",
                allow_discrete=kwargs["allow_discrete"],
                pdex_kwargs=kwargs["pdex_kwargs"],
            )
        )
        return None

    monkeypatch.setattr(upstream, "_build_de_comparison", de_stub)
    for pred_values in ([[0.0, 15.2377], [0.2, 0.7]], [[0.0, 16.0], [1.0, 2.0]]):
        real, pred = data([[0.0, 0.3], [0.2, 0.7]]), data(pred_values)
        real.obs["gene"] = pred.obs["gene"] = ["ctrl", "A"]
        before_real, before_pred = real.X.copy(), pred.X.copy()
        options, _ = metrics.explicit_log1p_options(real, pred)
        upstream.MetricsEvaluator(
            adata_real=real,
            adata_pred=pred,
            pert_col="gene",
            control_pert="ctrl",
            outdir=str(tmp_path / "upstream"),
            **options,
        )
        np.testing.assert_array_equal(real.X, before_real)
        np.testing.assert_array_equal(pred.X, before_pred)
        assert observed[-1]["is_log1p"] is True
