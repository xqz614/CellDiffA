"""The exact metric surface reported by PerturbDiff."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import r2_score

from .artifacts import sha256_file, write_manifest
from .contracts import validate_prediction_pair

CELL_EVAL_VERSION = "0.6.6"

# Paper label -> Cell-Eval 0.6.6 output column.
PAPER_METRIC_NAMES = {
    "DEOver": "overlap_at_N",
    "DEPrec": "precision_at_N",
    "ES": "de_spearman_sig",
    "DirAgr": "de_direction_match",
    "LFCSpear": "de_spearman_lfc_sig",
    "AUPRC": "pr_auc",
    "AUROC": "roc_auc",
    "PDCorr": "pearson_delta",
    "MSE": "mse",
    "MAE": "mae",
    "PDS_L1": "discrimination_score_l1",
    "PDS_L2": "discrimination_score_l2",
    "PDS_cos": "discrimination_score_cosine",
}


def _mean(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.mean(axis=0)).ravel()
    return np.asarray(matrix).mean(axis=0)


def cellflow_r2(real, pred) -> float:
    """CellFlow's exact R²: sklearn R² between population means."""
    return float(r2_score(_mean(real), _mean(pred)))


def _require_cell_eval_066() -> None:
    try:
        installed = metadata.version("cell-eval")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "cell-eval is not installed. Install the benchmark environment with "
            "`conda env create -f environments/benchmark.yaml`."
        ) from exc
    if installed != CELL_EVAL_VERSION:
        raise RuntimeError(
            f"PerturbDiff used cell-eval=={CELL_EVAL_VERSION}, but {installed} is installed."
        )


def expression_scale_summary(adata, *, block_rows: int = 1024) -> dict:
    """Audit every entry without changing it or densifying a full sparse matrix."""
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    total = int(adata.n_obs * adata.n_vars)
    if not total:
        raise ValueError("Cannot evaluate an empty expression matrix")
    negative = nonfinite = high = 0
    minimum, maximum, summed = float("inf"), -float("inf"), 0.0
    for start in range(0, adata.n_obs, block_rows):
        block = adata.X[start : start + block_rows]
        values = block.data if sparse.issparse(block) else np.asarray(block).reshape(-1)
        finite = np.isfinite(values)
        nonfinite += int((~finite).sum())
        values = values[finite]
        if values.size:
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            summed += float(values.sum(dtype=np.float64))
            negative += int((values < 0).sum())
            high += int((values >= 15).sum())
        if sparse.issparse(block) and block.nnz < block.shape[0] * block.shape[1]:
            minimum, maximum = min(minimum, 0.0), max(maximum, 0.0)
    return dict(
        shape=[adata.n_obs, adata.n_vars],
        entries=total,
        nonfinite=nonfinite,
        negative=negative,
        minimum=minimum if np.isfinite(minimum) else None,
        maximum=maximum if np.isfinite(maximum) else None,
        mean=summed / (total - nonfinite) if total > nonfinite else None,
        entries_ge_15=high,
        fraction_ge_15=high / total,
    )


def explicit_log1p_options(real, pred) -> tuple[dict, dict]:
    """Declare known log1p units, not a transform or an automatic scale repair.

    In Cell-Eval 0.6.6 allow_discrete=True disables the upper-bound heuristic
    AND defaults DE to is_log1p=False. Override the latter explicitly. The same
    option also prevents integer-valued log1p inputs from being renormalized.
    """
    summaries = {"real": expression_scale_summary(real), "pred": expression_scale_summary(pred)}
    for name, summary in summaries.items():
        if summary["nonfinite"] or summary["negative"]:
            raise ValueError(
                f"Explicit log1p input must be finite and nonnegative: {name} {summary}"
            )
    return {"allow_discrete": True, "pdex_kwargs": {"is_log1p": True}}, summaries


def evaluate_perturbdiff_protocol(
    *,
    real_path: str | Path,
    pred_path: str | Path,
    outdir: str | Path,
    pert_col: str,
    control_pert: str,
    num_threads: int = 16,
    break_on_error: bool = True,
    input_scale: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Cell-Eval 0.6.6 full profile plus CellFlow R².

    Returns per-perturbation and summary tables using the labels from the
    PerturbDiff paper. Cell-Eval's original outputs are retained in ``outdir``.
    """
    if input_scale not in {"auto", "log1p"}:
        raise ValueError("input_scale must be auto or log1p")
    _require_cell_eval_066()
    from cell_eval import MetricsEvaluator

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    real = ad.read_h5ad(real_path)
    pred = ad.read_h5ad(pred_path)
    validate_prediction_pair(
        real,
        pred,
        pert_col=pert_col,
        control_pert=control_pert,
    )

    options, audit = {}, None
    if input_scale == "log1p":
        options, summaries = explicit_log1p_options(real, pred)
        audit = dict(
            status="started",
            cell_eval_version=CELL_EVAL_VERSION,
            input_scale="log1p",
            validation_override="Skip automatic max>=15/counts heuristic; keep raw input values",
            pdex_is_log1p=True,
            input_values_changed=False,
            real_sha256=sha256_file(real_path),
            prediction_sha256=sha256_file(pred_path),
            summaries=summaries,
        )
        write_manifest(outdir / "input_scale_audit.json", audit)

    evaluator = MetricsEvaluator(
        adata_pred=pred,
        adata_real=real,
        control_pert=control_pert,
        pert_col=pert_col,
        num_threads=num_threads,
        outdir=str(outdir / "cell_eval_0.6.6"),
        **options,
    )
    results, _ = evaluator.compute(
        profile="full",
        basename="results.csv",
        write_csv=True,
        break_on_error=break_on_error,
    )
    frame = results.to_pandas()
    if "perturbation" not in frame:
        raise RuntimeError("Cell-Eval output is missing the perturbation column.")

    labels_real = real.obs[pert_col].astype(str).to_numpy()
    labels_pred = pred.obs[pert_col].astype(str).to_numpy()
    r2_values = {}
    for pert in frame["perturbation"].astype(str):
        r2_values[pert] = cellflow_r2(
            real.X[labels_real == pert],
            pred.X[labels_pred == pert],
        )

    missing = [column for column in PAPER_METRIC_NAMES.values() if column not in frame]
    if missing:
        raise RuntimeError(f"Cell-Eval 0.6.6 did not produce required metrics: {missing}.")

    paper = pd.DataFrame(
        {
            "perturbation": frame["perturbation"],
            "R2": frame["perturbation"].map(r2_values),
        }
    )
    for paper_name, cell_eval_name in PAPER_METRIC_NAMES.items():
        paper[paper_name] = frame[cell_eval_name]
    paper.to_csv(outdir / "perturbdiff_metrics_per_perturbation.csv", index=False)

    numeric = paper.drop(columns="perturbation")
    summary = numeric.agg(["count", "mean", "std", "min", "median", "max"])
    summary.index.name = "statistic"
    summary.to_csv(outdir / "perturbdiff_metrics_summary.csv")
    if audit is not None:
        audit["status"] = "complete"
        audit["perturbations"] = len(paper)
        audit["metrics_sha256"] = sha256_file(outdir / "perturbdiff_metrics_per_perturbation.csv")
        write_manifest(outdir / "input_scale_audit.json", audit)
    return paper, summary
