#!/usr/bin/env python
"""Offline, descriptive case studies from real/reference and generated populations.

No model fitting, cell-level pairing, pathway inference, or outcome adjustment is
performed. Cases are selected transparently after evaluation and are not evidence
of average performance; every eligible condition is retained in the output table.
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
from celldiffa.benchmark.metrics import expression_scale_summary


def mean_expression(matrix):
    if sparse.issparse(matrix):
        return np.asarray(matrix.astype(np.float64).mean(axis=0)).ravel()
    return np.asarray(matrix, dtype=np.float64).mean(axis=0)


def response_errors(reference, prediction, *, epsilon=1e-12):
    """Errors of population-mean contrasts, never errors of paired cells."""
    reference, prediction = np.asarray(reference), np.asarray(prediction)
    if reference.ndim != 1 or reference.shape != prediction.shape or not reference.size:
        raise ValueError("Response vectors must be nonempty, one-dimensional, and aligned")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Response vectors must be finite")
    rn, pn = float(np.linalg.norm(reference)), float(np.linalg.norm(prediction))
    angle = np.nan
    if rn > epsilon and pn > epsilon:
        cosine = float(np.dot(reference / rn, prediction / pn))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return {
        "magnitude_error": abs(pn - rn),
        "angle_error_degrees": angle,
        "response_mse": float(np.mean((reference - prediction) ** 2)),
        "reference_response_norm": rn,
        "predicted_response_norm": pn,
    }


def resolve_context_column(real, context_col):
    if context_col == "none":
        raise ValueError("Global pooling is not supported; supply matched context metadata")
    if context_col != "auto":
        if context_col not in real.obs:
            raise ValueError(f"Reference is missing context column {context_col!r}")
        return context_col
    available = [name for name in ("cell_line", "cell_type", "context") if name in real.obs]
    if not available:
        raise ValueError("Cannot infer matched controls: supply --context-col from reference obs")
    if len(available) > 1:
        raise ValueError(f"Multiple context columns {available}; set --context-col explicitly")
    return available[0]


def context_values(data, *, context_col, real_contexts):
    if context_col in data.obs:
        if data.obs[context_col].isna().any():
            raise ValueError(f"Missing context values in {context_col}")
        return data.obs[context_col].astype(str).to_numpy()
    # Standard benchmark predictions may retain only perturbation labels. This
    # inference is safe only when the complete reference contains ONE context.
    unique = np.unique(real_contexts)
    if len(unique) != 1:
        raise ValueError("Predictions lack multi-context metadata; refusing to guess cell contexts")
    return np.full(data.n_obs, unique[0], dtype=object)


def compute_case_responses(
    real, base, pred, *, pert_col="gene", control_pert="non-targeting", context_col="auto"
):
    context_col = resolve_context_column(real, context_col)
    if not real.var_names.is_unique or not real.n_vars:
        raise ValueError("Reference genes must be nonempty and unique")
    for data in (real, base, pred):
        if pert_col not in data.obs or data.obs[pert_col].isna().any():
            raise ValueError(f"Missing perturbation values in {pert_col}")
    for candidate in (base, pred):
        validate_prediction_pair(real, candidate, pert_col=pert_col, control_pert=control_pert)
    audits = {}
    for name, data in (("real", real), ("base", base), ("pred", pred)):
        audits[name] = expression_scale_summary(data)
        if audits[name]["nonfinite"] or audits[name]["negative"]:
            raise ValueError(f"Expression must be finite and nonnegative: {name}")
    real_contexts = context_values(
        real, context_col=context_col, real_contexts=np.array([], dtype=str)
    )
    labels, contexts = {}, {}
    for name, data in (("real", real), ("base", base), ("pred", pred)):
        labels[name] = data.obs[pert_col].astype(str).to_numpy()
        contexts[name] = context_values(data, context_col=context_col, real_contexts=real_contexts)

    def group_counts(name):
        return pd.Series(list(zip(labels[name], contexts[name]))).value_counts().to_dict()

    expected = group_counts("real")
    for name in ("base", "pred"):
        if group_counts(name) != expected:
            raise ValueError(f"Perturbation/context group coverage or counts differ: {name}")
    controls = {}
    for context in sorted(set(real_contexts)):
        mask = (labels["real"] == control_pert) & (real_contexts == context)
        if not mask.any():
            raise ValueError(f"No matched controls for context {context!r}")
        reference_controls = real.X[mask]
        controls[context] = mean_expression(reference_controls)
        reference_controls = (
            reference_controls.toarray()
            if sparse.issparse(reference_controls)
            else np.asarray(reference_controls)
        )
        # The shared contract checks controls globally. Recheck their assignment
        # to contexts to catch swapped context labels as well.
        for name, data in (("base", base), ("pred", pred)):
            cmask = (labels[name] == control_pert) & (contexts[name] == context)
            predicted_controls = data.X[cmask]
            predicted_controls = (
                predicted_controls.toarray()
                if sparse.issparse(predicted_controls)
                else np.asarray(predicted_controls)
            )
            if not np.array_equal(reference_controls, predicted_controls):
                raise ValueError(f"Controls differ within context {context!r}: {name}")
    rows, responses = [], {}
    for perturbation, context in sorted(expected):
        if perturbation == control_pert:
            continue
        if "::" in perturbation or "::" in context:
            raise ValueError("Context/perturbation labels cannot contain reserved separator '::'")
        case_id = f"{perturbation}::{context}"
        shifts = {}
        for name, data in (("real", real), ("base", base), ("pred", pred)):
            mask = (labels[name] == perturbation) & (contexts[name] == context)
            shifts[name] = mean_expression(data.X[mask]) - controls[context]
        row = {
            "case_id": case_id,
            "perturbation": perturbation,
            "context": context,
            "n_treated": expected[(perturbation, context)],
            "n_controls": expected[(control_pert, context)],
        }
        for name in ("base", "pred"):
            row.update(
                {
                    f"{name}_{key}": value
                    for key, value in response_errors(shifts["real"], shifts[name]).items()
                }
            )
        for metric in ("angle_error_degrees", "magnitude_error", "response_mse"):
            row[f"improvement_{metric}"] = row[f"base_{metric}"] - row[f"pred_{metric}"]
        rows.append(row)
        responses[case_id] = shifts
    if not rows:
        raise ValueError("No non-control perturbation/context groups")
    return pd.DataFrame(rows), responses, {"context_col": context_col, "expression": audits}


def select_cases(table, requested=None):
    """Select examples by an explicit deterministic rule, including a weak case."""
    if requested is not None:
        if not requested or len(requested) != len(set(requested)):
            raise ValueError("Requested case identifiers must be nonempty and unique")
        missing = sorted(set(requested) - set(table.case_id))
        if missing:
            raise ValueError(f"Unknown case identifiers: {missing}")
        return [{"case_id": name, "selection": "user_requested"} for name in requested], {
            "mode": "user_requested",
            "selection_metric": None,
        }
    metric = "improvement_angle_error_degrees"
    eligible = table[np.isfinite(table[metric])].copy()
    if eligible.empty:
        metric = "improvement_response_mse"
        eligible = table[np.isfinite(table[metric])].copy()
    ranked = eligible.sort_values([metric, "case_id"], ascending=[False, True])
    records = []

    def add(row, role):
        if row.case_id not in {item["case_id"] for item in records}:
            records.append(
                {"case_id": row.case_id, "selection": role, "selection_value": float(row[metric])}
            )

    add(
        ranked.iloc[0],
        "largest_improvement" if ranked.iloc[0][metric] > 0 else "least_deterioration",
    )
    positives = ranked[ranked[metric] > 0]
    if len(positives) > 1:
        add(positives.iloc[len(positives) // 2], "median_positive_improvement")
    add(ranked.iloc[-1], "failure_or_least_improvement")
    if len(records) < min(3, len(ranked)):
        add(ranked.iloc[len(ranked) // 2], "middle_ranked_condition")
    return records, {
        "mode": "deterministic_posthoc_examples",
        "selection_metric": metric,
        "eligible_conditions": len(ranked),
        "excluded_undefined_conditions": len(table) - len(ranked),
        "positive_is_improvement": True,
        "tie_break": "lexicographic case_id, stable sorted order",
        "not_representative_of_average_performance": True,
    }


def load_prior_cache(path, genes, perturbations):
    """Read the existing train-only cache format; do not compute test-derived priors."""
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        version = metadata.get("format_version")
        if version not in {3, 4} or not metadata.get("split_text"):
            raise ValueError("Prior cache must have v3/v4 training/split provenance")
        if (
            metadata.get("prior_mode", "full") != "full"
            or metadata.get("prior_fraction", 1.0) != 1.0
        ):
            raise ValueError("Case reference requires audited full, unmodified training priors")
        # A full prior with a non-default recorded seed has v4 metadata even
        # though no subsampling/shuffling takes place. Altered priors must not
        # be mislabeled as the reference prior in a full-method case study.
        if version == 4:
            seed = metadata.get("prior_seed")
            if (
                metadata.get("prior_robustness_version") != 1
                or metadata.get("prior_mode") != "full"
                or metadata.get("prior_fraction") != 1.0
                or not isinstance(seed, int)
                or isinstance(seed, bool)
                or seed < 0
            ):
                raise ValueError("v4 reference requires audited full, unmodified training priors")
        if metadata.get("genes") != list(genes):
            raise ValueError("Prior cache genes/order differ from evaluated expressions")
        names = [str(value) for value in cache["perturbations"]]
        shifts = np.asarray(cache["shifts"], dtype=np.float64)
        sources = [str(value) for value in cache["sources"]]
    if len(names) != len(set(names)) or shifts.shape != (len(names), len(genes)):
        raise ValueError("Malformed prior cache names/shape")
    if len(sources) != len(names) or not set(sources) <= {
        "direct_training_mean",
        "genept_dual_ridge",
    }:
        raise ValueError("Unrecognized prior provenance")
    if not np.isfinite(shifts).all() or set(perturbations) - set(names):
        raise ValueError("Prior cache has nonfinite values or lacks evaluated perturbations")
    return dict(zip(names, shifts)), metadata


def gene_table(genes, shifts, top_genes, prior=None):
    # This is an evaluation-only display selection, NEVER the steering signature.
    order = sorted(range(len(genes)), key=lambda i: (-abs(shifts["real"][i]), str(genes[i])))
    rank = np.empty(len(genes), dtype=int)
    rank[order] = np.arange(1, len(genes) + 1)
    result = pd.DataFrame(
        {
            "gene": list(genes),
            "reference_shift": shifts["real"],
            "baseline_shift": shifts["base"],
            "adacell_shift": shifts["pred"],
            "observed_effect_rank": rank,
            "display_gene": rank <= top_genes,
        }
    )
    if prior is not None:
        result["training_prior_shift"] = prior
    return result.sort_values("observed_effect_rank")


def plot_case(genes, row, outbase, *, baseline_label="PerturbDiff", pred_label="AdaCell"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import TwoSlopeNorm

    available = {font.name for font in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in available else "DejaVu Serif"
    colors = ["#5B7FA5", "#23866C"]
    with plt.rc_context(
        {
            "font.family": font,
            "font.size": 10,
            "axes.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    ):
        fig = plt.figure(figsize=(14.4, 4.7))
        grid = fig.add_gridspec(
            1,
            3,
            width_ratios=[1.05, 1.3, 1.0],
            left=0.06,
            right=0.975,
            bottom=0.27,
            top=0.80,
            wspace=0.55,
        )
        ax = fig.add_subplot(grid[0])
        for label, column, color in zip(
            (baseline_label, pred_label), ("baseline_shift", "adacell_shift"), colors
        ):
            ax.scatter(
                genes.reference_shift,
                genes[column],
                color=color,
                s=12,
                alpha=0.6,
                linewidth=0,
                label=label,
                rasterized=True,
            )
        bound = max(
            float(
                np.abs(
                    genes[["reference_shift", "baseline_shift", "adacell_shift"]].to_numpy()
                ).max()
            )
            * 1.08,
            0.01,
        )
        ax.plot([-bound, bound], [-bound, bound], "--", color="0.5", lw=0.8)
        ax.set(
            xlim=(-bound, bound),
            ylim=(-bound, bound),
            aspect="equal",
            xlabel="Reference response",
            ylabel="Predicted response",
            title="(a) Gene-level response",
        )
        ax.legend(frameon=False, loc="upper left", fontsize=9)
        heat = fig.add_subplot(grid[1])
        display = genes[genes.display_gene]
        values = display[["reference_shift", "baseline_shift", "adacell_shift"]].to_numpy().T
        limit = max(float(np.abs(values).max()), 0.01)
        picture = heat.imshow(
            values,
            cmap="RdBu_r",
            norm=TwoSlopeNorm(0, -limit, limit),
            aspect="auto",
            interpolation="none",
        )
        heat.set(
            yticks=range(3),
            yticklabels=["Reference", baseline_label, pred_label],
            xticks=range(len(display)),
            xticklabels=display.gene,
            title="(b) Largest observed responses",
            xlabel="Genes",
        )
        plt.setp(heat.get_xticklabels(), rotation=65, ha="right", fontsize=8)
        fig.colorbar(picture, ax=heat, fraction=0.045, pad=0.04, label="Expression shift")
        errors = grid[2].subgridspec(1, 2, wspace=0.85)
        for number, (metric, title, ylabel) in enumerate(
            (
                ("magnitude_error", "(c) Magnitude", "Norm error (expression units)"),
                ("angle_error_degrees", "Direction", "Angle error (degrees)"),
            )
        ):
            err = fig.add_subplot(errors[number])
            values = [row[f"base_{metric}"], row[f"pred_{metric}"]]
            for index, (value, color) in enumerate(zip(values, colors)):
                if np.isfinite(value):
                    err.bar(index, value, width=0.6, color=color, edgecolor="0.25", linewidth=0.7)
                else:
                    err.text(
                        index,
                        0.05,
                        "Undefined",
                        rotation=90,
                        ha="center",
                        va="bottom",
                        transform=err.get_xaxis_transform(),
                        fontsize=8,
                    )
            finite = [value for value in values if np.isfinite(value)]
            upper = 180.0 if metric.endswith("degrees") else max([0.01] + finite) * 1.2
            err.set(
                xticks=[0, 1],
                xticklabels=[baseline_label, pred_label],
                ylim=(0, upper),
                ylabel=ylabel,
                title=title,
            )
            plt.setp(err.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        fig.suptitle(f"{row['perturbation']} | {row['context']}", fontsize=13, y=0.96)
        fig.savefig(outbase.with_suffix(".pdf"))
        fig.savefig(outbase.with_suffix(".png"), dpi=300)
        plt.close(fig)
    return font


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("real", "base", "pred", "outdir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prior-cache", type=Path)
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control-pert", default="non-targeting")
    parser.add_argument("--context-col", default="auto")
    parser.add_argument("--cases", nargs="+", help="Exact PERTURBATION::CONTEXT identifiers")
    parser.add_argument("--top-genes", type=int, default=20)
    parser.add_argument("--baseline-label", default="PerturbDiff")
    parser.add_argument("--pred-label", default="AdaCell")
    args = parser.parse_args(argv)
    if args.top_genes < 1:
        parser.error("--top-genes must be positive")
    if args.outdir.exists():
        parser.error("Use a new --outdir; existing results are never overwritten")
    for name in ("real", "base", "pred"):
        if not getattr(args, name).is_file():
            parser.error(f"Missing --{name} file: {getattr(args, name)}")
    args.outdir.mkdir(parents=True, exist_ok=False)
    files = {key: getattr(args, key) for key in ("real", "base", "pred")}
    if args.prior_cache:
        files["prior_cache"] = args.prior_cache
    manifest = {
        "status": "started",
        "analysis_version": 1,
        "implementation_sha256": sha256_file(__file__),
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for key, path in files.items()
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "interpretation": "Population contrasts, not paired-cell effects; cases are "
        "outcome-selected examples, not aggregate evidence or pathway validation.",
        "gene_selection": "Display genes ranked by absolute reference test response; "
        "evaluation only, never used for fitting, sampling, rewards, or model selection.",
        "undefined_angle": "NaN when either vector norm <= 1e-12; never replaced with zero",
        "input_values_changed": False,
    }
    write_manifest(args.outdir / "manifest.json", manifest)
    datasets = [ad.read_h5ad(path, backed="r") for path in (args.real, args.base, args.pred)]
    try:
        table, responses, audit = compute_case_responses(
            *datasets,
            pert_col=args.pert_col,
            control_pert=args.control_pert,
            context_col=args.context_col,
        )
        genes = list(datasets[0].var_names)
    finally:
        for data in datasets:
            data.file.close()
    priors, prior_metadata = {}, None
    if args.prior_cache:
        priors, prior_metadata = load_prior_cache(args.prior_cache, genes, table.perturbation)
    selected, selection = select_cases(table, args.cases)
    table.to_csv(args.outdir / "all_condition_errors.csv", index=False)
    pd.DataFrame(selected).to_csv(args.outdir / "selected_cases.csv", index=False)
    plot_files, captions = [], []
    for index, case in enumerate(selected, 1):
        row = table.set_index("case_id").loc[case["case_id"]]
        frame = gene_table(
            genes, responses[case["case_id"]], args.top_genes, prior=priors.get(row.perturbation)
        )
        stem = args.outdir / f"case_{index:02d}"
        frame.to_csv(stem.with_name(stem.name + "_gene_responses.csv"), index=False)
        font = plot_case(
            frame, row, stem, baseline_label=args.baseline_label, pred_label=args.pred_label
        )
        plot_files.append({"case_id": case["case_id"], "stem": stem.name})
        captions.append(
            f"{stem.name}: {row.perturbation} in {row.context}. "
            f"Selection: {case['selection']}; the selection rule and every condition's "
            "errors are retained in the analysis outputs. "
            "(a) Gene-level population responses, defined as treated mean minus the "
            "matched reference control mean. The dashed line denotes equal responses. "
            f"(b) Signed responses for the {min(args.top_genes, len(genes))} genes with "
            "the largest absolute reference response, shared across all three rows. "
            "These display genes are selected after evaluation and are not a training "
            "or steering signature. (c) Absolute response-norm error and angular error "
            "are shown on separate axes in expression units and degrees, respectively. "
            "Zero-norm response angles are undefined. No cells are paired between "
            "the generated and reference populations. This illustrative case does not "
            "establish a pathway mechanism or replace the all-condition evaluation."
        )
    (args.outdir / "captions.txt").write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    manifest.update(
        status="complete",
        condition_count=len(table),
        selection=selection,
        plots=plot_files,
        audit=audit,
        prior_metadata=prior_metadata,
        font=font,
    )
    write_manifest(args.outdir / "manifest.json", manifest)
    outputs = {
        path.name: sha256_file(path) for path in sorted(args.outdir.iterdir()) if path.is_file()
    }
    temporary = args.outdir / ".COMPLETE.json.tmp"
    write_manifest(
        temporary,
        {
            "status": "complete",
            "outputs_sha256": outputs,
            "inputs": manifest["inputs"],
            "condition_count": len(table),
        },
    )
    temporary.replace(args.outdir / "COMPLETE.json")
    print(f"COMPLETE: {args.outdir.resolve()} ({len(table)} conditions, {len(selected)} cases)")


if __name__ == "__main__":
    main()
