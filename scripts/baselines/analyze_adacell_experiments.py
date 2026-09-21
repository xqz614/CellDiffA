#!/usr/bin/env python
"""Figures for contrast diagnosis, matched controls and independent diversity.

No parameter selection. Missing/undefined results are listed, never invented.
Timing from concurrent jobs is labelled descriptive and is not a fair isolated
GPU benchmark. Per-perturbation bootstrap intervals do not measure seed variance.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np
import pandas as pd

from celldiffa.benchmark.artifacts import sha256_file
from celldiffa.benchmark.backbone_experiments import atomic_json, dense
from celldiffa.benchmark.contracts import validate_prediction_pair


def bootstrap(values, seed=1729):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires finite values for every included condition")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    return float(values.mean()), *np.quantile(means, [0.025, 0.975])


def effect_strata(real):
    labels = real.obs.gene.astype(str)
    contexts = real.obs.cell_line.astype(str)
    means = {
        context: dense(
            real.X[((labels == "non-targeting") & (contexts == context)).to_numpy()]
        ).mean(0)
        for context in set(contexts)
    }
    rows = []
    for pert in sorted(set(labels) - {"non-targeting"}):
        total, weight = 0.0, 0
        for context in sorted(set(contexts[labels == pert])):
            x = dense(real.X[((labels == pert) & (contexts == context)).to_numpy()])
            total += len(x) * np.linalg.norm(x.mean(0) - means[context])
            weight += len(x)
        rows.append(dict(perturbation=pert, effect_norm=total / weight))
    frame = pd.DataFrame(rows).sort_values(["effect_norm", "perturbation"])
    frame["strength"] = ""
    for group, indices in zip(
        ("weak", "medium", "strong"), np.array_split(np.arange(len(frame)), 3)
    ):
        frame.iloc[indices, frame.columns.get_loc("strength")] = group
    return frame.set_index("perturbation")


def backbone_comparisons(tables):
    pairs = []
    for label, guided, matched, vanilla in (
        ("PerturbDiff", "scratch_main", "scratch_random16", None),
        ("Squidiff", "squidiff_adacell16", "squidiff_random16", "squidiff_vanilla"),
        (
            "Conditional DDPM",
            "conditional_ddpm_adacell16",
            "conditional_ddpm_random16",
            "conditional_ddpm_vanilla",
        ),
    ):
        baseline = matched if matched in tables else vanilla
        if guided in tables and baseline is not None and baseline in tables:
            pairs.append((label, guided, baseline, baseline == matched))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--main-pred", type=Path, required=True)
    parser.add_argument("--main-metrics", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    if args.outdir.exists():
        raise FileExistsError("Use a new analysis output directory")
    plan = json.loads(args.plan.read_text())
    root = Path(plan["output_root"])
    reference = Path(plan["repo"]) / "results/replogle/reference/real.h5ad"
    real = ad.read_h5ad(reference)
    expected = sorted(set(real.obs.gene.astype(str)) - {"non-targeting"})
    records = {
        "scratch_main": dict(
            prediction=args.main_pred, metrics=args.main_metrics, output=args.main_pred.parent
        )
    }
    missing = []
    for lane in plan["lanes"]:
        for job in lane:
            if job["kind"] == "train":
                continue
            output = root / "runs" / job["id"]
            prediction = output / (
                "celldiffa_scratch.h5ad" if job["kind"] == "perturbdiff" else "predictions.h5ad"
            )
            metrics = root / "metrics" / job["id"]
            if (
                not prediction.exists()
                or not (metrics / "perturbdiff_metrics_per_perturbation.csv").exists()
            ):
                missing.append(job["id"])
                continue
            records[job["id"]] = dict(prediction=prediction, metrics=metrics, output=output)
    args.outdir.mkdir(parents=True)
    tables, summaries, provenance, intervals = {}, [], {}, []
    for name, record in records.items():
        prediction = ad.read_h5ad(record["prediction"])
        validate_prediction_pair(real, prediction, pert_col="gene", control_pert="non-targeting")
        table_path = record["metrics"] / "perturbdiff_metrics_per_perturbation.csv"
        table = pd.read_csv(table_path)
        if table.perturbation.duplicated().any() or set(table.perturbation) != set(expected):
            raise ValueError(f"Incomplete metrics: {name}")
        table = table.set_index("perturbation").loc[expected]
        # Detect accidental use of a metric table from another prediction run.
        # These are the protocol's pseudobulk MSE and R², not paired-cell errors.
        from sklearn.metrics import r2_score

        for pert in expected:
            r = dense(real.X[real.obs.gene.astype(str).eq(pert).to_numpy()]).mean(0)
            p = dense(prediction.X[prediction.obs.gene.astype(str).eq(pert).to_numpy()]).mean(0)
            observed = [float(np.square(r - p).mean()), r2_score(r, p)]
            reported = table.loc[pert, ["MSE", "R2"]].to_numpy(float)
            if not np.allclose(observed, reported, atol=1e-6, rtol=1e-4):
                raise ValueError(
                    f"Metric table does not match prediction pseudobulk values: {name}/{pert}"
                )
        if not np.isfinite(table[["R2", "PDCorr", "PDS_cos", "DEOver", "MSE"]]).all().all():
            raise ValueError(
                f"Undefined key metric in {name}; report it explicitly before plotting"
            )
        # Recompute independent diagnostics so old files cannot silently mismatch predictions.
        from scripts.baselines.evaluate_population_diagnostics import main as diagnostics_main

        old_argv = sys.argv
        try:
            sys.argv = [
                "diagnostics",
                "--real",
                str(reference),
                "--pred",
                str(record["prediction"]),
                "--outdir",
                str(args.outdir / "diagnostics" / name),
            ]
            diagnostics_main()
        finally:
            sys.argv = old_argv
        diagnostics = pd.read_csv(
            args.outdir / "diagnostics" / name / "population_diagnostics.csv"
        ).set_index("perturbation")
        table = table.join(
            diagnostics[
                ["predicted_to_real_variance", "predicted_to_real_rank", "sliced_w1_to_real"]
            ]
        )
        tables[name] = table
        timing, ancestors = [], []
        for path in sorted((record["output"] / "shards").glob("group_*.diagnostics.json")):
            detail = json.loads(path.read_text())
            if "seconds" in detail or "sampling_seconds" in detail:
                timing.append(detail.get("seconds", detail.get("sampling_seconds")))
            if detail.get("distinct_initial_ancestors"):
                ancestors.append(detail["distinct_initial_ancestors"][-1])
        summary = dict(
            model=name,
            **table.mean(numeric_only=True).to_dict(),
            observed_sampling_seconds=sum(timing) if timing else None,
            timed_groups=len(timing),
            final_ancestors_mean=np.mean(ancestors) if ancestors else None,
        )
        summaries.append(summary)
        for metric in ("DEOver", "PDS_cos", "predicted_to_real_variance"):
            mean, low, high = bootstrap(table[metric])
            intervals.append(dict(model=name, metric=metric, mean=mean, low=low, high=high))
        provenance[name] = dict(
            prediction_sha256=sha256_file(record["prediction"]),
            metrics_sha256=sha256_file(table_path),
        )
    pd.DataFrame(summaries).to_csv(args.outdir / "summary.csv", index=False)
    pd.DataFrame(intervals).to_csv(args.outdir / "perturbation_bootstrap.csv", index=False)
    seed_deltas = []
    for seed, guided, baseline in (
        (42, "scratch_main", "scratch_random16"),
        (43, "scratch_seed43", "scratch_random16_seed43"),
        (44, "scratch_seed44", "scratch_random16_seed44"),
    ):
        if guided in tables and baseline in tables:
            for metric in ("DEOver", "PDCorr", "PDS_cos", "MSE"):
                delta = tables[guided][metric] - tables[baseline][metric]
                seed_deltas.append(dict(seed=seed, metric=metric, mean_paired_delta=delta.mean()))
    if seed_deltas:
        seed_frame = pd.DataFrame(seed_deltas)
        seed_frame.to_csv(args.outdir / "paired_seed_deltas.csv", index=False)
        seed_frame.groupby("metric")["mean_paired_delta"].agg(["count", "mean", "std"]).to_csv(
            args.outdir / "across_seed_summary.csv"
        )
    strata = effect_strata(real)
    strata.to_csv(args.outdir / "diagnostic_effect_strata.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    def save(fig, name):
        fig.savefig(args.outdir / f"{name}.pdf")
        fig.savefig(args.outdir / f"{name}.png", dpi=240)
        plt.close(fig)

    if "scratch_random16" in tables:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), constrained_layout=True)
        for name, color in (("scratch_random16", "#6b7280"), ("scratch_main", "#c44e52")):
            axes[0].scatter(
                tables[name].R2, tables[name].PDCorr, s=9, alpha=0.45, label=name, color=color
            )
        axes[0].set(xlabel="Expression fit (R²)", ylabel="Response accuracy (PDCorr)")
        axes[0].legend(fontsize=6)
        changes = []
        for i, group in enumerate(("weak", "medium", "strong")):
            names = strata.index[strata.strength == group]
            values = (
                tables["scratch_main"].loc[names, "PDCorr"]
                - tables["scratch_random16"].loc[names, "PDCorr"]
            )
            mean, low, high = bootstrap(values)
            changes.append(
                dict(group=group, perturbations=len(names), mean=mean, low=low, high=high)
            )
            axes[1].errorbar(
                i,
                mean,
                yerr=[[max(0, mean - low)], [max(0, high - mean)]],
                fmt="o",
                color="#c44e52",
                capsize=3,
            )
        axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.7)
        axes[1].set(
            xticks=[0, 1, 2],
            xticklabels=["Weak", "Medium", "Strong"],
            xlabel="Observed effect strength (diagnostic only)",
            ylabel="Δ PDCorr versus unselected16",
        )
        pd.DataFrame(changes).to_csv(args.outdir / "effect_stratified_changes.csv", index=False)
        save(fig, "contrast_diagnosis")
    order = [
        n
        for n in (
            "scratch_random16",
            "scratch_best16",
            "scratch_cellwise16",
            "scratch_mean_correction",
            "scratch_main",
        )
        if n in tables
    ]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.8), constrained_layout=True)
    for axis, metric in zip(axes, ("DEOver", "PDS_cos", "predicted_to_real_variance")):
        for i, name in enumerate(order):
            mean, low, high = bootstrap(tables[name][metric])
            axis.errorbar(
                i, mean, yerr=[[max(0, mean - low)], [max(0, high - mean)]], fmt="o", capsize=3
            )
        axis.set(
            xticks=list(range(len(order))),
            xticklabels=[n.replace("scratch_", "") for n in order],
            ylabel=metric,
        )
        axis.tick_params(axis="x", labelrotation=35, labelsize=7)
        if "variance" in metric:
            axis.axhline(1, color="gray", linestyle="--", linewidth=0.7)
    save(fig, "population_controls")
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), constrained_layout=True)
    for row in summaries:
        for axis, metric in zip(axes, ("predicted_to_real_variance", "sliced_w1_to_real")):
            axis.scatter(row["PDS_cos"], row[metric], s=25)
            axis.annotate(
                row["model"],
                (row["PDS_cos"], row[metric]),
                fontsize=5,
                xytext=(3, 3),
                textcoords="offset points",
            )
            axis.set(xlabel="Response accuracy (PDS-cos)", ylabel=metric)
    axes[0].axhline(1, color="gray", linestyle="--", linewidth=0.7)
    save(fig, "accuracy_diversity")
    backbone_pairs = backbone_comparisons(tables)
    if backbone_pairs:
        fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), constrained_layout=True)
        gains = []
        for axis, metric in zip(axes, ("DEOver", "PDS_cos")):
            for index, (label, guided, baseline, matched) in enumerate(backbone_pairs):
                mean, low, high = bootstrap(tables[guided][metric] - tables[baseline][metric])
                gains.append(
                    dict(
                        backbone=label,
                        metric=metric,
                        mean=mean,
                        low=low,
                        high=high,
                        guided=guided,
                        baseline=baseline,
                        matched_denoising_budget=matched,
                    )
                )
                axis.errorbar(
                    index,
                    mean,
                    yerr=[[max(0, mean - low)], [max(0, high - mean)]],
                    fmt="o",
                    capsize=3,
                )
            axis.axhline(0, color="gray", linestyle="--", linewidth=0.7)
            axis.set(
                xticks=list(range(len(backbone_pairs))),
                xticklabels=[
                    label
                    + "\nvs "
                    + ("random16 (matched)" if matched else "vanilla1 (unequal budget)")
                    for label, _, _, matched in backbone_pairs
                ],
                ylabel=f"Δ {metric} versus indicated reference",
            )
            axis.tick_params(axis="x", labelsize=7)
        pd.DataFrame(gains).to_csv(args.outdir / "paired_backbone_gains.csv", index=False)
        save(fig, "backbone_transfer")
    atomic_json(
        args.outdir / "analysis_manifest.json",
        dict(
            reference_sha256=sha256_file(reference),
            models=provenance,
            missing_jobs=missing,
            protocol_complete=not missing,
            test_based_parameter_selection=False,
            intervals="2000 perturbation bootstrap resamples, conditional on a generation seed",
            limitations=[
                "Effect strata are descriptive, not denoising-loss causal evidence",
                "Variance/rank/W1 do not prove biological realism or support preservation",
                "Concurrent-job timings in CSV are not isolated GPU cost benchmarks",
                "Backbone pairs versus vanilla1 are not matched-denoising-budget comparisons",
            ],
        ),
    )
    print(f"WROTE {args.outdir}; missing jobs: {missing}")


if __name__ == "__main__":
    main()
