#!/usr/bin/env python
"""Read-only experiment inventory; write a separate report, never run experiments.

Report all alphas, not a test-selected winner. Smoke/validation outputs are
excluded. Metric means require the full test condition set; NaNs are not dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from celldiffa.benchmark.artifacts import sha256_file  # noqa: E402
from celldiffa.benchmark.metrics import PAPER_METRIC_NAMES  # noqa: E402
from celldiffa.benchmark.streaming import (  # noqa: E402
    h5ad_expression_shape,
    read_h5ad_obs,
    read_h5ad_var,
)

METRICS = ["R2", *PAPER_METRIC_NAMES]
DISPLAY = ["DEOver", "PDCorr", "PDS_cos", "MSE"]
PER_FILE = "perturbdiff_metrics_per_perturbation.csv"
SUMMARY_FILE = "perturbdiff_metrics_summary.csv"
PRED_NAMES = ("celldiffa_scratch.h5ad", "celldiffa_finetuned.h5ad", "predictions.h5ad")
EXCLUDED = {
    "reference",
    "validation",
    "smoke",
    "_smoke",
    "reports",
    "analysis",
    "diagnostics",
    "workers",
    "wandb",
    "cell_eval_0.6.6",
    "launch_contracts",
    "maintext_completion",  # Partial isolated timing runs, not full-test results.
    "__pycache__",
    ".git",
}
SECTIONS = ("main", "ablation", "additional", "baseline", "other")
BASELINE_ALIASES = {"cpa_cpu": "cpa", "scouter_vectorized": "scouter"}
TITLES = {
    "main": "Main experiments / 主实验与 alpha 敏感性",
    "ablation": "Ablations / 消融实验",
    "additional": "Additional experiments / 新六队列实验",
    "baseline": "Baselines / 已有基线",
    "other": "Other discovered runs / 其他结果",
}


@dataclass
class Run:
    output: Path
    category: str
    prediction: Path | None = None
    metric_dirs: set[Path] = field(default_factory=set)
    kind: str = "prediction"
    plan_job: dict = field(default_factory=dict)


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def excluded(parts):
    return any(
        p in EXCLUDED or p.startswith(("smoke_", "_smoke", "parallel_analysis")) for p in parts
    )


def category_for(path, config=None):
    config = config or {}
    name = str(path).lower()
    weights = config.get("reward_weights", [1, 1, 1])
    ablation = (
        "ablation" in name
        or any(
            word in name
            for word in ("without_", "no_signature", "no_direction", "no_anchor", "no_norm")
        )
        or (isinstance(weights, list) and any(v == 0 for v in weights))
        or config.get("reward_normalization") == "none"
    )
    if ablation:
        return "ablation"
    if "test_sensitivity" in name or config.get("evaluation_split") == "test":
        return "main"
    if "perturbdiff" in name or any(
        name.endswith(p)
        for p in (
            "cpa_cpu",
            "gears_extended",
            "scouter_vectorized",
            "state",
            "cellflow",
            "squidiff_cpu",
        )
    ):
        return "baseline"
    return "other"


def run_config(output):
    path = output / "shards/run_config.json"
    return read_json(path if path.exists() else output / "run_config.json")


def discover(root):
    """Infer only unique metric associations. Ambiguity is reported, never guessed."""
    runs, metric_dirs, plans, warnings = {}, set(), [], []
    for current, directories, files in os.walk(root):
        directory = Path(current)
        relative = directory.relative_to(root)
        directories[:] = [d for d in directories if not excluded((*relative.parts, d))]
        if "shards" in directories:
            directories.remove("shards")
        if PER_FILE in files or SUMMARY_FILE in files:
            metric_dirs.add(directory)
        if "plan.json" in files:
            plans.append(directory / "plan.json")
        # Central prediction files have no per-run directory.
        if directory == root / "predictions":
            for filename in files:
                if filename.endswith(".h5ad"):
                    stem = Path(filename).stem
                    output = root / stem
                    runs[output] = Run(output, "baseline", directory / filename)
            continue
        markers = {"run_config.json", "progress.json", "training_progress.json", "job_status.json"}
        if not (
            markers.intersection(files)
            or any(p in files for p in PRED_NAMES)
            or (directory / "shards/run_config.json").is_file()
        ):
            continue
        try:
            config = run_config(directory)
            if config.get("smoke") or config.get("evaluation_split") == "validation":
                directories[:] = []
                continue
        except (ValueError, OSError) as error:
            warnings.append(f"Cannot read {directory}: {error}")
            config = {}
        runs.setdefault(directory, Run(directory, category_for(relative, config)))

    # These six explicitly requested main runs remain visible even if never started.
    if (root / "test_sensitivity").is_dir():
        for backbone in ("scratch", "finetuned"):
            for alpha in ("05", "1", "2"):
                output = root / "test_sensitivity" / f"{backbone}_alpha{alpha}"
                runs.setdefault(output, Run(output, "main"))
    for path in plans:
        try:
            plan = read_json(path)
            if "lanes" not in plan:
                continue
            # Plans are server-local contracts; never follow stale Mac paths.
            if Path(plan["output_root"]).resolve() != path.parent.resolve():
                warnings.append(f"Stale output_root in {path}; plan not followed")
                continue
            for lane in plan["lanes"]:
                for job in lane:
                    output = path.parent / "runs" / job["id"]
                    record = runs.setdefault(output, Run(output, "additional"))
                    record.category, record.kind, record.plan_job = "additional", job["kind"], job
                    record.metric_dirs.add(path.parent / "metrics" / job["id"])
        except (ValueError, OSError, KeyError, TypeError) as error:
            warnings.append(f"Cannot interpret {path}: {error}")

    claimed = set()
    alias_owners = {
        root / "metrics" / alias: root / run
        for run, alias in BASELINE_ALIASES.items()
        if (root / run / "predictions.h5ad").is_file()
    }
    for output, record in runs.items():
        candidates = [
            output,
            output / "metrics",
            output / "evaluation",
            output.parent / "metrics" / output.name,
            root / "metrics" / output.relative_to(root),
        ]
        if output.parent.name == "runs":
            candidates.append(output.parent.parent / "metrics" / output.name)
        candidates.extend(p for p, owner in alias_owners.items() if owner == output)
        record.metric_dirs.update(
            p for p in candidates if p in metric_dirs and alias_owners.get(p, output) == output
        )
        record.metric_dirs.intersection_update(metric_dirs)
        claimed.update(record.metric_dirs)
    # Metrics often live at root/metrics/<run-name>. Associate by basename only
    # when exactly one discovered run matches; otherwise keep an orphan record.
    for directory in sorted(metric_dirs - claimed):
        matching = [record for output, record in runs.items() if output.name == directory.name]
        if not matching:

            def canonical(name):
                return name.removeprefix("celldiffa_").removeprefix("adacell_")

            matching = [
                record
                for output, record in runs.items()
                if canonical(output.name) == canonical(directory.name)
            ]
        if len(matching) == 1:
            matching[0].metric_dirs.add(directory)
        else:
            output = directory
            runs[output] = Run(
                output,
                category_for(directory.relative_to(root)),
                metric_dirs={directory},
                kind="metrics_only",
            )
            if len(matching) > 1:
                warnings.append(f"Ambiguous metrics directory, not attached: {directory}")
    return sorted(
        runs.values(), key=lambda r: (SECTIONS.index(r.category), str(r.output))
    ), warnings


def reference_metadata(path):
    obs, genes = read_h5ad_obs(path), read_h5ad_var(path).index
    if "gene" not in obs or obs.gene.isna().any() or not genes.is_unique:
        raise ValueError("Reference requires non-null gene labels and unique expression genes")
    counts = obs.gene.astype(str).value_counts().sort_index()
    if "non-targeting" not in counts or len(counts) < 2:
        raise ValueError("Reference must contain controls and treated conditions")
    return obs, genes, counts


def validate_prediction_metadata(prediction, reference):
    obs, genes, counts = reference_metadata(prediction)
    real_obs, real_genes, real_counts = reference
    if not genes.equals(real_genes) or not counts.equals(real_counts):
        raise ValueError("Prediction gene order / condition cell counts differ from full reference")
    if h5ad_expression_shape(prediction, "X") != (len(obs), len(genes)):
        raise ValueError("Prediction expression shape differs from its metadata")
    if "cell_line" in real_obs and "cell_line" in obs:

        def contexts(frame):
            return (
                frame.assign(gene=frame.gene.astype(str), cell_line=frame.cell_line.astype(str))
                .groupby(["gene", "cell_line"])
                .size()
            )

        if not contexts(obs).equals(contexts(real_obs)):
            raise ValueError("Prediction cellular-context counts differ from reference")


def metrics_values(path, expected):
    table = pd.read_csv(path, dtype={"perturbation": str})
    if not {"perturbation", *METRICS}.issubset(table.columns):
        raise ValueError("Metric table does not contain all 14 protocol metrics")
    if (
        table.perturbation.isna().any()
        or table.perturbation.duplicated().any()
        or set(table.perturbation) != expected
    ):
        raise ValueError(
            f"Metric conditions differ from full test set ({len(table)}/{len(expected)} rows)"
        )
    numeric = table[METRICS].apply(pd.to_numeric, errors="raise")
    finite = np.isfinite(numeric)
    counts = finite.sum().to_dict()
    means = {key: float(numeric[key].mean()) if finite[key].all() else None for key in METRICS}
    summary = path.with_name(SUMMARY_FILE)
    if summary.is_file():
        saved = pd.read_csv(summary, index_col=0)
        if not set(METRICS).issubset(saved.columns) or not {"mean", "count"}.issubset(saved.index):
            raise ValueError("Incomplete summary CSV schema")
        if not np.allclose(
            saved.loc["count", METRICS], numeric.count(), equal_nan=True
        ) or not np.allclose(saved.loc["mean", METRICS], numeric.mean(), equal_nan=True):
            raise ValueError("Summary CSV is stale relative to per-perturbation metrics")
    return means, counts


def inspect_run(record, root, reference_path, reference, *, verify_hashes=False, hashes=None):
    hashes = hashes if hashes is not None else {}
    row = dict(
        category=record.category,
        experiment=str(record.output.relative_to(root)),
        status="NOT_STARTED",
        progress="",
        note="",
        prediction="",
        metrics_dir="",
        hash_check="not_requested",
        **{key: None for key in METRICS},
    )
    output = record.output
    try:
        if (
            output == root / "perturbdiff_finetuned"
            and (root / "perturbdiff_finetuned_fixed_ids/predictions.h5ad").is_file()
        ):
            row.update(
                status="SUPERSEDED",
                note="Legacy Finetuned run superseded by corrected category-ID results; "
                "not a valid comparator",
            )
            return row
        config = run_config(output)
        for key in (
            "variant",
            "backbone",
            "alpha",
            "num_particles",
            "alignment_mode",
            "reward_unit",
            "reward_weights",
            "reward_normalization",
            "seed",
            "native_blocks_per_population",
            "population_cells",
            "sampling_steps",
            "start_time",
        ):
            if key in config:
                row[key] = config[key]
        if config.get("smoke") or config.get("evaluation_split") == "validation":
            row.update(status="EXCLUDED", note="Smoke / validation, not a formal test result")
            return row
        state = read_json(output / "job_status.json") or read_json(
            output.parent / f"{output.name}.job_status.json"
        )
        progress = read_json(output / "training_progress.json") or read_json(
            output / "progress.json"
        )
        if "completed_steps" in progress:
            row["progress"] = (
                f"{progress['completed_steps']}/{progress.get('total_steps', '?')} steps"
            )
        elif "completed_groups" in progress:
            row["progress"] = (
                f"{progress['completed_groups']}/{progress.get('total_groups', '?')} groups"
            )
        else:
            workers = sorted((output / "shards").glob("worker_*.progress.json"))
            if workers:
                groups = list((output / "shards").glob("group_*.npz"))
                row["progress"] = f"{len(groups)} saved groups (total not recorded)"
        if record.kind in {"train", "train_squidiff"}:
            row.update(
                status="TRAINING_COMPLETE"
                if progress.get("status") == "training_complete" and (output / "best.pt").is_file()
                else "TRAINING_INCOMPLETE"
            )
            if state.get("status") == "failed":
                row.update(status="FAILED", note=state.get("error", "Training process failed"))
            return row
        predictions = (
            [record.prediction]
            if record.prediction is not None
            else [output / p for p in PRED_NAMES if (output / p).is_file()]
        )
        if len(predictions) > 1:
            raise ValueError("Multiple prediction H5ADs in one run; association is ambiguous")
        prediction = predictions[0] if predictions else None
        if prediction is None or not prediction.is_file():
            row["status"] = (
                "METRICS_ONLY"
                if record.metric_dirs
                else ("INCOMPLETE" if output.exists() else "NOT_STARTED")
            )
            if state.get("status") in {"failed", "waiting_for_training"}:
                row.update(status=state["status"].upper(), note=state.get("error", ""))
            row["note"] = row["note"] or "No full prediction H5AD; not counted as completed"
            return row
        row["prediction"] = str(prediction)
        validate_prediction_metadata(prediction, reference)
        row["perturbations"] = len(reference[2]) - 1
        row["cells"] = len(reference[0])
        if not record.metric_dirs:
            if state.get("status") == "failed":
                row.update(
                    status="FAILED",
                    note="Full prediction exists but queue failed: "
                    + state.get("error", "inspect evaluation.log"),
                )
                return row
            row.update(
                status="NEEDS_EVALUATION",
                note="Full prediction metadata matches; no protocol metric files",
            )
            return row
        if len(record.metric_dirs) != 1:
            raise ValueError("Multiple possible metric directories; refusing to choose one")
        directory = next(iter(record.metric_dirs))
        row["metrics_dir"] = str(directory)
        per = directory / PER_FILE
        if not per.is_file():
            row.update(
                status="SUMMARY_ONLY",
                note="Cannot verify coverage without per-perturbation metrics",
            )
            return row
        means, counts = metrics_values(per, set(reference[2].index) - {"non-targeting"})
        marker = read_json(output / "evaluated.json")
        if verify_hashes and marker:
            for key, path in (
                ("reference_sha256", reference_path),
                ("prediction_sha256", prediction),
                ("metrics_sha256", per),
            ):
                if path not in hashes:
                    hashes[path] = sha256_file(path)
                if marker.get(key) != hashes[path]:
                    raise ValueError(f"Evaluation hash mismatch: {key}")
            row["hash_check"] = "verified"
        elif verify_hashes:
            row["hash_check"] = "no_record"
            row["note"] = (
                "No evaluation hash record; coverage checked, prediction/metric identity not proven"
            )
        else:
            row["note"] = (
                "Coverage checked; use --verify-hashes to verify available evaluation records"
            )
        row.update(means)
        row.update({f"finite_{key}": value for key, value in counts.items()})
        undefined = [key for key, value in counts.items() if value != len(reference[2]) - 1]
        row["status"] = "EVALUATED_UNDEFINED" if undefined else "EVALUATED"
        if undefined:
            row["note"] += "; undefined metrics (macro-mean withheld): " + ", ".join(undefined)
        if state.get("status") == "failed":
            row["note"] += "; queue recorded failure, inspect run/evaluation/diagnostics logs"
        diagnostic = output / "diagnostics/population_diagnostics.csv"
        row["diagnostics_present"] = diagnostic.is_file()
        row["native_support_breakdown"] = (directory / "native_support_means.csv").is_file()
    except (ValueError, OSError, KeyError, TypeError) as error:
        row.update(status="CHECK_FAILED", note=str(error))
    return row


def display_value(value):
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.3e}" if value != 0 and abs(value) < 0.001 else f"{value:.4f}"


def markdown_table(rows):
    """A plain Markdown table without requiring the optional tabulate package."""
    columns = list(rows[0])
    cells = [
        [str(row[key]).replace("|", "\\|").replace("\n", " ") for key in columns] for row in rows
    ]
    widths = [max(len(column), *(len(row[i]) for row in cells)) for i, column in enumerate(columns)]

    def line(values):
        return "| " + " | ".join(value.ljust(width) for value, width in zip(values, widths)) + " |"

    return "\n".join([line(columns), line(["-" * width for width in widths]), *map(line, cells)])


def write_report(rows, warnings, root, reference, outdir):
    if outdir.exists():
        raise FileExistsError("Use a new --outdir; existing reports are never overwritten")
    outdir.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(outdir / "all_results.csv", index=False)
    text = [
        "# Replogle experiment inventory",
        "",
        f"Reference: `{reference}`",
        "",
        "Metrics are macro-averages over ALL test perturbations. "
        "No test-set alpha selection is performed.",
        "",
        "EVALUATED means full prediction metadata and metric coverage match. "
        "It does not mean training provenance or biological performance has been re-audited. "
        "`hash_check=verified` additionally verifies the saved prediction/reference/metric "
        "identities. Other statuses are not complete evaluated predictions. Undefined "
        "metrics are shown as --, never averaged over a smaller subset.",
        "",
    ]
    for category in SECTIONS:
        selected = [row for row in rows if row["category"] == category]
        text += [f"## {TITLES[category]}", ""]
        if not selected:
            text += ["No runs discovered / 未发现对应实验目录。", ""]
            continue
        pd.DataFrame(selected).to_csv(outdir / f"{category}.csv", index=False)
        compact = [
            {
                "Experiment": row["experiment"],
                "Status": row["status"],
                "Progress": row["progress"],
                **{key: display_value(row[key]) for key in DISPLAY},
            }
            for row in selected
        ]
        text += [markdown_table(compact), ""]
    text += ["## Checks and paths", ""]
    for row in rows:
        if row["note"]:
            text.append(f"- `{row['experiment']}`: {row['note']} (hash: {row['hash_check']}).")
    text += [
        "",
        "## Discovery warnings",
        "",
        *([f"- {w}" for w in warnings] or ["None."]),
        "",
        "Source experiments were not modified. Expression values were not re-scored; "
        "only H5AD metadata, CSV coverage/consistency and optional saved hashes were checked. "
        "Missing-evaluation commands below must be explicitly run by the user. Concurrent "
        "training/sampling can change files; rerun this inventory after completion.",
    ]
    (outdir / "report.md").write_text("\n".join(text) + "\n")
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Not executed by the report script. No training or sampling.",
        "# Only structurally complete predictions with absent metrics are listed.",
        f"cd {shlex.quote(str(REPO))}",
    ]
    for row in rows:
        if row["status"] != "NEEDS_EVALUATION" or row["category"] not in {
            "main",
            "ablation",
            "additional",
        }:
            continue
        destination = Path(row["prediction"]).parent / "metrics"
        commands += [
            f"# {row['experiment']}",
            shlex.join(
                [
                    sys.executable,
                    str(REPO / "scripts/baselines/evaluate.py"),
                    "--real",
                    str(reference),
                    "--pred",
                    row["prediction"],
                    "--outdir",
                    str(destination),
                    "--pert-col",
                    "gene",
                    "--control-pert",
                    "non-targeting",
                    "--num-threads",
                    "4",
                ]
            ),
        ]
    (outdir / "evaluate_missing.sh").write_text("\n".join(commands) + "\n")
    (outdir / "inventory.json").write_text(
        json.dumps(
            dict(root=str(root), reference=str(reference), warnings=warnings, rows=rows),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    print("\n".join(text[: text.index("## Checks and paths")]))
    print(f"REPORT: {outdir / 'report.md'}")
    print(f"ALL 14 METRICS: {outdir / 'all_results.csv'}")
    print(f"MISSING EVALUATION COMMANDS (not executed): {outdir / 'evaluate_missing.sh'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/replogle")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify available evaluated.json records (reads prediction files, no GPU)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    reference = (args.reference or root / "reference/real.h5ad").resolve()
    if not root.is_dir() or not reference.is_file():
        parser.error("Results directory and reference/real.h5ad must exist")
    metadata = reference_metadata(reference)
    records, warnings = discover(root)
    hashes = {}
    rows = [
        inspect_run(
            record, root, reference, metadata, verify_hashes=args.verify_hashes, hashes=hashes
        )
        for record in records
    ]
    outdir = args.outdir or root / "reports" / datetime.now(timezone.utc).strftime(
        "inventory_%Y%m%dT%H%M%S%fZ"
    )
    write_report(rows, warnings, root, reference, outdir.resolve())


if __name__ == "__main__":
    main()
