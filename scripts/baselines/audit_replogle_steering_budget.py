#!/usr/bin/env python
"""Audit actual recorded sampling work; partial runs are never full comparisons.

Does not run a model, read held-out expression values, or tune any parameters.
Checks recorded group metadata, not equality of unrecorded control-cell tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np

from celldiffa.benchmark.artifacts import write_manifest

MATCHED_SETTINGS = (
    "variant",
    "checkpoint",
    "checkpoint_size",
    "checkpoint_mtime_ns",
    "reference",
    "evaluation_split",
    "seed",
    "num_particles",
    "native_blocks_per_population",
    "particle_batch_cells",
    "cell_set",
    "start_time",
    "eta",
    "guidance_strength",
    "normalize_counts",
    "selected_genes_sha256",
    "split_config_sha256",
    "upstream_revision",
)


def summarize_records(records, config, expected):
    observed, groups, ancestors, resampled = {}, {}, [], []
    particles, steps = config["num_particles"], config["start_time"]
    for record in records:
        group, name = record["group"], record["perturbation"]
        if group in groups or name not in expected:
            raise ValueError("Duplicate group or unexpected condition")
        valid, padded = record["valid_cells"], record["padded_population_cells"]
        if not 0 < valid <= padded:
            raise ValueError("Invalid valid/padded population sizes")
        histories = [record["ess"], record["resampled"], record["distinct_initial_ancestors"]]
        if any(len(history) != steps for history in histories):
            raise ValueError("Incomplete denoising history")
        if record["denoised_cell_steps"] != particles * padded * steps:
            raise ValueError("Recorded model work does not match particles, padding and steps")
        observed[name] = observed.get(name, 0) + valid
        groups[group] = [name, valid, padded, record["denoised_cell_steps"]]
        ancestors.append(record["distinct_initial_ancestors"][-1])
        resampled.append(sum(record["resampled"]))
    if any(count > expected[name] for name, count in observed.items()):
        raise ValueError("Recorded cells exceed the reference")
    covered = observed == expected
    return {
        "sampling_coverage_complete": covered,
        "not_a_final_evaluation": True,
        "completed_groups": len(groups),
        "valid_cells": sum(observed.values()),
        "expected_valid_cells": sum(expected.values()),
        "denoised_cell_steps": sum(x[3] for x in groups.values()),
        "padded_cells": sum(x[2] for x in groups.values()),
        "final_ancestors_mean": float(np.mean(ancestors)) if ancestors else None,
        "final_ancestors_min": min(ancestors) if ancestors else None,
        "resampling_events_mean": float(np.mean(resampled)) if resampled else None,
        "groups": groups,
    }


def read_run(root):
    config = json.loads((root / "run_config.json").read_text())
    real = ad.read_h5ad(config["reference"], backed="r")
    labels = real.obs.gene.astype(str)
    expected = labels[labels != "non-targeting"].value_counts().to_dict()
    real.file.close()
    records, skipped = [], []
    for path in sorted(root.glob("group_*.diagnostics.json")):
        try:
            records.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            # A running job may currently be writing its latest diagnostic file.
            skipped.append(path.name)
    summary = summarize_records(records, config, expected)
    summary["incomplete_json_files"] = skipped
    if skipped:
        summary["sampling_coverage_complete"] = False
    return config, summary


def compare_runs(first_config, first, second_config, second):
    differences = [key for key in MATCHED_SETTINGS if first_config[key] != second_config[key]]
    if differences:
        raise ValueError(f"Unmatched comparison settings: {differences}")
    shared = set(first["groups"]) & set(second["groups"])
    if any(first["groups"][key] != second["groups"][key] for key in shared):
        raise ValueError("Observed group metadata or denoising budgets differ")
    complete = first["sampling_coverage_complete"] and second["sampling_coverage_complete"]
    if complete and first["groups"] != second["groups"]:
        raise ValueError("Full runs have different group partitions")
    return {
        "full_budget_match_verified": complete,
        "shared_groups_checked": len(shared),
        "qualification": "Only recorded metadata and denoised cell-steps are compared. "
        "This is not proof of identical control-cell tensors, wall time, or biological quality.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--compare-shard-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config, summary = read_run(args.shard_root)
    output = {"settings": config, "snapshot": summary}
    if args.compare_shard_root:
        other_config, other = read_run(args.compare_shard_root)
        output["comparison"] = compare_runs(config, summary, other_config, other)
    write_manifest(args.out, output)
    print(
        f"WROTE sampling audit: {summary['valid_cells']}/{summary['expected_valid_cells']} "
        f"cells; full coverage={summary['sampling_coverage_complete']}; {args.out}"
    )


if __name__ == "__main__":
    main()
