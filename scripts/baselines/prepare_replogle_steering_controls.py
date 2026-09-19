#!/usr/bin/env python
"""Prepare auditable validation commands without starting any model job.

All cases retain the same backbone, candidates, native blocks and denoising
schedule. Equal planned model work is not a measured wall-time equality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
from pathlib import Path

ENVIRONMENT_KEYS = {
    "num_particles": "CELLDIFFA_NUM_PARTICLES",
    "native_blocks_per_population": "CELLDIFFA_NATIVE_BLOCKS_PER_POPULATION",
    "particle_batch_cells": "CELLDIFFA_PARTICLE_BATCH_CELLS",
    "seed": "CELLDIFFA_SEED",
    "alpha": "CELLDIFFA_ALPHA",
    "alignment_mode": "CELLDIFFA_ALIGNMENT_MODE",
    "reward_normalization": "CELLDIFFA_REWARD_NORMALIZATION",
    "ess_threshold": "CELLDIFFA_ESS_THRESHOLD",
    "signature_weight": "CELLDIFFA_SIGNATURE_WEIGHT",
    "direction_weight": "CELLDIFFA_DIRECTION_WEIGHT",
    "anchor_weight": "CELLDIFFA_ANCHOR_WEIGHT",
    "anchor_bandwidth": "CELLDIFFA_ANCHOR_BANDWIDTH",
    "prior_ridge": "CELLDIFFA_PRIOR_RIDGE",
}
BUDGET_KEYS = ("num_particles", "native_blocks_per_population", "particle_batch_cells", "seed")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def build_plan(spec, *, repo, variant, device, environment):
    if spec["evaluation_split"] != "validation":
        raise ValueError("This preparatory plan must use validation, never test outcomes")
    if variant not in {"scratch", "finetuned"}:
        raise ValueError("Unknown released backbone")
    repo = Path(repo).resolve()
    reference = repo / "results/replogle/reference/validation.h5ad"
    shared = spec["shared"]
    if set(shared) != set(ENVIRONMENT_KEYS):
        raise ValueError("All steering settings must be explicit in the shared configuration")
    cases, seen = [], set()
    for case in spec["cases"]:
        name = case["id"]
        if not re.fullmatch(r"[a-z0-9_]+", name) or name in seen:
            raise ValueError("Case names must be unique safe identifiers")
        seen.add(name)
        if set(case["overrides"]) - set(shared):
            raise ValueError("Unknown steering setting")
        settings = {**shared, **case["overrides"]}
        if any(settings[key] != shared[key] for key in BUDGET_KEYS):
            raise ValueError("Compute-matched controls must preserve budget and seed")
        if settings["alpha"] <= 0 or settings["alignment_mode"] not in {
            "smc",
            "random",
            "best_of_n",
        }:
            raise ValueError("Invalid temperature or selection mode")
        output_name = (
            f"adacell_{variant}_alpha1" if name == "adacell_alpha1" else f"{name}_{variant}"
        )
        output = repo / "results/replogle/validation" / output_name
        env = {
            "CELLDIFFA_DATA_ROOT": str(repo / "data"),
            "CELLDIFFA_DEVICE": device,
            "CELLDIFFA_EVALUATION_SPLIT": "validation",
            "CELLDIFFA_REAL_TEST": str(reference),
            **{ENVIRONMENT_KEYS[key]: str(value) for key, value in settings.items()},
        }
        command = [
            "env",
            *[f"{key}={value}" for key, value in env.items()],
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            environment,
            "bash",
            str(repo / "scripts/baselines/run_celldiffa_replogle.sh"),
            variant,
            str(output),
            "0",
            "0",
            "1",
            "all",
            str(repo / "external/PerturbDiff"),
        ]
        cases.append(
            {
                "id": name,
                "role": case["role"],
                "settings": settings,
                "output": str(output),
                "command": command,
                "prediction": str(output / f"celldiffa_{variant}.h5ad"),
                "status": "prepared, not executed by this planner",
            }
        )
    if not set(spec["selection"]["eligible_cases"]).issubset(seen):
        raise ValueError("Selection candidates must be present")
    return {
        "version": spec["version"],
        "evaluation_split": "validation",
        "reference": str(reference),
        "variant": variant,
        "cases": cases,
        "selection": spec["selection"],
        "budget": {
            "reverse_steps": 100,
            "eta": 0.0,
            "guidance_strength": 1.0,
            "native_cell_set": 32,
            "particles": shared["num_particles"],
            "measure": "sum of denoised_cell_steps in per-group diagnostics, including padding",
            "verification": (
                "Compare identical group IDs, perturbations, valid/padded cell counts and total "
                "denoised_cell_steps after complete runs; report wall time separately."
            ),
            "qualification": (
                "Released single-sample PerturbDiff is a quality reference, not the "
                "equal-16-candidate compute control. random keeps the first exchangeable "
                "candidate without reward selection. All current engine modes still evaluate "
                "rewards at every step."
            ),
        },
        "test_policy": (
            "No test command is generated. Lock validation selection first; replicate selected "
            "settings across test controls and ablations, then evaluate all test conditions "
            "once per locked seed."
        ),
    }


def write_unchanged_or_new(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f"Refusing to replace a different plan: {path}")
        return
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/benchmark/replogle_steering_controls.json")
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--variant", choices=["scratch", "finetuned"], default="scratch")
    parser.add_argument("--device", choices=["mps", "cpu", "cuda:0"], default="mps")
    parser.add_argument("--environment", default="adacell-replogle")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    plan = build_plan(
        spec, repo=args.repo, variant=args.variant, device=args.device, environment=args.environment
    )
    plan["spec_sha256"] = digest(args.spec)
    plan["reference_sha256"] = digest(plan["reference"])
    plan["runner_sha256"] = digest(args.repo / "scripts/baselines/run_celldiffa_replogle.sh")
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_unchanged_or_new(args.outdir / "plan.json", json.dumps(plan, indent=2) + "\n")
    # Commands remain text, not an auto-starting queue. Copy only the selected case.
    lines = [
        "# Validation commands only. No experiment was launched by this planner.",
        "# Activate a working conda installation before copying a command.",
    ]
    for case in plan["cases"]:
        lines.extend(["", f"# {case['id']} ({case['role']})", shlex.join(case["command"])])
    write_unchanged_or_new(args.outdir / "commands.txt", "\n".join(lines) + "\n")
    print(f"Prepared {len(plan['cases'])} validation cases; no model job started: {args.outdir}")


if __name__ == "__main__":
    main()
