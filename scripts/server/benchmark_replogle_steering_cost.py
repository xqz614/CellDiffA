#!/usr/bin/env python
"""Sequential timing on an idle GPU; never launch beside formal model jobs."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from celldiffa.benchmark.backbone_experiments import atomic_json
from scripts.server.replogle_remaining import command_for, environment, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--groups", type=int, default=12)
    args = parser.parse_args()
    if args.groups < 3:
        parser.error("At least 3 groups required, including one discarded warm-up group")
    if args.outdir.exists():
        raise FileExistsError("Use a new timing directory")
    config = json.loads(args.plan.read_text())
    config.update(output_root=str(args.outdir.resolve()), gpus=[args.gpu])
    env, gpu = environment(config, 0)
    env["PATH"] = str(Path(config["python"]).parent) + ":" + env.get("PATH", "")
    cases = [
        dict(id="random16", kind="perturbdiff", mode="random"),
        dict(id="best16", kind="perturbdiff", mode="best_of_n"),
        dict(id="cellwise16", kind="perturbdiff", mode="smc", unit="cell"),
    ]
    cases += [
        dict(id=f"adacell{k}", kind="perturbdiff", mode="smc", particles=k) for k in (4, 8, 16)
    ]
    summaries, groups_reference = [], None
    for case in cases:
        users = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if users:
            raise RuntimeError(
                f"GPU {gpu} is not idle. Finish its other jobs first; found: {users}"
            )
        command, changes, output, _ = command_for(config, case, gpu)
        command[-1] = str(args.groups)
        execute(command, config, {**env, **changes}, output / "run.log")
        details = [
            json.loads(path.read_text())
            for path in sorted((output / "shards").glob("group_*.diagnostics.json"))
        ]
        if len(details) != args.groups:
            raise ValueError("Timing run did not complete the planned groups")
        signature = [
            (row["group"], row["perturbation"], row["valid_cells"], row["padded_population_cells"])
            for row in details
        ]
        if groups_reference is not None and signature != groups_reference:
            raise ValueError("Timing comparison changed population composition")
        groups_reference = signature
        timed = details[1:]  # Discard warm-up; timings include rewards and selection.
        seconds = sum(row["sampling_seconds"] for row in timed)
        cells = sum(row["valid_cells"] for row in timed)
        summaries.append(
            dict(
                method=case["id"],
                timed_groups=len(timed),
                cells=cells,
                seconds=seconds,
                seconds_per_1000_cells=seconds / cells * 1000,
                denoised_cell_steps=sum(row["denoised_cell_steps"] for row in timed),
            )
        )
        atomic_json(
            args.outdir / "timings.json",
            dict(
                gpu=gpu,
                results=summaries,
                checked_idle_before_each_job=True,
                interpretation=(
                    "Repeatedly time the same groups; one discarded warm-up group per method. "
                    "Sampling latency, excluding training, loading and evaluation; "
                    "not whole-test runtime."
                ),
            ),
        )
    import pandas as pd

    table = pd.DataFrame(summaries)
    table.to_csv(args.outdir / "timings.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(5.8, 2.8), constrained_layout=True)
    axis.bar(table.method, table.seconds_per_1000_cells, color="#4777a5")
    axis.set_ylabel("Sampling seconds / 1,000 output cells")
    axis.tick_params(axis="x", rotation=25)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.outdir / "sampling_cost.pdf")
    fig.savefig(args.outdir / "sampling_cost.png", dpi=240)
    plt.close(fig)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
