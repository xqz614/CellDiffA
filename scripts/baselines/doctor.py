#!/usr/bin/env python
"""Check data paths and reject unsupported model/dataset combinations."""

import argparse
from pathlib import Path

from celldiffa.benchmark.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--datasets-config", default="configs/benchmark/datasets.yaml")
    parser.add_argument("--baselines-config", default="configs/benchmark/baselines.yaml")
    parser.add_argument("--allow-missing-data", action="store_true")
    args = parser.parse_args()

    datasets = load_yaml(args.datasets_config)["datasets"]
    baselines = load_yaml(args.baselines_config)["baselines"]
    if args.dataset not in datasets:
        raise SystemExit(f"Unknown dataset {args.dataset!r}; choose from {sorted(datasets)}")
    spec = datasets[args.dataset]
    data_path = Path(spec["path"])
    if not data_path.exists() and not args.allow_missing_data:
        raise SystemExit(f"Dataset is missing: {data_path}")

    if args.baseline:
        if args.baseline not in baselines:
            raise SystemExit(f"Unknown baseline {args.baseline!r}; choose from {sorted(baselines)}")
        baseline = baselines[args.baseline]
        if args.dataset not in baseline["datasets"]:
            raise SystemExit(
                f"{args.baseline} is not a valid predictor for {args.dataset}. "
                f"Supported: {baseline['datasets']}"
            )
    data_status = "present" if data_path.exists() else "missing"
    print(f"dataset={args.dataset} path={data_path} status={data_status}")
    if args.baseline:
        baseline = baselines[args.baseline]
        print(
            f"baseline={args.baseline} tier={baseline['tier']} "
            f"status=applicable runner={baseline['runner']}"
        )


if __name__ == "__main__":
    main()
