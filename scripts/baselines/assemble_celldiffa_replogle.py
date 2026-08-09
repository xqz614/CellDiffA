#!/usr/bin/env python
"""Assemble completed Replogle CellDiffA shards into evaluator-ready H5AD."""

from __future__ import annotations

import argparse
import json

from celldiffa.benchmark.replogle_shards import assemble_replogle_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-test", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pert-col", default="gene")
    parser.add_argument("--control-pert", default="non-targeting")
    args = parser.parse_args()
    _, status = assemble_replogle_shards(
        args.real_test,
        args.shard_root,
        args.output,
        pert_col=args.pert_col,
        control_pert=args.control_pert,
        require_complete=True,
    )
    print(json.dumps(status, indent=2, sort_keys=True))
    print(f"Wrote evaluator-ready CellDiffA prediction: {args.output}")


if __name__ == "__main__":
    main()
