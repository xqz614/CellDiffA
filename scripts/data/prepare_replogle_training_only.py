#!/usr/bin/env python
"""Add a training reference without touching completed validation/test exports."""

import argparse
import fcntl
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    import anndata as ad
    import numpy as np

    from celldiffa.benchmark.artifacts import load_selected_genes, sha256_file
    from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
    from celldiffa.benchmark.streaming import iter_h5ad_expression, read_h5ad_obs, read_h5ad_var

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--selected-genes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        split = PerturbDiffSplit.from_yaml(args.split_config)
        obs = read_h5ad_obs(args.source)
        genes = load_selected_genes(args.selected_genes)
        mask = split.masks(obs)["train"]
        if args.output.exists():
            existing = ad.read_h5ad(args.output, backed="r")
            try:
                if (
                    list(existing.var_names) != genes
                    or list(existing.obs_names) != list(obs.index[mask])
                    or not split.masks(existing.obs)["train"].all()
                ):
                    raise ValueError("Existing training export differs; refusing overwrite")
            finally:
                existing.file.close()
            print(f"VERIFIED existing training reference: {args.output}")
            return
        var = read_h5ad_var(args.source)
        if "highly_variable" not in var or var.index[var.highly_variable].tolist() != genes:
            raise ValueError("Published HVG order does not match source")
        values = np.empty((int(mask.sum()), len(genes)), dtype=np.float32)
        offset = 0
        for start, stop, chunk in iter_h5ad_expression(args.source):
            selected = chunk[mask[start:stop]]
            values[offset : offset + len(selected)] = selected
            offset += len(selected)
        if offset != len(values):
            raise ValueError("Incomplete training export")
        data = ad.AnnData(values, obs=obs.loc[mask].copy())
        data.var_names = genes
        data.uns["split_config_sha256"] = sha256_file(args.split_config)
        data.uns["reference_split"] = "train"
        temporary = args.output.with_suffix(".partial.h5ad")
        data.write_h5ad(temporary, compression="gzip")
        temporary.replace(args.output)
        print(f"WROTE {args.output}: {data.shape}")


if __name__ == "__main__":
    main()
