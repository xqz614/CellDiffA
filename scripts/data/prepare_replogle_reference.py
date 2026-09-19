#!/usr/bin/env python
"""Export immutable official validation/test references without fitting on them."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from celldiffa.benchmark.artifacts import load_selected_genes, sha256_file
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import iter_h5ad_expression, read_h5ad_obs, read_h5ad_var


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--selected-genes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-training", action="store_true")
    args = parser.parse_args()
    split = PerturbDiffSplit.from_yaml(args.split_config)
    obs = read_h5ad_obs(args.source)
    genes = load_selected_genes(args.selected_genes)
    var = read_h5ad_var(args.source)
    if "highly_variable" not in var or var.index[var.highly_variable].tolist() != genes:
        raise ValueError("Source HVG metadata and published ordered genes disagree.")
    masks = split.masks(obs)
    controls = (
        obs[split.pert_col].astype(str).eq(split.control_pert)
        & obs[split.context_col].astype(str).isin(split.holdout_contexts)
    ).to_numpy()
    # Controls are observed baseline data; shared controls are explicitly documented.
    selected = {name: masks[name] | controls for name in ("validation", "test")}
    if args.with_training:
        selected["train"] = masks["train"]
    arrays = {
        name: np.empty((int(mask.sum()), len(genes)), dtype=np.float32)
        for name, mask in selected.items()
    }
    offsets = {name: 0 for name in selected}
    for start, stop, values in iter_h5ad_expression(args.source):
        if values.shape[1] != len(genes):
            raise ValueError("Source X_hvg does not match published evaluation genes.")
        for name, mask in selected.items():
            chunk = values[mask[start:stop]]
            offset = offsets[name]
            arrays[name][offset : offset + len(chunk)] = chunk
            offsets[name] += len(chunk)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        path = args.output_dir / ("real.h5ad" if name == "test" else f"{name}.h5ad")
        if path.exists():
            raise FileExistsError(f"Not overwriting reference: {path}")
        data = ad.AnnData(X=values, obs=obs.loc[selected[name]].copy())
        data.var_names = genes
        data.uns["reference_split"] = name
        data.uns["split_config_sha256"] = sha256_file(args.split_config)
        data.uns["control_policy"] = "observed controls from held-out context"
        if name == "test":
            split.validate_real_test(data)
        data.write_h5ad(path, compression="gzip")
        print(f"WROTE {path}: {data.n_obs} cells x {data.n_vars} genes", flush=True)
        if name == "test":
            ctrl_data = data[data.obs[split.pert_col].astype(str).eq(split.control_pert)].copy()
            ctrl_data.X = sparse.csr_matrix(ctrl_data.X)
            ctrl_data.var["highly_variable"] = True
            ctrl_data.write_h5ad(args.output_dir / "controls.h5ad", compression="gzip")
    report = {
        "source": str(args.source.resolve()),
        "source_bytes": args.source.stat().st_size,
        "split_config_sha256": sha256_file(args.split_config),
        "selected_genes_sha256": sha256_file(args.selected_genes),
        "rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "shared_observed_control_rows": int(controls.sum()),
        "contexts": obs[split.context_col].value_counts().to_dict(),
        "evaluation_genes": len(genes),
        "tuning_policy": "validation only; never use test response scores to choose settings",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
