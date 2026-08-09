#!/usr/bin/env python
"""Compute all PerturbDiff Mean variants without loading the source data."""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np

from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import iter_h5ad_expression


def _update(store, keys, values):
    for key in np.unique(keys):
        selected = values[keys == key]
        current_sum, current_count = store[str(key)]
        store[str(key)] = (current_sum + selected.sum(axis=0), current_count + len(selected))


def _means(store):
    return {key: total / count for key, (total, count) in store.items() if count}


def _source_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    files = sorted(source.glob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No H5AD files found in {source}.")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Official source H5AD or Tahoe directory")
    parser.add_argument("--real-test", required=True)
    parser.add_argument("--upstream-split-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expression-key", default="X_hvg")
    parser.add_argument("--selected-genes", help="Official selected-gene pickle")
    parser.add_argument("--split-axis", choices=["batch", "context"], required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()

    split = PerturbDiffSplit.from_yaml(
        args.upstream_split_config,
        split_axis=args.split_axis,
    )
    pert_col = split.pert_col
    control = split.control_pert
    context_col = split.context_col
    batch_col = split.batch_col

    stores = {
        "perturbation": defaultdict(lambda: (0.0, 0)),
        "cell_type": defaultdict(lambda: (0.0, 0)),
        "batch": defaultdict(lambda: (0.0, 0)),
        "overall": defaultdict(lambda: (0.0, 0)),
    }
    n_genes = None
    for path in _source_files(Path(args.source)):
        backed = ad.read_h5ad(path, backed="r")
        obs = backed.obs
        required = {pert_col, context_col, batch_col} - {None}
        missing = required - set(obs.columns)
        if missing:
            raise ValueError(f"{path} is missing obs columns: {sorted(missing)}")
        labels = obs[pert_col].astype(str).to_numpy()
        contexts_all = obs[context_col].astype(str).to_numpy()
        batches_all = obs[batch_col].astype(str).to_numpy()
        training = split.masks(obs, split_axis=args.split_axis)["train"]
        treated_training = training & (labels != control)

        for start, stop, values in iter_h5ad_expression(
            path,
            expression_key=args.expression_key,
            chunk_size=args.chunk_size,
        ):
            n_genes = values.shape[1]
            local_train = training[start:stop]
            local_treated = treated_training[start:stop]
            _update(stores["perturbation"], labels[start:stop][local_train], values[local_train])
            _update(
                stores["cell_type"],
                contexts_all[start:stop][local_treated],
                values[local_treated],
            )
            _update(
                stores["batch"],
                batches_all[start:stop][local_treated],
                values[local_treated],
            )
            _update(
                stores["overall"],
                np.repeat("overall", int(local_treated.sum())),
                values[local_treated],
            )
        backed.file.close()

    real = ad.read_h5ad(args.real_test)
    if n_genes != real.n_vars:
        raise ValueError(f"Source has {n_genes} expression genes; real test has {real.n_vars}.")
    if args.selected_genes:
        with Path(args.selected_genes).open("rb") as handle:
            genes = list(pickle.load(handle))
        if genes != list(real.var_names):
            raise ValueError("Selected-gene pickle order differs from the real test H5AD.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_labels = real.obs[pert_col].astype(str).to_numpy()
    for variant, store in stores.items():
        group_means = _means(store)
        if variant == "perturbation":
            shared_mean = np.stack(list(group_means.values())).mean(axis=0)
        elif variant == "overall":
            shared_mean = group_means["overall"]
        predictions = {}
        for pert in np.unique(real_labels):
            if pert == control:
                continue
            positions = np.flatnonzero(real_labels == pert)
            if variant in {"perturbation", "overall"}:
                values = np.repeat(shared_mean[None, :], len(positions), axis=0)
            else:
                column = context_col if variant == "cell_type" else batch_col
                contexts = real.obs.iloc[positions][column].astype(str).to_numpy()
                values = np.empty((len(positions), real.n_vars), dtype=np.float32)
                for context in np.unique(contexts):
                    if context not in group_means:
                        raise ValueError(f"No training mean for {column}={context!r}.")
                    values[contexts == context] = group_means[context]
            predictions[pert] = values
        pred = build_prediction_anndata(
            real,
            predictions,
            pert_col=pert_col,
            control_pert=control,
        )
        pred.write_h5ad(output_dir / f"mean_{variant}.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
