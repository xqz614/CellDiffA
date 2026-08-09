"""Resumable CellDiffA prediction shards and strict final assembly."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

from .contracts import build_prediction_anndata


def shard_path(root: str | Path, group_index: int) -> Path:
    return Path(root) / f"group_{group_index:07d}.npz"


def save_group_shard(
    root: str | Path,
    group_index: int,
    perturbation: str,
    values: np.ndarray,
) -> Path:
    """Atomically save one official cell-set prediction."""
    path = shard_path(root, group_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("A prediction shard must be a finite cells-by-genes matrix.")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        group_index=np.asarray(group_index),
        perturbation=np.asarray(perturbation),
        values=values,
    )
    temporary.replace(path)
    return path


def load_group_shard(path: str | Path) -> tuple[int, str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as shard:
        index = int(shard["group_index"].item())
        perturbation = str(shard["perturbation"].item())
        values = shard["values"].astype(np.float32, copy=False)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"Invalid prediction shard: {path}")
    return index, perturbation, values


def assemble_replogle_shards(
    real_path: str | Path,
    shard_root: str | Path,
    output_path: str | Path,
    *,
    pert_col: str,
    control_pert: str,
    require_complete: bool = True,
    write_output: bool = True,
) -> tuple[ad.AnnData | None, dict]:
    """Assemble shards only when every official treated cell is present."""
    real = ad.read_h5ad(real_path)
    labels = real.obs[pert_col].astype(str).to_numpy()
    expected = {pert: int(np.sum(labels == pert)) for pert in sorted(set(labels) - {control_pert})}
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    seen_indices: set[int] = set()
    for path in sorted(Path(shard_root).glob("group_*.npz")):
        index, perturbation, values = load_group_shard(path)
        if index in seen_indices:
            raise ValueError(f"Duplicate group index {index} in {path}.")
        seen_indices.add(index)
        if values.shape[1] != real.n_vars:
            raise ValueError(f"Shard {path} has {values.shape[1]} genes; expected {real.n_vars}.")
        grouped.setdefault(perturbation, []).append((index, values))

    observed = {
        pert: sum(values.shape[0] for _, values in groups) for pert, groups in grouped.items()
    }
    extras = sorted(set(observed) - set(expected))
    overfull = {pert: observed[pert] for pert in expected if observed.get(pert, 0) > expected[pert]}
    if extras or overfull:
        raise ValueError(f"Invalid shards; extra perturbations={extras}, overfull={overfull}.")
    missing = {
        pert: expected[pert] - observed.get(pert, 0)
        for pert in expected
        if observed.get(pert, 0) != expected[pert]
    }
    status = {
        "complete": not missing,
        "groups": len(seen_indices),
        "expected_cells": expected,
        "observed_cells": observed,
        "missing_cells": missing,
    }
    if missing:
        if require_complete:
            raise RuntimeError(
                "CellDiffA shards are incomplete; no evaluator-ready H5AD was written. "
                f"Missing cells for {len(missing)} perturbations."
            )
        return None, status
    if not write_output:
        return None, status

    predictions = {
        pert: np.concatenate(
            [values for _, values in sorted(grouped[pert], key=lambda item: item[0])],
            axis=0,
        )
        for pert in expected
    }
    pred = build_prediction_anndata(
        real,
        predictions,
        pert_col=pert_col,
        control_pert=control_pert,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred.write_h5ad(output_path, compression="gzip")
    return pred, status
