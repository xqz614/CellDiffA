"""Shared contracts for independent diffusion backbones and their controls."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from celldiffa.rewards import AnchorReward, CompositeReward, GeometricReward, TranscriptomicReward
from celldiffa.rewards.cellwise import IndependentCellReward


def dense(values):
    return np.asarray(values.toarray() if sparse.issparse(values) else values, dtype=np.float32)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def make_reward(
    priors,
    controls,
    *,
    weights=(1, 1, 1),
    unit="population",
    normalization="zscore",
    expression_scale=10.0,
):
    if unit not in {"population", "cell"}:
        raise ValueError("Unknown reward unit")
    if expression_scale <= 0:
        raise ValueError("Reward expression scale must be positive")
    ctrl_mean = dense(controls).mean(axis=0) / expression_scale
    shifts = {name: value / expression_scale for name, value in priors.shifts.items()}
    cls = CompositeReward if unit == "population" else IndependentCellReward
    return cls(
        [
            TranscriptomicReward(
                priors.de_genes, shifts, ctrl_mean, list(priors.genes), weight=weights[0], top_k=20
            ),
            GeometricReward(shifts, ctrl_mean, weight=weights[1]),
            AnchorReward(shifts, weight=weights[2], bandwidth=1.0),
        ],
        normalization=normalization,
    )


def expression_conditions(data, embeddings, control_means, *, control="non-targeting"):
    """Per-row descriptors and observed context means; no paired outcomes."""
    width = len(next(iter(embeddings.values())))
    descriptor_names = sorted(set(data.obs.gene.astype(str)))
    vectors = []
    for name in descriptor_names:
        if name == control:
            vectors.append(np.zeros(width, dtype=np.float32))
        elif name not in embeddings:
            raise ValueError(f"No perturbation descriptor for {name}")
        else:
            vector = np.asarray(embeddings[name], dtype=np.float32)
            norm = np.linalg.norm(vector)
            if not np.isfinite(vector).all() or norm <= 0:
                raise ValueError(f"Invalid descriptor for {name}")
            vectors.append(vector / norm)
    context_names = sorted(set(data.obs.cell_line.astype(str)))
    if set(context_names) - set(control_means):
        raise ValueError("No observed controls for one or more query contexts")
    # Return compact lookup tables rather than one embedding copy per cell.
    return dict(
        descriptors=np.stack(vectors),
        descriptor_index=np.array([descriptor_names.index(x) for x in data.obs.gene.astype(str)]),
        controls=np.stack([control_means[x] for x in context_names]),
        control_index=np.array([context_names.index(x) for x in data.obs.cell_line.astype(str)]),
    )


def training_control_means(train, control):
    labels, contexts = train.obs.gene.astype(str), train.obs.cell_line.astype(str)
    result = {}
    for context in sorted(set(contexts)):
        rows = (contexts == context) & (labels == control)
        if not rows.any():
            raise ValueError(f"Training context {context} has no observed controls")
        result[context] = dense(train.X[rows.to_numpy()]).mean(axis=0)
    return result


def plan_groups(obs, *, control, population_cells):
    if population_cells < 2:
        raise ValueError("Population size must be at least two")
    groups = []
    for (pert, context), rows in obs.groupby(["gene", "cell_line"], observed=True, sort=True):
        if str(pert) == control:
            continue
        remaining = len(rows)
        while remaining:
            count = min(remaining, population_cells)
            groups.append(
                dict(
                    index=len(groups),
                    perturbation=str(pert),
                    context=str(context),
                    cells=count,
                    sampled_cells=max(2, count),
                )
            )
            remaining -= count
    return groups


def load_weights(path):
    """Only accept tensor state dictionaries; never fall back to unsafe pickle."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(v, torch.Tensor) for v in state.values())
    ):
        raise ValueError("Expected a tensor state_dict checkpoint")
    return state
