"""Small helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_genes(path: str | Path) -> list[str]:
    with Path(path).open("rb") as handle:
        values = pickle.load(handle)
    genes = [str(value) for value in values]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Selected-gene pickle must contain a non-empty, unique gene list.")
    return genes


def load_embedding_dict(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate an upstream ``name -> vector`` pickle."""
    with Path(path).open("rb") as handle:
        values = pickle.load(handle)
    if not isinstance(values, Mapping) or not values:
        raise ValueError("Embedding pickle must contain a non-empty mapping.")
    result: dict[str, np.ndarray] = {}
    dimension = None
    for raw_name, raw_vector in values.items():
        name = str(raw_name)
        vector = np.asarray(raw_vector, dtype=np.float64).reshape(-1)
        if not name or not len(vector) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid embedding for {name!r}.")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ValueError(
                f"Embedding {name!r} has dimension {len(vector)}; expected {dimension}."
            )
        if name in result:
            raise ValueError(f"Duplicate embedding name {name!r}.")
        result[name] = vector
    return result


def write_manifest(path: str | Path, values: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")
