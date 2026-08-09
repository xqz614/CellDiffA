"""Small helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path


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


def write_manifest(path: str | Path, values: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")
