"""Chunked H5AD expression access for PBMC and Tahoe-scale baselines."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


def _encoding(obj) -> str | None:
    value = obj.attrs.get("encoding-type")
    return value.decode() if isinstance(value, bytes) else value


def _shape(obj) -> tuple[int, int]:
    if isinstance(obj, h5py.Dataset):
        return tuple(int(value) for value in obj.shape)
    value = obj.attrs.get("shape")
    if value is None and "shape" in obj:
        value = obj["shape"][()]
    if value is None:
        raise ValueError(f"Cannot infer matrix shape for {obj.name}.")
    return int(value[0]), int(value[1])


def iter_h5ad_expression(
    path: str | Path,
    *,
    expression_key: str = "X_hvg",
    chunk_size: int = 8192,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield dense row chunks without loading a whole H5AD expression matrix."""
    with h5py.File(path, "r") as handle:
        obj = handle["X"] if expression_key == "X" else handle["obsm"][expression_key]
        n_rows, n_columns = _shape(obj)
        encoding = _encoding(obj)
        csc_matrix = None
        if isinstance(obj, h5py.Group) and encoding == "csc_matrix":
            # CSC does not permit efficient HDF5 row slicing. Convert it once,
            # rather than rescanning every nonzero entry for every row chunk.
            csc_matrix = sparse.csc_matrix(
                (
                    np.asarray(obj["data"]),
                    np.asarray(obj["indices"]),
                    np.asarray(obj["indptr"]),
                ),
                shape=(n_rows, n_columns),
            ).tocsr()

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            if isinstance(obj, h5py.Dataset) or encoding == "array":
                values = np.asarray(obj[start:stop], dtype=np.float32)
            elif isinstance(obj, h5py.Group) and encoding == "csr_matrix":
                indptr = np.asarray(obj["indptr"][start : stop + 1], dtype=np.int64)
                data_start, data_stop = int(indptr[0]), int(indptr[-1])
                data = np.asarray(obj["data"][data_start:data_stop])
                indices = np.asarray(obj["indices"][data_start:data_stop])
                local_indptr = indptr - data_start
                values = sparse.csr_matrix(
                    (data, indices, local_indptr),
                    shape=(stop - start, n_columns),
                ).toarray()
                values = values.astype(np.float32, copy=False)
            elif csc_matrix is not None:
                values = csc_matrix[start:stop].toarray().astype(np.float32, copy=False)
            else:
                raise ValueError(
                    f"Unsupported H5AD encoding {encoding!r} for matrix {obj.name}."
                )
            yield start, stop, values
