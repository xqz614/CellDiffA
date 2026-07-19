import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from celldiffa.benchmark.streaming import iter_h5ad_expression


@pytest.mark.parametrize("matrix", [np.arange(30).reshape(10, 3), sparse.csr_matrix(np.eye(10, 3))])
def test_stream_h5ad_obsm_dense_and_csr(tmp_path, matrix):
    adata = ad.AnnData(
        X=sparse.csr_matrix((10, 2)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(10)]),
        var=pd.DataFrame(index=["unused1", "unused2"]),
    )
    adata.obsm["X_hvg"] = matrix
    path = tmp_path / "data.h5ad"
    adata.write_h5ad(path)

    chunks = list(iter_h5ad_expression(path, expression_key="X_hvg", chunk_size=4))
    observed = np.concatenate([values for _, _, values in chunks])
    expected = matrix.toarray() if sparse.issparse(matrix) else matrix
    np.testing.assert_array_equal(observed, expected)
    assert [(start, stop) for start, stop, _ in chunks] == [(0, 4), (4, 8), (8, 10)]
