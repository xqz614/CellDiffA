import numpy as np
import pytest
from scipy import sparse

from scripts.baselines.run_gears_replogle_local import coexpression_graph, dense


@pytest.mark.parametrize("sparse_input", [False, True])
def test_chunked_gears_graph_matches_dense_pearson(sparse_input):
    rng = np.random.default_rng(5)
    values = rng.normal(size=(70, 25))
    values[:, 2] = values[:, 0] + 0.05 * rng.normal(size=70)
    genes = [f"G{i}" for i in range(25)]
    actual = coexpression_graph(
        sparse.csr_matrix(values) if sparse_input else values, genes, chunk_size=13
    )
    correlation = np.abs(np.corrcoef(values.T))
    expected = {
        (genes[j], genes[i]): correlation[i, j]
        for i in range(25)
        for j in np.argsort(correlation[i])[-21:]
        if correlation[i, j] > 0.4
    }
    observed = {(row.source, row.target): row.importance for row in actual.itertuples()}
    assert observed.keys() == expected.keys()
    for key in expected:
        assert observed[key] == pytest.approx(expected[key], abs=1e-12)


def test_graph_dense_conversion_preserves_row_values():
    values = np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(dense(values), values)
    np.testing.assert_array_equal(dense(sparse.csr_matrix(values)), values)
