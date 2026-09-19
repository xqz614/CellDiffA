import numpy as np
import pandas as pd
import pytest
import torch
from anndata import AnnData

from celldiffa.benchmark.grouped_loss import ScouterGroupedLoss
from scripts.baselines.run_scouter_replogle import encode_conditions, training_nonzero_genes


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_vectorized_scouter_loss_and_gradient_match_condition_loop(device, gamma):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    torch.manual_seed(123)
    prediction = torch.randn(7, 5, device=device, requires_grad=True)
    target = torch.randn(7, 5, device=device)
    control = torch.randn(7, 5, device=device)
    groups = np.array(["A", "B", "A", "ctrl", "B", "B", "C"])
    indices = {"A": [0, 2], "B": [1, 3, 4], "ctrl": [0, 1, 2, 3, 4], "C": [4]}
    per_gene = (target - prediction).abs().pow(2 + gamma)
    per_gene = (
        per_gene + 0.5 * (torch.sign(target - control) - torch.sign(prediction - control)).square()
    )
    expected = sum(
        per_gene[np.flatnonzero(groups == name).tolist()][:, indices[name]].mean()
        for name in np.unique(groups)
    ) / len(np.unique(groups))
    actual = ScouterGroupedLoss(indices, 5, device, gamma=gamma)(
        prediction, target, control, groups
    )
    torch.testing.assert_close(actual, expected)
    expected_gradient = torch.autograd.grad(expected, prediction, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(actual, prediction)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_scouter_encoding_does_not_drop_conditions():
    data = AnnData(
        np.array([[0.0, 1.0], [2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame({"gene": ["non-targeting", "A", "B"]}, index=["a", "b", "c"]),
    )
    embeddings = {name: np.ones(3) for name in ["A", "B", "non-targeting"]}
    result = encode_conditions(data, embeddings)
    assert result.shape == (3, 3)
    assert data.n_obs == 3
    assert data.obs["condition"].tolist() == ["ctrl", "A", "B"]
    assert data.obs["embd_index"].tolist() == [[2], [0], [1]]
    selected = training_nonzero_genes(data)
    assert selected["A"].tolist() == [0]
    assert selected["B"].tolist() == [1]


def test_scouter_missing_embedding_is_an_error_not_silent_filtering():
    data = AnnData(np.zeros((1, 2)), obs=pd.DataFrame({"gene": ["unseen"]}))
    with pytest.raises(ValueError, match="missing perturbations"):
        encode_conditions(data, {"A": np.ones(3)})
    assert data.n_obs == 1
