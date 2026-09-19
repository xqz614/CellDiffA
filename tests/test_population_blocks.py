import numpy as np
import pytest
import torch

from celldiffa.benchmark.population_blocks import BlockPopulationSampler, pack_native_groups


class NativeSampler:
    num_timesteps = 3

    def __init__(self):
        self.shapes = []

    def denoise_step(self, x_t, t, condition, prev_pred=None):
        self.shapes.append(tuple(x_t.shape))
        assert x_t.shape[1] == 3
        assert condition["ds_name"] == ["replogle"] * len(x_t)
        result = x_t + x_t.mean(dim=1, keepdim=True) + condition["cont_emb"]
        return {"x_prev": result, "x0_pred": result + prev_pred + t[:, None, None]}


@pytest.mark.parametrize("batch_cells", [3, 9, 48])
def test_block_wrapper_preserves_native_attention_and_covariates(batch_cells):
    source = NativeSampler()
    control = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    condition = {"cont_emb": control, "ds_name": ["replogle"] * 2, "gene_emb": None}
    wrapper = BlockPopulationSampler(source, condition, cells_per_block=3, batch_cells=batch_cells)
    x = torch.arange(48, dtype=torch.float32).reshape(4, 6, 2)
    prev = x / 10
    t = torch.tensor([2, 1, 0, 2])
    result = wrapper.denoise_step(x, t, {}, prev)
    expected = source.denoise_step(
        x.reshape(8, 3, 2),
        t.repeat_interleave(2),
        {"cont_emb": control.repeat(4, 1, 1), "ds_name": ["replogle"] * 8},
        prev.reshape(8, 3, 2),
    )
    for key in result:
        torch.testing.assert_close(result[key], expected[key].reshape(4, 6, 2))


def test_pack_preserves_padding_and_every_native_control():
    groups = [
        dict(
            genes=["A", "B"],
            condition={
                "cont_emb": torch.full((1, 3, 2), float(i)),
                "ds_name": ["replogle"],
                "gene_emb": None,
            },
            mask=torch.tensor([True, False, True]),
            covariates=(np.array(["P", "P"]), np.array(["hepg2"] * 2), np.array([str(i)] * 2)),
        )
        for i in range(2)
    ]
    batch, flat, mask, covariates, native = pack_native_groups(groups)
    assert batch["col_genes"] == [["A", "B"]]
    assert mask.tolist() == [[True, False, True, True, False, True]]
    assert flat["cont_emb"].shape == (1, 6, 2)
    assert native[0]["cont_emb"].shape == (2, 3, 2)
    assert covariates[0][2].tolist() == ["0", "0", "1", "1"]
    groups[1]["covariates"][0][0] = "Q"
    with pytest.raises(ValueError, match="one perturbation"):
        pack_native_groups(groups)
