"""Contract tests for the upstream-aligned PerturbDiff DDIM adapter."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baselines.adapter_perturbdiff import PerturbDiffAdapter, PerturbDiffSampler


class MockCrossDiT:
    model_name = "Cross_DiT"
    model_cfg = SimpleNamespace(cutoff=0.0)

    def eval(self):
        return self

    def __call__(self, x, control, timestep, self_condition):
        batch = x.shape[0]
        genes = x.shape[-1] // 2
        # Conditional and unconditional values are both negative. Applying the
        # cutoff before CFG would erase the signal; correct CFG gives +1.
        value = -1.0 if "batch_emb" in self_condition else -3.0
        return {"x": torch.full((batch, 1, genes), value, device=x.device)}


class MockDiffusion:
    num_timesteps = 2
    alphas_cumprod = np.array([0.5, 0.25], dtype=np.float32)
    alphas_cumprod_prev = np.array([1.0, 0.5], dtype=np.float32)
    rescale_timesteps = False


def test_cfg_is_applied_before_expression_cutoff():
    model = MockCrossDiT()
    pl_model = SimpleNamespace(model=model)
    condition = {
        "batch_emb": torch.zeros(1, 2),
        "cont_emb": torch.zeros(1, 1, 3),
        "gene_emb": None,
        "ds_name": [["norman"]],
    }
    sampler = PerturbDiffSampler(
        pl_model=pl_model,
        diffusion=MockDiffusion(),
        condition_dict=condition,
        device="cpu",
        guidance_strength=1.0,
        start_time=2,
    )
    output = sampler.denoise_step(
        x_t=torch.zeros(4, 3),
        t=torch.zeros(4, dtype=torch.long),
        condition=condition,
        prev_pred=torch.zeros(4, 3),
    )
    assert torch.allclose(output["x0_pred"], torch.ones(4, 3), atol=1e-6)


def test_non_onehot_unknown_perturbation_never_uses_negative_index():
    adapter = PerturbDiffAdapter(device="cpu")
    adapter._pert_dict = {"known": 0}
    adapter._cov_cfg = {"replogle_gene_encoding": "genept"}
    with pytest.raises(ValueError, match="absent"):
        adapter._lookup_perturbation("unknown")

    adapter._pert_dict["control"] = 1
    assert adapter._lookup_perturbation("unknown") == 1
