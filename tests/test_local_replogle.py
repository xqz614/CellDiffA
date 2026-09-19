import numpy as np
import pytest
import torch

from celldiffa.benchmark.torch_compat import extract_into_tensor


def test_validation_reference_cannot_be_used_as_test():
    from types import SimpleNamespace

    import pandas as pd

    from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit

    split = PerturbDiffSplit(
        "gene", "ctrl", "context", None, ("B",), frozenset({"V"}), frozenset({"T"})
    )
    reference = SimpleNamespace(obs=pd.DataFrame({"gene": ["V"], "context": ["B"]}))
    split.validate_reference(reference, split_name="validation")
    with pytest.raises(ValueError, match="outside the PerturbDiff test split"):
        split.validate_real_test(reference)


@pytest.mark.parametrize("shape", [(3, 2), (3, 4, 5)])
def test_schedule_cast_matches_upstream_cpu(shape):
    schedule = np.linspace(0.000001, 0.999999, 1000, dtype=np.float64)
    timesteps = torch.tensor([0, 59, 999])
    expected = torch.from_numpy(schedule)[timesteps].float()
    expected = expected.reshape(3, *([1] * (len(shape) - 1))).expand(shape)
    assert torch.equal(extract_into_tensor(schedule, timesteps, shape), expected)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_schedule_mps_agrees_with_cpu():
    schedule = np.linspace(0.001, 0.999, 100, dtype=np.float64)
    indices = torch.tensor([0, 30, 99])
    expected = extract_into_tensor(schedule, indices, (3, 4, 5))
    actual = extract_into_tensor(schedule, indices.to("mps"), (3, 4, 5))
    assert torch.equal(actual.cpu(), expected)
