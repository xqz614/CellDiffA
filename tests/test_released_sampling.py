import numpy as np
import pytest
import torch

from celldiffa.benchmark.released_sampling import capture_rng, restore_rng


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_native_sampling_rng_round_trip(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    device = torch.device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    state = capture_rng(device)
    expected = (torch.randn(3, 5, device=device), torch.randn(5), np.random.rand(3))
    restore_rng(state, device)
    torch.testing.assert_close(torch.randn(3, 5, device=device), expected[0], atol=0, rtol=0)
    torch.testing.assert_close(torch.randn(5), expected[1], atol=0, rtol=0)
    np.testing.assert_array_equal(np.random.rand(3), expected[2])
