import numpy as np
import pytest
import torch

from celldiffa.benchmark.released_sampling import capture_rng, restore_rng, validate_sampling_scale


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


def test_valid_sampling_scale_preserves_values():
    values = np.array([[0, 0.05, 6.84]], dtype=np.float32)
    before = values.copy()
    validate_sampling_scale(values)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("value", [4006.125, -0.1, float("nan"), float("inf")])
def test_invalid_sampling_scale_errors_without_altering_predictions(value):
    values = np.array([[value]], dtype=np.float32)
    before = values.copy()
    with pytest.raises(ValueError):
        validate_sampling_scale(values)
    np.testing.assert_array_equal(values, before)
