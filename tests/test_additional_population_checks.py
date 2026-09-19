import numpy as np
import pandas as pd
import pytest

from scripts.baselines.evaluate_population_diagnostics import effective_rank, sliced_wasserstein
from scripts.baselines.run_squidiff_replogle import latent_shifts


def test_latent_shifts_remove_context_baselines_and_never_create_unknowns():
    obs = pd.DataFrame({"gene": ["ctrl", "A", "ctrl", "A"], "cell_line": ["x", "x", "y", "y"]})
    latent = np.array([[10, 0], [12, 3], [-20, 4], [-18, 7]], dtype=np.float32)
    result = latent_shifts(latent, obs, "ctrl")
    assert set(result) == {"A"}
    np.testing.assert_array_equal(result["A"], [2, 3])


def test_latent_shift_without_control_fails():
    obs = pd.DataFrame({"gene": ["A"], "cell_line": ["x"]})
    with pytest.raises(ValueError, match="No training control"):
        latent_shifts(np.ones((1, 2)), obs, "ctrl")


def test_descriptive_distribution_checks_detect_collapse_and_permutations():
    x = np.random.default_rng(42).normal(size=(50, 5))
    assert effective_rank(x) > 1
    assert effective_rank(np.repeat(x[:1], len(x), axis=0)) == 0
    assert sliced_wasserstein(x, x[::-1]) == 0
    assert sliced_wasserstein(x, x + 1) == pytest.approx(1)


def test_effective_rank_constant_float32_population_has_no_spurious_rank():
    # Accumulating a float32 mean can introduce an artificial nonzero direction.
    values = np.repeat(np.full((1, 64), 0.1, dtype=np.float32), 200, axis=0)
    assert effective_rank(values) == 0.0
