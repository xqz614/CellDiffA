"""Fast, model-free tests for CellDiffA's mathematical and shape contracts."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from celldiffa.evaluation.metrics import energy_distance, mmd_rbf, pearson_delta
from celldiffa.rewards import AnchorReward, CompositeReward, TranscriptomicReward
from celldiffa.smc import Resampler, ResamplingStrategy, SMCConfig, SMCEngine


class IdentitySampler:
    """Deterministic sampler whose cell values identify their particle."""

    def __init__(self, timesteps: int = 4):
        self._timesteps = timesteps

    @property
    def num_timesteps(self):
        return self._timesteps

    def sample_noise(self, shape, device):
        values = torch.arange(np.prod(shape), device=device, dtype=torch.float32)
        return values.reshape(shape) / 10.0

    def denoise_step(self, x_t, t, condition, prev_pred=None):
        return {"x_prev": x_t, "x0_pred": x_t}


class MeanReward:
    def compute(self, x_pred, condition, timestep, **kwargs):
        return x_pred.mean(dim=(1, 2)) + float(timestep)


def make_config(**overrides):
    values = dict(
        num_particles=4,
        cells_per_particle=3,
        ess_threshold=0.01,
        start_timestep=3,
        output_mode="map",
        device="cpu",
        batch_size_per_step=4,
        seed=7,
    )
    values.update(overrides)
    return SMCConfig(**values)


def test_default_config_is_population_level():
    config_path = Path(__file__).parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["smc"]["cells_per_particle"] >= 2
    assert config["smc"]["output_mode"] == "map"
    assert config["rewards"]["normalization"] == "zscore"


@pytest.mark.parametrize("strategy", list(ResamplingStrategy))
def test_resamplers_return_valid_indices(strategy):
    weights = torch.tensor([0.05, 0.15, 0.3, 0.5])
    indices = Resampler(strategy).resample(weights, 20)
    assert indices.shape == (20,)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < len(weights)


def test_smc_particle_is_a_cell_batch_and_map_returns_one_distribution():
    engine = SMCEngine(IdentitySampler(), MeanReward(), make_config())
    result = engine.sample_with_alignment(
        condition="A+B",
        condition_emb={},
        ctrl_cells=torch.zeros(3, 2),
        num_genes=2,
    )
    assert result["all_particles"].shape == (4, 3, 2)
    assert result["samples"].shape == (3, 2)
    assert result["cells_per_particle"] == 3
    assert result["resample_history"][-1] is False


def test_feynman_kac_potential_telescopes_to_terminal_reward():
    sampler = IdentitySampler(timesteps=4)
    engine = SMCEngine(sampler, MeanReward(), make_config())
    result = engine.sample_with_alignment(condition="A", condition_emb={}, num_genes=2)
    final_reward = result["all_particles"].mean(dim=(1, 2))
    expected = torch.softmax(final_reward, dim=0)
    assert torch.allclose(result["weights"], expected, atol=1e-6)


def test_condition_rows_follow_cells_across_forward_minibatches():
    class RecordingSampler(IdentitySampler):
        def __init__(self):
            super().__init__(timesteps=1)
            self.seen = []

        def denoise_step(self, x_t, t, condition, prev_pred=None):
            self.seen.extend(condition["cell_id"].cpu().tolist())
            return super().denoise_step(x_t, t, condition, prev_pred)

    sampler = RecordingSampler()
    config = make_config(
        num_particles=2,
        cells_per_particle=3,
        start_timestep=0,
        batch_size_per_step=2,
    )
    engine = SMCEngine(sampler, MeanReward(), config)
    engine.sample_with_alignment(
        condition="A",
        condition_emb={"cell_id": torch.tensor([0, 1, 2])},
        num_genes=2,
    )
    assert sampler.seen == [0, 1, 2, 0, 1, 2]


def reward_fixture():
    genes = ["g0", "g1", "g2"]
    ctrl = np.zeros(3, dtype=np.float32)
    shifts = {"A+ctrl": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    de_genes = {"A+ctrl": ["g0"]}
    return genes, ctrl, shifts, de_genes


def test_signature_rewards_score_batches_not_individual_cells():
    genes, ctrl, shifts, de_genes = reward_fixture()
    reward = TranscriptomicReward(de_genes, shifts, ctrl, genes)
    # Both particles have the same population mean but different cell states.
    batches = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    scores = reward.compute(batches, "A+ctrl", timestep=0)
    assert torch.allclose(scores[0], scores[1])


def test_anchor_prefers_training_derived_reference_distribution():
    _, _, shifts, _ = reward_fixture()
    ctrl_cells = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    reference = ctrl_cells + torch.tensor(shifts["A+ctrl"])
    far = reference + 5.0
    batches = torch.stack([reference, far])
    reward = AnchorReward(shifts, bandwidth=1.0)
    scores = reward.compute(batches, "A+ctrl", timestep=0, ctrl_cells=ctrl_cells)
    assert scores[0] > scores[1]


def test_composite_reward_normalizes_objective_scales():
    class FixedReward:
        def __init__(self, values, weight=1.0):
            self.values = torch.tensor(values, dtype=torch.float32)
            self.weight = weight

        def compute(self, *args, **kwargs):
            return self.values

    composite = CompositeReward(
        [FixedReward([0, 1, 2]), FixedReward([0, 1000, 2000])],
        normalization="zscore",
    )
    values = composite.compute(torch.zeros(3, 2, 1), "A", 0)
    assert values[0] < values[1] < values[2]
    assert abs(float(values.mean())) < 1e-6


def test_metrics_are_finite_and_deterministic():
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(250, 5))
    true = rng.normal(size=(260, 5))
    assert energy_distance(pred, true) == energy_distance(pred, true)
    assert mmd_rbf(pred, true) == mmd_rbf(pred, true)
    assert pearson_delta(np.ones(5), np.ones(5), np.zeros(5)) == 0.0


def test_invalid_configuration_fails_early():
    with pytest.raises(ValueError, match="alpha"):
        SMCEngine(IdentitySampler(), MeanReward(), make_config(alpha=0.0))
