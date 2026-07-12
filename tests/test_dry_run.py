"""
Dry-run integration test for CellDiffA pipeline.

Tests the entire SMC engine + adapter interface without real model weights
or real data. Uses mock objects to verify:
1. DiffusionSamplerProtocol compliance
2. SMCEngine execution (forward pass, resampling, weight update)
3. Reward function computation
4. Adapter factory and interface
5. Configuration loading

Run with: python tests/test_dry_run.py
"""

import os
import sys
import traceback

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Mock Sampler (satisfies DiffusionSamplerProtocol)
# ============================================================

class MockDiffusionSampler:
    """Mock sampler for testing the SMC engine without a real model."""

    def __init__(self, num_genes: int = 50, num_timesteps: int = 20):
        self._num_timesteps = num_timesteps
        self._num_genes = num_genes

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def denoise_step(self, x_t, t, condition, prev_pred=None):
        """Mock denoising: slightly move toward zero + small noise."""
        N, G = x_t.shape
        # Simulate gradual denoising
        t_val = t[0].item()
        alpha = 1.0 - (t_val / self._num_timesteps)
        noise = 0.1 * torch.randn_like(x_t)
        x_prev = alpha * x_t + (1 - alpha) * noise
        x0_pred = x_t * 0.5 + torch.randn_like(x_t) * 0.1
        return {"x_prev": x_prev, "x0_pred": x0_pred}


# ============================================================
# Test Functions
# ============================================================

def test_config_loading():
    """Test that default.yaml loads correctly."""
    import yaml
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "default.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert config["smc"]["guidance_strength"] == 1.0, (
        f"guidance_strength should be 1.0, got {config['smc']['guidance_strength']}"
    )
    assert config["smc"]["alpha"] == 1.0
    assert config["smc"]["num_particles"] == 100
    assert config["smc"]["tempering_schedule"] == "linear"
    assert len(config["rewards"]["rewards"]) == 3
    print("  [PASS] Config loading and guidance_strength fix verified")


def test_resampler():
    """Test resampling algorithms."""
    from celldiffa.smc import Resampler, ResamplingStrategy

    for strategy in ResamplingStrategy:
        resampler = Resampler(strategy=strategy)
        weights = torch.softmax(torch.randn(50), dim=0)
        indices = resampler.resample(weights, 50)
        assert indices.shape == (50,), f"Expected shape (50,), got {indices.shape}"
        assert indices.max() < 50
        assert indices.min() >= 0

    print("  [PASS] All resampling strategies work correctly")


def test_smc_engine_basic():
    """Test SMC engine with mock sampler."""
    from celldiffa.smc import SMCEngine, SMCConfig

    config = SMCConfig(
        num_particles=30,
        resampling_strategy="systematic",
        ess_threshold=0.5,
        tempering_schedule="linear",
        alpha=1.0,
        start_timestep=19,
        guidance_strength=1.0,
        output_mode="all",
        device="cpu",
        batch_size_per_step=30,
    )

    # Mock reward function
    class MockReward:
        def compute(self, x_pred, condition, timestep, **kwargs):
            # Simple reward: prefer cells with higher mean expression
            return x_pred.mean(dim=1)

    sampler = MockDiffusionSampler(num_genes=50, num_timesteps=20)
    reward = MockReward()
    engine = SMCEngine(sampler, reward, config)

    result = engine.sample_with_alignment(
        condition="GeneA+GeneB",
        condition_emb={"mock": torch.zeros(1)},
        ctrl_cells=torch.randn(20, 50),
        num_genes=50,
    )

    assert "samples" in result
    assert "weights" in result
    assert "ess_history" in result
    assert "resample_history" in result
    assert result["samples"].shape == (30, 50), f"Got shape {result['samples'].shape}"
    assert len(result["ess_history"]) == 20  # num_timesteps steps
    print("  [PASS] SMC engine basic execution verified")


def test_smc_engine_tempering_schedules():
    """Test all tempering schedules."""
    from celldiffa.smc import SMCEngine, SMCConfig

    class MockReward:
        def compute(self, x_pred, condition, timestep, **kwargs):
            return torch.zeros(x_pred.shape[0])

    for schedule in ["linear", "cosine", "adaptive"]:
        config = SMCConfig(
            num_particles=10,
            tempering_schedule=schedule,
            start_timestep=9,
            device="cpu",
            batch_size_per_step=10,
        )
        sampler = MockDiffusionSampler(num_genes=20, num_timesteps=10)
        engine = SMCEngine(sampler, MockReward(), config)
        result = engine.sample_with_alignment(
            condition="test",
            condition_emb={},
            num_genes=20,
        )
        assert result["samples"].shape == (10, 20)

    print("  [PASS] All tempering schedules work correctly")


def test_reward_functions():
    """Test reward function computation."""
    from celldiffa.rewards import TranscriptomicReward, GeometricReward, AnchorReward, CompositeReward

    G = 50
    gene_names = [f"Gene{i}" for i in range(G)]
    ctrl_mean = np.random.randn(G).astype(np.float32)

    # DE genes for known conditions
    de_genes = {
        "GeneA+ctrl": gene_names[:10],
        "GeneB+ctrl": gene_names[5:15],
    }

    # Perturbation shifts
    shifts = {
        "GeneA+ctrl": np.random.randn(G).astype(np.float32) * 0.5,
        "GeneB+ctrl": np.random.randn(G).astype(np.float32) * 0.5,
    }

    # Test TranscriptomicReward
    r_deg = TranscriptomicReward(
        de_genes=de_genes,
        perturbation_shifts=shifts,
        ctrl_mean=ctrl_mean,
        gene_names=gene_names,
        weight=1.0,
        top_k=10,
    )

    x_pred = torch.randn(20, G)
    reward = r_deg.compute(x_pred, "GeneA+ctrl", timestep=50)
    assert reward.shape == (20,), f"Expected (20,), got {reward.shape}"
    assert torch.all(reward <= 0), "TranscriptomicReward should be negative (neg MSE)"

    # Test for unseen combination
    reward_combo = r_deg.compute(x_pred, "GeneA+GeneB", timestep=50)
    assert reward_combo.shape == (20,)

    # Test GeometricReward
    r_geo = GeometricReward(
        perturbation_shifts=shifts,
        ctrl_mean=ctrl_mean,
        weight=1.0,
    )
    reward_geo = r_geo.compute(x_pred, "GeneA+ctrl", timestep=50)
    assert reward_geo.shape == (20,)

    # Test AnchorReward
    r_anchor = AnchorReward(
        ctrl_mean=ctrl_mean,
        weight=1.0,
    )
    reward_anchor = r_anchor.compute(x_pred, "GeneA+ctrl", timestep=50)
    assert reward_anchor.shape == (20,)

    # Test CompositeReward
    composite = CompositeReward(
        rewards=[r_deg, r_geo, r_anchor],
        aggregation="linear",
    )
    reward_total = composite.compute(x_pred, "GeneA+ctrl", timestep=50)
    assert reward_total.shape == (20,)

    # Test Pareto aggregation
    composite_pareto = CompositeReward(
        rewards=[r_deg, r_geo, r_anchor],
        aggregation="pareto",
    )
    reward_pareto = composite_pareto.compute(x_pred, "GeneA+ctrl", timestep=50)
    assert reward_pareto.shape == (20,)

    print("  [PASS] All reward functions compute correctly")


def test_build_reward_from_config():
    """Test the reward factory function."""
    from celldiffa.smc.utils import build_reward_from_config

    G = 50
    gene_names = [f"Gene{i}" for i in range(G)]
    ctrl_mean = np.random.randn(G).astype(np.float32)
    de_genes = {"cond1": gene_names[:10]}
    shifts = {"cond1": np.random.randn(G).astype(np.float32)}

    config = {
        "aggregation": "linear",
        "rewards": [
            {"type": "transcriptomic", "weight": 1.0, "top_k": 10},
            {"type": "geometric", "weight": 1.0},
            {"type": "anchor", "weight": 1.0},
        ],
    }

    composite = build_reward_from_config(
        config=config,
        de_genes=de_genes,
        shifts=shifts,
        ctrl_mean=ctrl_mean,
        gene_names=gene_names,
    )

    x_pred = torch.randn(10, G)
    reward = composite.compute(x_pred, "cond1", timestep=50)
    assert reward.shape == (10,)
    print("  [PASS] build_reward_from_config works correctly")


def test_adapter_imports():
    """Test that all adapters can be imported."""
    from baselines import (
        BaseAdapter,
        GEARSAdapter,
        CPAAdapter,
        PerturbDiffAdapter,
        ScDFMAdapter,
        SquidiffAdapter,
        CellFlowAdapter,
    )

    # Verify they all inherit from BaseAdapter
    for cls in [GEARSAdapter, CPAAdapter, PerturbDiffAdapter, ScDFMAdapter, SquidiffAdapter, CellFlowAdapter]:
        assert issubclass(cls, BaseAdapter), f"{cls.__name__} doesn't inherit from BaseAdapter"

    # Verify generative property
    pd = PerturbDiffAdapter(device="cpu")
    assert pd.is_generative is True

    sq = SquidiffAdapter(device="cpu")
    assert sq.is_generative is True

    cf = CellFlowAdapter(device="cpu")
    assert cf.is_generative is True

    gears = GEARSAdapter(device="cpu")
    assert gears.is_generative is False

    print("  [PASS] All adapters import and instantiate correctly")


def test_adapter_factory():
    """Test the get_adapter factory from evaluate_model.py."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
    ))
    # Import the factory function directly
    from scripts.evaluate_model import get_adapter

    for name in ["gears", "cpa", "perturbdiff", "scdfm", "squidiff", "cellflow"]:
        adapter = get_adapter(name, device="cpu")
        assert adapter.model_name is not None
        assert adapter.device == "cpu"

    print("  [PASS] Adapter factory works for all model names")


def test_smc_with_batched_denoise():
    """Test SMC engine with batch_size_per_step < num_particles."""
    from celldiffa.smc import SMCEngine, SMCConfig

    class MockReward:
        def compute(self, x_pred, condition, timestep, **kwargs):
            return x_pred.mean(dim=1)

    config = SMCConfig(
        num_particles=50,
        start_timestep=9,
        device="cpu",
        batch_size_per_step=15,  # Force batching
    )

    sampler = MockDiffusionSampler(num_genes=30, num_timesteps=10)
    engine = SMCEngine(sampler, MockReward(), config)

    result = engine.sample_with_alignment(
        condition="test",
        condition_emb={},
        num_genes=30,
    )
    assert result["samples"].shape == (50, 30)
    print("  [PASS] Batched denoising works correctly")


def test_evaluation_metrics():
    """Test evaluation metrics computation."""
    from celldiffa.evaluation import evaluate_all_conditions

    G = 50
    ctrl_mean = np.random.randn(G).astype(np.float32)

    predictions = {
        "cond1": np.random.randn(20, G).astype(np.float32),
        "cond2": np.random.randn(20, G).astype(np.float32),
    }
    ground_truth = {
        "cond1": np.random.randn(30, G).astype(np.float32),
        "cond2": np.random.randn(30, G).astype(np.float32),
    }

    aggregated, per_condition = evaluate_all_conditions(
        predictions=predictions,
        ground_truth=ground_truth,
        ctrl_mean=ctrl_mean,
    )

    assert "mse_all" in aggregated
    assert "pearson_delta" in aggregated
    assert "cond1" in per_condition
    assert "cond2" in per_condition
    print("  [PASS] Evaluation metrics compute correctly")


def test_squidiff_sampler_interface():
    """Test SquidiffSampler satisfies DiffusionSamplerProtocol."""
    from baselines.adapter_squidiff import SquidiffSampler

    # Create a mock model and diffusion for SquidiffSampler
    class MockSquidiffModel:
        def eval(self):
            pass

        def __call__(self, x, t, **kwargs):
            return torch.randn_like(x)  # Mock epsilon prediction

    class MockDiffusion:
        def __init__(self):
            self.num_timesteps = 100
            self.alphas_cumprod = np.linspace(0.9999, 0.001, 100)
            self.alphas_cumprod_prev = np.concatenate([[1.0], self.alphas_cumprod[:-1]])
            self.timestep_map = list(range(0, 1000, 10))  # 100 steps
            self.original_num_steps = 1000
            self.rescale_timesteps = False

    model = MockSquidiffModel()
    diffusion = MockDiffusion()
    z_mod = torch.randn(1, 60)

    sampler = SquidiffSampler(
        model=model,
        diffusion=diffusion,
        z_mod=z_mod,
        device="cpu",
        eta=0.0,
        clip_denoised=True,
    )

    # Test interface
    assert sampler.num_timesteps == 100

    noise = sampler.sample_noise((20, 50), torch.device("cpu"))
    assert noise.shape == (20, 50)

    x_t = torch.randn(20, 50)
    t = torch.full((20,), 50, dtype=torch.long)
    output = sampler.denoise_step(x_t, t, condition={})
    assert "x_prev" in output
    assert "x0_pred" in output
    assert output["x_prev"].shape == (20, 50)
    assert output["x0_pred"].shape == (20, 50)

    print("  [PASS] SquidiffSampler satisfies DiffusionSamplerProtocol")


def test_cellflow_sampler_interface():
    """Test CellFlowSampler basic interface (without JAX model)."""
    # We can't fully test without a trained CellFlow model,
    # but we can verify the adapter instantiation and interface
    from baselines.adapter_cellflow import CellFlowAdapter

    adapter = CellFlowAdapter(device="cpu", num_integration_steps=50)
    assert adapter.model_name == "CellFlow"
    assert adapter.is_generative is True
    assert adapter.num_integration_steps == 50

    print("  [PASS] CellFlowAdapter instantiation and interface verified")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  CellDiffA Dry-Run Integration Tests")
    print("=" * 60)

    tests = [
        ("Config Loading", test_config_loading),
        ("Resampler", test_resampler),
        ("SMC Engine Basic", test_smc_engine_basic),
        ("SMC Tempering Schedules", test_smc_engine_tempering_schedules),
        ("Reward Functions", test_reward_functions),
        ("Build Reward from Config", test_build_reward_from_config),
        ("Adapter Imports", test_adapter_imports),
        ("Adapter Factory", test_adapter_factory),
        ("SMC Batched Denoise", test_smc_with_batched_denoise),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Squidiff Sampler Interface", test_squidiff_sampler_interface),
        ("CellFlow Adapter Interface", test_cellflow_sampler_interface),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}...")
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e), traceback.format_exc()))
            print(f"  [FAIL] {name}: {e}")

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)

    if errors:
        print("\nFailure Details:")
        for name, msg, tb in errors:
            print(f"\n--- {name} ---")
            print(tb)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
