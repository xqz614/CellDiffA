"""
Model evaluation script for CellDiffA.

Supports two modes:
1. Standard inference: Run any baseline model's predict() directly.
2. CellDiffA inference: Wrap a diffusion model with SMC test-time alignment.

Usage:
    # Standard baseline evaluation
    python scripts/evaluate_model.py --model gears --checkpoint ./checkpoints/gears

    # CellDiffA test-time alignment on top of PerturbDiff
    python scripts/evaluate_model.py --model perturbdiff --checkpoint ./checkpoints/pd \
        --celldiffa --num_particles 100

    # CellDiffA on top of scDFM
    python scripts/evaluate_model.py --model scdfm --checkpoint ./checkpoints/scdfm \
        --celldiffa --num_particles 200 --tempering cosine
"""

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_adapter(model_name: str, device: str, **kwargs):
    """Factory function to instantiate the correct adapter."""
    from baselines import GEARSAdapter, CPAAdapter, PerturbDiffAdapter, ScDFMAdapter

    adapters = {
        "gears": GEARSAdapter,
        "cpa": CPAAdapter,
        "perturbdiff": PerturbDiffAdapter,
        "scdfm": ScDFMAdapter,
    }

    if model_name not in adapters:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(adapters.keys())}")

    return adapters[model_name](device=device, **kwargs)


def run_standard_evaluation(
    adapter,
    test_conditions: list,
    ground_truth: Dict[str, np.ndarray],
    ctrl_mean: np.ndarray,
    n_samples: int = 100,
) -> Dict[str, float]:
    """Run standard (non-CellDiffA) evaluation."""
    from celldiffa.evaluation import evaluate_all_conditions

    print(f"\n[Eval] Running standard inference with {adapter.model_name}...")
    start_time = time.time()

    predictions = adapter.predict(test_conditions, n_samples=n_samples)

    elapsed = time.time() - start_time
    print(f"[Eval] Inference completed in {elapsed:.1f}s")

    aggregated, per_condition = evaluate_all_conditions(
        predictions=predictions,
        ground_truth=ground_truth,
        ctrl_mean=ctrl_mean,
    )

    return aggregated, per_condition, predictions


def run_celldiffa_evaluation(
    adapter,
    test_conditions: list,
    ground_truth: Dict[str, np.ndarray],
    ctrl_mean: np.ndarray,
    ctrl_cells: np.ndarray,
    config: dict,
) -> Dict[str, float]:
    """Run CellDiffA test-time alignment evaluation."""
    import torch
    from celldiffa.smc import SMCEngine, SMCConfig
    from celldiffa.smc.utils import build_reward_from_config
    from celldiffa.evaluation import evaluate_all_conditions

    print(f"\n[CellDiffA] Running SMC test-time alignment on {adapter.model_name}...")

    # Get diffusion sampler from adapter
    sampler = adapter.get_diffusion_sampler()

    # Build reward function from config
    # (Requires pre-computed priors from DataManager)
    from data.data_manager import PerturbationDataManager
    dm = PerturbationDataManager(
        data_root=config["data"]["data_root"],
        dataset_name=config["data"]["dataset"],
        n_top_genes=config["data"]["n_top_genes"],
        seed=config["data"]["seed"],
    )
    dm.load_and_preprocess()
    dm.create_split(
        split_strategy=config["data"]["split_strategy"],
        fold=config["data"]["fold"],
    )
    de_genes = dm.compute_de_genes(top_k=config["rewards"]["rewards"][0].get("top_k", 20))
    shifts = dm.compute_perturbation_shifts()
    gene_names = list(dm.adata.var_names)

    reward_fn = build_reward_from_config(
        config=config["rewards"],
        de_genes=de_genes,
        shifts=shifts,
        ctrl_mean=ctrl_mean,
        gene_names=gene_names,
    )

    # Configure SMC engine
    smc_config = SMCConfig(
        num_particles=config["smc"]["num_particles"],
        resampling_strategy=config["smc"]["resampling_strategy"],
        ess_threshold=config["smc"]["ess_threshold"],
        tempering_schedule=config["smc"]["tempering_schedule"],
        initial_temperature=config["smc"]["initial_temperature"],
        final_temperature=config["smc"]["final_temperature"],
        output_mode=config["smc"]["output_mode"],
        top_k=config["smc"]["top_k"],
        device=config["training"]["device"],
        batch_size_per_step=config["smc"]["batch_size_per_step"],
    )

    engine = SMCEngine(
        model_sampler=sampler,
        reward_fn=reward_fn,
        config=smc_config,
    )

    # Run alignment for each test condition
    predictions = {}
    ctrl_tensor = torch.tensor(ctrl_cells, dtype=torch.float32, device=smc_config.device)

    start_time = time.time()
    for i, cond in enumerate(test_conditions):
        print(f"  [{i+1}/{len(test_conditions)}] Aligning: {cond}")

        # Encode condition for the base model
        cond_emb = adapter._encode_condition(cond)
        cond_emb["num_genes"] = len(gene_names)

        result = engine.sample_with_alignment(
            condition=cond,
            condition_emb=cond_emb,
            ctrl_cells=ctrl_tensor,
        )

        predictions[cond] = result["samples"].cpu().numpy()

    elapsed = time.time() - start_time
    print(f"[CellDiffA] Alignment completed in {elapsed:.1f}s")

    aggregated, per_condition = evaluate_all_conditions(
        predictions=predictions,
        ground_truth=ground_truth,
        ctrl_mean=ctrl_mean,
    )

    return aggregated, per_condition, predictions


def main():
    parser = argparse.ArgumentParser(description="CellDiffA Model Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="./configs/default.yaml", help="Config file")
    parser.add_argument("--celldiffa", action="store_true", help="Enable CellDiffA alignment")
    parser.add_argument("--num_particles", type=int, default=None, help="Override num_particles")
    parser.add_argument("--tempering", type=str, default=None, help="Override tempering schedule")
    parser.add_argument("--n_samples", type=int, default=100, help="Samples per condition")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.num_particles:
        config["smc"]["num_particles"] = args.num_particles
    if args.tempering:
        config["smc"]["tempering_schedule"] = args.tempering
    config["training"]["device"] = args.device

    # Setup data
    from data.data_manager import PerturbationDataManager

    dm = PerturbationDataManager(
        data_root=config["data"]["data_root"],
        dataset_name=config["data"]["dataset"],
        n_top_genes=config["data"]["n_top_genes"],
        seed=config["data"]["seed"],
    )
    dm.load_and_preprocess()
    adata_train, adata_test = dm.create_split(
        split_strategy=config["data"]["split_strategy"],
        fold=config["data"]["fold"],
    )

    ctrl_mean = dm.get_control_mean()
    ctrl_cells = dm.get_control_cells(n_cells=200)

    # Prepare ground truth
    test_conditions = [
        c for c in adata_test.obs["condition"].unique()
        if c not in ("ctrl", "control")
    ]

    ground_truth = {}
    for cond in test_conditions:
        cells = adata_test[adata_test.obs["condition"] == cond]
        expr = cells.X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        ground_truth[cond] = expr

    # Load model
    adapter = get_adapter(args.model, device=args.device)
    adapter.load_checkpoint(args.checkpoint)

    # Run evaluation
    if args.celldiffa:
        aggregated, per_condition, predictions = run_celldiffa_evaluation(
            adapter=adapter,
            test_conditions=test_conditions,
            ground_truth=ground_truth,
            ctrl_mean=ctrl_mean,
            ctrl_cells=ctrl_cells,
            config=config,
        )
        method_name = f"CellDiffA+{args.model}"
    else:
        aggregated, per_condition, predictions = run_standard_evaluation(
            adapter=adapter,
            test_conditions=test_conditions,
            ground_truth=ground_truth,
            ctrl_mean=ctrl_mean,
            n_samples=args.n_samples,
        )
        method_name = args.model

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"{method_name}_results.json")

    output = {
        "method": method_name,
        "dataset": config["data"]["dataset"],
        "split": config["data"]["split_strategy"],
        "aggregated_metrics": aggregated,
        "per_condition_metrics": {
            k: {mk: float(mv) for mk, mv in v.items()}
            for k, v in per_condition.items()
        },
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Results: {method_name}")
    print(f"{'='*60}")
    for metric, value in sorted(aggregated.items()):
        print(f"  {metric:25s}: {value:.4f}")
    print(f"\n  Results saved to: {result_file}")


if __name__ == "__main__":
    main()
