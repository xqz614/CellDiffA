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

    # Alternative tempering schedule
    python scripts/evaluate_model.py --model perturbdiff --checkpoint ./checkpoints/pd \
        --celldiffa --num_particles 200 --tempering cosine
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_adapter(model_name: str, device: str, **kwargs):
    """Factory function to instantiate the correct adapter."""
    from baselines import GEARSAdapter, PerturbDiffAdapter

    adapters = {
        "gears": GEARSAdapter,
        "perturbdiff": PerturbDiffAdapter,
    }

    if model_name not in adapters:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(adapters.keys())}")

    return adapters[model_name](device=device, **kwargs)


def run_standard_evaluation(
    adapter,
    test_conditions: List[str],
    ground_truth: Dict[str, np.ndarray],
    ctrl_mean: np.ndarray,
    n_samples: int = 100,
    ctrl_expr: Optional[torch.Tensor] = None,
    adapter_kwargs: Optional[dict] = None,
) -> tuple:
    """Run standard (non-CellDiffA) evaluation."""
    from celldiffa.evaluation import evaluate_all_conditions

    print(f"\n[Eval] Running standard inference with {adapter.model_name}...")
    start_time = time.time()

    predictions = adapter.predict(
        conditions=test_conditions,
        n_samples=n_samples,
        ctrl_expr=ctrl_expr,
        **(adapter_kwargs or {}),
    )

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
    test_conditions: List[str],
    ground_truth: Dict[str, np.ndarray],
    ctrl_mean: np.ndarray,
    ctrl_cells: np.ndarray,
    config: dict,
    gene_names: List[str],
    de_genes: Dict[str, List[str]],
    shifts: Dict[str, np.ndarray],
    ds_name: str = "norman",
    cell_type: str = "K562",
    batch_name: str = "default",
) -> tuple:
    """
    Run CellDiffA test-time alignment evaluation.

    This is the core pipeline:
        For each test condition:
            1. Build condition_dict via adapter.build_condition()
            2. Get step-by-step sampler via adapter.get_diffusion_sampler()
            3. Build reward function from training-set priors
            4. Run SMC engine to generate aligned samples
            5. Evaluate against ground truth
    """
    from celldiffa.evaluation import evaluate_all_conditions
    from celldiffa.smc import SMCConfig, SMCEngine
    from celldiffa.smc.utils import build_reward_from_config

    print(f"\n[CellDiffA] Running SMC test-time alignment on {adapter.model_name}...")

    # --- Build composite reward from pre-computed priors ---
    reward_fn = build_reward_from_config(
        config=config["rewards"],
        de_genes=de_genes,
        shifts=shifts,
        ctrl_mean=ctrl_mean,
        gene_names=gene_names,
    )

    # --- SMC configuration (DAS-style) ---
    smc_cfg = config["smc"]
    smc_config = SMCConfig(
        num_particles=smc_cfg["num_particles"],
        cells_per_particle=smc_cfg["cells_per_particle"],
        resampling_strategy=smc_cfg["resampling_strategy"],
        ess_threshold=smc_cfg["ess_threshold"],
        tempering_schedule=smc_cfg["tempering_schedule"],
        alpha=smc_cfg["alpha"],
        start_timestep=smc_cfg.get("start_timestep", None),
        use_ddim=smc_cfg.get("use_ddim", True),
        eta=smc_cfg.get("eta", 0.0),
        guidance_strength=smc_cfg.get("guidance_strength", 1.0),
        output_mode=smc_cfg["output_mode"],
        top_k=smc_cfg["top_k"],
        device=config["training"]["device"],
        batch_size_per_step=smc_cfg["batch_size_per_step"],
        seed=smc_cfg.get("seed", config["data"]["seed"]),
    )

    # --- Run alignment for each test condition ---
    predictions = {}
    ctrl_tensor = torch.tensor(ctrl_cells, dtype=torch.float32)

    start_time_total = time.time()

    for i, cond in enumerate(test_conditions):
        print(f"  [{i + 1}/{len(test_conditions)}] Aligning: {cond}")
        t0 = time.time()

        # Step 1: Build condition for this perturbation
        condition_ctrl = ctrl_tensor[: smc_config.cells_per_particle].to(smc_config.device)
        condition_dict = adapter.build_condition(
            perturbation=cond,
            cell_type=cell_type,
            batch_name=batch_name,
            ctrl_expr=condition_ctrl,
            ds_name=ds_name,
        )

        # Step 2: Get step-by-step sampler
        sampler = adapter.get_diffusion_sampler(
            condition_dict=condition_dict,
            guidance_strength=smc_config.guidance_strength,
            eta=smc_config.eta,
            start_time=smc_config.start_timestep or 100,
        )

        # Step 3: Create SMC engine with this sampler
        engine = SMCEngine(
            model_sampler=sampler,
            reward_fn=reward_fn,
            config=smc_config,
        )

        # Step 4: Run SMC-guided generation
        num_genes = len(gene_names)
        result = engine.sample_with_alignment(
            condition=cond,
            condition_emb=condition_dict,
            ctrl_cells=condition_ctrl,
            num_genes=num_genes,
        )

        predictions[cond] = result["samples"].cpu().numpy()

        t1 = time.time()
        ess_final = result["ess_history"][-1] if result["ess_history"] else 0
        n_resamples = sum(result["resample_history"])
        print(
            f"    Done in {t1 - t0:.1f}s | "
            f"Final ESS: {ess_final:.1f}/{smc_config.num_particles} | "
            f"Resampled: {n_resamples}/{len(result['resample_history'])} steps"
        )

    elapsed = time.time() - start_time_total
    print(f"\n[CellDiffA] All conditions aligned in {elapsed:.1f}s")

    # --- Evaluate ---
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
    parser.add_argument(
        "--tempering",
        choices=["linear", "cosine"],
        default=None,
        help="Override tempering schedule",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override reward temperature")
    parser.add_argument("--n_samples", type=int, default=100, help="Samples per condition")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--ds_name", type=str, default=None, help="Dataset name for gene embeddings"
    )
    parser.add_argument("--cell_type", type=str, default="K562")
    parser.add_argument("--batch_name", type=str, default="default")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.num_particles is not None:
        config["smc"]["num_particles"] = args.num_particles
    if args.tempering:
        config["smc"]["tempering_schedule"] = args.tempering
    if args.alpha is not None:
        config["smc"]["alpha"] = args.alpha
    config["training"]["device"] = args.device

    # ================================================================
    # Setup data (single DataManager instance for the entire pipeline)
    # ================================================================
    from data.data_manager import PerturbationDataManager

    dm = PerturbationDataManager(
        data_root=config["data"]["data_root"],
        dataset_name=config["data"]["dataset"],
        n_top_genes=config["data"]["n_top_genes"],
        seed=config["data"]["seed"],
        already_normalized=config["data"].get("already_normalized", True),
    )
    dm.load_and_preprocess()
    adata_train, adata_test = dm.create_split(
        split_strategy=config["data"]["split_strategy"],
        fold=config["data"]["fold"],
        n_folds=config["data"].get("n_folds", 5),
    )

    ctrl_mean = dm.get_control_mean()
    ctrl_cells = dm.get_control_cells(
        n_cells=max(200, args.n_samples, config["smc"]["cells_per_particle"])
    )
    gene_names = list(dm.adata.var_names)

    # Prepare ground truth from test set
    test_conditions = [
        c for c in adata_test.obs["condition"].unique() if c not in ("ctrl", "control")
    ]

    ground_truth = {}
    for cond in test_conditions:
        cells = adata_test[adata_test.obs["condition"] == cond]
        expr = cells.X
        if hasattr(expr, "toarray"):
            expr = expr.toarray()
        ground_truth[cond] = np.asarray(expr)

    print(f"\n[Setup] Dataset: {config['data']['dataset']}")
    print(f"[Setup] Split: {config['data']['split_strategy']}, fold {config['data']['fold']}")
    print(f"[Setup] Test conditions: {len(test_conditions)}")
    print(f"[Setup] Genes: {len(gene_names)}")

    # ================================================================
    # Pre-compute training-set priors (used by CellDiffA rewards)
    # ================================================================
    de_genes = dm.compute_de_genes(top_k=config["rewards"]["rewards"][0].get("top_k", 20))
    shifts = dm.compute_perturbation_shifts()

    # ================================================================
    # Load model
    # ================================================================
    adapter = get_adapter(args.model, device=args.device)

    # Load checkpoint with proper context
    adapter.load_checkpoint(
        checkpoint_path=args.checkpoint,
        gene_names=gene_names,
        ctrl_adata=dm.ctrl_adata,
        adata_train=adata_train,
        dataset_name=config["data"]["dataset"],
    )

    # ================================================================
    # Run evaluation
    # ================================================================
    if args.celldiffa:
        if not adapter.is_generative:
            raise ValueError(
                f"{args.model} is not a generative (diffusion/flow) model. "
                f"CellDiffA can only wrap a validated diffusion model."
            )

        aggregated, per_condition, predictions = run_celldiffa_evaluation(
            adapter=adapter,
            test_conditions=test_conditions,
            ground_truth=ground_truth,
            ctrl_mean=ctrl_mean,
            ctrl_cells=ctrl_cells,
            config=config,
            gene_names=gene_names,
            de_genes=de_genes,
            shifts=shifts,
            ds_name=args.ds_name or config["data"]["dataset"],
            cell_type=args.cell_type,
            batch_name=args.batch_name,
        )
        method_name = f"CellDiffA+{args.model}"
    else:
        # Standard evaluation
        ctrl_expr_tensor = torch.tensor(ctrl_cells[: args.n_samples], dtype=torch.float32)
        aggregated, per_condition, predictions = run_standard_evaluation(
            adapter=adapter,
            test_conditions=test_conditions,
            ground_truth=ground_truth,
            ctrl_mean=ctrl_mean,
            n_samples=args.n_samples,
            ctrl_expr=ctrl_expr_tensor,
            adapter_kwargs={
                "cell_type": args.cell_type,
                "batch_name": args.batch_name,
                "ds_name": args.ds_name or config["data"]["dataset"],
            },
        )
        method_name = args.model

    # ================================================================
    # Save results
    # ================================================================
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(args.output_dir, f"{method_name}_results.json")

    output = {
        "method": method_name,
        "dataset": config["data"]["dataset"],
        "split": config["data"]["split_strategy"],
        "fold": config["data"]["fold"],
        "config": {
            "smc": config["smc"] if args.celldiffa else None,
            "rewards": config["rewards"] if args.celldiffa else None,
        },
        "aggregated_metrics": {k: float(v) for k, v in aggregated.items()},
        "per_condition_metrics": {
            k: {mk: float(mv) for mk, mv in v.items()} for k, v in per_condition.items()
        },
    }

    with open(result_file, "w") as f:
        json.dump(output, f, indent=2)

    # Optionally save raw predictions
    if config["logging"].get("save_predictions", False):
        pred_file = os.path.join(args.output_dir, f"{method_name}_predictions.npz")
        np.savez_compressed(pred_file, **predictions)
        print(f"  Predictions saved to: {pred_file}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  Results: {method_name}")
    print(f"  Dataset: {config['data']['dataset']} | Split: {config['data']['split_strategy']}")
    print(f"{'=' * 60}")
    for metric, value in sorted(aggregated.items()):
        direction = "↑" if "pearson" in metric or "recall" in metric else "↓"
        print(f"  {metric:25s}: {value:.4f} {direction}")
    print(f"\n  Results saved to: {result_file}")


if __name__ == "__main__":
    main()
