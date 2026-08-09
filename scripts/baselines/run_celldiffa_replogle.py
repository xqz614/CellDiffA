#!/usr/bin/env python
"""Run population-native CellDiffA on PerturbDiff's released Replogle task."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PERTURBDIFF_REVISION = "f4e27c155be5325418c4cb3182453d4022754e91"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--real-test", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--selected-genes", required=True)
    parser.add_argument("--prior-cache", required=True)
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=["scratch", "finetuned"], required=True)
    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--particle-batch-cells", type=int, default=128)
    parser.add_argument("--ess-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--top-de", type=int, default=20)
    parser.add_argument("--anchor-bandwidth", type=float, default=1.0)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args, hydra_overrides = parser.parse_known_args()
    if args.num_workers < 1 or not 0 <= args.worker_index < args.num_workers:
        parser.error("worker-index must be in [0, num-workers).")
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("max-groups must be positive.")
    return args, hydra_overrides


def _group_value(value, index: int):
    import torch

    if isinstance(value, torch.Tensor):
        return value[index : index + 1]
    if isinstance(value, list):
        return [value[index]]
    return value


def main() -> None:
    args, hydra_overrides = parse_args()
    upstream_root = Path(os.environ.get("PERTURBDIFF_ROOT", "external/PerturbDiff")).resolve()
    if not (upstream_root / "configs").is_dir():
        raise SystemExit(f"Missing PerturbDiff checkout: {upstream_root}")
    revision = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PERTURBDIFF_REVISION:
        raise RuntimeError(f"PerturbDiff revision is {revision}; expected {PERTURBDIFF_REVISION}.")
    sys.path.insert(0, str(upstream_root))

    import anndata as ad
    import numpy as np
    import pytorch_lightning as pl
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pytorch_lightning.utilities import move_data_to_device
    from src.apps.sampling.sampling_generation_helpers import (
        build_gene_embedding_cache,
        build_self_condition,
        collect_batch_covariates,
    )
    from src.apps.sampling.sampling_setup import (
        build_sampling_datamodule,
        populate_covariate_cfg,
    )
    from src.apps.sampling.sampling_utils import setup_device
    from src.common.utils import setup_loggings

    from baselines.adapter_perturbdiff import PerturbDiffSampler
    from celldiffa.benchmark.artifacts import load_selected_genes, sha256_file, write_manifest
    from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
    from celldiffa.benchmark.replogle_priors import compute_replogle_training_priors
    from celldiffa.benchmark.replogle_shards import (
        assemble_replogle_shards,
        load_group_shard,
        save_group_shard,
        shard_path,
    )
    from celldiffa.rewards import (
        AnchorReward,
        CompositeReward,
        GeometricReward,
        ProjectedReward,
        TranscriptomicReward,
    )
    from celldiffa.smc import SMCConfig, SMCEngine
    from scripts.baselines.perturbdiff_sampling_entrypoint import (
        load_sampling_model_portable,
    )

    required = [
        args.source,
        args.real_test,
        args.split_config,
        args.selected_genes,
    ]
    for value in required:
        if not Path(value).exists():
            raise FileNotFoundError(value)
    selected_genes = load_selected_genes(args.selected_genes)
    split = PerturbDiffSplit.from_yaml(args.split_config, split_axis="context")
    real = ad.read_h5ad(args.real_test)
    split.validate_real_test(real)
    if list(real.var_names.astype(str)) != selected_genes:
        raise ValueError("Selected genes and real-test H5AD have different order.")
    real_labels = real.obs[split.pert_col].astype(str)
    test_perturbations = sorted(set(real_labels) - {split.control_pert})

    with initialize_config_dir(version_base=None, config_dir=str(upstream_root / "configs")):
        cfg = compose(config_name="rawdata_diffusion_sampling", overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    output_path = Path(args.output).resolve()
    shard_root = Path(args.shard_root).resolve()
    shard_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(cfg.model_checkpoint_path).resolve()
    run_config = {
        "format_version": 1,
        "variant": args.variant,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_path.stat().st_size,
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "source": str(Path(args.source).resolve()),
        "selected_genes": str(Path(args.selected_genes).resolve()),
        "selected_genes_sha256": sha256_file(args.selected_genes),
        "split_config_sha256": sha256_file(args.split_config),
        "upstream_revision": revision,
        "num_particles": args.num_particles,
        "particle_batch_cells": args.particle_batch_cells,
        "ess_threshold": args.ess_threshold,
        "alpha": args.alpha,
        "top_de": args.top_de,
        "anchor_bandwidth": args.anchor_bandwidth,
        "seed": args.seed,
        "normalize_counts": float(cfg.data.normalize_counts or 1.0),
        "cell_set": int(cfg.data.use_cell_set),
        "start_time": int(cfg.sampling.start_time),
        "eta": float(cfg.sampling.eta),
        "guidance_strength": float(cfg.sampling.guidance_strength),
    }
    run_config_path = shard_root / "run_config.json"
    # One worker establishes the run contract and prior cache; other GPU
    # workers wait, then reuse both. This prevents cache races and accidental
    # mixing of shards from different CellDiffA settings.
    with (shard_root / ".prepare.lock").open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        if run_config_path.exists():
            observed = json.loads(run_config_path.read_text(encoding="utf-8"))
            if observed != run_config:
                raise ValueError(
                    "Existing shard directory was created with different settings. "
                    "Use a new output directory instead of mixing runs."
                )
        else:
            write_manifest(run_config_path, run_config)
        priors = compute_replogle_training_priors(
            args.source,
            args.split_config,
            selected_genes,
            cache_path=args.prior_cache,
            expression_key="X_hvg",
            top_k=max(args.top_de, 50),
            target_perturbations=test_perturbations,
        )
        fcntl.flock(lock_handle, fcntl.LOCK_UN)

    logger = setup_loggings(cfg)
    pl.seed_everything(args.seed)
    datamodule = build_sampling_datamodule(cfg, logger)
    populate_covariate_cfg(cfg, datamodule)
    model = load_sampling_model_portable(cfg, logger, datamodule)
    device = setup_device(cfg, logger)
    model = model.to(device)
    model.eval()
    datamodule.setup_dataset()
    dataloader = datamodule.val_dataloader()[1]
    normalize_counts = float(cfg.data.normalize_counts or 1.0)
    scaled_shifts = {name: values / normalize_counts for name, values in priors.shifts.items()}

    progress_path = shard_root / f"worker_{args.worker_index:03d}.progress.json"
    processed_now = 0
    selected_now = 0
    skipped = 0
    global_group = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_data in dataloader:
            batch_data = move_data_to_device(batch_data, device)
            batch_data["batch_emb"] = model._encode_covariates(batch_data)
            gene_emb = build_gene_embedding_cache(model, batch_data, device)
            self_condition = build_self_condition(cfg, model, batch_data, gene_emb)
            valid_mask = ~batch_data["is_padded_list"].bool()
            covariates = collect_batch_covariates(batch_data, dataloader, datamodule, valid_mask)

            for item_index, covariate in enumerate(covariates):
                group_index = global_group
                global_group += 1
                if group_index % args.num_workers != args.worker_index:
                    continue
                if args.max_groups is not None and selected_now >= args.max_groups:
                    break
                selected_now += 1
                labels = np.asarray(covariate[0]).astype(str)
                unique_labels = np.unique(labels)
                if len(unique_labels) != 1:
                    raise ValueError(
                        f"Official group {group_index} mixes perturbations {unique_labels}."
                    )
                perturbation = str(unique_labels[0])
                path = shard_path(shard_root, group_index)
                if path.exists() and not args.no_resume:
                    saved_index, saved_pert, saved_values = load_group_shard(path)
                    if (
                        saved_index != group_index
                        or saved_pert != perturbation
                        or saved_values.shape != (len(labels), len(selected_genes))
                    ):
                        raise ValueError(f"Resume shard does not match group {group_index}: {path}")
                    skipped += 1
                    continue

                condition = {
                    key: _group_value(value, item_index) for key, value in self_condition.items()
                }
                controls = condition["cont_emb"][0]
                mask = valid_mask[item_index]
                col_genes = [str(value) for value in batch_data["col_genes"][item_index]]
                gene_lookup = {gene: index for index, gene in enumerate(col_genes)}
                missing_genes = [gene for gene in selected_genes if gene not in gene_lookup]
                if missing_genes:
                    raise ValueError(
                        f"Model group is missing evaluation genes: {missing_genes[:20]}"
                    )
                eval_indices = np.asarray(
                    [gene_lookup[gene] for gene in selected_genes], dtype=np.int64
                )
                eval_controls = (
                    controls[mask]
                    .index_select(-1, torch.as_tensor(eval_indices, device=device))
                    .float()
                )
                ctrl_mean = eval_controls.mean(dim=0).cpu().numpy()
                base_reward = CompositeReward(
                    [
                        TranscriptomicReward(
                            priors.de_genes,
                            scaled_shifts,
                            ctrl_mean,
                            selected_genes,
                            weight=1.0,
                            top_k=args.top_de,
                        ),
                        GeometricReward(
                            scaled_shifts,
                            ctrl_mean,
                            weight=1.0,
                        ),
                        AnchorReward(
                            scaled_shifts,
                            weight=1.0,
                            bandwidth=args.anchor_bandwidth,
                        ),
                    ],
                    normalization="zscore",
                )
                reward = ProjectedReward(
                    base_reward,
                    gene_indices=eval_indices,
                    cell_mask=mask,
                )
                sampler = PerturbDiffSampler(
                    model,
                    model.diffusion,
                    condition,
                    device=str(device),
                    guidance_strength=float(cfg.sampling.guidance_strength),
                    eta=float(cfg.sampling.eta),
                    start_time=int(cfg.sampling.start_time),
                    clip_denoised=bool(cfg.sampling.clip_denoised),
                )
                smc_config = SMCConfig(
                    num_particles=args.num_particles,
                    cells_per_particle=int(controls.shape[0]),
                    ess_threshold=args.ess_threshold,
                    alpha=args.alpha,
                    start_timestep=None,
                    eta=float(cfg.sampling.eta),
                    guidance_strength=float(cfg.sampling.guidance_strength),
                    output_mode="map",
                    device=str(device),
                    batch_size_per_step=args.particle_batch_cells,
                    seed=args.seed + group_index,
                )
                result = SMCEngine(sampler, reward, smc_config).sample_with_alignment(
                    perturbation,
                    condition,
                    ctrl_cells=controls,
                    num_genes=controls.shape[-1],
                )
                samples = result["samples"][mask]
                samples = samples.index_select(
                    -1, torch.as_tensor(eval_indices, device=samples.device)
                )
                values = torch.clamp(samples * normalize_counts, min=0).cpu().numpy()
                save_group_shard(
                    shard_root,
                    group_index,
                    perturbation,
                    values,
                )
                processed_now += 1
                logger.info(
                    "CellDiffA group %s: perturbation=%s cells=%s processed_now=%s",
                    group_index,
                    perturbation,
                    len(labels),
                    processed_now,
                )
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            write_manifest(
                progress_path,
                {
                    "worker_index": args.worker_index,
                    "num_workers": args.num_workers,
                    "processed_now": processed_now,
                    "selected_now": selected_now,
                    "skipped_existing": skipped,
                    "last_global_group": global_group - 1,
                    "elapsed_seconds": time.time() - start_time,
                },
            )
            if args.max_groups is not None and selected_now >= args.max_groups:
                break

    pred, status = assemble_replogle_shards(
        args.real_test,
        shard_root,
        output_path,
        pert_col=split.pert_col,
        control_pert=split.control_pert,
        require_complete=False,
        write_output=args.num_workers == 1 and args.max_groups is None,
    )
    manifest = {
        "method": "CellDiffA",
        "dataset": "Replogle-Nadig",
        "base_model": f"PerturbDiff {args.variant} released checkpoint",
        "upstream_revision": PERTURBDIFF_REVISION,
        "population_native": True,
        "official_split": str(Path(args.split_config).resolve()),
        "prior_policy": (
            "official training mask only; context-specific control centering; "
            "no held-out perturbed expression"
        ),
        "evaluation_genes": len(selected_genes),
        "normalization_scale": normalize_counts,
        "num_particles": args.num_particles,
        "cells_per_particle": int(cfg.data.use_cell_set),
        "ess_threshold": args.ess_threshold,
        "alpha": args.alpha,
        "top_de": args.top_de,
        "seed": args.seed,
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "max_groups": args.max_groups,
        "processed_now": processed_now,
        "selected_now": selected_now,
        "skipped_existing": skipped,
        "elapsed_seconds": time.time() - start_time,
        "status": status,
        "evaluator_ready_output_written": pred is not None,
    }
    write_manifest(shard_root / f"worker_{args.worker_index:03d}.manifest.json", manifest)
    print(json.dumps(status, indent=2, sort_keys=True))
    if pred is None:
        print("Partial run complete. No evaluator-ready H5AD was written.")
    else:
        print(f"Wrote evaluator-ready CellDiffA prediction: {output_path}")


if __name__ == "__main__":
    main()
