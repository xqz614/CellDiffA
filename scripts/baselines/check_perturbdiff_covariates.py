#!/usr/bin/env python
"""Paired numerical regression test for checkpoint vs. subset-local IDs.

This intentionally reproduces the old mapping only in an isolated diagnostic.
Both arms share weights, control cells, noise and native 32-cell attention sets.
No outcome accuracy metric, parameter selection, or prediction repair is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.perturbdiff_covariates import validate_checkpoint_covariates
from celldiffa.benchmark.released_sampling import (
    capture_rng,
    restore_rng,
    sample_native_batch,
    validate_sampling_scale,
)
from scripts.baselines.perturbdiff_sampling_entrypoint import (
    load_sampling_model_portable,
)


def value_stats(values):
    try:
        validate_sampling_scale(values)
        valid = True
    except ValueError:
        valid = False
    return {
        "cells": len(values),
        "finite": bool(np.isfinite(values).all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "entries_above_15": int((values > 15).sum()),
        "cells_above_15": int((values > 15).any(axis=1).sum()),
        "valid_scale": valid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/PerturbDiff"))
    parser.add_argument("--batch-indices", type=int, nargs="+", default=[0, 98, 302])
    parser.add_argument("--sets-per-batch", type=int, default=8)
    parser.add_argument("--sets-per-call", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda:0"], default="mps")
    args = parser.parse_args()
    if min(args.batch_indices) < 0 or min(args.sets_per_batch, args.sets_per_call) < 1:
        parser.error("Batch indices must be nonnegative and set counts positive.")
    # Exclusive directory prevents accidental overwriting of paired evidence.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import pytorch_lightning as pl
    from pytorch_lightning.utilities import move_data_to_device
    from src.apps.sampling.sampling_generation_helpers import (
        collect_batch_covariates,
        load_selected_genes,
    )
    from src.apps.sampling.sampling_setup import build_sampling_datamodule, populate_covariate_cfg

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("covariate_check")
    torch.set_num_threads(2)
    original = json.loads(args.contract.read_text())
    cfg = OmegaConf.create(original["config"])
    cfg.device = args.device
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.data.persistent_workers = False
    cfg.data.pin_memory = False
    pl.seed_everything(args.seed)
    dm = build_sampling_datamodule(cfg, logger)
    populate_covariate_cfg(cfg, dm)
    legacy_dicts = {
        key: dict(getattr(dm, key)) for key in ("pert_dict", "cell_type_dict", "batch_dict")
    }
    model = load_sampling_model_portable(cfg, logger, dm)
    device = torch.device(args.device)
    model.to(device).eval()
    dm.setup_dataset()
    validate_checkpoint_covariates(dm, model.cov_encoder.cov_cfg)
    loader = dm.val_dataloader()[0 if original["evaluation_split"] == "validation" else 1]
    if max(args.batch_indices) >= len(loader):
        raise ValueError("Diagnostic batch index is outside the original loader.")
    genes = [str(g) for g in load_selected_genes(cfg)]
    inverse = {}
    for tensor_key, dictionary in (
        ("cov_pert", "pert_dict"),
        ("cov_celltype", "cell_type_dict"),
        ("cov_batch", "batch_dict"),
    ):
        table = torch.full((len(getattr(dm, dictionary)),), -1, dtype=torch.long, device=device)
        for name, old_id in legacy_dicts[dictionary].items():
            table[getattr(dm, dictionary)[name]] = old_id
        inverse[tensor_key] = table
    report = {
        "status": "running",
        "source_contract": str(args.contract.resolve()),
        "source_contract_sha256": sha256_file(args.contract),
        "alignment": dm.checkpoint_covariate_alignment,
        "device": str(device),
        "seed": args.seed,
        "batch_indices": args.batch_indices,
        "sets_per_batch": args.sets_per_batch,
        "sets_per_call": args.sets_per_call,
        "paired_controls_and_noise": True,
        "accuracy_metrics_used": False,
        "same_native_sampler_as_full_run": True,
        "comparisons": [],
    }
    report_path = args.output_dir / "comparison.json"
    write_manifest(report_path, report)
    native_size = int(cfg.data.use_cell_set)
    wanted = set(args.batch_indices)
    with torch.no_grad():
        # Iterating the index sampler skips unwanted HDF5 reads and all unwanted
        # model calls while preserving the original batch/group membership.
        for batch_index, keys in enumerate(loader.batch_sampler):
            if batch_index > max(wanted):
                break
            if batch_index not in wanted:
                continue
            if len(keys) % native_size:
                raise ValueError("Diagnostic must retain whole native cell sets.")
            group_count = len(keys) // native_size
            selected = np.linspace(
                0, group_count - 1, min(args.sets_per_batch, group_count), dtype=int
            )
            for offset in range(0, len(selected), args.sets_per_call):
                groups = selected[offset : offset + args.sets_per_call]
                np.random.seed(args.seed + batch_index * 100 + offset)
                rows = [
                    key
                    for group in groups
                    for key in keys[group * native_size : (group + 1) * native_size]
                ]
                batch = loader.collate_fn([loader.dataset[key] for key in rows])
                batch = move_data_to_device(batch, device)
                mask = ~batch["is_padded_list"].bool()
                covs = collect_batch_covariates(batch, loader, dm, mask)
                labels = np.concatenate([entry[0] for entry in covs]).astype(str)
                record = {
                    "batch_index": batch_index,
                    "groups": groups.tolist(),
                    "perturbations": sorted(set(labels)),
                }
                noise = capture_rng(device)
                outputs = {}
                for arm in ("legacy_local_ids", "checkpoint_ids"):
                    paired = dict(batch)
                    if arm == "legacy_local_ids":
                        for key, table in inverse.items():
                            paired[key] = table[batch[key]]
                            if (paired[key] < 0).any():
                                raise ValueError("A diagnostic category has no legacy index.")
                    restore_rng(noise, device)
                    values = sample_native_batch(model, model.diffusion, cfg, device, paired, genes)
                    record[arm] = value_stats(values)
                    outputs[arm] = values
                    logger.info(
                        "Paired batch=%s groups=%s arm=%s stats=%s",
                        batch_index,
                        groups.tolist(),
                        arm,
                        record[arm],
                    )
                np.savez_compressed(
                    args.output_dir / f"batch_{batch_index:05d}_{offset:03d}.npz",
                    labels=labels,
                    genes=np.asarray(genes),
                    **outputs,
                )
                report["comparisons"].append(record)
                write_manifest(report_path, report)
    if not report["comparisons"]:
        raise RuntimeError("No paired comparisons ran.")
    passed = all(row["checkpoint_ids"]["valid_scale"] for row in report["comparisons"])
    report["status"] = "passed" if passed else "failed"
    report["legacy_scale_failure_reproduced"] = any(
        not row["legacy_local_ids"]["valid_scale"] for row in report["comparisons"]
    )
    write_manifest(report_path, report)
    if not passed:
        raise RuntimeError("Corrected samples still fail scale checks; do not launch a full rerun.")
    logger.info("Paired numerical checks passed: %s", report_path)


if __name__ == "__main__":
    main()
