"""Resumable Replogle orchestration around the author's unchanged sampler.

The native cell sets, covariates, DDIM implementation and random stream are
preserved. Per-batch training/test scores are deliberately not computed here;
the fixed evaluator runs after all predictions pass the shared contract.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from omegaconf import OmegaConf

from .artifacts import write_manifest
from .contracts import build_prediction_anndata
from .perturbdiff_covariates import validate_checkpoint_covariates
from .perturbdiff_split import PerturbDiffSplit


def capture_rng(device):
    state = {"cpu": torch.get_rng_state(), "numpy": np.random.get_state()}
    if device.type == "mps":
        state["accelerator"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["accelerator"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["accelerator"])
    elif device.type == "cuda":
        torch.cuda.set_rng_state(state["accelerator"], device)


def validate_sampling_scale(values):
    """Match the evaluator's log-expression scale guard; never repair values."""
    if not np.isfinite(values).all():
        raise ValueError("Sampling output contains non-finite expression values.")
    if values.size and (values.min() < 0 or values.max() > 15):
        raise ValueError(
            f"Sampling output is outside the evaluator's log-expression scale: "
            f"min={float(values.min()):.6g}, max={float(values.max()):.6g}. "
            "Raw values are preserved; do not clip or renormalize to bypass this check."
        )


def sample_native_batch(model, diffusion, cfg, device, batch, genes):
    """One unchanged upstream sample call, shared by inference and diagnostics."""
    from src.apps.sampling.sampling_generation_helpers import (
        build_gene_embedding_cache,
        build_self_condition,
        resolve_sampling_runner,
    )

    batch["batch_emb"] = model._encode_covariates(batch)
    gene_embeddings = build_gene_embedding_cache(model, batch, device)
    condition = build_self_condition(cfg, model, batch, gene_embeddings)
    sample_fn, sampling_kwargs = resolve_sampling_runner(cfg, diffusion, cfg.sampling.use_ddim)
    samples, _ = sample_fn(
        model.model,
        tuple(batch["pert_emb"].shape),
        self_condition=condition,
        clip_denoised=cfg.sampling.clip_denoised,
        device=device,
        progress=False,
        **sampling_kwargs,
    )
    if samples is None:
        raise RuntimeError("Native sampler returned no samples")
    columns = [str(g) for g in batch["col_genes"][0]]
    if any([str(g) for g in row] != columns for row in batch["col_genes"]):
        raise ValueError("Mixed gene order inside native batch")
    lookup = {name: i for i, name in enumerate(columns)}
    selected = torch.tensor([lookup[g] for g in genes], device=device)
    masks = ~batch["is_padded_list"].bool()
    values = samples[masks].index_select(-1, selected).float().cpu().numpy()
    values *= float(cfg.data.normalize_counts or 1.0)
    return values


def generate_samples(model, diffusion, cfg, device, logger, datamodule, **kwargs):
    from pytorch_lightning.utilities import move_data_to_device
    from src.apps.sampling.sampling_generation_helpers import (
        collect_batch_covariates,
        load_selected_genes,
    )

    device = torch.device(device)
    validate_checkpoint_covariates(datamodule, model.cov_encoder.cov_cfg)
    root = Path(cfg.sampling.output_dir).resolve()
    shards = root / "batches"
    shards.mkdir(parents=True, exist_ok=True)
    evaluation_split = os.environ.get("CELLDIFFA_EVALUATION_SPLIT", "test")
    if evaluation_split not in {"test", "validation"}:
        raise ValueError("Unknown evaluation split")
    contract = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "torch_version": torch.__version__,
        "evaluation_split": evaluation_split,
        "orchestration": "native_sampler_resumable_v1",
        "covariate_alignment": datamodule.checkpoint_covariate_alignment,
    }
    contract_path = root / "sampling_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Sampling settings changed; use a separate output directory")
    write_manifest(contract_path, contract)
    loader = datamodule.val_dataloader()[0 if evaluation_split == "validation" else 1]
    model.to(device).eval()
    genes = [str(g) for g in load_selected_genes(cfg)]
    if len(genes) != 2000:
        raise ValueError("Replogle requires the published 2000 evaluation genes")
    limit = cfg.sampling.num_sampled_batches
    complete_run = limit is None
    limit = len(loader) if limit is None else min(int(limit), len(loader))
    predictions, labels = [], []
    began = time.monotonic()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if index >= limit:
                break
            path, rng_path = shards / f"batch_{index:05d}.npz", shards / f"batch_{index:05d}.rng.pt"
            batch = move_data_to_device(batch, device)
            masks = ~batch["is_padded_list"].bool()
            covariates = collect_batch_covariates(batch, loader, datamodule, masks)
            batch_labels = np.concatenate([entry[0] for entry in covariates]).astype(str)
            if path.exists() and rng_path.exists():
                with np.load(path, allow_pickle=False) as stored:
                    values = stored["values"]
                    if not np.array_equal(stored["labels"], batch_labels):
                        raise ValueError(f"Resume batch labels changed: {path}")
                restore_rng(torch.load(rng_path, weights_only=False, map_location="cpu"), device)
            else:
                started = time.monotonic()
                values = sample_native_batch(model, diffusion, cfg, device, batch, genes)
                if values.shape != (len(batch_labels), 2000):
                    raise RuntimeError("Unexpected native sampling output shape")
                # Store the unmodified output. The final evaluator adapter clips
                # tiny negative numerical values consistently across baselines.
                temporary = path.with_suffix(".partial.npz")
                np.savez_compressed(temporary, values=values, labels=batch_labels)
                temporary.replace(path)
                rng_temporary = rng_path.with_suffix(".partial.pt")
                torch.save(capture_rng(device), rng_temporary)
                rng_temporary.replace(rng_path)
                logger.info(
                    "Saved native batch %s/%s, %s cells, %.2f seconds",
                    index + 1,
                    limit,
                    len(values),
                    time.monotonic() - started,
                )
            try:
                validate_sampling_scale(values)
            except ValueError as exc:
                write_manifest(
                    root / "invalid_output.json",
                    {
                        "batch_index": index,
                        "raw_batch": str(path),
                        "error": str(exc),
                    },
                )
                raise
            predictions.append(values)
            labels.append(batch_labels)
            write_manifest(
                root / "progress.json",
                {
                    "completed_batches": index + 1,
                    "total_batches": len(loader),
                    "complete_run": complete_run,
                    "elapsed_seconds": time.monotonic() - began,
                    "sampled_cells": sum(len(x) for x in labels),
                },
            )
    values, labels = np.concatenate(predictions), np.concatenate(labels)
    if complete_run:
        reference_path = os.environ.get("CELLDIFFA_REAL_TEST")
        if not reference_path:
            raise ValueError("CELLDIFFA_REAL_TEST must name the immutable evaluation reference")
        reference = ad.read_h5ad(reference_path)
        split = PerturbDiffSplit.from_yaml(
            Path(os.environ["PERTURBDIFF_ROOT"]) / "configs/data/perturb_data/replogle.yaml"
        )
        split.validate_reference(reference, split_name=evaluation_split)
        if reference.var_names.tolist() != genes:
            raise ValueError("Reference gene order differs from the published sampling order")
        output = build_prediction_anndata(
            reference,
            {name: values[labels == name].clip(min=0) for name in sorted(set(labels))},
            pert_col=split.pert_col,
            control_pert=split.control_pert,
        )
        output.uns["negative_value_policy"] = "clip normalized log expression at zero"
        temporary = root / "predictions.partial.h5ad"
        output.write_h5ad(temporary, compression="gzip")
        temporary.replace(root / "predictions.h5ad")
        logger.info("Full native sampling complete: %s", root / "predictions.h5ad")
    else:
        logger.info("Native sampling smoke complete; no full prediction file written")
    return None, values, [], None
