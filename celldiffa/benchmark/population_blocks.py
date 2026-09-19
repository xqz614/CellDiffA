"""Joint steering of native diffusion blocks without changing their attention.

Each candidate concatenates several original cell sets. Model forwards still
see the exact original set size and covariates, while rewards and resampling
operate on the union of valid cells. Padding is never counted as biological data.
"""

from __future__ import annotations

import numpy as np
import torch


def slice_condition(condition, index):
    return {
        key: value[index : index + 1] if isinstance(value, (torch.Tensor, list)) else value
        for key, value in condition.items()
    }


def pack_native_groups(groups):
    first = groups[0]
    if any(group["genes"] != first["genes"] for group in groups):
        raise ValueError("Cannot combine native sets with different gene orders.")
    covariates = tuple(
        np.concatenate([group["covariates"][i] for group in groups]) for i in range(3)
    )
    if len(np.unique(covariates[0])) != 1 or len(np.unique(covariates[1])) != 1:
        raise ValueError("A candidate population must have one perturbation and cell context.")
    condition = {}
    for key, value in first["condition"].items():
        values = [group["condition"][key] for group in groups]
        if isinstance(value, torch.Tensor):
            condition[key] = torch.cat(values, dim=0)
        elif isinstance(value, list):
            condition[key] = sum(values, [])
        elif value is None:
            condition[key] = None
        else:
            raise TypeError(f"Unsupported native condition {key}: {type(value)}")
    controls = condition["cont_emb"].flatten(0, 1)
    mask = torch.cat([group["mask"] for group in groups])
    return (
        {"col_genes": [first["genes"]]},
        {"cont_emb": controls.unsqueeze(0)},
        mask.unsqueeze(0),
        [covariates],
        [condition],
    )


def iter_sampling_populations(dataloader, model, cfg, device, datamodule, *, max_blocks=1):
    from pytorch_lightning.utilities import move_data_to_device
    from src.apps.sampling.sampling_generation_helpers import (
        build_gene_embedding_cache,
        build_self_condition,
        collect_batch_covariates,
    )

    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    pending, previous_key = [], None
    for batch in dataloader:
        batch = move_data_to_device(batch, device)
        batch["batch_emb"] = model._encode_covariates(batch)
        gene_emb = build_gene_embedding_cache(model, batch, device)
        condition = build_self_condition(cfg, model, batch, gene_emb)
        masks = ~batch["is_padded_list"].bool()
        covariates = collect_batch_covariates(batch, dataloader, datamodule, masks)
        if max_blocks == 1:
            yield (
                batch,
                condition,
                masks,
                covariates,
                [slice_condition(condition, i) for i in range(len(covariates))],
            )
            continue
        for i, covariate in enumerate(covariates):
            if not len(covariate[0]):
                raise ValueError("Empty native cell set")
            key = (str(covariate[0][0]), str(covariate[1][0]))
            if pending and (key != previous_key or len(pending) == max_blocks):
                yield pack_native_groups(pending)
                pending = []
            pending.append(
                dict(
                    condition=slice_condition(condition, i),
                    mask=masks[i],
                    genes=[str(value) for value in batch["col_genes"][i]],
                    covariates=covariate,
                )
            )
            previous_key = key
    if pending:
        yield pack_native_groups(pending)


class BlockPopulationSampler:
    """Evaluate many native sets jointly but denoise each original set intact."""

    population_native = True

    def __init__(self, sampler, native_condition, *, cells_per_block, batch_cells):
        self.sampler = sampler
        self.condition = native_condition
        self.cells_per_block = cells_per_block
        self.blocks = native_condition["cont_emb"].shape[0]
        self.batch_blocks = max(1, batch_cells // cells_per_block)

    @property
    def num_timesteps(self):
        return self.sampler.num_timesteps

    def sample_noise(self, shape, device):
        return self.sampler.sample_noise(shape, device)

    def denoise_step(self, x_t, t, condition, prev_pred=None):
        particles, cells, genes = x_t.shape
        if cells != self.blocks * self.cells_per_block:
            raise ValueError("Candidate population does not contain the expected native blocks")
        native_x = x_t.reshape(-1, self.cells_per_block, genes)
        native_prev = None if prev_pred is None else prev_pred.reshape_as(native_x)
        native_t = t.repeat_interleave(self.blocks)
        # Block IDs repeat once per particle. Indexing preserves batch/context
        # conditioning for every original set after particle resampling.
        block_ids = torch.arange(self.blocks, device=x_t.device).repeat(particles)
        outputs = {"x_prev": [], "x0_pred": []}
        for start in range(0, len(native_x), self.batch_blocks):
            stop = min(start + self.batch_blocks, len(native_x))
            indices = block_ids[start:stop]
            native_condition = {}
            for key, value in self.condition.items():
                if isinstance(value, torch.Tensor):
                    native_condition[key] = value.index_select(0, indices.to(value.device))
                elif isinstance(value, list):
                    native_condition[key] = [value[index] for index in indices.cpu().tolist()]
                else:
                    native_condition[key] = value
            result = self.sampler.denoise_step(
                native_x[start:stop],
                native_t[start:stop],
                native_condition,
                prev_pred=None if native_prev is None else native_prev[start:stop],
            )
            for key in outputs:
                outputs[key].append(result[key])
        return {
            key: torch.cat(values).reshape(particles, cells, genes)
            for key, values in outputs.items()
        }
