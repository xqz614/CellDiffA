#!/usr/bin/env python
"""Native STATE network and CellLoad on audited PerturbDiff row splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from omegaconf import OmegaConf

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import read_h5ad_obs

REVISION = "da4178c930dc917dac6b56faf10a33e21bd8e905"


def audit_splits(module, obs, split):
    masks = split.masks(obs)
    controls = obs.gene.astype(str).to_numpy() == split.control_pert
    report = {}
    for name, datasets in [
        ("train", module.train_datasets),
        ("validation", module.val_datasets),
        ("test", module.test_datasets),
    ]:
        indices = np.concatenate([np.asarray(ds.indices) for ds in datasets])
        response_indices = indices[~controls[indices]]
        expected = np.flatnonzero(masks[name] & ~controls)
        if not np.array_equal(np.sort(response_indices), expected):
            raise ValueError(f"STATE {name} response rows differ from the official split")
        if name == "train" and not masks["train"][indices].all():
            raise ValueError("STATE training contains held-out response rows")
        report[name] = {
            "responses": len(response_indices),
            "controls": int(controls[indices].sum()),
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/PerturbDiff_data/finetune_data/nadig_processed_data/replogle.h5ad"),
    )
    parser.add_argument("--reference-dir", type=Path, default=Path("results/replogle/reference"))
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/state"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/replogle/state"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--set-batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Unexpected STATE revision {revision}")
    sys.path.insert(0, str(args.upstream_root.resolve() / "src"))
    import lightning.pytorch as pl
    from cell_load.data_modules import PerturbationDataModule
    from hydra import compose, initialize_config_dir
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from state.tx.utils import get_lightning_module

    pl.seed_everything(args.seed)
    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    contract.update(
        revision=revision,
        torch_version=torch.__version__,
        split_sha256=sha256_file(args.split_config),
    )
    path = args.output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("STATE output settings changed; use a new directory")
    write_manifest(path, contract)
    split = PerturbDiffSplit.from_yaml(args.split_config)
    toml = (
        "[datasets]\nreplogle = "
        + json.dumps(str(args.source.resolve()))
        + '\n[training]\nreplogle = "train"\n[fewshot."replogle.hepg2"]\nval = '
        + json.dumps(sorted(split.validation_perts))
        + "\ntest = "
        + json.dumps(sorted(split.test_perts))
        + "\n"
    )
    toml_path = args.output_dir / "official_split.toml"
    toml_path.write_text(toml)
    with initialize_config_dir(
        version_base=None, config_dir=str(args.upstream_root.resolve() / "src/state/configs")
    ):
        cfg = compose(config_name="config")
    data_cfg = OmegaConf.to_container(cfg.data.kwargs, resolve=True)
    data_cfg.update(
        toml_config_path=str(toml_path),
        embed_key="X_hvg",
        output_space="gene",
        cell_type_key="cell_line",
        control_pert=split.control_pert,
        num_workers=0,
        pin_memory=False,
        val_subsample_fraction=1.0,
    )
    module = PerturbationDataModule(
        **data_cfg, batch_size=args.set_batch_size, cell_sentence_len=512, random_seed=args.seed
    )
    module.setup("fit")
    audit = audit_splits(module, read_h5ad_obs(args.source), split)
    dimensions = module.get_var_dims()
    reference_genes = ad.read_h5ad(args.reference_dir / "controls.h5ad").var_names.tolist()
    if list(dimensions["gene_names"]) != reference_genes:
        raise ValueError("STATE output genes differ from the published evaluation order")
    write_manifest(args.output_dir / "split_audit.json", audit)
    print(f"STATE official split verified: {audit}", flush=True)
    if args.prepare_only:
        return
    training = OmegaConf.to_container(cfg.training, resolve=True)
    training.update(batch_size=args.set_batch_size, max_steps=args.max_steps, train_seed=args.seed)
    model_cfg = OmegaConf.to_container(cfg.model.kwargs, resolve=True)
    model = get_lightning_module("state", data_cfg, model_cfg, training, dimensions)
    checkpoint = ModelCheckpoint(
        dirpath=str(args.output_dir / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-{step}",
    )

    class Progress(pl.Callback):
        def on_train_start(self, trainer, module):
            self.started = time.monotonic()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"STATE optimizer_step={trainer.global_step} batches={batch_idx + 1} "
                    f"seconds={time.monotonic() - self.started:.1f}",
                    flush=True,
                )

    trainer = pl.Trainer(
        accelerator=args.device,
        devices=1,
        max_steps=2 if args.smoke else args.max_steps,
        accumulate_grad_batches=1 if args.smoke else args.accumulate,
        val_check_interval=2 if args.smoke else 1000 * args.accumulate,
        check_val_every_n_epoch=None,
        limit_val_batches=1 if args.smoke else 1.0,
        num_sanity_val_steps=0,
        gradient_clip_val=training["gradient_clip_val"],
        logger=False,
        enable_progress_bar=False,
        use_distributed_sampler=False,
        callbacks=[
            checkpoint,
            Progress(),
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ],
        default_root_dir=str(args.output_dir),
    )
    last = args.output_dir / "checkpoints/last.ckpt"
    trainer.fit(model, module, ckpt_path=str(last) if last.exists() else None)
    if checkpoint.best_model_path:
        model.load_state_dict(
            torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
    model.to(args.device).eval()
    controls = ad.read_h5ad(args.reference_dir / "controls.h5ad")
    target_obs = read_h5ad_obs(args.reference_dir / "real.h5ad")
    rng = np.random.default_rng(args.seed)
    predictions = {}
    names = sorted(set(target_obs.gene.astype(str)) - {split.control_pert})
    if args.smoke:
        names = names[:1]
    with torch.no_grad():
        for name in names:
            count = int((target_obs.gene.astype(str) == name).sum())
            parts = []
            for start in range(0, count, 512):
                size = min(512, count - start)
                values = controls.X[rng.integers(controls.n_obs, size=size)]
                values = values.toarray() if hasattr(values, "toarray") else np.asarray(values)
                batch = {
                    "ctrl_cell_emb": torch.tensor(values, dtype=torch.float32, device=args.device),
                    "pert_emb": module.pert_onehot_map[name].to(args.device).expand(size, -1),
                }
                parts.append(model.forward(batch, padded=False).cpu().numpy())
            predictions[name] = np.concatenate(parts)
    if args.smoke:
        print("STATE native training/prediction smoke complete; no formal predictions", flush=True)
        return
    real = ad.read_h5ad(args.reference_dir / "real.h5ad")
    result = build_prediction_anndata(
        real, predictions, pert_col="gene", control_pert=split.control_pert
    )
    result.uns["baseline"] = "STATE (official architecture, trained from scratch)"
    temporary = args.output_dir / "predictions.partial.h5ad"
    result.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, args.output_dir / "predictions.h5ad")


if __name__ == "__main__":
    main()
