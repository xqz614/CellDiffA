#!/usr/bin/env python
"""Pinned CPA with official row splits and control-only prediction inputs.

Use the isolated adacell-cpa environment. Native one-hot perturbation embeddings
are retained, including their inability to learn truly unseen intervention IDs.
Metadata-only rows register those IDs but never enter training or validation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import read_h5ad_obs

REVISION = "fbd7c0250edc23eff003a10c99655579c53afd63"


def install_mps_device_parser():
    """Map the legacy scvi device argument to Lightning's existing MPS support."""
    import cpa._data
    import scvi.train._trainrunner
    from scvi.model._utils import parse_use_gpu_arg

    def parse(value=None, return_device=True):
        if value == "mps":
            result = ("mps", 1, torch.device("mps"))
            return result if return_device else result[:2]
        return parse_use_gpu_arg(value, return_device=return_device)

    cpa._data.parse_use_gpu_arg = parse
    scvi.train._trainrunner.parse_use_gpu_arg = parse


def prediction_inputs(controls, target_obs, *, seed):
    """Construct inputs without receiving the target expression matrix."""
    rng = np.random.default_rng(seed)
    chosen = rng.integers(controls.n_obs, size=len(target_obs))
    query = ad.AnnData(X=controls.X[chosen].copy(), obs=target_obs.copy(), var=controls.var.copy())
    query.obs = query.obs[["gene", "cell_line"]].astype(str)
    return query


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=Path("results/replogle/reference"))
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/cpa"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/replogle/cpa"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda:0"])
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected CPA {REVISION}, found {revision}")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import cpa
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint

    torch.set_num_threads(8)
    pl.seed_everything(args.seed)
    if args.device == "mps":
        install_mps_device_parser()
    contract = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    contract.update(
        revision=revision,
        torch_version=torch.__version__,
        split_sha256=sha256_file(args.split_config),
        representation="native one-hot perturbations and cell-line covariates",
        unknown_perturbation_policy="registered untrained embedding, never dropped",
        query_input="sampled observed hepg2 controls only",
        epochs_policy="native CPA size-based default rounds to 13 on this training set",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != contract:
        raise ValueError("Output settings differ; use a new directory")
    write_manifest(config_path, contract)
    train = ad.read_h5ad(args.reference_dir / "train.h5ad")
    validation = ad.read_h5ad(args.reference_dir / "validation.h5ad")
    split = PerturbDiffSplit.from_yaml(args.split_config)
    if not split.masks(train.obs)["train"].all():
        raise ValueError("Held-out expression in training input")
    split.validate_reference(validation, split_name="validation")
    if not train.var_names.equals(validation.var_names):
        raise ValueError("Train and validation genes differ")
    validation = validation[validation.obs.gene.astype(str) != split.control_pert].copy()
    if args.smoke:
        train = train[
            np.random.default_rng(args.seed).choice(train.n_obs, 512, replace=False)
        ].copy()
        validation = validation[:128].copy()
    target_obs = read_h5ad_obs(args.reference_dir / "real.h5ad")
    vocabulary = sorted(set(target_obs.gene.astype(str)))
    metadata = ad.AnnData(
        X=sparse.csr_matrix((len(vocabulary), train.n_vars), dtype=np.float32),
        obs=pd.DataFrame(
            {"gene": vocabulary, "cell_line": split.holdout_contexts[0]},
            index=[f"metadata_only_{i}" for i in range(len(vocabulary))],
        ),
        var=train.var.copy(),
    )
    for data, label in [(train, "train"), (validation, "validation"), (metadata, "metadata")]:
        data.obs = data.obs[["gene", "cell_line"]].astype(str)
        data.obs["official_split"] = label
    data = ad.concat([train, validation, metadata], join="inner", merge="same")
    del train, validation, metadata
    gc.collect()
    print(f"CPA registry input {data.shape}; metadata rows excluded from fitting", flush=True)
    cpa.CPA.setup_anndata(
        data,
        perturbation_key="gene",
        control_group=split.control_pert,
        categorical_covariate_keys=["cell_line"],
        is_count_data=False,
        max_comb_len=1,
    )
    model = cpa.CPA(
        data,
        split_key="official_split",
        train_split="train",
        valid_split="validation",
        test_split="metadata",
        recon_loss="gauss",
        seed=args.seed,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(args.output_dir / "checkpoints"),
        monitor="cpa_metric",
        mode="max",
        save_last=True,
        save_top_k=1,
        filename="best-{epoch}",
    )

    class Progress(pl.Callback):
        def on_train_epoch_start(self, trainer, module):
            self.started = time.monotonic()

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if (batch_idx + 1) % 200 == 0:
                print(
                    f"CPA epoch={trainer.current_epoch + 1} batch={batch_idx + 1} "
                    f"seconds={time.monotonic() - self.started:.1f}",
                    flush=True,
                )

    # Preserve native TrainRunner bookkeeping while enabling resumable Lightning fits.
    from scvi.train import TrainRunner

    native_trainer = TrainRunner._trainer_cls
    last = args.output_dir / "checkpoints/last.ckpt"

    class ResumableTrainer(native_trainer):
        def fit(self, *fit_args, **kwargs):
            if last.exists():
                kwargs["ckpt_path"] = str(last)
            return super().fit(*fit_args, **kwargs)

    TrainRunner._trainer_cls = ResumableTrainer
    kwargs = {"limit_train_batches": 2, "limit_val_batches": 1} if args.smoke else {}
    model.train(
        max_epochs=1 if args.smoke else args.epochs,
        use_gpu=False if args.device == "cpu" else args.device,
        batch_size=args.batch_size,
        check_val_every_n_epoch=1,
        save_path=False,
        callbacks=[checkpoint, Progress()],
        enable_checkpointing=True,
        enable_progress_bar=False,
        default_root_dir=str(args.output_dir),
        plan_kwargs={"n_epochs_verbose": 1},
        **kwargs,
    )
    model.epoch_history.to_csv(args.output_dir / "training_history.csv", index=False)
    if not args.smoke:
        model.save(str(args.output_dir / "model"), overwrite=True)
    controls = ad.read_h5ad(args.reference_dir / "controls.h5ad")
    treated = target_obs.loc[target_obs.gene.astype(str) != split.control_pert]
    if args.smoke:
        treated = treated.iloc[:16]
    query = prediction_inputs(controls, treated, seed=args.seed)
    cpa.CPA.setup_anndata(
        query,
        perturbation_key="gene",
        control_group=split.control_pert,
        categorical_covariate_keys=["cell_line"],
        is_count_data=False,
        max_comb_len=1,
    )
    model.predict(query, batch_size=args.batch_size)
    values = np.asarray(query.obsm["CPA_pred"], dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("CPA predictions are not finite")
    if args.smoke:
        print(f"CPA train/predict smoke finished: {values.shape}; no formal output", flush=True)
        return
    real = ad.read_h5ad(args.reference_dir / "real.h5ad")
    split.validate_real_test(real)
    predictions = {
        name: values[query.obs.gene.to_numpy() == name].clip(min=0)
        for name in sorted(set(query.obs.gene))
    }
    output = build_prediction_anndata(
        real, predictions, pert_col="gene", control_pert=split.control_pert
    )
    output.uns["baseline"] = "CPA (official one-hot architecture)"
    output.uns["negative_value_policy"] = "clip normalized log expression at zero"
    temporary = args.output_dir / "predictions.partial.h5ad"
    output.write_h5ad(temporary, compression="gzip")
    os.replace(temporary, args.output_dir / "predictions.h5ad")
    print(f"WROTE {args.output_dir / 'predictions.h5ad'}", flush=True)


if __name__ == "__main__":
    main()
