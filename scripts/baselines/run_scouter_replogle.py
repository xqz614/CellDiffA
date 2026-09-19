#!/usr/bin/env python
"""Train the pinned official Scouter architecture/loss on the official split.

Only the orchestration differs from the author loop: explicit validation rows,
atomic best/last checkpoints (the author loop retains a mutable state_dict),
and prediction from observed held-out-context controls rather than test inputs.
The fixed 40-epoch/Adam defaults and balanced control pairing follow upstream.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from celldiffa.benchmark.artifacts import load_embedding_dict, sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.grouped_loss import ScouterGroupedLoss
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit

REVISION = "bf763aaf87f162fdf28bec3145d254daaddfef84"


def encode_conditions(data, embeddings, *, control="non-targeting"):
    """Validate complete coverage before invoking upstream's filtering helpers."""
    names = sorted(embeddings)
    lookup = {name: index for index, name in enumerate(names)}
    labels = data.obs["gene"].astype(str)
    missing = sorted(set(labels) - set(lookup))
    if missing:
        raise ValueError(f"Scouter embeddings are missing perturbations: {missing}")
    data.obs["condition"] = labels.replace({control: "ctrl"})
    data.obs["embd_index"] = labels.map(lambda name: [lookup[name]])
    data.X = sparse.csr_matrix(data.X, dtype=np.float32)
    return torch.tensor(np.stack([embeddings[name] for name in names]), dtype=torch.float32)


def training_nonzero_genes(train):
    labels = train.obs["condition"].astype(str).to_numpy()
    return {
        name: np.flatnonzero(np.asarray(train.X[labels == name].sum(axis=0)).ravel() != 0)
        for name in np.unique(labels)
    }


def save_checkpoint(path, payload):
    temporary = path.with_suffix(path.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=Path("results/replogle/reference"))
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/scouter"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/replogle/scouter"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Scouter {REVISION}, found {revision}")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    from scouter._datasets import BalancedDataset
    from scouter._model import ScouterModel

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    contract.update(
        source_revision=revision,
        split_sha256=sha256_file(args.split_config),
        embedding_sha256=sha256_file(args.embeddings),
        torch_version=torch.__version__,
        embedding_source="published GenePT perturbation descriptors",
        context_policy="pooled training; observed hepg2 controls at inference",
        loss_implementation="vectorized exact official condition-balanced reduction",
        unknown_validation_gene_filter="all expression genes, no validation DE selection",
    )
    contract_path = args.output_dir / "run_config.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Output directory has different settings; use a new directory.")
    write_manifest(contract_path, contract)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    train = ad.read_h5ad(args.reference_dir / "train.h5ad")
    validation = ad.read_h5ad(args.reference_dir / "validation.h5ad")
    split = PerturbDiffSplit.from_yaml(args.split_config)
    if not split.masks(train.obs)["train"].all():
        raise ValueError("Training artifact contains held-out response rows.")
    split.validate_reference(validation, split_name="validation")
    if not train.var_names.equals(validation.var_names):
        raise ValueError("Training and validation genes differ.")
    embeddings = load_embedding_dict(args.embeddings)
    embedding_tensor = encode_conditions(train, embeddings)
    encode_conditions(validation, embeddings)
    nonzero = training_nonzero_genes(train)
    for label in validation.obs["condition"].unique():
        nonzero.setdefault(label, np.arange(train.n_vars))
    if any(len(indices) == 0 for indices in nonzero.values()):
        raise ValueError("An observed training perturbation has no expressed genes.")
    train_data = BalancedDataset(train, "condition", "embd_index", seed=args.seed)
    val_data = BalancedDataset(validation, "condition", "embd_index", seed=args.seed)
    loaders = [
        DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_data, batch_size=args.batch_size, shuffle=False),
    ]
    if not all(len(loader) for loader in loaders):
        raise ValueError("Empty training/validation loader.")
    labels = [data.obs["condition"].astype(str).to_dict() for data in (train, validation)]
    loss_function = ScouterGroupedLoss(nonzero, train.n_vars, args.device)
    network = ScouterModel(
        train.n_vars,
        embedding_tensor,
        n_encoder=(2048, 512),
        n_out_encoder=64,
        n_decoder=(2048,),
        use_batch_norm=True,
        use_layer_norm=False,
        dropout_rate=0.0,
    ).to(args.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_loss, failures, start_epoch, history = float("inf"), 0, 0, []
    last_path, best_path = args.output_dir / "last.pt", args.output_dir / "best.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_loss, failures = state["best_loss"], state["failures"]
        start_epoch, history = state["epoch"] + 1, state["history"]
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if args.device == "mps":
            torch.mps.set_rng_state(state["device_rng"])
    for epoch in range(start_epoch, args.epochs):
        if failures >= args.patience:
            break
        began = time.monotonic()
        losses = []
        for phase, loader in enumerate(loaders):
            network.train(phase == 0)
            accumulated = 0.0
            for batch_index, (ctrl, target, pert_index, barcodes) in enumerate(loader):
                ctrl, target, pert_index = (
                    value.to(args.device) for value in (ctrl, target, pert_index)
                )
                with torch.set_grad_enabled(phase == 0):
                    if phase == 0:
                        optimizer.zero_grad()
                    prediction = network(pert_index, ctrl)
                    loss = loss_function(
                        prediction,
                        target,
                        ctrl,
                        [labels[phase][barcode] for barcode in barcodes],
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Scouter produced a non-finite loss.")
                    if phase == 0:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                        optimizer.step()
                accumulated += float(loss.detach().cpu())
                if phase == 0 and (batch_index + 1) % 200 == 0:
                    print(
                        f"epoch={epoch + 1} batch={batch_index + 1}/{len(loader)} "
                        f"seconds={time.monotonic() - began:.1f}",
                        flush=True,
                    )
            losses.append(accumulated / len(loader))
        scheduler.step()
        if best_loss - losses[1] > 0.001:
            best_loss, failures = losses[1], 0
            save_checkpoint(
                best_path,
                {key: value.detach().cpu().clone() for key, value in network.state_dict().items()},
            )
        else:
            failures += 1
        record = dict(
            epoch=epoch + 1,
            train_loss=losses[0],
            validation_loss=losses[1],
            seconds=time.monotonic() - began,
        )
        history.append(record)
        print(json.dumps(record), flush=True)
        save_checkpoint(
            last_path,
            dict(
                model=copy.deepcopy(network.state_dict()),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                best_loss=best_loss,
                failures=failures,
                epoch=epoch,
                history=history,
                torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                device_rng=torch.mps.get_rng_state() if args.device == "mps" else None,
            ),
        )
        write_manifest(args.output_dir / "training_history.json", {"epochs": history})
    if not best_path.exists():
        raise RuntimeError("No validated Scouter checkpoint is available.")
    network.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    network.eval()
    # Test expression is accessed only to copy observed controls into the output.
    # No held-out treated cell is ever used as the network's input.
    real = ad.read_h5ad(args.reference_dir / "real.h5ad")
    split.validate_real_test(real)
    real_labels = real.obs["gene"].astype(str).to_numpy()
    controls = real.X[real_labels == split.control_pert]
    controls = controls.toarray() if sparse.issparse(controls) else np.asarray(controls)
    rng = np.random.default_rng(args.seed)
    lookup = {name: index for index, name in enumerate(sorted(embeddings))}
    predictions = {}
    with torch.no_grad():
        for name in sorted(set(real_labels) - {split.control_pert}):
            if name not in lookup:
                raise ValueError(f"Missing test perturbation embedding {name}")
            count = int((real_labels == name).sum())
            chosen = rng.integers(len(controls), size=count)
            parts = []
            for start in range(0, count, args.batch_size):
                ctrl = torch.tensor(
                    controls[chosen[start : start + args.batch_size]], device=args.device
                )
                index = torch.full(
                    (len(ctrl), 1), lookup[name], device=args.device, dtype=torch.long
                )
                parts.append(network(index, ctrl).cpu().numpy())
            predictions[name] = np.concatenate(parts).clip(min=0)
    output = build_prediction_anndata(
        real, predictions, pert_col="gene", control_pert=split.control_pert
    )
    output.uns["baseline"] = "Scouter (official architecture, GenePT descriptors)"
    output.uns["negative_value_policy"] = "clip to zero in normalized log expression space"
    output.write_h5ad(args.output_dir / "predictions.h5ad", compression="gzip")
    print(f"WROTE {args.output_dir / 'predictions.h5ad'}", flush=True)


if __name__ == "__main__":
    main()
