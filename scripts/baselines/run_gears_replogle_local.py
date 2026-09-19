#!/usr/bin/env python
"""GEARS with lazy native cell graphs and strict official validation rows.

No experiment is removed because a perturbation lacks GO annotation. Extra IDs
are isolated nodes in the author's default graph (SGConv adds its usual self
loops). This explicit extended-node policy is not an exact paper reproduction.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.grouped_loss import ScouterGroupedLoss
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import read_h5ad_obs
from scripts.baselines.run_scouter_replogle import save_checkpoint, training_nonzero_genes

REVISION = "f374e43e197b295016d80395d7a54ddb81cc6769"


def dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


class LazyGraphs(Dataset):
    def __init__(self, data, controls, node_map, de_indices, seed):
        self.data, self.controls = data, controls
        self.labels = data.obs.condition.astype(str).to_numpy()
        self.map, self.de = node_map, de_indices
        self.control_rows = np.random.default_rng(seed).integers(controls.n_obs, size=data.n_obs)

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, index):
        from gears import PertData

        label = self.labels[index]
        y = dense(self.data.X[index]).reshape(1, -1)
        x = (
            y
            if label == "ctrl"
            else dense(self.controls.X[self.control_rows[index]]).reshape(1, -1)
        )
        indices = None if label == "ctrl" else [self.map[label.split("+")[0]]]
        return PertData.create_cell_graph(
            None, x, y, self.de.get(label, np.arange(20)), label, indices
        )


def coexpression_graph(matrix, genes, *, chunk_size=8192):
    """Same Pearson/top-21/0.4 graph as GEARS, without a full dense copy."""
    n, width = matrix.shape
    sums = np.zeros(width, dtype=np.float64)
    gram = np.zeros((width, width), dtype=np.float64)
    for start in range(0, n, chunk_size):
        values = dense(matrix[start : start + chunk_size]).astype(np.float64)
        sums += values.sum(axis=0)
        gram += values.T @ values
    centered = gram - np.outer(sums, sums) / n
    norm = np.sqrt(np.maximum(np.diag(centered), 0))
    correlation = np.abs(
        np.divide(
            centered,
            np.outer(norm, norm),
            out=np.zeros_like(centered),
            where=np.outer(norm, norm) > 0,
        )
    )
    ranking = np.argsort(correlation, axis=1)[:, -21:]
    edges = [
        (genes[j], genes[i], correlation[i, j])
        for i, row in enumerate(ranking)
        for j in row
        if correlation[i, j] > 0.4
    ]
    return pd.DataFrame(edges, columns=["source", "target", "importance"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=Path("results/replogle/reference"))
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/GEARS"))
    parser.add_argument("--asset-dir", type=Path, default=Path("data/gears_assets"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/replogle/gears_extended"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Unexpected GEARS source {revision}")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    from gears import GEARS
    from gears.data_utils import get_DE_genes
    from gears.inference import compute_metrics, evaluate
    from gears.utils import GeneSimNetwork, create_cell_graph_dataset_for_prediction
    from torch_geometric.loader import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = ad.read_h5ad(args.reference_dir / "train.h5ad")
    validation = ad.read_h5ad(args.reference_dir / "validation.h5ad")
    split = PerturbDiffSplit.from_yaml(args.split_config)
    if not split.masks(train.obs)["train"].all():
        raise ValueError("GEARS training includes held-out rows")
    split.validate_reference(validation, split_name="validation")
    target_obs = read_h5ad_obs(args.reference_dir / "real.h5ad")
    for data in [train, validation]:
        labels = data.obs.gene.astype(str)
        data.obs["condition"] = labels.map(
            lambda g: "ctrl" if g == split.control_pert else g + "+ctrl"
        )
        data.obs["cell_type"] = "pooled"
        data.var["gene_name"] = data.var_names
    with (args.asset_dir / "essential_all_data_pert_genes.pkl").open("rb") as f:
        essential = set(pickle.load(f))
    with (args.asset_dir / "gene2go_all.pkl").open("rb") as f:
        gene2go = pickle.load(f)
    native_genes = essential & set(gene2go)
    required = (
        set(train.obs.gene.astype(str))
        | set(validation.obs.gene.astype(str))
        | set(target_obs.gene.astype(str))
    )
    required.discard(split.control_pert)
    extra = sorted(required - native_genes)
    perts = sorted(native_genes | required)
    pert_map = {g: i for i, g in enumerate(perts)}
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    contract.update(
        revision=revision,
        split_sha256=sha256_file(args.split_config),
        extra_isolated_nodes=extra,
        graph_policy="official default GO edges; missing perturbations are isolated nodes",
        context_policy="pooled training; matched hepg2 controls at inference",
    )
    config = args.output_dir / "run_config.json"
    if config.exists() and json.loads(config.read_text()) != contract:
        raise ValueError("GEARS settings changed; use a new directory")
    write_manifest(config, contract)
    print(f"GEARS retains every condition; isolated additional graph nodes: {extra}", flush=True)
    if args.smoke:
        # Small row subset is only for execution testing, never a reported model.
        ctrl_idx = np.flatnonzero(train.obs.condition == "ctrl")[:64]
        pert_idx = np.flatnonzero(train.obs.condition != "ctrl")[:128]
        train = train[np.concatenate([ctrl_idx, pert_idx])].copy()
    train = get_DE_genes(train, skip_calc_de=True)
    nonzero = training_nonzero_genes(train)
    nonzero["ctrl"] = np.arange(train.n_vars)
    train.uns["non_zeros_gene_idx"] = {
        full: nonzero[label]
        for full, label in train.obs[["condition_name", "condition"]].drop_duplicates().values
    }
    # Validation DEs are used only for checkpoint selection, never the training loss.
    validation = get_DE_genes(validation, skip_calc_de=False)
    full_to_cond = dict(validation.obs[["condition_name", "condition"]].values)
    de = {
        full_to_cond[name]: np.flatnonzero(validation.var_names.isin(genes[:20]))
        for name, genes in validation.uns["rank_genes_groups_cov_all"].items()
    }
    if args.smoke:
        ctrl_idx = np.flatnonzero(validation.obs.condition == "ctrl")[:32]
        pert_idx = np.flatnonzero(validation.obs.condition != "ctrl")[:64]
        validation = validation[np.concatenate([ctrl_idx, pert_idx])].copy()
    controls = train[train.obs.condition == "ctrl"].copy()
    val_controls = ad.read_h5ad(args.reference_dir / "controls.h5ad")
    val_controls.X = sparse.csr_matrix(val_controls.X)
    training_loader = DataLoader(
        LazyGraphs(train, controls, pert_map, {}, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        LazyGraphs(validation, val_controls, pert_map, de, args.seed), batch_size=args.batch_size
    )
    genes = train.var_names.tolist()
    gene_map = {g: i for i, g in enumerate(genes)}
    graph_path = args.output_dir / "coexpression.csv"
    if graph_path.exists():
        coexpress = pd.read_csv(graph_path)
    else:
        print("Building exact training-only Pearson coexpression graph", flush=True)
        coexpress = coexpression_graph(train.X, genes)
        coexpress.to_csv(graph_path, index=False)
    go = pd.read_csv(args.asset_dir / "go_essential_all/go_essential_all.csv")
    go = go.groupby("target", group_keys=False).apply(lambda x: x.nlargest(21, "importance"))
    if not set(go.source).union(go.target) <= set(perts):
        raise ValueError("GO edges reference genes outside the native graph vocabulary")
    co_graph, go_graph = (
        GeneSimNetwork(coexpress, genes, gene_map),
        GeneSimNetwork(go, perts, pert_map),
    )
    pd_data = SimpleNamespace(
        dataloader={"train_loader": training_loader, "val_loader": validation_loader},
        adata=train,
        node_map=gene_map,
        node_map_pert=pert_map,
        data_path=str(args.output_dir),
        dataset_name="replogle",
        split="no_test",
        seed=args.seed,
        train_gene_set_size=1.0,
        set2conditions={"train": train.obs.condition.unique().tolist()},
        subgroup=None,
        gene_names=train.var.gene_name,
        pert_names=np.asarray(perts),
        default_pert_graph=True,
    )
    gears = GEARS(pd_data, device=args.device, weight_bias_track=False)
    gears.model_initialize(
        G_go=go_graph.edge_index,
        G_go_weight=go_graph.edge_weight,
        G_coexpress=co_graph.edge_index,
        G_coexpress_weight=co_graph.edge_weight,
    )
    network = gears.model
    loss_fn = ScouterGroupedLoss(nonzero, train.n_vars, args.device, gamma=2, direction_weight=0.1)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    last, best = args.output_dir / "last.pt", args.output_dir / "best.pt"
    minimum, first_epoch, history = float("inf"), 0, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        network.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        minimum, first_epoch, history = state["minimum"], state["epoch"] + 1, state["history"]
        torch.set_rng_state(state["rng"])
        if args.device == "mps":
            torch.mps.set_rng_state(state["mps_rng"])
    for epoch in range(first_epoch, 1 if args.smoke else args.epochs):
        network.train()
        started = time.monotonic()
        for batch_index, batch in enumerate(training_loader):
            batch.to(args.device)
            optimizer.zero_grad()
            loss = loss_fn(network(batch), batch.y, gears.ctrl_expression, batch.pert)
            if not torch.isfinite(loss):
                raise RuntimeError("GEARS training produced a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_value_(network.parameters(), 1.0)
            optimizer.step()
            if (batch_index + 1) % 200 == 0:
                print(
                    f"GEARS epoch={epoch + 1} batch={batch_index + 1} "
                    f"seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
            if args.smoke and batch_index == 1:
                break
        scheduler.step()
        metrics, _ = compute_metrics(evaluate(validation_loader, network, False, args.device))
        if metrics["mse_de"] < minimum:
            minimum = float(metrics["mse_de"])
            save_checkpoint(
                best, {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
            )
        history.append(
            {
                "epoch": epoch + 1,
                "validation_mse_de": float(metrics["mse_de"]),
                "seconds": time.monotonic() - started,
            }
        )
        print(history[-1], flush=True)
        save_checkpoint(
            last,
            dict(
                model=copy.deepcopy(network.state_dict()),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                minimum=minimum,
                epoch=epoch,
                history=history,
                rng=torch.get_rng_state(),
                mps_rng=torch.mps.get_rng_state() if args.device == "mps" else None,
            ),
        )
        write_manifest(args.output_dir / "training_history.json", {"epochs": history})
    network.load_state_dict(torch.load(best, map_location="cpu", weights_only=True))
    network.eval()
    predictions = {}
    names = sorted(set(target_obs.gene.astype(str)) - {split.control_pert})
    with torch.no_grad():
        for name in names[:1] if args.smoke else names:
            # Native GEARS reports a mean over 300 sampled controls per condition.
            graphs = create_cell_graph_dataset_for_prediction(
                [name], val_controls, perts, args.device, num_samples=300
            )
            values = np.concatenate(
                [
                    network(batch.to(args.device)).cpu().numpy()
                    for batch in DataLoader(graphs, batch_size=args.batch_size)
                ]
            )
            count = int((target_obs.gene.astype(str) == name).sum())
            predictions[name] = np.repeat(values.mean(0, keepdims=True), count, axis=0).clip(min=0)
    if args.smoke:
        print("GEARS native training/prediction smoke complete; no formal predictions", flush=True)
        return
    real = ad.read_h5ad(args.reference_dir / "real.h5ad")
    output = build_prediction_anndata(
        real, predictions, pert_col="gene", control_pert=split.control_pert
    )
    output.uns["baseline"] = "GEARS (official model, extended isolated-node vocabulary)"
    output.write_h5ad(args.output_dir / "predictions.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
