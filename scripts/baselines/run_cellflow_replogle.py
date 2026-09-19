#!/usr/bin/env python
"""Native CellFlow OTFM on official Replogle rows, with resumable fitting.

Uses the author's network defaults and public GenePT descriptors. This is a
documented dataset adaptation, not a claim to reproduce unreported paper knobs.
The training sampler, OT matching, optimizer, EMA and ODE solver are upstream.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from celldiffa.benchmark.artifacts import load_embedding_dict, sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import read_h5ad_obs

REVISION = "446ed6073c60ac2e8db13c4ea096a43cdec288b2"


def dense(values):
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def decorate(data, embeddings, contexts, control):
    data.obs["gene"] = data.obs.gene.astype("category")
    data.obs["cell_line"] = data.obs.cell_line.astype("category")
    data.obs["is_control"] = data.obs.gene.astype(str) == control
    data.uns["gene_embedding"] = {
        name: np.asarray(value, dtype=np.float32)[:, None] for name, value in embeddings.items()
    }
    data.uns["context_embedding"] = {
        name: np.eye(len(contexts), dtype=np.float32)[index, :, None]
        for index, name in enumerate(contexts)
    }


def save_payload(path, payload):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def predict_condition(model, controls, name, *, cells, seed, batch_size=128, ode_kwargs=None):
    """No target expression input; pad independent cells to avoid repeated JITs."""
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, cells, batch_size):
        selected = rng.integers(controls.n_obs, size=batch_size)
        query = controls[selected].copy()
        metadata = pd.DataFrame({"gene": [name], "cell_line": [str(query.obs.cell_line.iloc[0])]})
        metadata["gene"] = metadata.gene.astype("category")
        metadata["cell_line"] = metadata.cell_line.astype("category")
        metadata["is_control"] = False
        output = model.predict(query, covariate_data=metadata, sample_rep="X", **(ode_kwargs or {}))
        if len(output) != 1:
            raise RuntimeError("A CellFlow query produced multiple condition outputs")
        prediction = np.asarray(next(iter(output.values())), dtype=np.float32)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite CellFlow prediction for {name}")
        values.append(prediction[: min(batch_size, cells - start)])
    return np.concatenate(values).clip(min=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=Path("results/replogle/reference"))
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/CellFlow"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/replogle/cellflow"))
    parser.add_argument("--iterations", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validation-every", type=int, default=20000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Unexpected CellFlow source {revision}")
    sys.path.insert(0, str((args.upstream_root / "src").resolve()))
    import cellflow
    import diffrax
    import jax
    from cellflow.data._dataloader import TrainSampler
    from flax import serialization

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    contract.update(
        revision=revision,
        split_sha256=sha256_file(args.split_config),
        embedding_sha256=sha256_file(args.embeddings),
        architecture="native OTFM defaults, explicit empty pre/post pooling layers",
        validation_selection="mean MSE over all validation conditions; 128 generated controls each",
        prediction_input="observed matched controls only",
        optimizer="native Adam 5e-5 with 20-step gradient accumulation",
        device=[str(x) for x in jax.devices()],
    )
    config = args.output_dir / "run_config.json"
    if config.exists() and json.loads(config.read_text()) != contract:
        raise ValueError("CellFlow run settings changed; use a new output directory")
    write_manifest(config, contract)
    train = ad.read_h5ad(args.reference_dir / "train.h5ad")
    validation = ad.read_h5ad(args.reference_dir / "validation.h5ad")
    controls = ad.read_h5ad(args.reference_dir / "controls.h5ad")
    target_obs = read_h5ad_obs(args.reference_dir / "real.h5ad")
    split = PerturbDiffSplit.from_yaml(args.split_config)
    if not split.masks(train.obs)["train"].all():
        raise ValueError("Held-out rows in CellFlow training input")
    split.validate_reference(validation, split_name="validation")
    embeddings = load_embedding_dict(args.embeddings)
    required = set(train.obs.gene.astype(str)) | set(validation.obs.gene.astype(str))
    required |= set(target_obs.gene.astype(str))
    if required - embeddings.keys():
        raise ValueError(f"Missing CellFlow descriptors: {sorted(required - embeddings.keys())}")
    contexts = sorted(set(train.obs.cell_line.astype(str)))
    if args.smoke:
        train = train[
            np.random.default_rng(args.seed).choice(train.n_obs, 2048, replace=False)
        ].copy()
    for data in [train, controls]:
        decorate(data, embeddings, contexts, split.control_pert)
    means = {
        name: dense(validation.X[validation.obs.gene.astype(str) == name]).mean(axis=0)
        for name in sorted(set(validation.obs.gene.astype(str)) - {split.control_pert})
    }
    del validation
    gc.collect()
    model = cellflow.model.CellFlow(train, solver="otfm")
    model.prepare_data(
        sample_rep="X",
        control_key="is_control",
        perturbation_covariates={"gene": ["gene"]},
        perturbation_covariate_reps={"gene": "gene_embedding"},
        sample_covariates=["cell_line"],
        sample_covariate_reps={"cell_line": "context_embedding"},
        split_covariates=["cell_line"],
        max_combination_length=1,
    )
    model.prepare_model(
        layers_before_pool=[], layers_after_pool=[], conditioning_kwargs={}, seed=args.seed
    )
    parameter_count = sum(x.size for x in jax.tree.leaves(model.solver.vf_state.params))
    print(f"CellFlow parameters={parameter_count}; native OTFM on {train.shape}", flush=True)
    sampler = TrainSampler(model.train_data, batch_size=args.batch_size)
    rng_jax, rng_np = jax.random.PRNGKey(0), np.random.default_rng(0)
    last, best = args.output_dir / "last.pkl", args.output_dir / "best.pkl"
    first, minimum, stale, history = 0, float("inf"), 0, []
    if last.exists():
        with last.open("rb") as handle:
            state = pickle.load(handle)
        model.solver.vf_state = serialization.from_state_dict(model.solver.vf_state, state["train"])
        model.solver.vf_state_inference = serialization.from_state_dict(
            model.solver.vf_state_inference, state["inference"]
        )
        first, minimum, stale = state["step"], state["minimum"], state["stale"]
        rng_jax, history = state["rng_jax"], state["history"]
        rng_np.bit_generator.state = state["rng_np"]
    started = time.monotonic()
    # Exercise at least one real optimizer update (native accumulation is 20).
    total = 20 if args.smoke else args.iterations
    smoke_ode = (
        dict(
            solver=diffrax.Heun(),
            dt0=0.05,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=32,
        )
        if args.smoke
        else None
    )
    for step in range(first, total):
        if stale >= args.patience:
            break
        rng_jax, rng_step = jax.random.split(rng_jax, 2)
        loss = float(model.solver.step_fn(rng_step, sampler.sample(rng_np)))
        if not np.isfinite(loss):
            raise RuntimeError("CellFlow training loss is non-finite")
        if (step + 1) % 100 == 0 or args.smoke:
            print(
                f"CellFlow step={step + 1} loss={loss:.6f} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
        if (step + 1) % args.validation_every == 0 or step + 1 == total:
            model.solver.is_trained = True
            names = list(means)[:1] if args.smoke else list(means)
            errors = [
                float(
                    np.mean(
                        (
                            predict_condition(
                                model,
                                controls,
                                name,
                                cells=128,
                                seed=args.seed,
                                ode_kwargs=smoke_ode,
                            ).mean(axis=0)
                            - means[name]
                        )
                        ** 2
                    )
                )
                for name in names
            ]
            score = float(np.mean(errors))
            stale = stale + 1 if score >= minimum else 0
            if score < minimum:
                minimum = score
                save_payload(
                    best,
                    jax.device_get(serialization.to_state_dict(model.solver.vf_state_inference)),
                )
            history.append({"step": step + 1, "validation_mse": score, "stale": stale})
            print(history[-1], flush=True)
            write_manifest(args.output_dir / "training_history.json", {"checks": history})
        if (step + 1) % 1000 == 0 or step + 1 == total:
            save_payload(
                last,
                dict(
                    train=jax.device_get(serialization.to_state_dict(model.solver.vf_state)),
                    inference=jax.device_get(
                        serialization.to_state_dict(model.solver.vf_state_inference)
                    ),
                    step=step + 1,
                    minimum=minimum,
                    stale=stale,
                    history=history,
                    rng_jax=jax.device_get(rng_jax),
                    rng_np=rng_np.bit_generator.state,
                ),
            )
    with best.open("rb") as handle:
        model.solver.vf_state_inference = serialization.from_state_dict(
            model.solver.vf_state_inference, pickle.load(handle)
        )
    model.solver.is_trained = True
    if args.smoke:
        print("CellFlow native training/prediction smoke complete; no formal output", flush=True)
        return
    predictions = {}
    shards = args.output_dir / "prediction_shards"
    shards.mkdir(exist_ok=True)
    for index, name in enumerate(sorted(set(target_obs.gene.astype(str)) - {split.control_pert})):
        path = shards / f"{index:04d}.npz"
        if path.exists():
            with np.load(path) as data:
                if str(data["name"]) != name:
                    raise ValueError("CellFlow shard condition mismatch")
                values = data["values"]
        else:
            count = int((target_obs.gene.astype(str) == name).sum())
            values = predict_condition(model, controls, name, cells=count, seed=args.seed + index)
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(temporary, name=name, values=values)
            os.replace(temporary, path)
        predictions[name] = values
        print(f"CellFlow prediction {index + 1}/380 {name}", flush=True)
    real = ad.read_h5ad(args.reference_dir / "real.h5ad")
    output = build_prediction_anndata(
        real, predictions, pert_col="gene", control_pert=split.control_pert
    )
    output.uns["baseline"] = "CellFlow (native OTFM; GenePT and cell-line descriptors)"
    output.write_h5ad(args.output_dir / "predictions.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
