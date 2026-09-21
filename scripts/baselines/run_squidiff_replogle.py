#!/usr/bin/env python
"""Native Squidiff autoencoder/diffusion with training-only latent shifts.

The author's genetic example transfers shifts learned from observed responses.
For a completely unobserved intervention that rule is undefined. The optional,
explicit zero-shift fallback retains those rows but is NOT a learned prediction
for the unknown gene. It is recorded and must be identified in result tables.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse

from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.contracts import build_prediction_anndata
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.streaming import read_h5ad_obs
from celldiffa.benchmark.torch_compat import extract_into_tensor
from scripts.baselines.run_scouter_replogle import save_checkpoint

REVISION = "abdfc27d84947dcccd745d1067c0840a41d32eb8"


def prediction_metadata(reference_dir, train_obs, split, policy):
    """Read test labels only for prediction, never to enable training."""
    from types import SimpleNamespace

    obs = read_h5ad_obs(reference_dir / "real.h5ad")
    split.validate_real_test(SimpleNamespace(obs=obs))
    unknown = sorted(set(obs.gene.astype(str)) - set(train_obs.gene.astype(str)))
    if unknown and policy == "error":
        raise ValueError(f"Native latent shifts undefined for {unknown}; explicit policy required")
    return obs, unknown


def training_contract(args, train, validation):
    """Prediction policy and runtime device do not change the fitted model's contract."""
    operational = {"stage", "unseen_policy", "device", "num_threads"}
    contract = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in operational
    }
    contract.update(
        contract_version=2,
        revision=REVISION,
        split_sha256=sha256_file(args.split_config),
        train_sha256=sha256_file(args.reference_dir / "train.h5ad"),
        validation_sha256=sha256_file(args.reference_dir / "validation.h5ad"),
        ordered_genes_sha256=hashlib.sha256("\n".join(train.var_names).encode()).hexdigest(),
        training_cells=train.n_obs,
        validation_cells=validation.n_obs,
        genes=train.n_vars,
        architecture="native 2048-wide/3-layer/60-latent; 1000-step linear diffusion",
        selection="official validation split denoising MSE of EMA model; no test outcomes",
    )
    return contract


def save_progress(output_dir, values):
    temporary = output_dir / "training_progress.partial.json"
    write_manifest(temporary, values)
    os.replace(temporary, output_dir / "training_progress.json")


def tensor(values, device):
    dense = values.toarray() if sparse.issparse(values) else np.asarray(values)
    return torch.as_tensor(dense, dtype=torch.float32, device=device)


def rng_state(device):
    return dict(
        cpu=torch.get_rng_state(),
        numpy=np.random.get_state(),
        mps=torch.mps.get_rng_state() if device == "mps" else None,
        cuda=torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
    )


def restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    np.random.set_state(state["numpy"])
    if device == "mps":
        torch.mps.set_rng_state(state["mps"])
    if device.startswith("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def latent_shifts(latents, obs, control):
    """Condition effects centered on controls in the same observed cell line."""
    genes, contexts = obs.gene.astype(str).to_numpy(), obs.cell_line.astype(str).to_numpy()
    centered = np.empty_like(latents)
    for context in np.unique(contexts):
        mask = contexts == context
        ctrl = mask & (genes == control)
        if not ctrl.any():
            raise ValueError(f"No training control for {context}")
        centered[mask] = latents[mask] - latents[ctrl].mean(axis=0)
    return {g: centered[genes == g].mean(axis=0) for g in np.unique(genes) if g != control}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=Path("results/replogle/reference"))
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=Path("external/Squidiff"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/replogle/squidiff"))
    parser.add_argument("--device", default="mps")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--stage", choices=["train", "predict", "all"], default="all")
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-every", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--unseen-policy", choices=["error", "zero_shift"], default="error")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for key in ("iterations", "batch_size", "validation_every", "patience", "num_threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_lock = (args.output_dir / ".squidiff.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.stage == "predict" and not (args.output_dir / "training_progress.json").exists():
        raise ValueError("Prediction requires a completed training run in --output-dir")
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Unexpected Squidiff source {revision}")
    sys.path.insert(0, str(args.upstream_root.resolve()))
    from Squidiff import diffusion as diffusion_module
    from Squidiff.resample import UniformSampler
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "mps":
        diffusion_module._extract_into_tensor = extract_into_tensor
    train = ad.read_h5ad(args.reference_dir / "train.h5ad")
    validation = ad.read_h5ad(args.reference_dir / "validation.h5ad")
    split = PerturbDiffSplit.from_yaml(args.split_config)
    if not split.masks(train.obs)["train"].all():
        raise ValueError("Held-out rows in Squidiff training input")
    split.validate_reference(validation, split_name="validation")
    if not train.var_names.equals(validation.var_names) or not train.var_names.is_unique:
        raise ValueError("Training and validation must have the same ordered, unique genes")
    target_obs, unknown = None, []
    if args.stage != "train":
        target_obs, unknown = prediction_metadata(
            args.reference_dir, train.obs, split, args.unseen_policy
        )
    contract = training_contract(args, train, validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = args.output_dir / "run_config.json"
    if args.stage == "predict":
        progress = json.loads((args.output_dir / "training_progress.json").read_text())
        if not config.exists() or progress.get("status") != "training_complete":
            raise ValueError("Prediction requires completed training and its saved contract")
    if config.exists() and json.loads(config.read_text()) != contract:
        raise ValueError("Squidiff settings changed; use a new directory")
    write_manifest(config, contract)
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(train.n_obs, 2048, replace=False)
        train = train[indices].copy()
        validation = validation[:128].copy()
    kwargs = model_and_diffusion_defaults()
    kwargs.update(gene_size=train.n_vars, output_dim=train.n_vars, use_encoder=True)
    model, diffusion = create_model_and_diffusion(**kwargs)
    model = model.to(args.device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    sampler = UniformSampler(diffusion)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    last, best = args.output_dir / "last.pt", args.output_dir / "best.pt"
    minimum, stale, first, cursor, history = float("inf"), 0, 0, 0, []
    order_rng = np.random.default_rng(args.seed)
    order = order_rng.permutation(train.n_obs)
    if last.exists() and args.stage != "predict":
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if (
            saved.get("training_device", args.device) != args.device
            or saved.get("num_threads", args.num_threads) != args.num_threads
        ):
            raise ValueError("Resume training with its original device and thread count")
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], args.device)
        first, minimum, stale, history = (
            saved["step"],
            saved["minimum"],
            saved["stale"],
            saved["history"],
        )
        cursor, order = saved["cursor"], saved["order"]
        order_rng.bit_generator.state = saved["order_rng"]
    total, started = (2 if args.smoke else args.iterations), time.monotonic()
    print(
        f"Squidiff stage={args.stage} training={train.shape} validation={validation.shape} "
        f"device={args.device} threads={args.num_threads}",
        flush=True,
    )
    completed = first
    if args.stage != "predict":
        save_progress(
            args.output_dir,
            dict(
                status="training",
                completed_steps=completed,
                total_steps=total,
                device=args.device,
                num_threads=args.num_threads,
                test_responses_accessed=False,
                smoke=args.smoke,
            ),
        )
    for step in range(first, total) if args.stage != "predict" else ():
        if stale >= args.patience:
            break
        if cursor == train.n_obs:
            order, cursor = order_rng.permutation(train.n_obs), 0
        rows = order[cursor : cursor + args.batch_size]
        cursor += len(rows)
        model.train()
        x = tensor(train.X[rows], args.device)
        t, weights = sampler.sample(len(x), args.device)
        optimizer.zero_grad()
        losses = diffusion.training_losses(
            model, x, t, model_kwargs=dict(group=None, drug_dose=None, control_feature=None)
        )
        loss = (losses["loss"] * weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Squidiff non-finite training loss")
        loss.backward()
        optimizer.step()
        completed = step + 1
        with torch.no_grad():
            for target, source in zip(ema.parameters(), model.parameters()):
                target.mul_(0.9999).add_(source, alpha=0.0001)
            for target, source in zip(ema.buffers(), model.buffers()):
                target.copy_(source)
        for group in optimizer.param_groups:
            group["lr"] = 1e-4 * (1 - step / args.iterations)
        if (step + 1) % 200 == 0 or args.smoke:
            print(
                f"Squidiff step={step + 1} loss={loss.item():.5f} "
                f"seconds={time.monotonic() - started:.1f}",
                flush=True,
            )
            save_progress(
                args.output_dir,
                dict(
                    status="training",
                    completed_steps=completed,
                    total_steps=total,
                    elapsed_this_session_seconds=time.monotonic() - started,
                    last_loss=float(loss.detach().cpu()),
                    device=args.device,
                    num_threads=args.num_threads,
                    test_responses_accessed=False,
                    smoke=args.smoke,
                ),
            )
        if (step + 1) % args.validation_every == 0 or step + 1 == total:
            state = rng_state(args.device)
            torch.manual_seed(args.seed + 917)
            np.random.seed(args.seed + 917)
            scores = []
            with torch.no_grad():
                for offset in range(0, validation.n_obs, 128):
                    x = tensor(validation.X[offset : offset + 128], args.device)
                    t, weights = sampler.sample(len(x), args.device)
                    losses = diffusion.training_losses(
                        ema,
                        x,
                        t,
                        model_kwargs=dict(group=None, drug_dose=None, control_feature=None),
                    )
                    scores.extend((losses["loss"] * weights).cpu().tolist())
            restore_rng(state, args.device)
            score = float(np.mean(scores))
            if not np.isfinite(score):
                raise RuntimeError("Squidiff non-finite validation loss")
            stale = stale + 1 if score >= minimum else 0
            if score < minimum:
                minimum = score
                save_checkpoint(
                    best, {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
                )
            history.append(dict(step=step + 1, validation_denoising_mse=score, stale=stale))
            write_manifest(args.output_dir / "training_history.json", {"checks": history})
            print(history[-1], flush=True)
        if (step + 1) % 1000 == 0 or step + 1 == total:
            save_checkpoint(
                last,
                dict(
                    model=model.state_dict(),
                    ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step + 1,
                    minimum=minimum,
                    stale=stale,
                    history=history,
                    rng=rng_state(args.device),
                    cursor=cursor,
                    order=order,
                    order_rng=order_rng.bit_generator.state,
                    training_device=args.device,
                    num_threads=args.num_threads,
                ),
            )
    if args.stage != "predict":
        if not best.exists():
            raise RuntimeError("Training ended without a validation-selected checkpoint")
        save_progress(
            args.output_dir,
            dict(
                status="training_complete",
                completed_steps=completed,
                total_steps=total,
                stop_reason="early_stopping" if stale >= args.patience else "step_limit",
                best_validation_denoising_mse=minimum,
                best_step=min(history, key=lambda row: row["validation_denoising_mse"])["step"],
                elapsed_this_session_seconds=time.monotonic() - started,
                device=args.device,
                num_threads=args.num_threads,
                test_responses_accessed=False,
                smoke=args.smoke,
            ),
        )
    if args.stage == "train":
        print(
            f"Training complete. Best checkpoint: {best}. No test predictions requested.",
            flush=True,
        )
        return
    prediction_contract = dict(
        unseen_policy=args.unseen_policy,
        unknown_test_interventions=unknown,
        checkpoint_sha256=sha256_file(best),
        response_rule="training context-centered latent effect added to encoded matched controls",
        seed=args.seed,
        smoke=args.smoke,
    )
    prediction_config = args.output_dir / "prediction_config.json"
    if (
        prediction_config.exists()
        and json.loads(prediction_config.read_text()) != prediction_contract
    ):
        raise ValueError("Prediction checkpoint/policy changed; do not reuse previous shards")
    write_manifest(prediction_config, prediction_contract)
    ema.load_state_dict(torch.load(best, map_location="cpu", weights_only=True))
    ema.eval()
    with torch.no_grad():
        z = np.concatenate(
            [
                ema.encoder(tensor(train.X[i : i + 512], args.device)).cpu().numpy()
                for i in range(0, train.n_obs, 512)
            ]
        )
        shifts = latent_shifts(z, train.obs, split.control_pert)
        controls = ad.read_h5ad(args.reference_dir / "controls.h5ad")
        z_control = np.concatenate(
            [
                ema.encoder(tensor(controls.X[i : i + 512], args.device)).cpu().numpy()
                for i in range(0, controls.n_obs, 512)
            ]
        )
        predictions = {}
        shards = args.output_dir / "prediction_shards"
        shards.mkdir(exist_ok=True)
        names = sorted(set(target_obs.gene.astype(str)) - {split.control_pert})
        for index, name in enumerate(names[:1] if args.smoke else names):
            path = shards / f"{index:04d}.npz"
            count = int((target_obs.gene.astype(str) == name).sum())
            if path.exists() and not args.smoke:
                with np.load(path) as saved:
                    if str(saved["name"]) != name:
                        raise ValueError("Squidiff shard label mismatch")
                    predictions[name] = saved["values"]
                continue
            torch.manual_seed(args.seed + index)
            selected = np.random.default_rng(args.seed + index).integers(len(z_control), size=count)
            delta = shifts.get(name, np.zeros(z_control.shape[1], dtype=np.float32))
            if name not in shifts and args.unseen_policy == "error":
                raise ValueError(f"Missing native Squidiff latent effect for {name}")
            latent = tensor(z_control[selected] + delta, args.device)
            values = []
            for offset in range(0, count, args.batch_size):
                cond = latent[offset : offset + args.batch_size]
                output = diffusion.ddim_sample_loop(
                    ema,
                    shape=(len(cond), train.n_vars),
                    device=args.device,
                    model_kwargs={"z_mod": cond},
                    clip_denoised=False,
                )
                if not torch.isfinite(output).all():
                    raise RuntimeError(f"Squidiff non-finite prediction for {name}")
                values.append(output.clamp(min=0).cpu().numpy())
            predictions[name] = np.concatenate(values)
            if not args.smoke:
                temporary = path.with_suffix(".partial.npz")
                np.savez_compressed(temporary, name=name, values=predictions[name])
                os.replace(temporary, path)
            print(f"Squidiff predictions {index + 1}/{len(names)} {name}", flush=True)
    if args.smoke:
        print("Squidiff native training/1000-step prediction smoke complete; no formal output")
        return
    real = ad.read_h5ad(args.reference_dir / "real.h5ad")
    output = build_prediction_anndata(
        real, predictions, pert_col="gene", control_pert=split.control_pert
    )
    output.uns["baseline"] = "Squidiff (training latent shifts; explicit unknown-ID fallback)"
    output.uns["unknown_interventions"] = unknown
    output.write_h5ad(args.output_dir / "predictions.h5ad", compression="gzip")


if __name__ == "__main__":
    main()
