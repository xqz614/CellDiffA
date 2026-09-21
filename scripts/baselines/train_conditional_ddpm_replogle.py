#!/usr/bin/env python
"""Train an explicitly labelled conditional-DDPM reference on official rows."""

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import anndata as ad
import numpy as np
import torch

from baselines.conditional_ddpm import ConditionalDDPM
from celldiffa.benchmark.artifacts import load_embedding_dict, sha256_file
from celldiffa.benchmark.backbone_experiments import (
    atomic_json,
    dense,
    expression_conditions,
    training_control_means,
)
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from scripts.baselines.run_scouter_replogle import save_checkpoint


def validation_loss(model, values, conditions, device, batch_size):
    generator = torch.Generator().manual_seed(1729)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            end = min(start + batch_size, len(values))
            x = values[start:end].to(device)
            t = torch.randint(len(model.alphas), (len(x),), generator=generator).to(device)
            noise = torch.randn(x.shape, generator=generator).to(device)
            alpha = model.alphas[t, None]
            noisy = alpha.sqrt() * x + (1 - alpha).sqrt() * noise
            descriptor = conditions["descriptors"][conditions["descriptor_index"][start:end]].to(
                device
            )
            control = conditions["controls"][conditions["control_index"][start:end]].to(device)
            total += (model(noisy, t, descriptor, control) - noise).square().mean(1).sum().item()
    return total / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--validation-every", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for key in (
        "steps",
        "validation_every",
        "patience",
        "batch_size",
        "width",
        "depth",
        "num_threads",
    ):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.diffusion_steps < 2 or args.learning_rate <= 0:
        parser.error("Invalid diffusion steps or learning rate")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    train = ad.read_h5ad(args.reference_dir / "train.h5ad")
    validation = ad.read_h5ad(args.reference_dir / "validation.h5ad")
    split = PerturbDiffSplit.from_yaml(args.split_config)
    if not split.masks(train.obs)["train"].all():
        raise ValueError("Training data contain held-out response rows")
    split.validate_reference(validation, split_name="validation")
    if not train.var_names.equals(validation.var_names) or not train.var_names.is_unique:
        raise ValueError("Ordered genes differ across train/validation")
    embeddings = load_embedding_dict(args.embeddings)
    controls = training_control_means(train, split.control_pert)
    # Validation controls are observed inputs, not intervention outcomes.
    controls.update(
        training_control_means(
            validation[validation.obs.gene.astype(str) == split.control_pert], split.control_pert
        )
    )
    tables = [
        expression_conditions(x, embeddings, controls, control=split.control_pert)
        for x in (train, validation)
    ]
    arrays = [dense(x.X) for x in (train, validation)]
    mean = arrays[0].mean(0, dtype=np.float64).astype(np.float32)
    scale = arrays[0].std(0, dtype=np.float64).clip(0.1).astype(np.float32)
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Non-finite expression values")
    values = [torch.from_numpy((x - mean) / scale) for x in arrays]
    conditions = []
    for table in tables:
        table["controls"] = (table["controls"] - mean) / scale
        conditions.append({key: torch.from_numpy(value) for key, value in table.items()})
    architecture = dict(
        genes=train.n_vars,
        descriptor_dim=tables[0]["descriptors"].shape[1],
        width=args.width,
        depth=args.depth,
        timesteps=args.diffusion_steps,
    )
    contract = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"device", "num_threads"}
    }
    contract.update(
        method="Conditional DDPM (in-house reference, not scDiff)",
        architecture=architecture,
        train_sha256=sha256_file(args.reference_dir / "train.h5ad"),
        validation_sha256=sha256_file(args.reference_dir / "validation.h5ad"),
        split_sha256=sha256_file(args.split_config),
        embedding_sha256=sha256_file(args.embeddings),
        selection="minimum fixed-noise validation epsilon MSE; no test outcomes",
        implementation_sha256=sha256_file(
            Path(__file__).parents[2] / "baselines/conditional_ddpm.py"
        ),
        trainer_sha256=sha256_file(__file__),
    )
    contract_path = args.output_dir / "run_config.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing training settings differ; use a new output directory")
    atomic_json(contract_path, contract)
    progress_path = args.output_dir / "training_progress.json"
    if (
        progress_path.exists()
        and json.loads(progress_path.read_text()).get("status") == "training_complete"
    ):
        if not (args.output_dir / "best.pt").exists():
            raise ValueError("Completed training marker but best checkpoint is absent")
        print("Training already complete; preserving its selected checkpoint", flush=True)
        return
    model = ConditionalDDPM(**architecture).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_loss, stale, first, best_step = float("inf"), 0, 0, 0
    history = []
    last_path = args.output_dir / "last.pt"
    if last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first, best_loss, stale, best_step = (
            state["step"],
            state["best_loss"],
            state["stale"],
            state["best_step"],
        )
        history = state["history"]
        torch.set_rng_state(state["rng_cpu"])
        if args.device.startswith("cuda") and state["rng_cuda"]:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    total_steps = min(args.steps, 2) if args.smoke else args.steps
    validation_every = 1 if args.smoke else args.validation_every
    completed = first
    started = time.monotonic()
    for step in range(first, total_steps):
        if stale >= args.patience:
            break
        model.train()
        index = torch.randint(len(values[0]), (args.batch_size,))
        x = values[0][index].to(args.device)
        table = conditions[0]
        descriptor = table["descriptors"][table["descriptor_index"][index]].to(args.device)
        control = table["controls"][table["control_index"][index]].to(args.device)
        t = torch.randint(args.diffusion_steps, (len(x),), device=args.device)
        noise = torch.randn_like(x)
        alpha = model.alphas[t, None]
        prediction = model(alpha.sqrt() * x + (1 - alpha).sqrt() * noise, t, descriptor, control)
        loss = (prediction - noise).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite conditional-DDPM loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if completed % 100 == 0 or args.smoke:
            print(
                f"ConditionalDDPM step={completed}/{total_steps} loss={loss.item():.6f}", flush=True
            )
        if completed % validation_every == 0 or completed == total_steps:
            score = validation_loss(model, values[1], conditions[1], args.device, args.batch_size)
            if not np.isfinite(score):
                raise RuntimeError("Non-finite validation loss")
            stale += 1
            if score < best_loss:
                best_loss, stale, best_step = score, 0, completed
                save_checkpoint(
                    args.output_dir / "best.pt",
                    dict(
                        model=model.state_dict(),
                        architecture=architecture,
                        genes=list(train.var_names),
                        mean=torch.from_numpy(mean),
                        scale=torch.from_numpy(scale),
                        contract=contract,
                        step=completed,
                        validation_loss=score,
                    ),
                )
            history.append(dict(step=completed, validation_mse=score))
            atomic_json(args.output_dir / "training_history.json", {"checks": history})
            print(f"Validation step={completed} epsilon_MSE={score:.6f}", flush=True)
        if completed % 1000 == 0 or completed % validation_every == 0 or completed == total_steps:
            save_checkpoint(
                last_path,
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=completed,
                    best_loss=best_loss,
                    stale=stale,
                    best_step=best_step,
                    history=history,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state_all()
                    if args.device.startswith("cuda")
                    else [],
                ),
            )
            atomic_json(
                progress_path, dict(status="training", step=completed, total_steps=total_steps)
            )
    atomic_json(
        progress_path,
        dict(
            status="smoke_complete" if args.smoke else "training_complete",
            step=completed,
            total_steps=total_steps,
            best_step=best_step,
            best_validation_mse=best_loss,
            stop_reason="early_stopping" if stale >= args.patience else "step_limit",
            test_responses_accessed=False,
            seconds_this_session=time.monotonic() - started,
        ),
    )
    print(
        "Conditional-DDPM training finished; test predictions have not been generated", flush=True
    )


if __name__ == "__main__":
    main()
