#!/usr/bin/env python
"""One audited population-steering runner for Squidiff and conditional DDPM.

Test response values are never read for conditioning or steering. Only query
metadata, observed controls, train-derived priors and frozen weights are used.
Final assembly reads the reference to copy controls and attach evaluator labels.
"""

# ruff: noqa: E402 -- Direct-script use requires the repository path bootstrap.

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import anndata as ad
import numpy as np
import torch

from celldiffa.benchmark.artifacts import load_embedding_dict, load_selected_genes, sha256_file
from celldiffa.benchmark.backbone_experiments import (
    atomic_json,
    dense,
    load_weights,
    make_reward,
    plan_groups,
)
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.benchmark.replogle_priors import (
    _impute_shifts_with_embeddings,
    compute_replogle_training_priors,
)
from celldiffa.benchmark.replogle_shards import (
    assemble_replogle_shards,
    load_group_shard,
    save_group_shard,
    shard_path,
)
from celldiffa.benchmark.streaming import read_h5ad_obs, read_h5ad_var
from celldiffa.rewards.cellwise import AffineExpressionReward
from celldiffa.smc import SMCConfig, SMCEngine


def load_squidiff(args, genes, split, targets, embeddings):
    import subprocess

    from baselines.adapter_squidiff import SquidiffSampler
    from scripts.baselines.run_squidiff_replogle import REVISION, latent_shifts

    revision = subprocess.check_output(
        ["git", "-C", str(args.squidiff_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected pinned Squidiff revision {REVISION}, found {revision}")
    sys.path.insert(0, str(args.squidiff_root.resolve()))
    from Squidiff import diffusion as diffusion_module
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    if args.device == "mps":
        from celldiffa.benchmark.torch_compat import extract_into_tensor

        diffusion_module._extract_into_tensor = extract_into_tensor
    config_path = args.model_config or args.checkpoint.parent / "run_config.json"
    if not config_path.exists():
        raise ValueError(
            "Squidiff requires training run_config.json or --model-config; "
            "do not guess architecture"
        )
    config = json.loads(config_path.read_text())
    if config.get("smoke"):
        raise ValueError("A smoke-only Squidiff checkpoint cannot produce formal test results")
    gene_hash = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    if config.get("ordered_genes_sha256") != gene_hash and config.get("genes") != genes:
        raise ValueError("Squidiff gene-order provenance missing/mismatched in model config")
    if config.get("split_sha256") != sha256_file(args.split_config):
        raise ValueError("Squidiff training split provenance differs from evaluation split")
    kwargs = model_and_diffusion_defaults()
    kwargs.update(gene_size=len(genes), output_dim=len(genes), use_encoder=True)
    if "model_kwargs" in config:
        kwargs.update(config["model_kwargs"])
    elif config.get("revision") != REVISION:
        raise ValueError("External Squidiff config must explicitly declare model_kwargs")
    kwargs["timestep_respacing"] = f"ddim{args.sampling_steps}"
    model, diffusion = create_model_and_diffusion(**kwargs)
    model.load_state_dict(load_weights(args.checkpoint), strict=True)
    model = model.to(args.device).eval()
    train_path = args.reference_dir / "train.h5ad"
    if config.get("train_sha256") != sha256_file(train_path):
        raise ValueError("Squidiff checkpoint and training reference have different provenance")
    train = ad.read_h5ad(train_path)
    if list(train.var_names) != genes or not split.masks(train.obs)["train"].all():
        raise ValueError("Invalid Squidiff training reference")
    policy = args.unseen_policy
    if policy == "auto":
        prediction_path = args.checkpoint.parent / "prediction_config.json"
        policy = (
            json.loads(prediction_path.read_text())["unseen_policy"]
            if prediction_path.exists()
            else "error"
        )
    if policy not in {"error", "zero_shift", "ridge"}:
        raise ValueError(f"Unsupported Squidiff unseen policy: {policy}")
    unknown = sorted(set(targets) - set(train.obs.gene.astype(str)))
    if unknown and policy == "error":
        raise ValueError(
            f"Squidiff has no native condition for {unknown}. Specify the SAME explicit "
            "--unseen-policy zero_shift or ridge for ALL paired Squidiff runs."
        )
    with torch.no_grad():
        latents = np.concatenate(
            [
                model.encoder(torch.as_tensor(dense(train.X[i : i + 512]), device=args.device))
                .cpu()
                .numpy()
                for i in range(0, train.n_obs, 512)
            ]
        )
    shifts = latent_shifts(latents, train.obs, split.control_pert)
    if unknown and policy == "ridge":
        shifts.update(
            _impute_shifts_with_embeddings(shifts, embeddings, unknown, ridge_penalty=1.0)
        )
    elif unknown and policy == "zero_shift":
        shifts.update({name: np.zeros(latents.shape[1], dtype=np.float32) for name in unknown})
    metadata = dict(
        unseen_policy=policy,
        unsupported_native_conditions=unknown,
        unknown_response_extension=bool(unknown),
        upstream_revision=revision,
        model_config_sha256=sha256_file(config_path),
        native_model_kwargs=kwargs,
        training_reference_sha256=sha256_file(train_path),
    )
    return SquidiffSampler(model, diffusion, eta=args.eta), shifts, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=["squidiff", "conditional_ddpm"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument("--selected-genes", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--squidiff-root", type=Path, default=REPO / "external/Squidiff")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--unseen-policy", choices=["auto", "error", "zero_shift", "ridge"], default="auto"
    )
    parser.add_argument("--alignment-mode", choices=["smc", "best_of_n", "random"], default="smc")
    parser.add_argument("--reward-unit", choices=["population", "cell"], default="population")
    parser.add_argument("--reward-normalization", choices=["zscore", "none"], default="zscore")
    parser.add_argument("--num-particles", type=int, default=16)
    parser.add_argument("--population-cells", type=int, default=512)
    parser.add_argument("--batch-cells", type=int, default=1024)
    parser.add_argument("--sampling-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--reward-expression-scale", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-groups", type=int)
    args = parser.parse_args()
    if args.max_groups is not None and args.max_groups < 1:
        parser.error("max-groups must be positive")
    if args.num_threads < 1 or args.sampling_steps < 2 or not 0 <= args.eta <= 1:
        parser.error("Invalid threads, sampling steps or eta")
    torch.set_num_threads(args.num_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    split = PerturbDiffSplit.from_yaml(args.split_config)
    genes = load_selected_genes(args.selected_genes)
    reference = args.reference_dir / "real.h5ad"
    obs = read_h5ad_obs(reference)
    split.validate_real_test(SimpleNamespace(obs=obs))
    if list(read_h5ad_var(reference).index) != genes:
        raise ValueError("Reference and published ordered genes differ")
    controls = ad.read_h5ad(args.reference_dir / "controls.h5ad")
    if (
        list(controls.var_names) != genes
        or not controls.obs.gene.astype(str).eq(split.control_pert).all()
    ):
        raise ValueError("Invalid observed control reference")
    # Reject a control file imported from a different query/reference export.
    expected_controls = obs.index[obs.gene.astype(str) == split.control_pert]
    if list(controls.obs_names) != list(expected_controls):
        raise ValueError("Observed control identities differ from the fixed reference")
    embeddings = load_embedding_dict(args.embeddings)
    targets = sorted(set(obs.gene.astype(str)) - {split.control_pert})
    groups = plan_groups(obs, control=split.control_pert, population_cells=args.population_cells)
    base_metadata, shifts = {}, None
    if args.backbone == "squidiff":
        sampler, shifts, base_metadata = load_squidiff(args, genes, split, targets, embeddings)
        mean, scale = np.zeros(len(genes), np.float32), np.ones(len(genes), np.float32)
    else:
        from baselines.conditional_ddpm import ConditionalDDPM, ConditionalDDPMSampler

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if state["genes"] != genes or state["contract"]["smoke"]:
            raise ValueError("Conditional-DDPM gene mismatch or smoke-only checkpoint")
        if state["contract"]["split_sha256"] != sha256_file(args.split_config) or state["contract"][
            "embedding_sha256"
        ] != sha256_file(args.embeddings):
            raise ValueError("Conditional-DDPM split or descriptor provenance mismatch")
        model = ConditionalDDPM(**state["architecture"]).to(args.device)
        model.load_state_dict(state["model"], strict=True)
        sampler = ConditionalDDPMSampler(model, sampling_steps=args.sampling_steps, eta=args.eta)
        mean, scale = state["mean"].numpy(), state["scale"].numpy()
        base_metadata = dict(training_contract=state["contract"], selected_step=state["step"])
    # Freeze the full contract before resume; reject stale/incompatible shards.
    source_stat = args.source.stat()
    contract = dict(
        version=1,
        backbone=args.backbone,
        checkpoint_sha256=sha256_file(args.checkpoint),
        reference_sha256=sha256_file(reference),
        controls_sha256=sha256_file(args.reference_dir / "controls.h5ad"),
        selected_genes_sha256=sha256_file(args.selected_genes),
        split_sha256=sha256_file(args.split_config),
        embeddings_sha256=sha256_file(args.embeddings),
        source=str(args.source.resolve()),
        source_size=source_stat.st_size,
        source_mtime_ns=source_stat.st_mtime_ns,
        alignment_mode=args.alignment_mode,
        reward_unit=args.reward_unit,
        reward_normalization=args.reward_normalization,
        num_particles=args.num_particles,
        population_cells=args.population_cells,
        batch_cells=args.batch_cells,
        sampling_steps=args.sampling_steps,
        alpha=args.alpha,
        eta=args.eta,
        seed=args.seed,
        reward_expression_scale=args.reward_expression_scale,
        device=args.device,
        torch_version=str(torch.__version__),
        base=base_metadata,
        groups=groups,
        implementation_sha256={
            str(path.relative_to(REPO)): sha256_file(path)
            for path in (
                Path(__file__),
                REPO / "celldiffa/smc/engine.py",
                REPO / "celldiffa/rewards/cellwise.py",
                REPO / "baselines/adapter_squidiff.py",
                REPO / "baselines/conditional_ddpm.py",
                REPO / "celldiffa/benchmark/backbone_experiments.py",
                REPO / "celldiffa/benchmark/replogle_priors.py",
                REPO / "celldiffa/rewards/base.py",
                REPO / "celldiffa/rewards/transcriptomic.py",
                REPO / "celldiffa/rewards/geometric.py",
                REPO / "celldiffa/rewards/anchor.py",
                REPO / "celldiffa/smc/resampler.py",
            )
        },
    )
    contract_path = args.output_dir / "run_config.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Existing run settings differ; use a new output directory")
    atomic_json(contract_path, contract)
    priors = compute_replogle_training_priors(
        args.source,
        args.split_config,
        genes,
        cache_path=args.output_dir / "training_priors.npz",
        top_k=20,
        target_perturbations=targets,
        perturbation_embeddings=embeddings,
        embedding_signature=sha256_file(args.embeddings),
    )
    atomic_json(args.output_dir / "prior_provenance.json", priors.sources)
    shard_root = args.output_dir / "shards"
    shard_root.mkdir(exist_ok=True)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    processed = 0
    for group in groups:
        if args.max_groups is not None and processed >= args.max_groups:
            break
        index, name, count = group["index"], group["perturbation"], group["cells"]
        path = shard_path(shard_root, index)
        if path.exists():
            saved_index, saved_name, saved = load_group_shard(path)
            if (saved_index, saved_name, saved.shape) != (index, name, (count, len(genes))):
                raise ValueError(f"Invalid resume shard: {path}")
            continue
        pool = controls[controls.obs.cell_line.astype(str).eq(group["context"])].X
        if pool.shape[0] < 2:
            raise ValueError(f"No matched control population for {group['context']}")
        rng = np.random.default_rng(args.seed + index)
        selected = rng.integers(pool.shape[0], size=group["sampled_cells"])
        ctrl = dense(pool[selected])
        native_controls = torch.as_tensor((ctrl - mean) / scale, device=args.device)
        if args.backbone == "squidiff":
            with torch.no_grad():
                z = sampler.model.encoder(torch.as_tensor(ctrl, device=args.device))
            condition = {"z_mod": z + torch.as_tensor(shifts[name], device=args.device)}
        else:
            if name not in embeddings:
                raise ValueError(f"Missing query descriptor for {name}")
            vector = embeddings[name] / np.linalg.norm(embeddings[name])
            context_mean = dense(pool).mean(0)
            condition = dict(
                descriptor=torch.as_tensor(vector[None], device=args.device, dtype=torch.float32),
                control_mean=torch.as_tensor(
                    ((context_mean - mean) / scale)[None], device=args.device
                ),
            )
        reward = make_reward(
            priors,
            ctrl,
            unit=args.reward_unit,
            normalization=args.reward_normalization,
            expression_scale=args.reward_expression_scale,
        )
        reward = AffineExpressionReward(
            reward, mean / args.reward_expression_scale, scale / args.reward_expression_scale
        )
        config = SMCConfig(
            num_particles=args.num_particles,
            cells_per_particle=len(ctrl),
            batch_size_per_step=args.batch_cells,
            device=args.device,
            alignment_mode=args.alignment_mode,
            alpha=args.alpha,
            eta=args.eta,
            seed=args.seed + index,
            output_mode="map",
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        group_start = time.monotonic()
        result = SMCEngine(sampler, reward, config).sample_with_alignment(
            name, condition, ctrl_cells=native_controls, num_genes=len(genes)
        )
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        values = result["samples"][:count].cpu().numpy() * scale + mean
        save_group_shard(shard_root, index, name, np.maximum(values, 0))
        atomic_json(
            path.with_suffix(".diagnostics.json"),
            dict(
                **group,
                seconds=time.monotonic() - group_start,
                ess=result["ess_history"],
                resampled=result["resample_history"],
                final_weights=result["weights"].tolist(),
                distinct_initial_ancestors=result["ancestor_history"],
                denoised_cell_steps=result["denoised_cell_steps"],
                max_cuda_memory_bytes=torch.cuda.max_memory_allocated()
                if args.device.startswith("cuda")
                else None,
            ),
        )
        processed += 1
        atomic_json(
            args.output_dir / "progress.json",
            dict(
                completed_groups=len(list(shard_root.glob("group_*.npz"))),
                total_groups=len(groups),
                last_perturbation=name,
                seconds_this_session=time.monotonic() - start,
            ),
        )
        print(
            f"{args.backbone}/{args.alignment_mode} group={index + 1}/{len(groups)} "
            f"perturbation={name} cells={count}",
            flush=True,
        )
    _, status = assemble_replogle_shards(
        reference,
        shard_root,
        args.output_dir / "predictions.h5ad",
        pert_col=split.pert_col,
        control_pert=split.control_pert,
        require_complete=False,
        write_output=args.max_groups is None,
    )
    atomic_json(
        args.output_dir / "completion.json",
        dict(
            **status,
            formal_output_written=bool(status["complete"] and args.max_groups is None),
            test_outcomes_used_for_steering=False,
            evaluation_completed=False,
        ),
    )
    print(
        "COMPLETE predictions"
        if status["complete"] and args.max_groups is None
        else "PARTIAL CHECK ONLY: no evaluator-ready output",
        flush=True,
    )


if __name__ == "__main__":
    main()
