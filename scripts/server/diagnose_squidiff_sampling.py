#!/usr/bin/env python
"""Small validation-only native Squidiff vs adapter sampling audit.

Uses the existing completed checkpoint. Does not train, steer, read real.h5ad,
change a plan, or replace any benchmark prediction. Native runs call the pinned
author implementation, not a reimplementation of the DDIM update.
"""

# ruff: noqa: E402
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import anndata as ad
import numpy as np
import torch

from baselines.adapter_squidiff import SquidiffSampler
from celldiffa.benchmark.artifacts import sha256_file, write_manifest
from celldiffa.benchmark.backbone_experiments import dense, load_weights
from celldiffa.benchmark.metrics import expression_scale_summary
from celldiffa.benchmark.perturbdiff_split import PerturbDiffSplit
from celldiffa.smc import SMCConfig, SMCEngine
from scripts.baselines.run_squidiff_replogle import REVISION, latent_shifts
from scripts.server.replogle_remaining import paths


class FixedNoiseSampler(SquidiffSampler):
    def __init__(self, model, diffusion, noise):
        super().__init__(model, diffusion, eta=0.0)
        self.initial_noise = noise

    def sample_noise(self, shape, device):
        if tuple(shape) != (1, *self.initial_noise.shape):
            raise ValueError("Diagnostic noise and requested population differ")
        return self.initial_noise.to(device).clone().unsqueeze(0)


class NoReward:
    def compute(self, x_pred, **kwargs):
        # The shared engine evaluates rewards even in random mode. A constant
        # reward makes this an exact unguided path with no biological scoring.
        return torch.zeros(x_pred.shape[0], device=x_pred.device)


@torch.no_grad()
def native_sample(model, diffusion, noise, condition):
    return diffusion.ddim_sample_loop(
        model,
        shape=tuple(noise.shape),
        noise=noise.clone(),
        clip_denoised=False,
        model_kwargs={"z_mod": condition},
        device=noise.device,
        progress=False,
        eta=0.0,
    )


@torch.no_grad()
def adapter_sample(model, diffusion, noise, condition, seed):
    sampler = FixedNoiseSampler(model, diffusion, noise)
    config = SMCConfig(
        num_particles=1,
        cells_per_particle=len(noise),
        batch_size_per_step=len(noise),
        alignment_mode="random",
        device=str(noise.device),
        seed=seed,
    )
    return SMCEngine(sampler, NoReward(), config).sample_with_alignment(
        "diagnostic",
        {"z_mod": condition},
        num_genes=noise.shape[1],
    )["samples"]


def array_summary(values):
    values = np.asarray(values)
    summary = expression_scale_summary(ad.AnnData(values))
    finite = values[np.isfinite(values)]
    summary["p50_p99_p999"] = (
        np.quantile(finite, [0.5, 0.99, 0.999]).tolist() if finite.size else None
    )
    return summary


def choose_conditions(train_obs, validation_obs, control, count):
    """Deterministic metadata-only selection, never choose cases by output quality."""
    training_genes = set(train_obs.gene.astype(str)) - {control}
    rows = (
        validation_obs[validation_obs.gene.astype(str).isin(training_genes)][["gene", "cell_line"]]
        .astype(str)
        .drop_duplicates()
    )
    rows = rows.sort_values(["gene", "cell_line"])
    contexts = set(
        validation_obs.loc[validation_obs.gene.astype(str).eq(control), "cell_line"].astype(str)
    )
    return [(row.gene, row.cell_line) for row in rows.itertuples() if row.cell_line in contexts][
        :count
    ]


@torch.no_grad()
def encode(model, values, device):
    parts = []
    for start in range(0, values.shape[0], 256):
        batch = torch.as_tensor(
            dense(values[start : start + 256]), dtype=torch.float32, device=device
        )
        parts.append(model.encoder(batch).cpu().numpy())
    result = np.concatenate(parts)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite semantic encoder output")
    return result


def verify_training(config, checkpoint, train, validation, split_path):
    if config.get("smoke") or config.get("revision") != REVISION:
        raise ValueError("Need a non-smoke checkpoint from the pinned upstream training run")
    progress = json.loads((checkpoint.parent / "training_progress.json").read_text())
    if progress.get("status") != "training_complete" or progress.get("smoke"):
        raise ValueError("Checkpoint training is not complete")
    for key, path in (
        ("train_sha256", Path(train.filename)),
        ("validation_sha256", Path(validation.filename)),
        ("split_sha256", split_path),
    ):
        if config.get(key) != sha256_file(path):
            raise ValueError(f"Training provenance mismatch: {key}")
    gene_hash = hashlib.sha256("\n".join(train.var_names).encode()).hexdigest()
    if config.get("ordered_genes_sha256") != gene_hash:
        raise ValueError("Checkpoint gene order differs from training input")
    if not train.var_names.equals(validation.var_names):
        raise ValueError("Training and validation gene orders differ")
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO / "results/replogle/maintext_train_v1/plan.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cells", type=int, default=32)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.cells < 2 or args.groups < 1 or args.num_threads < 1:
        parser.error("Need cells>=2, groups>=1, num-threads>=1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; activate the server experiment environment")
    if args.device == "mps":
        parser.error("Use CUDA on the server (or CPU for unit checks); this is not an MPS run")
    plan = json.loads(args.config.read_text())
    p = paths(plan)
    checkpoint = Path(plan["squidiff_checkpoint"])
    model_config_path = Path(
        plan.get("squidiff_model_config") or checkpoint.parent / "run_config.json"
    )
    config = json.loads(model_config_path.read_text())
    root = Path(plan["repo"]) / "external/Squidiff"
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Unexpected Squidiff revision: {revision}")
    sys.path.insert(0, str(root.resolve()))
    from Squidiff.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    torch.set_num_threads(args.num_threads)
    train = ad.read_h5ad(p["reference"] / "train.h5ad", backed="r")
    validation = ad.read_h5ad(p["reference"] / "validation.h5ad", backed="r")
    try:
        progress = verify_training(config, checkpoint, train, validation, p["split"])
        split = PerturbDiffSplit.from_yaml(p["split"])
        if not split.masks(train.obs)["train"].all():
            raise ValueError("Training input contains held-out rows")
        split.validate_reference(validation, split_name="validation")
        selected = choose_conditions(train.obs, validation.obs, split.control_pert, args.groups)
        if not selected:
            raise ValueError("No validation conditions with a training-derived latent effect")
        output = args.output_dir or REPO / "results/replogle/diagnostics" / (
            "squidiff_sampling_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        )
        output.mkdir(parents=True, exist_ok=False)
        report = dict(
            status="running",
            purpose="validation-only sampling diagnostic, not benchmark results",
            test_responses_accessed=False,
            seed=args.seed,
            cells=args.cells,
            checkpoint_sha256=sha256_file(checkpoint),
            model_config_sha256=sha256_file(model_config_path),
            upstream_revision=revision,
            training_progress=progress,
            device=args.device,
            torch_version=str(torch.__version__),
            input_scales={
                "train": expression_scale_summary(train),
                "validation": expression_scale_summary(validation),
            },
            settings=dict(
                eta=0,
                clip_denoised=False,
                steering=False,
                condition="encoded validation controls + observed training latent shift",
            ),
            cases=[],
        )
        write_manifest(output / "report.json", report)
        print("Input ranges:", json.dumps(report["input_scales"], indent=2), flush=True)
        for name, summary in report["input_scales"].items():
            if summary["nonfinite"] or summary["negative"]:
                raise ValueError(f"Invalid {name} input: {summary}")
        kwargs = model_and_diffusion_defaults()
        kwargs.update(gene_size=train.n_vars, output_dim=train.n_vars, use_encoder=True)
        kwargs.update(config.get("model_kwargs", {}))
        if kwargs["diffusion_steps"] != 1000:
            raise ValueError("This comparison requires the existing 1000-step training schedule")
        kwargs["timestep_respacing"] = ""
        model, full = create_model_and_diffusion(**kwargs)
        model.load_state_dict(load_weights(checkpoint), strict=True)
        model = model.to(args.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        kwargs["timestep_respacing"] = "ddim100"
        unused_model, short = create_model_and_diffusion(**kwargs)
        del unused_model
        if full.num_timesteps != 1000 or short.num_timesteps != 100:
            raise ValueError("Unexpected native sampling schedule")
        report["schedules"] = {"native1000": full.timestep_map, "native100": short.timestep_map}
        needed = {split.control_pert, *(name for name, _ in selected)}
        train_mask = train.obs.gene.astype(str).isin(needed).to_numpy()
        subset = train[train_mask].to_memory()
        z_train = encode(model, subset.X, args.device)
        shifts = latent_shifts(z_train, subset.obs, split.control_pert)
        del subset, z_train
        cases = [(split.control_pert, selected[0][1]), *selected]
        for index, (name, context) in enumerate(cases):
            ctrl_mask = (
                validation.obs.gene.astype(str).eq(split.control_pert)
                & validation.obs.cell_line.astype(str).eq(context)
            ).to_numpy()
            indices = np.flatnonzero(ctrl_mask)
            chosen = np.random.default_rng(args.seed + index).choice(
                indices, size=args.cells, replace=len(indices) < args.cells
            )
            # Backed HDF5 cannot slice repeated/unsorted indices directly.
            pool = validation[indices].to_memory()
            local_indices = np.searchsorted(indices, chosen)
            control_values = dense(pool.X[local_indices])
            z = encode(model, control_values, args.device)
            delta = (
                np.zeros(z.shape[1], dtype=np.float32)
                if name == split.control_pert
                else shifts[name]
            )
            condition = torch.as_tensor(z + delta, dtype=torch.float32, device=args.device)
            generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
            noise = torch.randn((args.cells, train.n_vars), generator=generator).to(args.device)
            result = dict(
                perturbation=name,
                context=context,
                control_ids=validation.obs_names[chosen].tolist(),
                controls=array_summary(control_values),
                outputs={},
            )
            arrays = {"initial_noise": noise.cpu().numpy(), "z_mod": condition.cpu().numpy()}
            for variant, diffusion in (
                ("native1000", full),
                ("native100", short),
                ("adapter100", short),
            ):
                print(f"Case {index + 1}/{len(cases)} {name}/{context}: {variant}", flush=True)
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.monotonic()
                values = (
                    adapter_sample(model, diffusion, noise, condition, args.seed + index)
                    if variant == "adapter100"
                    else native_sample(model, diffusion, noise, condition)
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                arrays[variant] = values.detach().cpu().numpy()
                summary = array_summary(arrays[variant])
                summary["seconds"] = time.monotonic() - started
                result["outputs"][variant] = summary
                print(json.dumps(summary), flush=True)
            a, b = arrays["native100"], arrays["adapter100"]
            finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
            result["native_adapter_100"] = dict(
                finite=finite,
                max_abs_difference=float(np.max(np.abs(a - b))) if finite else None,
                allclose=bool(np.allclose(a, b, rtol=1e-5, atol=1e-5)) if finite else False,
                rtol=1e-5,
                atol=1e-5,
            )
            np.savez_compressed(output / f"case_{index:02d}.npz", **arrays)
            report["cases"].append(result)
            write_manifest(output / "report.json", report)
        report["status"] = "diagnostic_complete"
        report["note"] = (
            "Agreement isolates the adapter path, not biological validity. If both native "
            "schedules remain abnormal, inspect weights/training/semantic conditions before "
            "retraining. Never repair output by arbitrary scaling/clipping."
        )
        write_manifest(output / "report.json", report)
        print(f"DIAGNOSTIC COMPLETE: {output / 'report.json'}", flush=True)
        for case in report["cases"]:
            print(case["perturbation"], case["native_adapter_100"], flush=True)
    finally:
        train.file.close()
        validation.file.close()


if __name__ == "__main__":
    main()
