#!/usr/bin/env python
"""Run PerturbDiff sampling with portable checkpoint asset paths.

Released PerturbDiff checkpoints retain absolute covariate-asset paths from
the authors' cluster. The upstream sampling loader reuses those paths instead
of the runtime Hydra values. This wrapper patches only those path fields in
memory, then delegates the rest of sampling to the pinned upstream entrypoint.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

COVARIATE_PATH_KEYS = (
    "celltype_embedding_path",
    "gene_embedding_path",
    "pert_embedding_path",
    "drug_embedding_path",
    "replogle_gene_embedding_path",
)


def patch_covariate_paths(checkpoint_cfg: Any, runtime_cfg: Any) -> Any:
    """Return checkpoint covariate config with runtime asset paths."""
    if checkpoint_cfg is None:
        raise ValueError("Checkpoint has no cov_encoding_cfg.")
    patched = copy.deepcopy(checkpoint_cfg)
    for key in COVARIATE_PATH_KEYS:
        value = runtime_cfg.get(key, None)
        if value is not None:
            patched[key] = value

    # Preserve the official sampling behavior: this is the only non-path
    # covariate setting that upstream explicitly replaces at runtime.
    patched["celltype_encoding"] = runtime_cfg.celltype_encoding
    return patched


def main() -> None:
    upstream_root_value = os.environ.get("PERTURBDIFF_ROOT")
    if not upstream_root_value:
        raise SystemExit("PERTURBDIFF_ROOT must point to the PerturbDiff checkout.")

    upstream_root = Path(upstream_root_value).expanduser().resolve()
    upstream_entrypoint = upstream_root / "src/apps/run/rawdata_diffusion_sampling.py"
    if not upstream_entrypoint.is_file():
        raise SystemExit(f"Missing PerturbDiff sampling entrypoint: {upstream_entrypoint}")

    sys.path.insert(0, str(upstream_root))

    import torch
    from hydra import compose, initialize_config_dir
    from src.apps.run import rawdata_diffusion_sampling as upstream
    from src.models.lightning.lightning_module import PlModel

    def load_sampling_model(cfg, logger, datamodule):
        checkpoint = torch.load(
            cfg.model_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        hparams = checkpoint.get("hyper_parameters", {})
        covariate_cfg = patch_covariate_paths(
            hparams.get("cov_encoding_cfg"),
            cfg.cov_encoding,
        )
        model_cfg = hparams.get("model_cfg", cfg.model)
        optimizer_cfg = hparams.get("optimizer_cfg", cfg.optimization)
        del checkpoint

        return PlModel.load_from_checkpoint(
            cfg.model_checkpoint_path,
            cov_encoding_cfg=covariate_cfg,
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
            py_logger=logger,
            trainer_cfg=cfg.trainer,
            all_split_names=datamodule.all_split_names,
            map_location="cuda:0" if torch.cuda.is_available() else "cpu",
            weights_only=False,
        )

    # rawdata_diffusion_sampling imported the loader into its module namespace;
    # replacing that reference keeps the official model and sampling flow.
    upstream.load_sampling_model = load_sampling_model

    # Calling the decorated upstream main after importing it makes Hydra treat
    # ../../../configs as a Python package. PerturbDiff's configs directory is
    # not a package, so compose from its absolute filesystem path and invoke
    # the original task function retained by functools.wraps.
    config_dir = upstream_root / "configs"
    if not config_dir.is_dir():
        raise SystemExit(f"Missing PerturbDiff config directory: {config_dir}")
    task_function = getattr(upstream.main, "__wrapped__", None)
    if task_function is None:
        raise RuntimeError("Unable to access the upstream Hydra task function.")

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="rawdata_diffusion_sampling",
            overrides=sys.argv[1:],
        )
    task_function(cfg)


if __name__ == "__main__":
    main()
