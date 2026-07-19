#!/usr/bin/env python
"""Selectively download the released PerturbDiff checkpoints."""

import argparse
import os
from pathlib import Path

CHECKPOINTS = {
    "pretrained": "pretrained.ckpt",
    "pbmc_scratch": "from_scratch_pbmc.ckpt",
    "pbmc_finetuned": "finetuned_pbmc.ckpt",
    "tahoe100m_scratch": "from_scratch_tahoe100m.ckpt",
    "tahoe100m_finetuned": "finetuned_tahoe100m.ckpt",
    "replogle_scratch": "from_scratch_replogle.ckpt",
    "replogle_finetuned": "finetuned_replogle.ckpt",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=[*CHECKPOINTS, "all"], required=True)
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "CELLDIFFA_DATA_ROOT", "/data/users/jchengak/DiffA/CellDiffA/data"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = list(CHECKPOINTS.values()) if args.checkpoint == "all" else [
        CHECKPOINTS[args.checkpoint]
    ]
    destination = Path(args.data_root) / "checkpoints" / "PerturbDiff_release_ckpt"
    print("repository: katarinayuan/PerturbDiff_release_ckpt")
    print(f"destination: {destination}")
    print(f"files: {selected}")
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="katarinayuan/PerturbDiff_release_ckpt",
        repo_type="model",
        local_dir=destination,
        allow_patterns=selected,
    )


if __name__ == "__main__":
    main()
