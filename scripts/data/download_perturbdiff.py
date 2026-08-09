#!/usr/bin/env python
"""Download released PerturbDiff data into the server data root.

The command deliberately requires acknowledgement for PBMC/Tahoe because the
official README reports roughly 750 GB and 3 TB after decompression.
"""

import argparse
import os
from pathlib import Path

PATTERNS = {
    "pbmc": ["finetune_data/pbmc_new/**"],
    "tahoe100m": ["finetune_data/tahoe100m_full_selected_processed_new/**"],
    "replogle": ["finetune_data/nadig_processed_data/**"],
    "assets": [
        "gene_names/**",
        "indices_cache/**",
        "selected_genes/**",
        "meta_data/**",
        "tmp_pbmc_ctrl.h5ad*",
        "tmp_ourtahoe_ctrl.h5ad*",
    ],
}
LARGE = {"pbmc": "~750 GB extracted", "tahoe100m": "~3 TB extracted"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*PATTERNS, "all"], required=True)
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "CELLDIFFA_DATA_ROOT", "/data/users/jchengak/DiffA/CellDiffA/data"
        ),
    )
    parser.add_argument("--confirm-large-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = list(PATTERNS) if args.dataset == "all" else [args.dataset]
    large = {name: LARGE[name] for name in selected if name in LARGE}
    destination = Path(args.data_root) / "PerturbDiff_data"
    patterns = [pattern for name in selected for pattern in PATTERNS[name]]

    print("repository: katarinayuan/PerturbDiff_data")
    print(f"destination: {destination}")
    print(f"include: {patterns}")
    if large:
        print(f"large datasets: {large}")
    if args.dry_run:
        return
    if large and not args.confirm_large_download:
        raise SystemExit(
            "Refusing the large download without --confirm-large-download. "
            "Check your storage quota first."
        )

    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="katarinayuan/PerturbDiff_data",
        repo_type="dataset",
        local_dir=destination,
        allow_patterns=patterns,
    )


if __name__ == "__main__":
    main()
