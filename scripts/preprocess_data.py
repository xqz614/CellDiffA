"""
Data preprocessing script for CellDiffA.

Downloads and preprocesses single-cell perturbation datasets into a
standardized format for all downstream experiments.

Usage:
    python scripts/preprocess_data.py --dataset norman --n_top_genes 2000
    python scripts/preprocess_data.py --dataset norman --split additive --fold 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scanpy as sc
import numpy as np


def download_norman(data_root: str) -> str:
    """
    Download Norman et al. (2019) K562 CRISPRa dataset.

    Source: https://github.com/snap-stanford/GEARS (preprocessed version)
    Original: Norman et al., Science 2019.
    """
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "norman.h5ad")

    if os.path.exists(output_path):
        print(f"[Download] Norman dataset already exists at {output_path}")
        return output_path

    print("[Download] Downloading Norman dataset...")

    # Method 1: Try GEARS data download
    try:
        from gears import PertData
        pert_data = PertData(raw_dir)
        pert_data.load(data_name="norman")
        adata = pert_data.adata
        adata.write_h5ad(output_path)
        print(f"[Download] Saved to {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"[Download] GEARS download failed: {e}")

    # Method 2: Direct download from public source
    import urllib.request
    url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
    print(f"[Download] Downloading from Harvard Dataverse...")
    urllib.request.urlretrieve(url, output_path)
    print(f"[Download] Saved to {output_path}")
    return output_path


def download_replogle(data_root: str) -> str:
    """
    Download Replogle et al. (2022) K562 CRISPRi dataset.

    Source: https://github.com/snap-stanford/GEARS
    Original: Replogle et al., Nature Biotechnology 2022.
    """
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    output_path = os.path.join(raw_dir, "replogle_k562.h5ad")

    if os.path.exists(output_path):
        print(f"[Download] Replogle dataset already exists at {output_path}")
        return output_path

    print("[Download] Downloading Replogle K562 dataset...")
    try:
        from gears import PertData
        pert_data = PertData(raw_dir)
        pert_data.load(data_name="replogle_k562")
        adata = pert_data.adata
        adata.write_h5ad(output_path)
        print(f"[Download] Saved to {output_path}")
        return output_path
    except (ImportError, Exception) as e:
        print(f"[Download] Download failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="CellDiffA Data Preprocessing")
    parser.add_argument(
        "--dataset",
        type=str,
        default="norman",
        choices=["norman", "replogle_k562", "adamson"],
        help="Dataset to preprocess",
    )
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of HVGs")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["additive", "combinations", "unseen"],
        help="Generate a specific split (optional)",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold index for split")
    parser.add_argument("--n_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compute_priors", action="store_true", help="Compute DE genes and shifts")
    args = parser.parse_args()

    # Step 1: Download
    print(f"\n{'='*60}")
    print(f"  CellDiffA Data Preprocessing: {args.dataset}")
    print(f"{'='*60}\n")

    if args.dataset == "norman":
        download_norman(args.data_root)
    elif args.dataset == "replogle_k562":
        download_replogle(args.data_root)
    else:
        raise ValueError(f"Download not implemented for {args.dataset}")

    # Step 2: Preprocess
    from data.data_manager import PerturbationDataManager

    dm = PerturbationDataManager(
        data_root=args.data_root,
        dataset_name=args.dataset,
        n_top_genes=args.n_top_genes,
        seed=args.seed,
    )
    adata = dm.load_and_preprocess()
    print(f"\n[Preprocess] Final AnnData: {adata.n_obs} cells x {adata.n_vars} genes")

    # Step 3: Generate split (if requested)
    if args.split:
        adata_train, adata_test = dm.create_split(
            split_strategy=args.split,
            fold=args.fold,
            n_folds=args.n_folds,
        )

        # Step 4: Compute priors (if requested)
        if args.compute_priors:
            print("\n[Priors] Computing DE genes from training set...")
            de_genes = dm.compute_de_genes(top_k=20)
            print(f"[Priors] DE genes computed for {len(de_genes)} conditions")

            print("[Priors] Computing perturbation shift vectors...")
            shifts = dm.compute_perturbation_shifts()
            print(f"[Priors] Shift vectors computed for {len(shifts)} conditions")

    print(f"\n{'='*60}")
    print("  Preprocessing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
