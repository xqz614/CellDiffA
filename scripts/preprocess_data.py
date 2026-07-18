"""
Data preprocessing script for CellDiffA.

Downloads and preprocesses single-cell perturbation datasets into a
standardized format for all downstream experiments.

Usage:
    python scripts/preprocess_data.py --dataset norman --n_top_genes 2000
    python scripts/preprocess_data.py --dataset norman --split additive --fold 0 --compute_priors
"""

import argparse
import os
import shutil
import sys
from zipfile import BadZipFile, ZipFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# Download Functions
# ============================================================


def dataverse_download(url: str, save_path: str) -> None:
    """
    Download a file from Harvard Dataverse using requests (same as GEARS).
    Uses requests.get with stream=True to handle large files.
    """
    import requests
    from tqdm import tqdm

    if os.path.exists(save_path):
        print(f"[Download] Found local copy: {save_path}")
        return

    print(f"[Download] Downloading from {url}...")
    response = requests.get(url, stream=True, timeout=(30, 300))
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    partial_path = f"{save_path}.part"
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(partial_path, "wb") as f:
        for data in response.iter_content(block_size):
            if data:
                progress_bar.update(len(data))
                f.write(data)
    progress_bar.close()
    os.replace(partial_path, save_path)
    print(f"[Download] Saved to {save_path}")


def extract_h5ad(zip_path: str, output_path: str, preferred_parent: str) -> None:
    """Safely copy the expected H5AD member without extracting arbitrary paths."""
    try:
        with ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".h5ad")]
            preferred = [
                name
                for name in members
                if preferred_parent.lower() in name.lower()
                and name.lower().endswith("perturb_processed.h5ad")
            ]
            candidates = preferred or members
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one H5AD in {zip_path}, found {len(candidates)}: {candidates}"
                )
            with archive.open(candidates[0]) as source, open(output_path, "wb") as target:
                shutil.copyfileobj(source, target)
    except BadZipFile as exc:
        raise RuntimeError(f"Dataverse response is not a valid ZIP: {zip_path}") from exc


def download_norman(data_root: str) -> str:
    """
    Download Norman et al. (2019) K562 CRISPRa dataset.

    This follows the exact same logic as GEARS:
        1. Download zip from Harvard Dataverse
        2. Extract to get perturb_processed.h5ad
        3. Copy/symlink to our expected path

    Source: https://github.com/snap-stanford/GEARS (preprocessed version)
    Original: Norman et al., Science 2019.
    """
    raw_dir = os.path.join(data_root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Final output path for CellDiffA
    output_path = os.path.join(raw_dir, "norman.h5ad")
    if os.path.exists(output_path):
        print(f"[Download] Norman dataset already exists at {output_path}")
        return output_path

    # Method 1: Try GEARS package (if installed)
    try:
        from gears import PertData

        print("[Download] Using GEARS package to download Norman dataset...")
        pert_data = PertData(raw_dir)
        pert_data.load(data_name="norman")
        adata = pert_data.adata
        adata.write_h5ad(output_path)
        print(f"[Download] Saved to {output_path}")
        return output_path
    except ImportError:
        print("[Download] GEARS not installed, using direct download...")
    except Exception as e:
        print(f"[Download] GEARS download failed: {e}, trying direct download...")

    # Method 2: Direct download from Harvard Dataverse (same URL as GEARS)
    # GEARS downloads this as a zip file containing norman/perturb_processed.h5ad
    url = "https://dataverse.harvard.edu/api/access/datafile/6154020"
    zip_path = os.path.join(raw_dir, "norman.zip")

    dataverse_download(url, zip_path)

    # Extract zip file (same as GEARS's zip_data_download_wrapper)
    print("[Download] Extracting H5AD...")
    extract_h5ad(zip_path, output_path, preferred_parent="norman")
    print(f"[Download] Saved to {output_path}")

    # Cleanup zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

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

    # Method 1: Try GEARS
    try:
        from gears import PertData

        print("[Download] Using GEARS package to download Replogle dataset...")
        pert_data = PertData(raw_dir)
        pert_data.load(data_name="replogle_k562_essential")
        adata = pert_data.adata
        adata.write_h5ad(output_path)
        print(f"[Download] Saved to {output_path}")
        return output_path
    except ImportError:
        print("[Download] GEARS not installed, using direct download...")
    except Exception as e:
        print(f"[Download] GEARS download failed: {e}, trying direct download...")

    # Method 2: Direct download
    url = "https://dataverse.harvard.edu/api/access/datafile/7458695"
    zip_path = os.path.join(raw_dir, "replogle_k562.zip")

    dataverse_download(url, zip_path)

    print("[Download] Extracting H5AD...")
    extract_h5ad(zip_path, output_path, preferred_parent="replogle_k562_essential")
    print(f"[Download] Saved to {output_path}")

    if os.path.exists(zip_path):
        os.remove(zip_path)

    return output_path


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="CellDiffA Data Preprocessing")
    parser.add_argument(
        "--dataset",
        type=str,
        default="norman",
        choices=["norman", "replogle_k562"],
        help="Dataset to preprocess",
    )
    parser.add_argument("--data_root", type=str, default="./data", help="Data root directory")
    parser.add_argument("--n_top_genes", type=int, default=2000, help="Number of HVGs")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["additive", "unseen"],
        help="Generate a specific split (optional)",
    )
    parser.add_argument("--fold", type=int, default=0, help="Fold index for split")
    parser.add_argument("--n_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--raw_counts",
        action="store_true",
        help="Normalize/log-transform X; omit for GEARS-distributed log data",
    )
    parser.add_argument("--compute_priors", action="store_true", help="Compute DE genes and shifts")
    args = parser.parse_args()
    if args.compute_priors and args.split is None:
        parser.error("--compute_priors requires --split.")

    # Step 1: Download
    print(f"\n{'=' * 60}")
    print(f"  CellDiffA Data Preprocessing: {args.dataset}")
    print(f"{'=' * 60}\n")

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
        already_normalized=not args.raw_counts,
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

    print(f"\n{'=' * 60}")
    print("  Preprocessing complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
