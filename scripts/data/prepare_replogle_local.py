#!/usr/bin/env python
"""Fetch the published Replogle artifact, not the entire pretraining corpus.

Remote paths are resolved from the Hub API and stored without perturb_data/.
Downloads resume and must pass size and SHA256 checks before atomic publication.
This fetches only published assets; it does not invent missing embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA_REPO = "katarinayuan/PerturbDiff_data"
MODEL_REPO = "katarinayuan/PerturbDiff_release_ckpt"
DATA_REVISION = "10654f2"  # Published dataset layout inspected before this run.
MODEL_REVISION = "c33e578"


def hub_files(endpoint: str, repo: str, kind: str, revision: str, folder: str = ""):
    url = f"{endpoint}/api/{kind}/{repo}/tree/{revision}/{folder}?recursive=true"
    result = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as response:
            result.extend(json.load(response))
            links = response.headers.get("Link", "")
        url = None
        for item in links.split(","):
            if 'rel="next"' in item:
                url = item.split("<", 1)[1].split(">", 1)[0]
    return [row for row in result if row["type"] == "file"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(item: dict) -> dict:
    path = Path(item["local_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Existing file failed verification; not overwriting: {path}")
        print(f"VERIFIED {path.name}", flush=True)
        return item
    partial = path.with_name(path.name + ".partial")
    print(f"DOWNLOADING {path.name} ({item['bytes'] / 1e9:.3f} GB)", flush=True)
    command = [
        "curl",
        "--fail",
        "--http1.1",
        "--location",
        "--silent",
        "--show-error",
        "--retry",
        "0",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "1024",
        "--speed-time",
        "120",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        item["url"],
    ]
    # Retry in a new curl invocation: internal curl retries may truncate back
    # to the original resume offset, losing progress on unstable connections.
    for attempt in range(12):
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            break
        if partial.exists() and partial.stat().st_size == item["bytes"]:
            break
        if attempt == 11:
            completed.check_returncode()
        time.sleep(min(5 * (attempt + 1), 30))
    if partial.stat().st_size != item["bytes"] or sha256(partial) != item["sha256"]:
        raise RuntimeError(f"Downloaded file failed size/SHA256 verification: {partial}")
    os.replace(partial, path)
    print(f"VERIFIED {path.name}", flush=True)
    return item


def unpack(path: Path) -> None:
    import zstandard

    target = path.with_suffix("")
    with path.open("rb") as source:
        content_size = zstandard.frame_content_size(source.read(32))
    if content_size in (zstandard.CONTENTSIZE_UNKNOWN, zstandard.CONTENTSIZE_ERROR):
        raise RuntimeError("Refusing decompression without a known storage requirement.")
    if target.exists():
        if target.stat().st_size != content_size:
            raise RuntimeError(f"Existing extracted file has the wrong size: {target}")
        print(f"EXISTS {target} ({content_size / 1e9:.3f} GB)", flush=True)
        return
    if shutil.disk_usage(path.parent).free < content_size + 20 * 1024**3:
        raise RuntimeError("Insufficient storage to extract while keeping 20 GiB free.")
    partial = target.with_name(target.name + ".extracting")
    print(f"EXTRACTING {target.name} ({content_size / 1e9:.3f} GB)", flush=True)
    with path.open("rb") as source, partial.open("wb") as destination:
        zstandard.ZstdDecompressor().copy_stream(source, destination)
    if partial.stat().st_size != content_size:
        raise RuntimeError("Extracted size differs from the zstd frame header.")
    os.replace(partial, target)
    print(f"EXTRACTED {target}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    manifest = []
    collections = [
        (DATA_REPO, "datasets", DATA_REVISION, "perturb_data"),
        (MODEL_REPO, "models", MODEL_REVISION, ""),
    ]
    for repo, kind, revision, folder in collections:
        for row in hub_files(endpoint, repo, kind, revision, folder):
            remote = row["path"]
            if kind == "datasets":
                relative = remote.removeprefix("perturb_data/")
                selected = (
                    relative == "finetune_data/nadig_processed_data/replogle.h5ad.zst"
                    or relative.startswith(("gene_names/", "meta_data/", "selected_genes/"))
                    or (relative.startswith("indices_cache/") and "replogle" in relative)
                )
                local = root / "PerturbDiff_data" / relative
                prefix = "datasets/"
            else:
                selected = remote in ("from_scratch_replogle.ckpt", "finetuned_replogle.ckpt")
                local = root / "checkpoints/PerturbDiff_release_ckpt" / remote
                prefix = ""
            if not selected:
                continue
            # Selected genes include a small non-LFS CSV, which is not used.
            checksum = row.get("lfs", {}).get("oid")
            if checksum is None:
                continue
            manifest.append(
                {
                    "repository": repo,
                    "revision": revision,
                    "remote_path": remote,
                    "url": f"{endpoint}/{prefix}{repo}/resolve/{revision}/"
                    f"{urllib.parse.quote(remote, safe='/')}",
                    "local_path": str(local),
                    "bytes": row["size"],
                    "sha256": checksum,
                }
            )
    if not any(x["remote_path"].endswith("replogle.h5ad.zst") for x in manifest):
        raise RuntimeError("No Replogle artifact matched. Refusing an empty download.")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"TOTAL DOWNLOAD {sum(x['bytes'] for x in manifest) / 1e9:.3f} GB", flush=True)
    if not args.download:
        return
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 60 * 1024**3:
        raise RuntimeError(
            "At least 60 GiB free disk space is required for first-time preparation."
        )
    # Save the exact manifest even if a later network request is interrupted.
    (root / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(download, manifest))
    unpack(root / "PerturbDiff_data/finetune_data/nadig_processed_data/replogle.h5ad.zst")


if __name__ == "__main__":
    main()
