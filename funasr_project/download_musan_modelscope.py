# -*- coding: utf-8 -*-
"""
Download MUSAN dataset (11GB) from ModelScope (魔搭社区) using multi-threading.

After downloading, the archive `musan.tar.gz` will be placed in `data/public/augmentations/musan/`,
and you can run:
    python prepare_augmentation_assets.py --assets musan --skip-download
"""
import argparse
import shutil
import sys
from pathlib import Path
from modelscope.hub.snapshot_download import snapshot_download


def download_musan(max_workers=8, target_root="data/public/augmentations"):
    target_dir = Path(target_root) / "musan"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_archive = target_dir / "musan.tar.gz"

    if target_archive.exists():
        print(f"[ModelScope Download] Existing archive found: {target_archive}")
        return target_archive

    temp_dir = Path(target_root) / "musan_modelscope_tmp"
    print(f"[ModelScope Download] Downloading musan.tar.gz from OmniData/MUSAN using {max_workers} worker threads...")
    
    snapshot_download(
        repo_id="OmniData/MUSAN",
        repo_type="dataset",
        allow_patterns=["raw/musan.tar.gz"],
        local_dir=str(temp_dir),
        max_workers=max_workers,
    )

    downloaded_file = temp_dir / "raw" / "musan.tar.gz"
    if not downloaded_file.exists():
        raise RuntimeError(f"Download completed but file not found at: {downloaded_file}")

    print(f"[ModelScope Download] Moving {downloaded_file} -> {target_archive}")
    shutil.move(str(downloaded_file), str(target_archive))

    if temp_dir.exists():
        shutil.rmtree(str(temp_dir), ignore_errors=True)

    print(f"[ModelScope Download] Download complete: {target_archive}")
    return target_archive


def main():
    parser = argparse.ArgumentParser(description="Download MUSAN dataset from ModelScope with multi-threading.")
    parser.add_argument("--max-workers", type=int, default=8, help="Number of concurrent download worker threads (default: 8)")
    parser.add_argument("--root", default="data/public/augmentations", help="Augmentation assets root directory")
    args = parser.parse_args()

    download_musan(max_workers=args.max_workers, target_root=args.root)
    print("\nNext step: Run extraction and MD5 verification with:")
    print("  python prepare_augmentation_assets.py --assets musan --skip-download\n")


if __name__ == "__main__":
    main()
