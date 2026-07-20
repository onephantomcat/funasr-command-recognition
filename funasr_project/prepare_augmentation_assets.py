# -*- coding: utf-8 -*-
"""Download MUSAN and RIRS_NOISES for external audio augmentation.

The assets are never mixed with DataSetA. They are consumed by
``build_external_trainset.py`` to create external positive noise/reverb samples
and retain the competition's pos/neg JSONL format.
"""
import argparse
import zipfile
from pathlib import Path

from prepare_public_dataset import download_from_candidates, extract_tar


ASSETS = {
    "musan": {
        "archive_name": "musan.tar.gz",
        "official_url": "https://www.openslr.org/resources/17/musan.tar.gz",
        "mirror_url": "https://openslr.magicdatatech.com/resources/17/musan.tar.gz",
        "size_hint": "11G",
        "archive_type": "tar",
    },
    "rirs_noises": {
        "archive_name": "rirs_noises.zip",
        "official_url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "mirror_url": "https://openslr.magicdatatech.com/resources/28/rirs_noises.zip",
        "size_hint": "1.3G",
        "archive_type": "zip",
    },
}


def extract_zip(archive, extract_root):
    archive = Path(archive)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extract_complete"
    if marker.exists():
        print(f"Extraction already complete: {extract_root}")
        return
    print(f"Extracting {archive} -> {extract_root}")
    with zipfile.ZipFile(archive) as bundle:
        base = extract_root.resolve()
        for member in bundle.infolist():
            target = (extract_root / member.filename).resolve()
            if base not in (target, *target.parents):
                raise RuntimeError(f"Unsafe archive member path: {member.filename}")
        bundle.extractall(extract_root)
    marker.write_text(str(archive), encoding="utf-8")


def candidate_urls(spec, source):
    if source == "official":
        return [spec["official_url"]]
    if source == "mirror":
        return [spec["mirror_url"]]
    return [spec["mirror_url"], spec["official_url"]]


def prepare_one(asset, args):
    spec = ASSETS[asset]
    asset_root = Path(args.root) / asset
    archive = asset_root / spec["archive_name"]
    extract_root = asset_root / "extracted"
    print(f"\n[{asset}] archive size: {spec['size_hint']}")
    if not args.skip_download and not archive.exists():
        download_from_candidates(candidate_urls(spec, args.source), archive, proxy=args.proxy)
    elif archive.exists():
        print(f"Using existing archive: {archive}")
    else:
        raise SystemExit(f"Archive not found for --skip-download: {archive}")

    if args.no_extract:
        return extract_root
    if spec["archive_type"] == "tar":
        extract_tar(archive, extract_root)
    else:
        extract_zip(archive, extract_root)
    return extract_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets",
        default="musan,rirs_noises",
        help=f"Comma-separated assets: {', '.join(sorted(ASSETS))}",
    )
    parser.add_argument("--root", default="data/public/augmentations")
    parser.add_argument("--source", choices=("auto", "mirror", "official"), default="auto")
    parser.add_argument("--proxy", default=None,
                        help="Optional HTTP/HTTPS proxy, e.g. http://127.0.0.1:7890.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    assets = [item.strip() for item in args.assets.split(",") if item.strip()]
    unknown = sorted(set(assets) - set(ASSETS))
    if unknown:
        raise SystemExit(f"Unknown assets: {', '.join(unknown)}")
    for asset in assets:
        root = prepare_one(asset, args)
        print(f"[{asset}] ready: {root}")


if __name__ == "__main__":
    main()
