# -*- coding: utf-8 -*-
"""Download MUSAN and RIRS_NOISES for external audio augmentation.

The assets are never mixed with DataSetA. They are consumed by
``build_external_trainset.py`` to create external positive noise/reverb samples
and retain the competition's pos/neg JSONL format.
"""
import argparse
import hashlib
import socket
import zipfile
from pathlib import Path

from prepare_public_dataset import download_from_candidates, extract_tar


LOCAL_PROXY_PORTS = (7890, 7891, 10809, 10808, 1080)


ASSETS = {
    "musan": {
        "archive_name": "musan.tar.gz",
        "official_url": "https://www.openslr.org/resources/17/musan.tar.gz",
        # The mirror's HTTPS certificate is currently expired; this public-data
        # mirror also serves the same file over HTTP (as used for AISHELL here).
        "mirror_url": "http://openslr.magicdatatech.com/resources/17/musan.tar.gz",
        "size_hint": "11G",
        "archive_type": "tar",
        "md5": "0c472d4fc0c5141eca47ad1ffeb2a7df",
    },
    "rirs_noises": {
        "archive_name": "rirs_noises.zip",
        "official_url": "https://www.openslr.org/resources/28/rirs_noises.zip",
        "mirror_url": "https://openslr.magicdatatech.com/resources/28/rirs_noises.zip",
        "size_hint": "1.3G",
        "archive_type": "zip",
        "md5": "e6f48e257286e05de56413b4779d8ffb",
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


def detect_local_proxy():
    """Return a reachable common local HTTP proxy, if one is running."""
    for port in LOCAL_PROXY_PORTS:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return f"http://127.0.0.1:{port}"
        except OSError:
            continue
    return None


def verify_md5(path, expected):
    """Validate an archive before extraction, without loading it into memory."""
    digest = hashlib.md5()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise RuntimeError(
            f"Archive checksum mismatch for {path}: expected {expected}, got {actual}. "
            "The file is corrupt; remove or rename it, then run this command again."
        )
    print(f"Checksum verified (MD5): {actual}")


def prepare_one(asset, args):
    spec = ASSETS[asset]
    asset_root = Path(args.root) / asset
    archive = asset_root / spec["archive_name"]
    extract_root = asset_root / "extracted"
    print(f"\n[{asset}] archive size: {spec['size_hint']}")
    if not args.skip_download and not archive.exists():
        download_from_candidates(
            candidate_urls(spec, args.source),
            archive,
            proxy=args.proxy,
            retries=args.retries,
            timeout=args.download_timeout,
        )
    elif archive.exists():
        print(f"Using existing archive: {archive}")
    else:
        raise SystemExit(f"Archive not found for --skip-download: {archive}")

    verify_md5(archive, spec["md5"])

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
    parser.add_argument(
        "--proxy",
        default="auto",
        help=(
            "Proxy URL, 'auto' (default; detect a running local proxy), or "
            "'direct' to disable proxy use."
        ),
    )
    parser.add_argument("--retries", type=int, default=5,
                        help="Retry each source this many times; partial downloads resume automatically.")
    parser.add_argument("--download-timeout", type=float, default=60,
                        help="Socket timeout in seconds for each download request.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    if args.proxy.lower() == "auto":
        args.proxy = detect_local_proxy()
        if args.proxy:
            print(f"Detected local proxy: {args.proxy}")
        else:
            print("No local proxy detected; downloading directly.")
    elif args.proxy.lower() == "direct":
        args.proxy = None
        print("Proxy disabled; downloading directly.")

    assets = [item.strip() for item in args.assets.split(",") if item.strip()]
    unknown = sorted(set(assets) - set(ASSETS))
    if unknown:
        raise SystemExit(f"Unknown assets: {', '.join(unknown)}")
    for asset in assets:
        root = prepare_one(asset, args)
        print(f"[{asset}] ready: {root}")


if __name__ == "__main__":
    main()
