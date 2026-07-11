# -*- coding: utf-8 -*-
"""
Download a public speech dataset and convert it to the competition train format.

Currently implemented:
  aishell1: OpenSLR SLR33 AISHELL-1 Mandarin speech corpus.

The output shape matches DataSetA, but it is external training/tuning data:
  pos/
  neg/
  pos.jsonl
  neg.jsonl
  phrase_bank.txt

DataSetA itself must remain test-only.
"""
import argparse
import tarfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from build_external_trainset import build


DATASETS = {
    "aishell1": {
        "homepage": "https://www.openslr.org/33/",
        "url": "https://www.openslr.org/resources/33/data_aishell.tgz",
        "archive_name": "data_aishell.tgz",
        "size_hint": "15G",
    },
}


def human_size(n):
    if n is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def human_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"


def print_progress(done, total, started_at, last_update=False):
    elapsed = max(time.time() - started_at, 1e-6)
    speed = done / elapsed
    if total:
        ratio = min(max(done / total, 0.0), 1.0)
        bar_width = 30
        filled = int(round(bar_width * ratio))
        bar = "#" * filled + "-" * (bar_width - filled)
        eta = (total - done) / speed if speed > 0 else None
        line = (
            f"\r[{bar}] {ratio * 100:6.2f}% "
            f"{human_size(done)} / {human_size(total)} "
            f"{human_size(speed)}/s ETA {human_duration(eta)}"
        )
    else:
        line = (
            f"\rDownloaded {human_size(done)} "
            f"at {human_size(speed)}/s elapsed {human_duration(elapsed)}"
        )
    print(line, end="\n" if last_update else "", flush=True)


def download_with_resume(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    if start:
        headers["Range"] = f"bytes={start}-"
        print(f"Resuming download at {human_size(start)}")
    else:
        print(f"Downloading {url}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        status = getattr(resp, "status", None)
        if start and status == 200:
            print("Server did not honor Range; restarting download from 0.")
            start = 0
        mode = "ab" if start else "wb"
        total_header = resp.headers.get("Content-Length")
        total = int(total_header) + start if total_header else None
        done = start
        started_at = time.time()
        last_print_at = 0.0
        print_progress(done, total, started_at)
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last_print_at >= 0.5:
                    last_print_at = now
                    print_progress(done, total, started_at)
        print_progress(done, total, started_at, last_update=True)
    tmp.replace(dest)
    print(f"Downloaded: {dest}")


def extract_tar(archive, extract_root):
    archive = Path(archive)
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    marker = extract_root / ".extract_complete"
    if marker.exists():
        print(f"Extraction already complete: {extract_root}")
        return
    print(f"Extracting {archive} -> {extract_root}")
    with tarfile.open(archive, "r:*") as tar:
        base = extract_root.resolve()
        for member in tar.getmembers():
            target = (extract_root / member.name).resolve()
            if base not in (target, *target.parents):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")
        tar.extractall(extract_root)
    marker.write_text(str(archive), encoding="utf-8")


def first_existing(candidates):
    for item in candidates:
        if item and Path(item).exists():
            return Path(item)
    return None


def find_aishell_paths(extract_root):
    root = Path(extract_root)
    wav_candidates = (
        list(root.glob("**/wav/train"))
        + list(root.glob("**/wav/dev"))
        + list(root.glob("**/wav/test"))
        + list(root.glob("**/aishell_test"))
    )
    transcript_candidates = (
        list(root.glob("**/aishell_transcript*.txt"))
        + list(root.glob("**/aishell*.csv"))
    )
    wav_root = first_existing(wav_candidates)
    transcript = first_existing(transcript_candidates)
    if not wav_root or not transcript:
        raise SystemExit(
            "Could not locate AISHELL wav/transcript files after extraction. "
            f"wav_root={wav_root}, transcript={transcript}"
        )
    return wav_root, transcript


def prepare(args):
    spec = DATASETS[args.dataset]
    public_root = Path(args.public_root) / args.dataset
    archive = Path(args.archive) if args.archive else public_root / spec["archive_name"]
    extract_root = Path(args.extract_root) if args.extract_root else public_root / "extracted"

    if args.use_existing_local:
        wav_root = Path(args.local_wav_root)
        transcript = Path(args.local_transcript)
    else:
        if not args.skip_download and not archive.exists():
            print(f"{args.dataset} archive is large ({spec['size_hint']}).")
            download_with_resume(args.url or spec["url"], archive)
        elif archive.exists():
            print(f"Using existing archive: {archive}")
        else:
            raise SystemExit(f"Archive not found: {archive}")

        if not args.no_extract:
            extract_tar(archive, extract_root)
        wav_root, transcript = find_aishell_paths(extract_root)

    print(f"Converting public dataset:")
    print(f"  wav_root   = {wav_root}")
    print(f"  transcript = {transcript}")
    print(f"  out        = {args.out}")

    build(SimpleNamespace(
        wav_root=str(wav_root),
        csv=str(transcript),
        out=args.out,
        target_speakers=args.target_speakers,
        interferer_speakers=args.interferer_speakers,
        enroll_count=args.enroll_count,
        trials_per_speaker=args.trials_per_speaker,
        wake_text=args.wake_text,
        seed=args.seed,
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="aishell1")
    parser.add_argument("--public-root", default="data/public")
    parser.add_argument("--out", default="data/public_train/aishell1")
    parser.add_argument("--url", default=None, help="Override dataset download URL.")
    parser.add_argument("--archive", default=None, help="Existing or target archive path.")
    parser.add_argument("--extract-root", default=None)
    parser.add_argument("--skip-download", action="store_true",
                        help="Use an existing archive instead of downloading.")
    parser.add_argument("--no-extract", action="store_true",
                        help="Assume files are already extracted.")
    parser.add_argument("--use-existing-local", action="store_true",
                        help="Use the local AISHELL subset already in data/aishell_test.")
    parser.add_argument("--local-wav-root", default="data/aishell_test")
    parser.add_argument("--local-transcript", default="data/aishell1_test.csv")
    parser.add_argument("--target-speakers", type=int, default=8)
    parser.add_argument("--interferer-speakers", type=int, default=8)
    parser.add_argument("--enroll-count", type=int, default=3)
    parser.add_argument("--trials-per-speaker", type=int, default=4)
    parser.add_argument("--wake-text", default="hi colmo")
    parser.add_argument("--seed", type=int, default=2026)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
