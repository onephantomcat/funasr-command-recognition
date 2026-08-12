#!/usr/bin/env python3
"""Prepare the nine source manifests consumed by build_p1_v2_b1.py.

Speech is split by speaker, while MUSAN and RIRS assets follow the existing
v2_b1 path-hash partition rule. The script only indexes source files; it does
not copy, alter, or synthesize audio.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


SPLITS = ("train", "dev", "holdout")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def path_partition(relative_path: str) -> str:
    """Existing contract: stable 80/10/10 partition from the relative path."""
    bucket = int(hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "holdout"


def split_speakers(speakers: list[str], seed: int) -> dict[str, set[str]]:
    if len(speakers) < 6:
        raise ValueError("at least six AISHELL speakers are required (two per split)")
    ordered = sorted(speakers)
    random.Random(seed).shuffle(ordered)
    n_train = max(2, min(len(ordered) - 4, round(len(ordered) * 0.75)))
    remaining = len(ordered) - n_train
    n_dev = remaining // 2
    result = {
        "train": set(ordered[:n_train]),
        "dev": set(ordered[n_train:n_train + n_dev]),
        "holdout": set(ordered[n_train + n_dev:]),
    }
    if any(len(value) < 2 for value in result.values()):
        raise AssertionError(f"speaker split too small: { {k: len(v) for k, v in result.items()} }")
    return result


def discover_speech(root: Path, seed: int) -> tuple[dict[str, list[dict]], dict[str, set[str]]]:
    wavs = sorted(path for path in root.rglob("*.wav") if path.is_file())
    if not wavs:
        raise FileNotFoundError(f"no AISHELL wav files found under {root}")
    by_speaker: dict[str, list[Path]] = {}
    for path in wavs:
        speaker = path.parent.name
        by_speaker.setdefault(speaker, []).append(path)
    too_small = sorted(speaker for speaker, paths in by_speaker.items() if len(paths) < 2)
    if too_small:
        raise ValueError(f"speakers with fewer than two utterances: {too_small[:8]}")
    speaker_sets = split_speakers(list(by_speaker), seed)
    rows = {split: [] for split in SPLITS}
    for split, speakers in speaker_sets.items():
        for speaker in sorted(speakers):
            for path in by_speaker[speaker]:
                rows[split].append({
                    "audio_path": str(path.resolve()),
                    "speaker_id": speaker,
                    "utterance_id": path.stem,
                })
    return rows, speaker_sets


def discover_assets(root: Path, kind: str) -> dict[str, list[dict]]:
    wavs = sorted(path for path in root.rglob("*.wav") if path.is_file())
    if kind == "rir":
        wavs = [
            path for path in wavs
            if "pointsource_noises" not in {part.lower() for part in path.parts}
            and ("simulated_rirs" in {part.lower() for part in path.parts}
                 or "rir" in path.stem.lower())
        ]
    if not wavs:
        raise FileNotFoundError(f"no usable {kind} wav files found under {root}")
    rows = {split: [] for split in SPLITS}
    for path in wavs:
        rel = path.relative_to(root).as_posix()
        rows[path_partition(rel)].append({"relative_path": rel})
    if any(not rows[split] for split in SPLITS):
        raise ValueError(f"empty {kind} partition: { {k: len(v) for k, v in rows.items()} }")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aishell-root", type=Path, required=True,
                        help="directory containing speaker subdirectories of WAV files")
    parser.add_argument("--musan-root", type=Path, required=True,
                        help="MUSAN root whose children are music/noise/speech")
    parser.add_argument("--rir-root", type=Path, required=True,
                        help="RIRS_NOISES root")
    parser.add_argument("--output", type=Path, required=True,
                        help="new input root; writes manifests/splits/*.jsonl")
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    split_dir = output / "manifests" / "splits"
    targets = [split_dir / f"{kind}_{split}.jsonl"
               for kind in ("speech", "noise", "rir") for split in SPLITS]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing split manifests: {existing[0]}")
    split_dir.mkdir(parents=True, exist_ok=True)

    speech, speaker_sets = discover_speech(args.aishell_root.resolve(), args.seed)
    noise = discover_assets(args.musan_root.resolve(), "noise")
    rir = discover_assets(args.rir_root.resolve(), "rir")

    counts = Counter()
    for kind, groups in (("speech", speech), ("noise", noise), ("rir", rir)):
        for split in SPLITS:
            counts[f"{kind}_{split}"] = write_jsonl(
                split_dir / f"{kind}_{split}.jsonl", groups[split]
            )

    if any(speaker_sets[a] & speaker_sets[b]
           for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]):
        raise AssertionError("speaker partitions overlap")
    print("P1_V2_INPUT_PREP_STATUS=PASS")
    print("SPEAKERS=" + json.dumps({k: len(v) for k, v in speaker_sets.items()}, sort_keys=True))
    print("ROWS=" + json.dumps(dict(sorted(counts.items())), sort_keys=True))
    print(f"OUTPUT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
