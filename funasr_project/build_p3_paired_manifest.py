"""Build an auditable P3 paired-CER manifest from frozen P1 manifests.

P1 publishes acoustic metadata with ``*_wav`` field names and AISHELL
utterance IDs.  P3's paired evaluator consumes a smaller canonical contract
with reference text attached.  This bridge performs only deterministic field
mapping; it never synthesizes or alters audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path


CONTRACT_VERSION = "p1_to_p3_paired_manifest_v1"
AUDIO_FIELDS = ("mixture", "target", "enrollment")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_reference_text(text):
    """Remove transcript token separators without changing characters."""
    if not isinstance(text, str):
        raise TypeError(f"transcript text must be str, got {type(text).__name__}")
    return "".join(text.split())


def read_transcripts(path):
    transcripts = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_number} has no transcript text")
            utterance_id, text = parts
            if utterance_id in transcripts:
                raise ValueError(f"duplicate transcript utterance ID: {utterance_id}")
            text = compact_reference_text(text)
            if not text:
                raise ValueError(f"empty transcript for utterance ID: {utterance_id}")
            transcripts[utterance_id] = text
    if not transcripts:
        raise ValueError(f"transcript file is empty: {path}")
    return transcripts


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def convert_row(row, transcripts):
    required = (
        "sample_id",
        "target_utt",
        "mixture_wav",
        "target_wav",
        "enroll_wav",
    )
    missing = [name for name in required if row.get(name) in (None, "")]
    if missing:
        raise ValueError(
            f"P1 row {row.get('sample_id')!r} missing fields: {', '.join(missing)}"
        )
    target_utt = row["target_utt"]
    if target_utt not in transcripts:
        raise KeyError(
            f"P1 row {row['sample_id']!r} target_utt not found in transcript: "
            f"{target_utt}"
        )

    scenario = str(row.get("scenario") or "UNSPECIFIED")
    return {
        "sample_id": row["sample_id"],
        "ref_text": transcripts[target_utt],
        "mixture": row["mixture_wav"],
        "target": row["target_wav"],
        "enrollment": row["enroll_wav"],
        "scene": "SINGLE" if scenario.lower() == "single" else scenario,
        "split": row.get("split"),
        "sir_db": row.get("measured_sir_db"),
        "snr_db": row.get("measured_snr_db"),
        "overlap_ratio": row.get("measured_overlap"),
        "seed": row.get("seed"),
        "target_speaker": row.get("target_speaker"),
        "interferer_speaker": row.get("interferer_speaker"),
        "target_utt": target_utt,
        "enroll_utt": row.get("enroll_utt"),
        "p1_schema_version": row.get("schema_version"),
        "p1_generator_version": row.get("generator_version"),
        "p1_output_hashes": row.get("output_hashes"),
    }


def resolve_audio_path(value, data_root):
    path = Path(value)
    return path if path.is_absolute() else data_root / path


def readiness(rows, data_root, example_limit=20):
    missing_counts = Counter()
    missing_examples = []
    present_counts = Counter()
    for row in rows:
        for field in AUDIO_FIELDS:
            path = resolve_audio_path(row[field], data_root)
            if path.is_file():
                present_counts[field] += 1
            else:
                missing_counts[field] += 1
                if len(missing_examples) < example_limit:
                    missing_examples.append({
                        "sample_id": row["sample_id"],
                        "field": field,
                        "path": str(path.resolve()),
                    })
    missing_total = sum(missing_counts.values())
    return {
        "ready_for_paired_eval": missing_total == 0,
        "required_audio_files": len(rows) * len(AUDIO_FIELDS),
        "present_by_field": dict(present_counts),
        "missing_by_field": dict(missing_counts),
        "missing_total": missing_total,
        "missing_examples": missing_examples,
    }


def write_jsonl_atomic(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def build_manifest(manifest_paths, transcript_path, data_root):
    transcripts = read_transcripts(transcript_path)
    converted = []
    inputs = []
    seen = set()
    for manifest_path in manifest_paths:
        source_rows = read_jsonl(manifest_path)
        inputs.append({
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "rows": len(source_rows),
        })
        for source_row in source_rows:
            row = convert_row(source_row, transcripts)
            if row["sample_id"] in seen:
                raise ValueError(f"duplicate sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
            converted.append(row)

    report = {
        "contract": CONTRACT_VERSION,
        "input_manifests": inputs,
        "transcript": str(transcript_path.resolve()),
        "transcript_sha256": sha256_file(transcript_path),
        "data_root": str(data_root.resolve()),
        "rows": len(converted),
        "unique_sample_ids": len(seen),
        "split_counts": dict(Counter(row["split"] for row in converted)),
        "scene_counts": dict(Counter(row["scene"] for row in converted)),
        "audio_readiness": readiness(converted, data_root),
    }
    return converted, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1-manifest", action="append", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--require-audio",
        action="store_true",
        help="Abort instead of writing when any mixture/target/enrollment WAV is absent.",
    )
    args = parser.parse_args()

    manifest_paths = [Path(path).resolve() for path in args.p1_manifest]
    transcript_path = Path(args.transcript).resolve()
    data_root = Path(args.data_root).resolve()
    rows, report = build_manifest(manifest_paths, transcript_path, data_root)

    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    write_jsonl_atomic(output_path, rows)
    report["output"] = str(output_path)
    report["output_sha256"] = sha256_file(output_path)
    write_json_atomic(report_path, report)
    print(f"paired manifest: {output_path}")
    print(f"readiness report: {report_path}")
    print(
        "ready_for_paired_eval="
        f"{report['audio_readiness']['ready_for_paired_eval']} "
        f"missing={report['audio_readiness']['missing_total']}"
    )
    if args.require_audio and not report["audio_readiness"]["ready_for_paired_eval"]:
        missing = report["audio_readiness"]["missing_total"]
        raise FileNotFoundError(f"paired CER audio is incomplete: {missing} paths missing")


if __name__ == "__main__":
    main()
