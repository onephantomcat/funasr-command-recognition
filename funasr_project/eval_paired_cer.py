"""Run B0/ORACLE/B1 paired CER with one frozen Paraformer instance.

Canonical JSONL manifest fields:
``sample_id``, ``ref_text``, ``mixture``, ``target``, ``enrollment``.
Optional analysis fields include ``scene``, ``sir_db``, ``snr_db``,
``overlap_ratio`` and ``seed``. Supply either ``tse_target`` per row or one
``--p2-tse-checkpoint`` to generate B1 through P2's frozen public API.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from asr_demo import ASR_DIR, VAD_DIR, build_model, recognize_result
from cer import CerStats, cer_stats
from p2_tse_runtime import P2TSERuntime, sha256_file
from p3_eval_contracts import recognize_result_safely, require_unique_sample_ids
from speaker_verify import build_sv_model, extract_embedding


CONTRACT_VERSION = "p3_paired_cer_v1-rc2"
CONDITIONS = ("B0_MIXTURE", "ORACLE_TARGET", "B1_P2_TARGET")


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
    return rows


def resolve_path(value, data_root):
    if not isinstance(value, str) or not value:
        raise ValueError(f"audio path must be a non-empty string, got {value!r}")
    path = Path(value)
    if not path.is_absolute():
        path = data_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_manifest(rows, *, require_precomputed_tse):
    if not rows:
        raise ValueError("paired CER manifest is empty")
    require_unique_sample_ids(rows)
    required = {"sample_id", "ref_text", "mixture", "target"}
    if require_precomputed_tse:
        required.add("tse_target")
    else:
        required.add("enrollment")
    for index, row in enumerate(rows):
        missing = sorted(key for key in required if row.get(key) in (None, ""))
        if missing:
            raise ValueError(f"row {index} missing required fields: {', '.join(missing)}")
        if not isinstance(row["ref_text"], str):
            raise TypeError(
                f"row {index} ref_text must be str, got {type(row['ref_text']).__name__}"
            )


def analysis_bucket(row):
    scene = str(row.get("scene") or "UNSPECIFIED").upper()
    overlap = row.get("overlap_ratio")
    sir = row.get("sir_db")
    buckets = [f"scene:{scene}"]
    is_single = scene == "SINGLE" or overlap in (0, 0.0)
    if is_single:
        buckets.append("SINGLE")
    else:
        buckets.append("OVERLAP")
    if overlap is not None and float(overlap) >= 0.999:
        buckets.append("OVERLAP_100")
        if sir is not None and abs(float(sir) - (-5.0)) < 1e-6:
            buckets.append("OVERLAP_100_SIR_-5DB")
    if sir is not None:
        buckets.append(f"sir_db:{float(sir):g}")
    if row.get("snr_db") is not None:
        buckets.append(f"snr_db:{float(row['snr_db']):g}")
    return list(dict.fromkeys(buckets))


def safe_sample_name(sample_id):
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in str(sample_id)
    )


def _aggregate(records):
    total = CerStats()
    for record in records:
        total += CerStats(
            substitutions=record["substitutions"],
            deletions=record["deletions"],
            insertions=record["insertions"],
            reference_chars=record["reference_chars"],
        )
    return {
        **total.to_dict(),
        "samples": len(records),
        "asr_errors": sum(record["asr_status"] == "ERROR" for record in records),
    }


def summarize_predictions(predictions):
    by_condition = defaultdict(list)
    by_bucket = defaultdict(lambda: defaultdict(list))
    for record in predictions:
        condition = record["condition"]
        by_condition[condition].append(record)
        for bucket in record["buckets"]:
            # ``seed`` in the paired manifest controls mixture construction.  It
            # is not a model-training seed and must never drive the 2/3-model
            # replication decision.  Ignore legacy seed buckets defensively so
            # rc1 prediction files can be resummarized without rerunning ASR.
            if bucket.startswith("seed:"):
                continue
            by_bucket[bucket][condition].append(record)

    overall = {condition: _aggregate(by_condition[condition]) for condition in CONDITIONS}
    buckets = {
        bucket: {
            condition: _aggregate(condition_records)
            for condition, condition_records in conditions.items()
        }
        for bucket, conditions in sorted(by_bucket.items())
    }

    comparisons = {}
    for bucket, condition_summary in {"OVERALL": overall, **buckets}.items():
        if "B0_MIXTURE" not in condition_summary or "B1_P2_TARGET" not in condition_summary:
            continue
        baseline = condition_summary["B0_MIXTURE"]["cer"]
        candidate = condition_summary["B1_P2_TARGET"]["cer"]
        comparisons[bucket] = {
            "b0_cer": baseline,
            "b1_cer": candidate,
            "absolute_change_pp": (candidate - baseline) * 100.0,
            "absolute_reduction_pp": (baseline - candidate) * 100.0,
            "relative_reduction": None if baseline == 0 else (baseline - candidate) / baseline,
        }
    return overall, buckets, comparisons


def acceptance_summary(
    comparisons,
    *,
    result_valid,
    asr_errors,
    p2_output_quality,
    expected_samples,
):
    """Evaluate one trained checkpoint on one paired manifest.

    Replication across independently trained checkpoints is intentionally left
    to ``aggregate_paired_cer_seeds.py``.  The per-row manifest ``seed`` is a
    mixture-construction seed and is unrelated to that replication criterion.
    """
    high_overlap_key = "OVERLAP_100" if "OVERLAP_100" in comparisons else "OVERLAP"
    high_overlap = comparisons.get(high_overlap_key)
    single = comparisons.get("SINGLE")
    extreme = comparisons.get("OVERLAP_100_SIR_-5DB")
    quality_available = isinstance(p2_output_quality, dict)
    quality_samples = p2_output_quality.get("samples") if quality_available else None
    near_silent_samples = (
        p2_output_quality.get("near_silent_samples") if quality_available else None
    )

    checks = {
        "result_valid": bool(result_valid),
        "asr_errors_zero": asr_errors == 0,
        "p2_quality_available": quality_available,
        "p2_quality_complete": bool(
            quality_available and quality_samples == expected_samples
        ),
        "p2_near_silent_zero": None if not quality_available else bool(
            near_silent_samples == 0
        ),
        "high_overlap_bucket": high_overlap_key if high_overlap else None,
        "high_overlap_gain": None if high_overlap is None else bool(
            high_overlap["absolute_reduction_pp"] >= 5.0
            or (
                high_overlap["relative_reduction"] is not None
                and high_overlap["relative_reduction"] >= 0.15
            )
        ),
        "single_degradation_le_2pp": None if single is None else bool(
            single["absolute_change_pp"] <= 2.0
        ),
        "extreme_direction_not_reversed": None if extreme is None else bool(
            extreme["absolute_reduction_pp"] >= 0.0
        ),
    }
    required = (
        checks["result_valid"],
        checks["asr_errors_zero"],
        checks["p2_quality_available"],
        checks["p2_quality_complete"],
        checks["p2_near_silent_zero"],
        checks["high_overlap_gain"],
        checks["single_degradation_le_2pp"],
        checks["extreme_direction_not_reversed"],
    )
    if any(value is None for value in required):
        verdict = "INCONCLUSIVE_MISSING_REQUIRED_EVIDENCE"
    elif all(required):
        verdict = "PASS_SINGLE_RUN_THRESHOLDS"
    else:
        verdict = "FAIL_SINGLE_RUN_THRESHOLDS"
    return {"verdict": verdict, "checks": checks}


def summarize_p2_quality(predictions):
    """Summarize non-scoring P2 waveform diagnostics for B1 outputs."""
    infos = [
        record["p2_info"]
        for record in predictions
        if record["condition"] == "B1_P2_TARGET" and record.get("p2_info")
    ]
    if not infos:
        return None
    ratios = [
        float(info["output_to_input_rms_ratio"])
        for info in infos
        if info.get("output_to_input_rms_ratio") is not None
    ]
    near_silent = sum(bool(info.get("output_near_silent")) for info in infos)
    return {
        "samples": len(infos),
        "near_silent_samples": near_silent,
        "near_silent_rate": near_silent / len(infos),
        "rms_ratio_min": min(ratios) if ratios else None,
        "rms_ratio_median": statistics.median(ratios) if ratios else None,
        "rms_ratio_max": max(ratios) if ratios else None,
    }


def write_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--p2-tse-checkpoint", default=None)
    parser.add_argument("--p2-tse-sha256", default=None)
    parser.add_argument(
        "--training-seed",
        type=int,
        required=True,
        help="Seed used to train the evaluated checkpoint (not the manifest mixture seed)",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve() if args.data_root else manifest_path.parent
    rows = read_jsonl(manifest_path)
    if args.limit is not None:
        rows = rows[:args.limit]
    validate_manifest(rows, require_precomputed_tse=not bool(args.p2_tse_checkpoint))

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to reuse paired CER output directory: {out_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    p2_runtime = None
    sv_model = None
    if args.p2_tse_checkpoint:
        p2_runtime = P2TSERuntime(
            args.p2_tse_checkpoint,
            device=device,
            expected_sha256=args.p2_tse_sha256,
        )
        sv_model = build_sv_model(device=device)

    asr_model = build_model(with_punc=False, device=device)
    target_dir = out_dir / "p2_targets"
    out_dir.mkdir(parents=True)
    if p2_runtime:
        target_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    started_at = time.time()
    for index, row in enumerate(rows, 1):
        sample_id = row["sample_id"]
        mixture = resolve_path(row["mixture"], data_root)
        oracle = resolve_path(row["target"], data_root)
        if p2_runtime:
            enrollment = resolve_path(row["enrollment"], data_root)
            embedding = extract_embedding(sv_model, str(enrollment))
            tse_target = target_dir / (
                f"{safe_sample_name(sample_id)}_{p2_runtime.checkpoint_sha256[:12]}.wav"
            )
            p2_info = p2_runtime.extract_file(mixture, embedding, tse_target)
        else:
            tse_target = resolve_path(row["tse_target"], data_root)
            p2_info = None

        condition_paths = {
            "B0_MIXTURE": mixture,
            "ORACLE_TARGET": oracle,
            "B1_P2_TARGET": tse_target,
        }
        buckets = analysis_bucket(row)
        for condition, audio_path in condition_paths.items():
            outcome = recognize_result_safely(
                recognize_result, asr_model, str(audio_path)
            )
            stats = cer_stats(row["ref_text"], outcome.text)
            predictions.append({
                "sample_id": sample_id,
                "condition": condition,
                "audio": str(audio_path),
                "ref_text": row["ref_text"],
                "hyp_text": outcome.text,
                "raw_text": outcome.raw_text,
                "normalized_text": outcome.normalized_text,
                "final_text": outcome.final_text,
                "asr_status": outcome.status,
                "asr_error": outcome.error,
                "asr_latency_sec": outcome.elapsed_sec,
                "scene": row.get("scene"),
                "sir_db": row.get("sir_db"),
                "snr_db": row.get("snr_db"),
                "overlap_ratio": row.get("overlap_ratio"),
                "seed": row.get("seed"),
                "buckets": buckets,
                "p2_info": p2_info if condition == "B1_P2_TARGET" else None,
                **stats.to_dict(),
            })
        if index % 50 == 0 or index == len(rows):
            print(f"paired CER {index}/{len(rows)}")

    predictions_path = out_dir / "paired_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    overall, buckets, comparisons = summarize_predictions(predictions)
    errors = sum(record["asr_status"] == "ERROR" for record in predictions)
    p2_output_quality = summarize_p2_quality(predictions)
    acceptance = acceptance_summary(
        comparisons,
        result_valid=errors == 0,
        asr_errors=errors,
        p2_output_quality=p2_output_quality,
        expected_samples=len(rows),
    )
    summary = {
        "contract": CONTRACT_VERSION,
        "result_valid": errors == 0,
        "training_seed": args.training_seed,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "data_root": str(data_root),
        "asr": {
            "model_id": ASR_DIR,
            "vad_model_id": VAD_DIR,
            "with_punctuation": False,
            "device": device,
        },
        "p2": None if p2_runtime is None else p2_runtime.metadata(),
        "precomputed_tse": p2_runtime is None,
        "samples": len(rows),
        "asr_errors": errors,
        "elapsed_sec": round(time.time() - started_at, 3),
        "overall": overall,
        "buckets": buckets,
        "b1_vs_b0": comparisons,
        "p2_output_quality": p2_output_quality,
        "acceptance": acceptance,
        "predictions_sha256": sha256_file(predictions_path),
    }
    summary_path = out_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, summary_path)
    print(f"paired predictions: {predictions_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
