"""Rebuild an rc2 paired-CER summary from an existing prediction file."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from eval_paired_cer import (
    CONDITIONS,
    CONTRACT_VERSION,
    acceptance_summary,
    read_jsonl,
    summarize_p2_quality,
    summarize_predictions,
)
from p2_tse_runtime import sha256_file


def validate_prediction_completeness(predictions, expected_samples):
    by_sample = defaultdict(list)
    for index, record in enumerate(predictions):
        sample_id = record.get("sample_id")
        condition = record.get("condition")
        if sample_id in (None, ""):
            raise ValueError(f"prediction record {index} has no sample_id")
        if condition not in CONDITIONS:
            raise ValueError(f"prediction record {index} has invalid condition {condition!r}")
        by_sample[str(sample_id)].append(condition)

    if len(by_sample) != expected_samples:
        raise ValueError(
            f"prediction sample count mismatch: expected {expected_samples}, got {len(by_sample)}"
        )
    expected_conditions = sorted(CONDITIONS)
    for sample_id, conditions in by_sample.items():
        if sorted(conditions) != expected_conditions:
            raise ValueError(
                f"sample {sample_id!r} conditions mismatch: "
                f"expected {expected_conditions}, got {sorted(conditions)}"
            )


def build_resummary(base_summary, predictions, *, predictions_path, training_seed):
    expected_samples = base_summary.get("samples")
    if not isinstance(expected_samples, int) or expected_samples <= 0:
        raise ValueError("base summary must contain a positive integer samples field")
    validate_prediction_completeness(predictions, expected_samples)

    predictions_sha256 = sha256_file(predictions_path)
    recorded_sha256 = base_summary.get("predictions_sha256")
    if recorded_sha256 and recorded_sha256 != predictions_sha256:
        raise ValueError("prediction file SHA-256 does not match the base summary")

    overall, buckets, comparisons = summarize_predictions(predictions)
    errors = sum(record.get("asr_status") == "ERROR" for record in predictions)
    p2_output_quality = summarize_p2_quality(predictions)
    acceptance = acceptance_summary(
        comparisons,
        result_valid=errors == 0,
        asr_errors=errors,
        p2_output_quality=p2_output_quality,
        expected_samples=expected_samples,
    )

    summary = dict(base_summary)
    summary.update({
        "contract": CONTRACT_VERSION,
        "source_contract": base_summary.get("contract"),
        "result_valid": errors == 0,
        "training_seed": training_seed,
        "asr_errors": errors,
        "overall": overall,
        "buckets": buckets,
        "b1_vs_b0": comparisons,
        "p2_output_quality": p2_output_quality,
        "acceptance": acceptance,
        "predictions": str(Path(predictions_path).resolve()),
        "predictions_sha256": predictions_sha256,
    })
    return summary


def write_json_atomic(path, payload):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_summary_path = Path(args.base_summary).resolve()
    predictions_path = Path(args.predictions).resolve()
    with open(base_summary_path, encoding="utf-8") as stream:
        base_summary = json.load(stream)
    predictions = read_jsonl(predictions_path)
    summary = build_resummary(
        base_summary,
        predictions,
        predictions_path=predictions_path,
        training_seed=args.training_seed,
    )
    summary["resummarized_from"] = str(base_summary_path)
    summary["source_summary_sha256"] = sha256_file(base_summary_path)

    out_path = Path(args.out).resolve()
    write_json_atomic(out_path, summary)
    print(f"resummarized paired CER: {out_path}")
    print(f"single-run verdict: {summary['acceptance']['verdict']}")


if __name__ == "__main__":
    main()
