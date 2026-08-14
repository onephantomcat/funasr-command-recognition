"""Aggregate exactly three independent P2 training-seed paired-CER runs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval_paired_cer import CONTRACT_VERSION
from p2_tse_runtime import sha256_file


AGGREGATE_CONTRACT_VERSION = "p3_paired_cer_training_seed_aggregate_v1"
PASS_VERDICT = "PASS_SINGLE_RUN_THRESHOLDS"


def _require_valid_summary(summary, index):
    prefix = f"summary {index}"
    if summary.get("contract") != CONTRACT_VERSION:
        raise ValueError(
            f"{prefix} contract must be {CONTRACT_VERSION}, got {summary.get('contract')!r}"
        )
    training_seed = summary.get("training_seed")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise ValueError(f"{prefix} has no integer training_seed")
    if summary.get("result_valid") is not True or summary.get("asr_errors") != 0:
        raise ValueError(f"{prefix} is not a valid zero-error ASR result")

    samples = summary.get("samples")
    quality = summary.get("p2_output_quality")
    if not isinstance(quality, dict):
        raise ValueError(f"{prefix} has no P2 output-quality evidence")
    if quality.get("samples") != samples or quality.get("near_silent_samples") != 0:
        raise ValueError(f"{prefix} has incomplete or near-silent P2 outputs")

    checkpoint_sha256 = (summary.get("p2") or {}).get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise ValueError(f"{prefix} has no valid P2 checkpoint SHA-256")
    if not isinstance(summary.get("manifest_sha256"), str):
        raise ValueError(f"{prefix} has no manifest SHA-256")
    if not isinstance(summary.get("asr"), dict):
        raise ValueError(f"{prefix} has no ASR configuration")


def aggregate_summaries(summaries, *, frozen_training_seed, sources=None):
    if len(summaries) != 3:
        raise ValueError(f"exactly three training-seed summaries are required, got {len(summaries)}")
    for index, summary in enumerate(summaries, 1):
        _require_valid_summary(summary, index)

    training_seeds = [summary["training_seed"] for summary in summaries]
    if len(set(training_seeds)) != 3:
        raise ValueError("training_seed values must be distinct")
    if frozen_training_seed not in training_seeds:
        raise ValueError("frozen training seed is absent from the three summaries")

    checkpoint_shas = [summary["p2"]["checkpoint_sha256"] for summary in summaries]
    if len(set(checkpoint_shas)) != 3:
        raise ValueError("P2 checkpoint SHA-256 values must be distinct")

    common_fields = ("manifest_sha256", "samples", "asr")
    for field in common_fields:
        values = [summary[field] for summary in summaries]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"all summaries must share the same {field}")

    per_seed = []
    for index, summary in enumerate(summaries):
        comparisons = summary["b1_vs_b0"]
        per_seed.append({
            "training_seed": summary["training_seed"],
            "is_frozen_candidate": summary["training_seed"] == frozen_training_seed,
            "checkpoint_sha256": summary["p2"]["checkpoint_sha256"],
            "single_run_verdict": summary["acceptance"]["verdict"],
            "passed": summary["acceptance"]["verdict"] == PASS_VERDICT,
            "overall_absolute_reduction_pp": comparisons["OVERALL"]["absolute_reduction_pp"],
            "overlap_100_absolute_reduction_pp": comparisons["OVERLAP_100"]["absolute_reduction_pp"],
            "single_absolute_change_pp": comparisons["SINGLE"]["absolute_change_pp"],
            "extreme_absolute_reduction_pp": comparisons[
                "OVERLAP_100_SIR_-5DB"
            ]["absolute_reduction_pp"],
            "source": None if sources is None else str(sources[index]),
        })

    passed_count = sum(item["passed"] for item in per_seed)
    frozen_passed = next(
        item["passed"] for item in per_seed if item["is_frozen_candidate"]
    )
    accepted = frozen_passed and passed_count >= 2
    return {
        "contract": AGGREGATE_CONTRACT_VERSION,
        "paired_run_contract": CONTRACT_VERSION,
        "manifest_sha256": summaries[0]["manifest_sha256"],
        "samples_per_training_seed": summaries[0]["samples"],
        "frozen_training_seed": frozen_training_seed,
        "training_seed_count": 3,
        "passed_training_seed_count": passed_count,
        "required_passed_training_seed_count": 2,
        "frozen_candidate_passed": frozen_passed,
        "verdict": "ACCEPT_B1_CANDIDATE" if accepted else "REJECT_CURRENT_TSE",
        "per_training_seed": sorted(per_seed, key=lambda item: item["training_seed"]),
    }


def write_json_atomic(path, payload):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing aggregate: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--frozen-training-seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summary_paths = [Path(value).resolve() for value in args.summary]
    summaries = []
    for path in summary_paths:
        with open(path, encoding="utf-8") as stream:
            summary = json.load(stream)
        summary["source_summary_sha256"] = sha256_file(path)
        summaries.append(summary)

    aggregate = aggregate_summaries(
        summaries,
        frozen_training_seed=args.frozen_training_seed,
        sources=summary_paths,
    )
    aggregate["source_summaries"] = [
        {"path": str(path), "sha256": summary["source_summary_sha256"]}
        for path, summary in zip(summary_paths, summaries)
    ]

    out_path = Path(args.out).resolve()
    write_json_atomic(out_path, aggregate)
    print(f"training-seed aggregate: {out_path}")
    print(f"aggregate verdict: {aggregate['verdict']}")


if __name__ == "__main__":
    main()
