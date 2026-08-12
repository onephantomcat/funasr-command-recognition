"""Strict P3 evaluation contracts shared by DatasetA and paired CER runners."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real


class RecognitionContractError(TypeError):
    """Raised when an ASR caller violates the frozen return-value contract."""


@dataclass(frozen=True)
class RecognitionOutcome:
    text: str
    elapsed_sec: float
    status: str
    error: str | None = None
    raw_text: str | None = None
    normalized_text: str | None = None
    final_text: str | None = None


def unpack_recognition(result):
    """Validate and unpack the legacy ``recognize() -> (text, elapsed)`` API."""
    if not isinstance(result, tuple) or len(result) != 2:
        raise RecognitionContractError(
            "recognize() must return exactly (text: str, elapsed_sec: float)"
        )
    text, elapsed = result
    if not isinstance(text, str):
        raise RecognitionContractError(
            f"recognize() text must be str, got {type(text).__name__}"
        )
    if isinstance(elapsed, bool) or not isinstance(elapsed, Real):
        raise RecognitionContractError(
            f"recognize() elapsed_sec must be a real number, got {type(elapsed).__name__}"
        )
    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RecognitionContractError(
            f"recognize() elapsed_sec must be finite and non-negative, got {elapsed!r}"
        )
    return text, elapsed


def recognize_safely(recognizer, model, wav_path):
    """Run ASR while preserving runtime errors and failing on contract errors.

    A model/runtime exception is returned as ``status=ERROR`` so it cannot be
    counted as a correct negative rejection. A malformed return value raises
    immediately because all downstream CER/RR values would otherwise be invalid.
    """
    try:
        raw_result = recognizer(model, wav_path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return RecognitionOutcome(
            text="",
            elapsed_sec=0.0,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
            raw_text="",
            normalized_text="",
            final_text="",
        )

    text, elapsed = unpack_recognition(raw_result)
    return RecognitionOutcome(
        text=text,
        elapsed_sec=elapsed,
        status="OK",
        raw_text=text,
        normalized_text=text,
        final_text=text,
    )


def recognize_result_safely(recognizer, model, wav_path):
    """Run the structured ASR API and validate all text stages."""
    try:
        result = recognizer(model, wav_path)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        return RecognitionOutcome(
            text="",
            elapsed_sec=0.0,
            status="ERROR",
            error=f"{type(exc).__name__}: {exc}",
            raw_text="",
            normalized_text="",
            final_text="",
        )

    required = ("raw_text", "normalized_text", "final_text", "elapsed_sec")
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise RecognitionContractError(
            f"structured ASR result missing fields: {', '.join(missing)}"
        )
    for name in ("raw_text", "normalized_text", "final_text"):
        value = getattr(result, name)
        if not isinstance(value, str):
            raise RecognitionContractError(
                f"structured ASR {name} must be str, got {type(value).__name__}"
            )
    _, elapsed = unpack_recognition((result.final_text, result.elapsed_sec))
    status = getattr(result, "status", "OK")
    if status != "OK":
        raise RecognitionContractError(f"unexpected structured ASR status: {status!r}")
    return RecognitionOutcome(
        text=result.final_text,
        elapsed_sec=elapsed,
        status="OK",
        raw_text=result.raw_text,
        normalized_text=result.normalized_text,
        final_text=result.final_text,
    )


def negative_is_rejected(*, emit_allowed, outcome):
    """Return RR decision without treating an ASR failure as a rejection."""
    if not emit_allowed:
        return True
    if outcome.status == "ERROR":
        return False
    if outcome.status != "OK":
        raise ValueError(f"unexpected ASR status for emitted sample: {outcome.status!r}")
    return not outcome.text.strip()


def require_unique_sample_ids(rows, key="sample_id"):
    """Validate the one-to-one sample identity required by paired CER."""
    seen = set()
    for index, row in enumerate(rows):
        sample_id = row.get(key)
        if sample_id is None or sample_id == "":
            raise ValueError(f"row {index} is missing non-empty {key!r}")
        if sample_id in seen:
            raise ValueError(f"duplicate {key}={sample_id!r}")
        seen.add(sample_id)
    return seen
