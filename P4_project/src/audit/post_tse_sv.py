"""Post-TSE Speaker Verification (P4-10, P4-11).

Computes multi-window SV scores on TSE target and residual outputs.

Key principles:
  - SV only on TSE target output, never on original mixture
  - Each window must pass validity checks (duration, energy, clipping, input quality)
  - Invalid SV → missing (null), never 0
  - delta_s defined only when BOTH target and residual SV valid
  - RobustAgg uses duration-weighted median, NEVER max single window
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from ..sv.campplus_backend import load_model, compute_embedding, cosine_similarity
from ..quality.enroll_quality import slice_windows, WINDOW_LENGTH, WINDOW_SHIFT, MIN_WINDOW_LENGTH

logger = logging.getLogger(__name__)

# Validity thresholds
MIN_SPEECH_DURATION = 0.3   # seconds
MIN_ENERGY = 1e-6
MAX_CLIPPING = 0.5          # max fraction of clipped samples


def _check_window_valid(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> Tuple[bool, str]:
    """Check if a window of audio is valid for SV computation."""
    if len(audio) == 0:
        return False, "EMPTY_WINDOW"

    duration = len(audio) / sample_rate
    if duration < MIN_SPEECH_DURATION:
        return False, "WINDOW_TOO_SHORT"

    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-10)
    if rms < MIN_ENERGY:
        return False, "WINDOW_ENERGY_TOO_LOW"

    if not np.all(np.isfinite(audio)):
        return False, "WINDOW_NONFINITE"

    # Clipping check
    abs_max = np.max(np.abs(audio))
    if abs_max > 1e-6:
        clip_ratio = np.mean(np.abs(audio) >= 0.99 * abs_max)
        if clip_ratio > MAX_CLIPPING:
            return False, "WINDOW_EXCESSIVE_CLIPPING"

    return True, "OK"


def _robust_aggregate(
    cos_scores: List[float],
    durations: List[float],
) -> float:
    """Robust aggregation: duration-weighted median.

    Never max; max single window is forbidden per manual.
    """
    if not cos_scores:
        return 0.0
    if len(cos_scores) == 1:
        return cos_scores[0]

    # Sort by score, accumulate durations to find weighted median
    pairs = sorted(zip(cos_scores, durations))
    total_dur = sum(durations)
    cumulative = 0.0
    for score, dur in pairs:
        cumulative += dur
        if cumulative >= total_dur / 2:
            return score

    return pairs[-1][0]  # fallback: highest (shouldn't reach)


def compute_post_tse_sv(
    audio: np.ndarray,
    enrollment_embedding: np.ndarray,
    sample_rate: int = 16000,
    label: str = "tgt",
) -> dict:
    """Compute multi-window SV scores on post-TSE audio.

    Args:
        audio: Post-TSE target or residual waveform (mono, float32, 16kHz).
        enrollment_embedding: Frozen enrollment embedding.
        sample_rate: Audio sample rate.
        label: "tgt" or "res" for logging/reason codes.

    Returns:
        Dict with:
          - valid: bool
          - score: float | None
          - window_scores: list of per-window scores
          - valid_window_count: int
          - reason_codes: list of str
    """
    if audio is None or len(audio) == 0:
        return {
            "valid": False,
            "score": None,
            "window_scores": [],
            "valid_window_count": 0,
            "window_count": 0,
            "reason_codes": [f"{label.upper()}_SV_EMPTY_INPUT"],
        }

    # Slice into windows
    windows = slice_windows(audio, sample_rate)

    model, frontend = load_model()
    cos_scores = []
    durations = []
    valid_count = 0

    for win_audio, win_info in windows:
        win_valid, win_reason = _check_window_valid(win_audio, sample_rate)
        if not win_valid:
            continue

        # Compute embedding for this window
        try:
            import torch
            waveform = torch.from_numpy(win_audio.astype(np.float32)).unsqueeze(0)
            emb = compute_embedding(waveform, model, frontend)
            cos = cosine_similarity(enrollment_embedding, emb)
            cos_scores.append(cos)
            durations.append(win_info["speech_duration"])
            valid_count += 1
        except Exception as e:
            logger.warning(f"Window SV failed: {e}")
            continue

    window_count = len(windows)

    if valid_count == 0:
        return {
            "valid": False,
            "score": None,
            "window_scores": [],
            "valid_window_count": 0,
            "window_count": window_count,
            "reason_codes": [f"{label.upper()}_SV_NO_VALID_WINDOW"],
        }

    # RobustAgg: duration-weighted median
    score = _robust_aggregate(cos_scores, durations)

    return {
        "valid": True,
        "score": float(score),
        "window_scores": cos_scores,
        "valid_window_count": valid_count,
        "window_count": window_count,
        "reason_codes": [],
    }


def compute_delta_s(
    tgt_result: dict,
    res_result: dict,
) -> dict:
    """Compute delta_s = s_tgt - s_res only when both sides valid.

    Returns dict with valid_delta, delta_s, and reason_codes.
    """
    if not tgt_result.get("valid") or not res_result.get("valid"):
        return {
            "valid_delta": False,
            "delta_s": None,
            "reason_codes": ["DELTA_ONE_SIDE_INVALID"],
        }

    s_tgt = tgt_result["score"]
    s_res = res_result["score"]

    if s_tgt is None or s_res is None:
        return {
            "valid_delta": False,
            "delta_s": None,
            "reason_codes": ["DELTA_SCORE_MISSING"],
        }

    return {
        "valid_delta": True,
        "delta_s": float(s_tgt - s_res),
        "reason_codes": [],
    }


def compute_energy_ratio(
    target_audio: np.ndarray,
    mixture_audio: np.ndarray,
    eps: float = 1e-10,
) -> float:
    """Compute rho_s = ||s_tgt||^2 / (||x||^2 + eps)."""
    tgt_energy = np.sum(target_audio.astype(np.float64) ** 2)
    mix_energy = np.sum(mixture_audio.astype(np.float64) ** 2)
    return float(tgt_energy / (mix_energy + eps))
