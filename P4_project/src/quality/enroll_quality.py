"""Enrollment Quality Assessment (P4-07).

Estimates q_enroll ∈ [0,1] representing the reliability of enrollment evidence.

Key principle:
  - High q_enroll: enrollment evidence is stable
  - Low q_enroll: enrollment evidence is unreliable
  - Low q_enroll can push toward GRAY, but NOT auto-EMPTY
  - High q_enroll does NOT auto-PRESENT

First version uses low-capacity rules on:
  - Effective speech duration
  - Speech ratio
  - Clipping ratio
  - Waveform RMS / loudness
  - Multi-window embedding cosine median and IQR
  - Number of valid windows
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Frozen window parameters
WINDOW_LENGTH = 1.6   # seconds
WINDOW_SHIFT = 0.8    # seconds
MIN_WINDOW_LENGTH = 0.3  # minimum valid tail window
MIN_ENERGY = 1e-6
CLIP_FRAC = 0.95  # fraction of max amplitude


def _compute_rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-10))


def _estimate_snr(audio: np.ndarray) -> float:
    """Naive SNR estimate using silence frames as noise reference."""
    rms = _compute_rms(audio)
    if rms < 1e-6:
        return 0.0
    # Use lowest-energy frames as noise estimate
    frame_len = 400  # 25ms at 16kHz
    n_frames = len(audio) // frame_len
    if n_frames < 4:
        return 10.0
    frame_rms = np.array([
        _compute_rms(audio[i * frame_len:(i + 1) * frame_len])
        for i in range(n_frames)
    ])
    # Bottom 10% as noise
    noise_rms = np.percentile(frame_rms[frame_rms > 1e-8], 10) if np.any(frame_rms > 1e-8) else 1e-6
    if noise_rms < 1e-6:
        return 20.0
    return float(20 * np.log10(rms / noise_rms))


def _clipping_ratio(audio: np.ndarray) -> float:
    """Fraction of samples at or near the maximum amplitude."""
    if audio.size == 0:
        return 0.0
    abs_audio = np.abs(audio)
    max_val = np.max(abs_audio)
    if max_val < 1e-6:
        return 0.0
    return float(np.mean(abs_audio >= CLIP_FRAC * max_val))


def _speech_ratio(audio: np.ndarray, sample_rate: int = 16000) -> float:
    """Fraction of audio above a simple energy threshold."""
    rms = _compute_rms(audio)
    if rms < 1e-6:
        return 0.0
    frame_len = int(0.025 * sample_rate)  # 25ms
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return 0.0
    frame_rms = np.array([
        _compute_rms(audio[i * frame_len:(i + 1) * frame_len])
        for i in range(n_frames)
    ])
    threshold = 0.1 * rms
    return float(np.mean(frame_rms > threshold))


def compute_enroll_quality(
    audio: np.ndarray,
    sample_rate: int = 16000,
    multi_window_embeddings: Optional[List[np.ndarray]] = None,
    multi_window_info: Optional[List[dict]] = None,
) -> Tuple[float, dict]:
    """Compute enrollment quality q_enroll ∈ [0,1].

    Args:
        audio: Mono float32 audio at sample_rate.
        sample_rate: Audio sample rate.
        multi_window_embeddings: Optional list of per-window embeddings.
        multi_window_info: Optional list of per-window metadata dicts.

    Returns:
        (quality_score, diagnostics_dict)
    """
    duration = len(audio) / sample_rate if sample_rate > 0 else 0

    # Basic features
    rms = _compute_rms(audio)
    snr_est = _estimate_snr(audio)
    clip_ratio = _clipping_ratio(audio)
    speech_ratio = _speech_ratio(audio, sample_rate)

    diagnostics = {
        "duration": duration,
        "rms": rms,
        "snr_estimate": snr_est,
        "clipping_ratio": clip_ratio,
        "speech_ratio": speech_ratio,
    }

    # Multi-window consistency
    if multi_window_embeddings and len(multi_window_embeddings) > 1:
        n_windows = len(multi_window_embeddings)
        valid_windows = sum(
            1 for info in (multi_window_info or [])
            if info.get("embedding_valid", True)
        )

        if valid_windows >= 2:
            cos_matrix = np.zeros((n_windows, n_windows))
            for i in range(n_windows):
                for j in range(n_windows):
                    a, b = multi_window_embeddings[i], multi_window_embeddings[j]
                    cos_matrix[i][j] = np.dot(a, b) / (
                        np.linalg.norm(a) * np.linalg.norm(b) + 1e-10
                    )
            pairwise_cos = cos_matrix[np.triu_indices(n_windows, k=1)]
            median_cos = float(np.median(pairwise_cos)) if len(pairwise_cos) > 0 else 0.0
            iqr_cos = float(np.subtract(*np.percentile(pairwise_cos, [75, 25]))) if len(pairwise_cos) > 1 else 0.0

            diagnostics.update({
                "num_windows": n_windows,
                "num_valid_windows": valid_windows,
                "window_cosine_median": median_cos,
                "window_cosine_iqr": iqr_cos,
            })
        else:
            diagnostics.update({
                "num_windows": n_windows,
                "num_valid_windows": valid_windows,
            })
    else:
        diagnostics.update({
            "num_windows": 1 if multi_window_embeddings else 0,
            "num_valid_windows": 1 if multi_window_embeddings else 0,
        })

    # Quality heuristic (low-capacity, pre-registered)
    # Each feature contributes 0-1, combined via weighted geometric mean
    dur_score = min(duration / 3.0, 1.0)  # 3s = full score
    speech_score = speech_ratio
    clip_penalty = max(0.0, 1.0 - clip_ratio * 10)  # 10% clipping → 0
    snr_score = min(max(snr_est / 20.0, 0.0), 1.0)  # 20dB → 1.0

    # Multi-window consistency
    if diagnostics.get("window_cosine_median") is not None:
        median_cos = diagnostics["window_cosine_median"]
        iqr_cos = diagnostics.get("window_cosine_iqr", 0.5)
        window_consistency = median_cos * max(0.0, 1.0 - iqr_cos)
    else:
        window_consistency = 0.8  # default: unknown

    # Combined score (low-capacity geometric mean)
    scores = [dur_score, speech_score, clip_penalty, snr_score, window_consistency]
    scores = [max(s, 0.01) for s in scores]  # avoid zero
    quality = float(np.exp(np.mean(np.log(scores))))

    # Clamp to [0, 1]
    quality = max(0.0, min(1.0, quality))

    diagnostics["quality"] = quality
    diagnostics["quality_components"] = {
        "duration_score": dur_score,
        "speech_score": speech_score,
        "clip_penalty": clip_penalty,
        "snr_score": snr_score,
        "window_consistency": window_consistency,
    }

    return quality, diagnostics


def slice_windows(
    audio: np.ndarray,
    sample_rate: int = 16000,
) -> List[Tuple[np.ndarray, dict]]:
    """Slice audio into fixed-length overlapping windows for multi-window SV.

    Returns list of (window_audio, window_info) tuples.
    """
    window_samples = int(WINDOW_LENGTH * sample_rate)
    shift_samples = int(WINDOW_SHIFT * sample_rate)

    if len(audio) < window_samples:
        # Single short window
        info = {
            "window_start": 0.0,
            "window_end": len(audio) / sample_rate,
            "speech_duration": len(audio) / sample_rate,
            "energy": _compute_rms(audio) ** 2 * len(audio),
            "clipping": _clipping_ratio(audio),
            "embedding_valid": len(audio) / sample_rate >= MIN_WINDOW_LENGTH,
        }
        return [(audio, info)]

    windows = []
    for start in range(0, len(audio) - window_samples + 1, shift_samples):
        end = start + window_samples
        window_audio = audio[start:end]
        duration = len(window_audio) / sample_rate
        info = {
            "window_start": start / sample_rate,
            "window_end": end / sample_rate,
            "speech_duration": duration,
            "energy": _compute_rms(window_audio) ** 2 * duration,
            "clipping": _clipping_ratio(window_audio),
            "embedding_valid": (
                duration >= MIN_WINDOW_LENGTH
                and _compute_rms(window_audio) >= MIN_ENERGY
            ),
        }
        windows.append((window_audio, info))

    # Handle tail
    tail_start = start + shift_samples
    if tail_start < len(audio):
        tail_audio = audio[tail_start:]
        tail_duration = len(tail_audio) / sample_rate
        info = {
            "window_start": tail_start / sample_rate,
            "window_end": len(audio) / sample_rate,
            "speech_duration": tail_duration,
            "energy": _compute_rms(tail_audio) ** 2 * tail_duration,
            "clipping": _clipping_ratio(tail_audio),
            "embedding_valid": (
                tail_duration >= MIN_WINDOW_LENGTH
                and _compute_rms(tail_audio) >= MIN_ENERGY
            ),
        }
        windows.append((tail_audio, info))

    return windows
