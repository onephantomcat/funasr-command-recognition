"""Target Activity Post-Processing (P4-09).

Correct processing order (frozen):
  1. Hysteresis detection
  2. Merge short gaps
  3. Check merged core segment length
  4. Add guard on both sides
  5. Convert to waveform sample indices

All thresholds from versioned YAML config, NEVER hardcoded.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_ACTIVITY_CONFIG_PATH = "configs/activity_postprocess_v1.yaml"


def _find_project_root() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(current, "configs")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return os.path.dirname(os.path.dirname(current))


@dataclass
class ActivityConfig:
    """Frozen activity post-processing config loaded from YAML.

    ALL thresholds come from config file. No hardcoded defaults.
    """
    tau_on: float
    tau_off: float
    min_gap_merge: float
    min_core_duration: float
    guard_before: float
    guard_after: float
    frame_rate: float
    version: str = "activity_postprocess_v1"

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "ActivityConfig":
        if config_path is None:
            project_root = _find_project_root()
            config_path = os.path.join(project_root, _ACTIVITY_CONFIG_PATH)

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        tau_on = float(raw["hysteresis"]["tau_on"])
        tau_off = float(raw["hysteresis"]["tau_off"])
        if tau_on <= tau_off:
            raise ValueError(f"tau_on ({tau_on}) must be > tau_off ({tau_off})")

        return cls(
            tau_on=tau_on,
            tau_off=tau_off,
            min_gap_merge=float(raw["merge"]["min_gap_merge"]),
            min_core_duration=float(raw["core"]["min_core_duration"]),
            guard_before=float(raw["guard"]["guard_before"]),
            guard_after=float(raw["guard"]["guard_after"]),
            frame_rate=float(raw["frame_rate"]),
        )


@dataclass
class ActivitySegment:
    """A detected target activity segment."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    mean_activity: float
    is_core: bool


@dataclass
class ActivityResult:
    """Result of activity post-processing."""
    segments: List[ActivitySegment] = field(default_factory=list)
    core_segments: List[ActivitySegment] = field(default_factory=list)
    target_activity_duration: float = 0.0
    longest_segment_duration: float = 0.0
    mean_activity_prob: float = 0.0
    activity_entropy: float = 0.0
    has_valid_activity: bool = False
    diagnostics: dict = field(default_factory=dict)


def hysteresis_detect(
    probs: np.ndarray,
    config: ActivityConfig,
) -> List[Tuple[int, int]]:
    """Apply hysteresis thresholding to activity probabilities.

    p[t] >= tau_on  → ON
    p[t] <  tau_off → OFF
    tau_on > tau_off required.
    """
    if config.tau_on <= config.tau_off:
        raise ValueError(
            f"tau_on ({config.tau_on}) must be > tau_off ({config.tau_off})"
        )

    n_frames = len(probs)
    is_on = np.zeros(n_frames, dtype=bool)
    state = False  # OFF

    for i in range(n_frames):
        p = float(probs[i])
        if state:
            # Currently ON: stay ON unless below tau_off
            if p < config.tau_off:
                state = False
        else:
            # Currently OFF: turn ON if above tau_on
            if p >= config.tau_on:
                state = True
        is_on[i] = state

    # Extract ON segments
    segments = []
    i = 0
    while i < n_frames:
        if is_on[i]:
            start = i
            while i < n_frames and is_on[i]:
                i += 1
            segments.append((start, i - 1))
        else:
            i += 1

    return segments


def merge_segments(
    raw_segments: List[Tuple[int, int]],
    config: ActivityConfig,
) -> List[Tuple[int, int]]:
    """Merge segments separated by gaps shorter than min_gap_merge."""
    if not raw_segments:
        return []

    gap_frames = int(config.min_gap_merge * config.frame_rate)
    merged = [list(raw_segments[0])]

    for start, end in raw_segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_frames:
            merged[-1][1] = end  # extend
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def add_guard(
    segments: List[Tuple[int, int]],
    config: ActivityConfig,
    total_frames: int,
) -> List[Tuple[int, int]]:
    """Add guard frames before and after each segment."""
    guard_before = int(config.guard_before * config.frame_rate)
    guard_after = int(config.guard_after * config.frame_rate)

    guarded = []
    for start, end in segments:
        g_start = max(0, start - guard_before)
        g_end = min(total_frames - 1, end + guard_after)
        guarded.append((g_start, g_end))

    return guarded


def frames_to_samples(
    segments: List[Tuple[int, int]],
    config: ActivityConfig,
    sample_rate: int = 16000,
) -> List[Tuple[int, int]]:
    """Convert frame indices to waveform sample indices."""
    samples_per_frame = sample_rate / config.frame_rate
    sample_segments = []
    for start_frame, end_frame in segments:
        start_sample = int(start_frame * samples_per_frame)
        end_sample = int((end_frame + 1) * samples_per_frame)  # inclusive → exclusive
        sample_segments.append((start_sample, end_sample))
    return sample_segments


def compute_activity_features(
    probs: np.ndarray,
    config: ActivityConfig,
) -> dict:
    """Compute activity-related features from probability array."""
    n_frames = len(probs)

    # Basic stats
    mean_prob = float(np.mean(probs)) if n_frames > 0 else 0.0
    activity_entropy = 0.0
    if n_frames > 0:
        p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
        activity_entropy = float(-np.mean(
            p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip)
        ))

    # Quantiles
    q25, q50, q75 = np.percentile(probs, [25, 50, 75]) if n_frames > 0 else (0, 0, 0)

    return {
        "mean_activity_prob": mean_prob,
        "activity_entropy": activity_entropy,
        "activity_q25": float(q25),
        "activity_q50": float(q50),
        "activity_q75": float(q75),
        "n_frames": n_frames,
    }


def process_activity(
    probs: np.ndarray,
    config: Optional[ActivityConfig] = None,
    sample_rate: int = 16000,
    total_samples: Optional[int] = None,
) -> ActivityResult:
    """Full activity post-processing pipeline.

    Args:
        probs: [T] numpy array of activity probabilities in [0, 1].
        config: Activity configuration (uses default if None).
        sample_rate: Audio sample rate.
        total_samples: Total waveform length in samples (for boundary checks).

    Returns:
        ActivityResult with segments, features, and validity.
    """
    if config is None:
        config = ActivityConfig.from_yaml()

    n_frames = len(probs)

    # Validate input
    if n_frames == 0:
        return ActivityResult(diagnostics={"error": "Empty activity array"})

    if np.any(np.isnan(probs)):
        return ActivityResult(diagnostics={"error": "NaN in activity array"})

    if np.any(probs < 0) or np.any(probs > 1):
        logger.warning("Activity probabilities outside [0, 1]")

    # Step 1: Hysteresis
    raw_segments = hysteresis_detect(probs, config)

    # Step 2: Merge short gaps
    merged = merge_segments(raw_segments, config)

    # Step 3: Check core duration (after merge, before guard)
    min_core_frames = int(config.min_core_duration * config.frame_rate)
    core_before_guard = [
        (s, e) for s, e in merged
        if (e - s + 1) >= min_core_frames
    ]

    # Step 4: Add guard
    guarded = add_guard(core_before_guard, config, n_frames)

    # Step 5: Convert to samples
    sample_segments = frames_to_samples(guarded, config, sample_rate)
    if total_samples is not None:
        sample_segments = [
            (max(0, s), min(total_samples - 1, e))
            for s, e in sample_segments
        ]

    # Build result
    result_segments = []
    core_segments = []
    total_duration = 0.0
    max_duration = 0.0

    seg_idx = 0
    for (gs, ge), (cs, ce) in zip(guarded, core_before_guard):
        start_time = gs / config.frame_rate
        end_time = (ge + 1) / config.frame_rate
        duration = end_time - start_time

        # Mean activity in core region
        core_probs = probs[cs:ce + 1] if ce >= cs else np.array([0])
        mean_act = float(np.mean(core_probs)) if len(core_probs) > 0 else 0.0

        seg = ActivitySegment(
            start_frame=gs,
            end_frame=ge,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            mean_activity=mean_act,
            is_core=True,
        )
        result_segments.append(seg)
        core_segments.append(seg)

        total_duration += (ce - cs + 1) / config.frame_rate
        core_dur = (ce - cs + 1) / config.frame_rate
        max_duration = max(max_duration, core_dur)
        seg_idx += 1

    # Activity features
    features = compute_activity_features(probs, config)

    has_valid = len(core_segments) > 0

    return ActivityResult(
        segments=result_segments,
        core_segments=core_segments,
        target_activity_duration=total_duration,
        longest_segment_duration=max_duration,
        mean_activity_prob=features["mean_activity_prob"],
        activity_entropy=features["activity_entropy"],
        has_valid_activity=has_valid,
        diagnostics=features,
    )


def compute_adaptive_duration(
    speech_duration: float,
    rho_min: float = 0.1,
    d_floor: float = 0.3,
    d_ceil: float = 3.0,
) -> float:
    """Adaptive required duration for activity validity.

    d_req = clip(rho_min * T_speech, d_floor, d_ceil)
    """
    return float(np.clip(rho_min * speech_duration, d_floor, d_ceil))
