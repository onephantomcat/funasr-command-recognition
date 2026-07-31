"""Activity post-processing unit tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.activity.postprocess import (
    ActivityConfig, hysteresis_detect, merge_segments, add_guard,
    frames_to_samples, process_activity, compute_adaptive_duration,
)

CONFIG = ActivityConfig.from_yaml()


def test_hysteresis_basic():
    probs = np.array([0.1, 0.3, 0.8, 0.75, 0.6, 0.2, 0.1, 0.9, 0.85, 0.2])
    segs = hysteresis_detect(probs, CONFIG)
    assert len(segs) == 2, f"Expected 2 segments, got {len(segs)}"
    # Segment 1 starts at index 2 (0.8 >= 0.7)
    assert segs[0] == (2, 4), f"Seg1: {segs[0]}"
    # Segment 2 starts at index 7 (0.9 >= 0.7)
    assert segs[1] == (7, 8), f"Seg2: {segs[1]}"
    print("  PASS: hysteresis basic")


def test_hysteresis_stays_on():
    """Hysteresis: once ON, stays ON until below tau_off."""
    probs = np.array([0.8, 0.5, 0.35, 0.4, 0.25])
    segs = hysteresis_detect(probs, CONFIG)
    # 0.8 -> ON, 0.5 still ON (>= tau_off=0.3), 0.35 still ON, 0.4 ON, 0.25 -> OFF
    assert segs == [(0, 3)], f"Got: {segs}"
    print("  PASS: hysteresis stays on above tau_off")


def test_hysteresis_tau_on_gt_tau_off():
    """tau_on <= tau_off should raise error."""
    try:
        bad = ActivityConfig(tau_on=0.3, tau_off=0.5, min_gap_merge=0.15,
                             min_core_duration=0.2, guard_before=0.1,
                             guard_after=0.1, frame_rate=100.0)
        assert False, "Should have raised"
    except (ValueError, AssertionError):
        pass
    print("  PASS: tau_on > tau_off enforced")


def test_merge_gaps():
    probs = np.zeros(200)
    probs[10:30] = 0.9    # seg1: 10-29
    probs[40:60] = 0.85   # seg2: 40-59, gap=10 frames (0.1s < 0.15s merge)
    probs[90:110] = 0.9   # seg3: 90-109, gap=30 frames (0.3s > 0.15s no merge)
    segs = hysteresis_detect(probs, CONFIG)
    merged = merge_segments(segs, CONFIG)
    assert len(merged) == 2, f"Expected 2 segments after merge, got {len(merged)}"
    print("  PASS: merge gaps")


def test_core_duration_filter():
    """Core segments shorter than min_core_duration are filtered."""
    probs = np.zeros(200)
    probs[10:15] = 0.8    # 5 frames = 0.05s < 0.2s -> filtered
    probs[50:100] = 0.9   # 50 frames = 0.5s > 0.2s -> kept
    segs = hysteresis_detect(probs, CONFIG)
    merged = merge_segments(segs, CONFIG)
    min_frames = int(CONFIG.min_core_duration * CONFIG.frame_rate)
    core = [(s, e) for s, e in merged if (e - s + 1) >= min_frames]
    assert len(core) == 1
    assert core[0] == (50, 99)
    print("  PASS: core duration filter")


def test_guard():
    segs = [(50, 99)]
    guarded = add_guard(segs, CONFIG, total_frames=200)
    guard_frames = int(CONFIG.guard_before * CONFIG.frame_rate)
    assert guarded[0][0] == 50 - guard_frames
    assert guarded[0][1] == 99 + guard_frames
    print("  PASS: guard added")


def test_guard_boundary():
    """Guard should not go below 0 or above total_frames."""
    segs = [(2, 5)]
    guarded = add_guard(segs, CONFIG, total_frames=20)
    assert guarded[0][0] == 0  # clamped at 0
    print("  PASS: guard boundary")


def test_empty_activity():
    result = process_activity(np.array([]), CONFIG)
    assert not result.has_valid_activity
    print("  PASS: empty activity")


def test_all_zeros():
    result = process_activity(np.zeros(100), CONFIG)
    assert not result.has_valid_activity
    print("  PASS: all zeros")


def test_all_ones():
    result = process_activity(np.ones(200), CONFIG)
    assert result.has_valid_activity
    assert result.target_activity_duration > 0
    print("  PASS: all ones")


def test_adaptive_duration():
    d = compute_adaptive_duration(3.0, rho_min=0.1, d_floor=0.3, d_ceil=3.0)
    assert np.isclose(d, 0.3)  # 0.1*3.0=0.3, clipped at floor
    d2 = compute_adaptive_duration(40.0, rho_min=0.1, d_floor=0.3, d_ceil=3.0)
    assert np.isclose(d2, 3.0)  # clipped at ceil
    print("  PASS: adaptive duration")


def test_from_yaml():
    cfg = ActivityConfig.from_yaml()
    assert cfg.tau_on == 0.7
    assert cfg.tau_off == 0.3
    assert cfg.tau_on > cfg.tau_off
    print("  PASS: from_yaml")


if __name__ == "__main__":
    print("=== Activity Postprocess Tests ===")
    test_hysteresis_basic()
    test_hysteresis_stays_on()
    test_hysteresis_tau_on_gt_tau_off()
    test_merge_gaps()
    test_core_duration_filter()
    test_guard()
    test_guard_boundary()
    test_empty_activity()
    test_all_zeros()
    test_all_ones()
    test_adaptive_duration()
    test_from_yaml()
    print("=== 12/12 PASSED ===")
