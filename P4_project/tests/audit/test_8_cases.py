"""P4 三态决策器 — 8 项要求测试 (要求.txt).

用构造的分数和假数据验证所有规则。
不依赖真实 TSE 输出、P1/P2/P3 交付。
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
from src.audit.three_state import IdentityFeatures, AuditConfig, audit
from src.decision.gray_strategy import GrayConfig, decide_gray_v1


def test_1_clear_present():
    """Test 1: 明确目标存在 → PRESENT.

    Strong activity, energy, quality, SV, multi-window all aligned.
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_01",
        q_enroll=0.85,
        activity_mean=0.9,             # A+
        target_activity_duration=2.0,
        s_tgt=0.78,                    # >= tau_tgt_hi (0.5)
        target_energy_ratio=0.35,       # G+ (>= 0.05)
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.12,                    # <= tau_res_lo (0.2) → clean
        valid_delta=True,
        delta_s=0.66,                  # >= tau_delta_hi (0.3)
        window_consistent=True,
        window_count=3,
        valid_window_count=3,
    )
    r = audit(f, config)
    assert r["audit_state"] == "PRESENT", f"Expected PRESENT, got {r['audit_state']}"
    assert r["hard_reject"] is False
    assert r["final_action"] == "EMIT"
    print("PASS: Test 1 — 明确目标存在 → PRESENT")


def test_2_clear_empty():
    """Test 2: 明确没有目标 → EMPTY.

    Activity and energy both deeply absent (A- ∧ G-).
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_02",
        q_enroll=0.8,
        activity_mean=0.02,            # A- (< 0.3)
        target_activity_duration=0.05,
        s_tgt=None,
        target_energy_ratio=0.001,     # G- (< 0.01)
        valid_tgt_sv=False,
        valid_res_sv=True,
        s_res=0.05,
        valid_delta=False,
        delta_s=None,
        window_consistent=False,
        window_count=0,
        valid_window_count=0,
    )
    r = audit(f, config)
    assert r["audit_state"] == "EMPTY", f"Expected EMPTY, got {r['audit_state']}"
    assert r["hard_reject"] is True
    assert r["final_action"] == "EMPTY"
    assert "TARGET_ABSENT_ACTIVITY_ENERGY" in r["reason_codes"]
    print("PASS: Test 2 — 明确没有目标 → EMPTY")


def test_3_target_in_residual():
    """Test 3: 目标进入残余 → EMPTY.

    Activity/energy strong but SV evidence points to residual side.
    Target SV low, residual SV high, delta_s deeply negative.
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_03",
        q_enroll=0.8,
        activity_mean=0.85,            # A+
        target_activity_duration=1.8,
        s_tgt=0.12,                    # <= tau_tgt_lo (0.2)
        target_energy_ratio=0.25,      # G+
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.65,                    # >= tau_res_hi (0.4)
        valid_delta=True,
        delta_s=-0.53,                 # <= tau_delta_lo (-0.1)
        window_consistent=True,
        window_count=3,
        valid_window_count=3,
    )
    r = audit(f, config)
    assert r["audit_state"] == "EMPTY", f"Expected EMPTY, got {r['audit_state']}"
    assert r["hard_reject"] is True
    assert "TARGET_MISROUTED_TO_RESIDUAL" in r["reason_codes"]
    print("PASS: Test 3 — 目标进入残余 → EMPTY")


def test_4_low_energy_but_activity_positive():
    """Test 4: 低能量但活动证据为正 → GRAY.

    rho_s = 0.005 (very low), but activity_mean = 0.8 (high).
    This is an evidence conflict → GRAY.
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_04",
        q_enroll=0.8,
        activity_mean=0.8,             # A+
        target_activity_duration=1.5,
        s_tgt=0.55,
        target_energy_ratio=0.005,     # G- (< 0.01)
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.15,
        valid_delta=True,
        delta_s=0.40,
        window_consistent=True,
        window_count=3,
        valid_window_count=3,
    )
    r = audit(f, config)
    assert r["audit_state"] == "GRAY", f"Expected GRAY, got {r['audit_state']}"
    assert "EVIDENCE_CONFLICT" in r["reason_codes"]
    print("PASS: Test 4 — 低能量但活动证据为正 → GRAY")


def test_5_tgt_res_both_high():
    """Test 5: 目标与残余同时高分 → GRAY.

    Both s_tgt and s_res are above their respective high thresholds.
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_05",
        q_enroll=0.85,
        activity_mean=0.85,
        target_activity_duration=1.5,
        s_tgt=0.72,                    # >= tau_tgt_hi (0.5)
        target_energy_ratio=0.30,
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.55,                    # >= tau_res_hi (0.4)
        valid_delta=True,
        delta_s=0.17,
        window_consistent=True,
        window_count=3,
        valid_window_count=3,
    )
    r = audit(f, config)
    assert r["audit_state"] == "GRAY", f"Expected GRAY, got {r['audit_state']}"
    assert "TGT_RES_BOTH_HIGH" in r["reason_codes"]
    print("PASS: Test 5 — 目标与残余同时高分 → GRAY")


def test_6_multiwindow_inconsistent():
    """Test 6: 多窗声纹不一致 → GRAY.

    SV evidence otherwise looks good but windows disagree.
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_06",
        q_enroll=0.8,
        activity_mean=0.9,
        target_activity_duration=2.0,
        s_tgt=0.70,
        target_energy_ratio=0.40,
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.10,
        valid_delta=True,
        delta_s=0.60,
        window_consistent=False,       # <-- inconsistent!
        window_count=4,
        valid_window_count=3,
    )
    r = audit(f, config)
    assert r["audit_state"] == "GRAY", f"Expected GRAY, got {r['audit_state']}"
    assert "MULTIWINDOW_INCONSISTENT" in r["reason_codes"]
    print("PASS: Test 6 — 多窗声纹不一致 → GRAY")


def test_7_low_enrollment_quality():
    """Test 7: 注册音频质量不足 → GRAY.

    q_enroll below threshold pushes to GRAY.
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_07",
        q_enroll=0.12,                 # < tau_q_enroll (0.3)
        activity_mean=0.85,
        target_activity_duration=1.5,
        s_tgt=0.78,
        target_energy_ratio=0.35,
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.10,
        valid_delta=True,
        delta_s=0.68,
        window_consistent=True,
        window_count=3,
        valid_window_count=3,
    )
    r = audit(f, config)
    assert r["audit_state"] == "GRAY", f"Expected GRAY, got {r['audit_state']}"
    assert "ENROLLMENT_LOW_QUALITY" in r["reason_codes"]
    print("PASS: Test 7 — 注册音频质量不足 → GRAY")


def test_8_activity_energy_conflict():
    """Test 8: 活动与能量证据冲突 → GRAY.

    Activity says absent (A-) but energy ratio is high (G+).
    """
    config = AuditConfig.from_yaml()
    f = IdentityFeatures(
        sample_id="test_08",
        q_enroll=0.8,
        activity_mean=0.05,            # A- (< 0.3)
        target_activity_duration=0.1,
        s_tgt=0.45,
        target_energy_ratio=0.40,      # G+ (>= 0.05)
        valid_tgt_sv=True,
        valid_res_sv=True,
        s_res=0.15,
        valid_delta=True,
        delta_s=0.30,
        window_consistent=True,
        window_count=2,
        valid_window_count=2,
    )
    r = audit(f, config)
    assert r["audit_state"] == "GRAY", f"Expected GRAY, got {r['audit_state']}"
    assert "EVIDENCE_CONFLICT" in r["reason_codes"]
    print("PASS: Test 8 — 活动与能量证据冲突 → GRAY")


if __name__ == "__main__":
    print("=" * 60)
    print("P4 三态决策器 — 8 项要求测试 (要求.txt)")
    print("=" * 60)
    test_1_clear_present()
    test_2_clear_empty()
    test_3_target_in_residual()
    test_4_low_energy_but_activity_positive()
    test_5_tgt_res_both_high()
    test_6_multiwindow_inconsistent()
    test_7_low_enrollment_quality()
    test_8_activity_energy_conflict()
    print("=" * 60)
    print("ALL 8 TESTS PASSED")
