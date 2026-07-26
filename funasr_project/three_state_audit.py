"""Three-State Audit (PRESENT / EMPTY / GRAY) Framework.

Evaluates evidence from target similarity, residual leakage similarity, and multi-window consistency.
Replaces binary hard thresholds with transparent three-state auditing.
"""
from __future__ import annotations

from typing import Any


class AuditReasonCode:
    TARGET_PRESENT_HIGH_CONF = "TARGET_PRESENT_HIGH_CONF"
    TARGET_ABSENT_LOW_SIM = "TARGET_ABSENT_LOW_SIM"
    TARGET_LEAKED_TO_RESIDUAL = "TARGET_LEAKED_TO_RESIDUAL"
    GRAY_EVIDENCE_CONFLICT = "GRAY_EVIDENCE_CONFLICT"
    GRAY_LOW_SNR_AMBIGUOUS = "GRAY_LOW_SNR_AMBIGUOUS"


class ThreeStateAudit:
    """Three-State Auditor for speaker verification and TSE dual-output evidence."""

    def __init__(
        self,
        present_threshold: float = 0.35,
        empty_threshold: float = 0.18,
        residual_leak_threshold: float = 0.40,
    ):
        self.present_threshold = float(present_threshold)
        self.empty_threshold = float(empty_threshold)
        self.residual_leak_threshold = float(residual_leak_threshold)

    def audit(
        self,
        target_sim: float,
        residual_sim: float | None = None,
        multi_window_sims: list[float] | None = None,
    ) -> dict[str, Any]:
        """Audits evidence and returns (state, reason_code, confidence_score).
        
        States:
        - "PRESENT": Target confirmed present. Pass to ASR.
        - "EMPTY": Target confirmed absent or mis-routed. Reject immediately.
        - "GRAY": Evidence ambiguous or conflicting. Fallback to Gate v2 / phrase check.
        """
        # Rule 1: High confidence PRESENT
        if target_sim >= self.present_threshold:
            if residual_sim is not None and residual_sim >= self.residual_leak_threshold:
                return {
                    "state": "GRAY",
                    "reason_code": AuditReasonCode.GRAY_EVIDENCE_CONFLICT,
                    "confidence": round(target_sim, 4),
                    "emit_allowed": True,
                }
            if multi_window_sims:
                window_std = float(max(multi_window_sims) - min(multi_window_sims))
                if window_std > 0.25:
                    return {
                        "state": "GRAY",
                        "reason_code": AuditReasonCode.GRAY_LOW_SNR_AMBIGUOUS,
                        "confidence": round(target_sim, 4),
                        "emit_allowed": True,
                    }
            return {
                "state": "PRESENT",
                "reason_code": AuditReasonCode.TARGET_PRESENT_HIGH_CONF,
                "confidence": round(target_sim, 4),
                "emit_allowed": True,
            }

        # Rule 2: High confidence EMPTY (target absent)
        if target_sim < self.empty_threshold:
            return {
                "state": "EMPTY",
                "reason_code": AuditReasonCode.TARGET_ABSENT_LOW_SIM,
                "confidence": round(1.0 - target_sim, 4),
                "emit_allowed": False,
            }

        # Rule 3: High confidence EMPTY (target leaked to residual)
        if residual_sim is not None and residual_sim > target_sim + 0.15:
            return {
                "state": "EMPTY",
                "reason_code": AuditReasonCode.TARGET_LEAKED_TO_RESIDUAL,
                "confidence": round(residual_sim, 4),
                "emit_allowed": False,
            }

        # Rule 4: Borderline / GRAY zone
        return {
            "state": "GRAY",
            "reason_code": AuditReasonCode.GRAY_LOW_SNR_AMBIGUOUS,
            "confidence": round(target_sim, 4),
            "emit_allowed": True,  # Fallback to secondary gate/phrase check
        }


def smoke_test():
    auditor = ThreeStateAudit(present_threshold=0.35, empty_threshold=0.18)
    
    # Test 1: Present
    r1 = auditor.audit(target_sim=0.45)
    assert r1["state"] == "PRESENT", f"Expected PRESENT, got {r1}"
    
    # Test 2: Empty (low sim)
    r2 = auditor.audit(target_sim=0.10)
    assert r2["state"] == "EMPTY", f"Expected EMPTY, got {r2}"
    
    # Test 3: Empty (leaked to residual)
    r3 = auditor.audit(target_sim=0.25, residual_sim=0.45)
    assert r3["state"] == "EMPTY", f"Expected EMPTY, got {r3}"
    
    # Test 4: Gray (borderline)
    r4 = auditor.audit(target_sim=0.25)
    assert r4["state"] == "GRAY", f"Expected GRAY, got {r4}"

    print("ThreeStateAudit 4/4 test cases passed successfully!")


if __name__ == "__main__":
    smoke_test()
