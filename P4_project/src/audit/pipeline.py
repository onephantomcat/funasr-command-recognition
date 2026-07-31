"""P4 Audit Pipeline — Top-Level Entry Point.

This is the single entry point that takes the 4 module inputs and returns
the 4 module outputs as specified in 要求.txt:

Inputs:
  1. enrollment embedding + quality
  2. TSE target waveform
  3. TSE residual waveform
  4. activity / energy / quality diagnostics

Outputs:
  1. audit_state: "PRESENT" | "EMPTY" | "GRAY" | "ERROR"
  2. final_action: "EMIT" | "EMPTY"
  3. reason_codes: list[str]
  4. diagnostics: dict (full trace)
"""

import logging
from typing import Optional

import numpy as np

from .three_state import (
    IdentityFeatures,
    AuditConfig,
    audit,
    _safe_logit,
)
from ..decision.gray_strategy import GrayConfig, decide_gray_v1

logger = logging.getLogger(__name__)


def audit_and_decide(
    sample_id: str,
    enrollment_embedding: np.ndarray,
    q_enroll: float,
    target_waveform: Optional[np.ndarray],
    residual_waveform: Optional[np.ndarray],
    activity_mean: float,
    target_activity_duration: float,
    target_energy_ratio: float,
    window_consistent: bool,
    window_count: int,
    valid_window_count: int,
    asr_text: Optional[str] = None,
    audit_config: Optional[AuditConfig] = None,
    gray_config: Optional[GrayConfig] = None,
) -> dict:
    """Run full identity audit + GRAY decision in one call.

    This is the single function P5 calls for each sample.

    Args:
        sample_id: Unique sample identifier.
        enrollment_embedding: [D] float32 enrollment embedding.
        q_enroll: Enrollment quality score in [0, 1].
        target_waveform: Post-TSE target waveform (mono float32 16kHz).
        residual_waveform: Post-TSE residual waveform (mono float32 16kHz).
        activity_mean: Mean target activity probability.
        target_activity_duration: Cumulative target activity duration (seconds).
        target_energy_ratio: rho_s = ||target||^2 / (||mixture||^2 + eps).
        window_consistent: Whether multi-window SV is consistent.
        window_count: Total number of SV windows.
        valid_window_count: Number of valid SV windows.
        asr_text: ASR output text (only for GRAY decision; P5 calls after ASR).
        audit_config: AuditConfig (loaded from YAML if None).
        gray_config: GrayConfig (loaded from YAML if None).

    Returns:
        Dict with audit_state, final_action, reason_codes, diagnostics.
    """
    if audit_config is None:
        audit_config = AuditConfig.from_yaml()
    if gray_config is None:
        gray_config = GrayConfig.from_yaml()

    # --- Pre-audit: compute SV scores on target and residual ---
    from .post_tse_sv import compute_post_tse_sv, compute_delta_s, compute_energy_ratio

    # Target SV
    tgt_sv = {"valid": False, "score": None, "valid_window_count": 0, "window_count": 0, "reason_codes": []}
    if target_waveform is not None and len(target_waveform) > 0:
        tgt_sv = compute_post_tse_sv(target_waveform, enrollment_embedding, 16000, "tgt")

    # Residual SV
    res_sv = {"valid": False, "score": None, "valid_window_count": 0, "window_count": 0, "reason_codes": []}
    if residual_waveform is not None and len(residual_waveform) > 0:
        res_sv = compute_post_tse_sv(residual_waveform, enrollment_embedding, 16000, "res")

    # Delta
    delta_info = compute_delta_s(tgt_sv, res_sv)

    # --- Build identity features ---
    features = IdentityFeatures(
        sample_id=sample_id,
        q_enroll=q_enroll,
        activity_mean=activity_mean,
        target_activity_duration=target_activity_duration,
        s_tgt=tgt_sv["score"],
        target_energy_ratio=target_energy_ratio,
        valid_tgt_sv=tgt_sv["valid"],
        valid_res_sv=res_sv["valid"],
        s_res=res_sv["score"],
        valid_delta=delta_info["valid_delta"],
        delta_s=delta_info["delta_s"],
        window_consistent=window_consistent,
        window_count=max(tgt_sv.get("window_count", 0), res_sv.get("window_count", 0)),
        valid_window_count=tgt_sv.get("valid_window_count", 0),
    )

    # --- Run three-state audit ---
    audit_result = audit(features, audit_config)

    # --- If GRAY and we have ASR, run GRAY decision ---
    if audit_result["audit_state"] == "GRAY" and asr_text is not None:
        gray_result = decide_gray_v1(
            sample_id=sample_id,
            audit_state="GRAY",
            hard_reject=audit_result["hard_reject"],
            identity_score=audit_result["identity_score"],
            identity_log_lr=audit_result["identity_log_likelihood_ratio"],
            asr_text=asr_text,
            asr_features=None,
            config=gray_config,
        )
        audit_result["final_action"] = gray_result["final_action"]
        audit_result["reason_codes"].extend(gray_result.get("reason_codes", []))
        audit_result["diagnostics"]["gray_decision"] = gray_result.get("diagnostics", {})

    # For PRESENT, final_action is EMIT (P5 runs ASR and emits if non-empty)
    # For EMPTY, final_action is EMPTY (P5 outputs empty, no ASR)
    # For ERROR, final_action is EMPTY (fail-closed)

    return audit_result
