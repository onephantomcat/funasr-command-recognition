"""P4 Speaker Verification Type Definitions.

Schema version: p4_enrollment.v1
All types are frozen per manual section 5.2.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class EnrollmentOutput:
    """Output of encode_enrollment().

    Schema: p4_enrollment.v1
    """
    sample_id: str
    embedding: Optional[np.ndarray]  # float32[D] or None
    embedding_dim: int
    embedding_sha256: Optional[str]
    embedding_l2_normalized: bool
    quality: Optional[float]  # q_enroll in [0,1], or None before P4-07
    speech_duration: float
    valid: bool
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SVScoreResult:
    """Result of computing speaker verification scores on a TSE target window."""
    valid_tgt_sv: bool
    s_tgt: Optional[float]
    valid_res_sv: bool
    s_res: Optional[float]
    valid_delta: bool
    delta_s: Optional[float]
    window_count: int
    window_consistent: bool
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class AuditResult:
    """Output of audit() identity review.

    Schema: p4_audit.v1
    """
    sample_id: str
    state: str  # "EMPTY" | "GRAY" | "PRESENT" | "ERROR"
    hard_reject: bool
    identity_score: Optional[float]
    identity_log_likelihood_ratio: Optional[float]
    features: dict = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)


@dataclass
class GrayDecision:
    """Output of decide_gray().

    Schema: p4_gray_decision.v1
    """
    sample_id: str
    action: str  # "EMIT" | "EMPTY"
    policy_version: str
    reason_codes: List[str] = field(default_factory=list)


# Error codes per manual section (P4-06)
ERROR_CODES = {
    "SV_EMPTY_INPUT": "Input audio is empty (zero samples)",
    "SV_INVALID_SAMPLE_RATE": "Sample rate not 16000 Hz",
    "SV_NONFINITE_AUDIO": "Audio contains NaN or Inf values",
    "SV_TOO_SHORT": "Audio duration below minimum threshold",
    "SV_TOO_LONG": "Audio duration exceeds maximum threshold",
    "SV_NO_VALID_SPEECH": "No valid speech detected in audio",
    "SV_MODEL_LOAD_FAILED": "CAM++ model failed to load",
    "SV_INFERENCE_FAILED": "Embedding inference failed",
    "SV_MULTICHANNEL_DOWNMIXED": "Multi-channel audio was downmixed",
}
