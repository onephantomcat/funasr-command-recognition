r"""Three-State Identity Audit (P4-13).

Implements EMPTY / GRAY / PRESENT tri-state review with explainable rules.

Core formula (manual equation):
  Audit = EMPTY   if R_abs ∨ R_route
          PRESENT if H_target-present
          GRAY    otherwise

where:
  H_target-present ⟺ A⁺ ∧ G⁺ ∧ Q⁺ ∧ I⁺

Hard EMPTY only allowed for:
  1. Activity AND energy jointly confirm target absence (A⁻ ∧ G⁻)
  2. Valid bilateral evidence confirms target misrouted to residual (I⁻)

All thresholds loaded from config file. NO hardcoded values.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Path to threshold config file relative to project root
_AUDIT_CONFIG_PATH = "configs/audit_thresholds_v1.yaml"


def _find_project_root() -> str:
    """Locate project root by searching for configs/ directory."""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(current, "configs")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    # Fallback: assume we're in src/audit/ under project root
    return os.path.dirname(os.path.dirname(current))


@dataclass
class AuditConfig:
    """Audit configuration loaded from YAML file.

    ALL thresholds come from the config file. No hardcoded defaults.
    Use AuditConfig.from_yaml() to instantiate.
    """
    tau_activity_high: float
    tau_activity_low: float
    tau_energy_high: float
    tau_energy_low: float
    tau_tgt_hi: float
    tau_tgt_lo: float
    tau_res_hi: float
    tau_res_lo: float
    tau_delta_hi: float
    tau_delta_lo: float
    tau_q_enroll: float
    min_valid_windows: int
    tau_iqr_max: float
    version: str = "audit_v1"
    dataset: str = "DEBUG_ONLY"

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "AuditConfig":
        """Load thresholds from YAML config file.

        Args:
            config_path: Path to audit_thresholds_v1.yaml.
                         If None, auto-discovers from project root.

        Returns:
            AuditConfig with all thresholds populated from file.
        """
        if config_path is None:
            project_root = _find_project_root()
            config_path = os.path.join(project_root, _AUDIT_CONFIG_PATH)

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            tau_activity_high=float(raw["activity"]["tau_activity_high"]),
            tau_activity_low=float(raw["activity"]["tau_activity_low"]),
            tau_energy_high=float(raw["energy"]["tau_energy_high"]),
            tau_energy_low=float(raw["energy"]["tau_energy_low"]),
            tau_tgt_hi=float(raw["sv_score"]["tau_tgt_hi"]),
            tau_tgt_lo=float(raw["sv_score"]["tau_tgt_lo"]),
            tau_res_hi=float(raw["sv_score"]["tau_res_hi"]),
            tau_res_lo=float(raw["sv_score"]["tau_res_lo"]),
            tau_delta_hi=float(raw["delta"]["tau_delta_hi"]),
            tau_delta_lo=float(raw["delta"]["tau_delta_lo"]),
            tau_q_enroll=float(raw["quality"]["tau_q_enroll"]),
            min_valid_windows=int(raw["multi_window"]["min_valid_windows"]),
            tau_iqr_max=float(raw["multi_window"]["tau_iqr_max"]),
        )


@dataclass
class IdentityFeatures:
    """Frozen identity feature vector (P4-12).

    z_I = [q_enroll, p̄_tgt, D_tgt, s_tgt, ρ_s, valid_sv]
    """
    sample_id: str
    q_enroll: float
    activity_mean: float          # p̄_tgt
    target_activity_duration: float  # D_tgt
    s_tgt: Optional[float]        # null if invalid
    target_energy_ratio: float    # ρ_s
    valid_tgt_sv: bool

    # Residual features
    valid_res_sv: bool = False
    s_res: Optional[float] = None
    valid_delta: bool = False
    delta_s: Optional[float] = None

    # Multi-window
    window_consistent: bool = False
    window_count: int = 0
    valid_window_count: int = 0
    window_cosine_median: Optional[float] = None
    window_cosine_iqr: Optional[float] = None

    # Hashes
    tse_checkpoint_sha256: Optional[str] = None
    sv_model_sha256: Optional[str] = None
    feature_version: str = "identity_features_v1"


def _check_activity(
    features: IdentityFeatures,
    config: AuditConfig,
) -> str:
    """Classify activity state: A+, A-, or UNCERTAIN."""
    p = features.activity_mean
    if p >= config.tau_activity_high:
        return "A_PLUS"
    elif p < config.tau_activity_low:
        return "A_MINUS"
    else:
        return "A_UNCERTAIN"


def _check_energy(
    features: IdentityFeatures,
    config: AuditConfig,
) -> str:
    """Classify energy/gain state: G+, G-, or UNCERTAIN."""
    rho = features.target_energy_ratio
    if rho >= config.tau_energy_high:
        return "G_PLUS"
    elif rho < config.tau_energy_low:
        return "G_MINUS"
    else:
        return "G_UNCERTAIN"


def _check_identity_positive(
    features: IdentityFeatures,
    config: AuditConfig,
) -> bool:
    """I⁺: Identity evidence supports target presence.

    Requires:
      - Valid target SV
      - Multi-window consistency
      - s_tgt ≥ tau_tgt_hi
      - AND either:
        (a) residual not valid, OR
        (b) residual clean AND delta sufficient
    """
    if not features.valid_tgt_sv:
        return False
    if not features.window_consistent:
        return False
    if features.s_tgt is None or features.s_tgt < config.tau_tgt_hi:
        return False

    # If residual is not valid, we can't use it
    if not features.valid_res_sv:
        # Residual unobservable → don't claim residual is clean
        # But we still require the above conditions
        return True

    # Residual valid: need both residual clean AND delta margin
    residual_clean = (
        features.s_res is not None
        and features.s_res <= config.tau_res_lo
    )
    delta_ok = (
        features.valid_delta
        and features.delta_s is not None
        and features.delta_s >= config.tau_delta_hi
    )
    return residual_clean and delta_ok


def _check_misroute(
    features: IdentityFeatures,
    config: AuditConfig,
) -> bool:
    """I⁻: Identity evidence indicates target misrouted to residual.

    Requires:
      - Valid delta (both sides valid)
      - s_tgt ≤ tau_tgt_lo
      - s_res ≥ tau_res_hi
      - Δs ≤ tau_delta_lo
    """
    if not features.valid_delta:
        return False
    if features.s_tgt is None or features.s_tgt > config.tau_tgt_lo:
        return False
    if features.s_res is None or features.s_res < config.tau_res_hi:
        return False
    if features.delta_s is None or features.delta_s > config.tau_delta_lo:
        return False
    return True


def _check_target_present(
    features: IdentityFeatures,
    config: AuditConfig,
) -> bool:
    """H_target-present ⟺ A⁺ ∧ G⁺ ∧ Q⁺ ∧ I⁺"""
    activity_state = _check_activity(features, config)
    energy_state = _check_energy(features, config)
    quality_ok = features.q_enroll >= config.tau_q_enroll
    identity_ok = _check_identity_positive(features, config)

    return (
        activity_state == "A_PLUS"
        and energy_state == "G_PLUS"
        and quality_ok
        and identity_ok
    )


def _check_absent_activity_energy(
    features: IdentityFeatures,
    config: AuditConfig,
) -> bool:
    """R_abs: Activity AND energy jointly confirm target absence. A⁻ ∧ G⁻"""
    activity_state = _check_activity(features, config)
    energy_state = _check_energy(features, config)
    return activity_state == "A_MINUS" and energy_state == "G_MINUS"


def _must_be_gray(
    features: IdentityFeatures,
    config: AuditConfig,
) -> List[str]:
    """Check conditions that force GRAY regardless of other signals.

    Per manual: low-energy high-cosine, activity-energy conflict,
    low quality, tgt+res both high, multi-window inconsistency,
    single short high-scoring window, insufficient residual validity,
    target SV uncomputable but activity present, threshold equality,
    undefined boundary.
    """
    reasons = []

    # Low energy, high cosine
    if (
        features.target_energy_ratio < config.tau_energy_low
        and features.s_tgt is not None
        and features.s_tgt >= config.tau_tgt_hi
    ):
        reasons.append("LOW_ENERGY_HIGH_COSINE")

    # Activity-energy conflict
    activity_state = _check_activity(features, config)
    energy_state = _check_energy(features, config)
    if (
        (activity_state == "A_PLUS" and energy_state == "G_MINUS")
        or (activity_state == "A_MINUS" and energy_state == "G_PLUS")
    ):
        reasons.append("EVIDENCE_CONFLICT")

    # Low enrollment quality
    if features.q_enroll < config.tau_q_enroll:
        reasons.append("ENROLLMENT_LOW_QUALITY")

    # Target and residual both high
    if (
        features.s_tgt is not None and features.s_tgt >= config.tau_tgt_hi
        and features.s_res is not None and features.s_res >= config.tau_res_hi
    ):
        reasons.append("TGT_RES_BOTH_HIGH")

    # Multi-window inconsistent
    if (
        features.valid_tgt_sv
        and features.window_count >= config.min_valid_windows
        and not features.window_consistent
    ):
        reasons.append("MULTIWINDOW_INCONSISTENT")

    # Single short high-scoring window
    if features.valid_window_count == 1 and features.window_count > 1:
        if features.s_tgt is not None and features.s_tgt >= config.tau_tgt_hi:
            reasons.append("SINGLE_SHORT_HIGH_WINDOW")

    # Residual validity insufficient but other evidence incomplete
    if (
        not features.valid_res_sv
        and features.valid_tgt_sv
        and not _check_identity_positive(features, config)
    ):
        reasons.append("INSUFFICIENT_RESIDUAL_EVIDENCE")

    # Target SV uncomputable but activity exists
    if (
        not features.valid_tgt_sv
        and activity_state != "A_MINUS"
    ):
        reasons.append("TGT_SV_INVALID_ACTIVITY_PRESENT")

    return reasons


def audit(
    features: IdentityFeatures,
    config: Optional[AuditConfig] = None,
) -> dict:
    """Run three-state identity audit.

    Args:
        features: IdentityFeatures with all computed evidence.
        config: Audit thresholds (uses default DEBUG_ONLY if None).

    Returns:
        Dict with state, hard_reject, identity_score, reason_codes.
        Schema: p4_audit.v1
    """
    if config is None:
        config = AuditConfig.from_yaml()

    reason_codes = []
    state = "GRAY"
    hard_reject = False

    # Collect diagnostics throughout the audit
    diagnostics = {
        "activity_state": "UNKNOWN",
        "energy_state": "UNKNOWN",
        "quality_ok": False,
        "identity_positive": False,
        "misroute_detected": False,
        "absent_activity_energy": False,
        "gray_reasons": [],
    }

    # Check input validity first
    if features.q_enroll is None:
        diagnostics["error"] = "INVALID_ENROLLMENT"
        return {
            "sample_id": features.sample_id,
            "audit_state": "ERROR",
            "final_action": "EMPTY",
            "hard_reject": False,
            "identity_score": None,
            "identity_log_likelihood_ratio": None,
            "reason_codes": ["INVALID_ENROLLMENT"],
            "diagnostics": diagnostics,
            "feature_version": features.feature_version,
            "audit_version": config.version,
        }

    # Check for forced GRAY conditions
    gray_reasons = _must_be_gray(features, config)

    # Check target presence
    is_present = _check_target_present(features, config)

    # Check hard absence: A⁻ ∧ G⁻
    absent_activity_energy = _check_absent_activity_energy(features, config)

    # Check misroute: A⁺ ∧ G⁺ ∧ Q⁺ ∧ I⁻
    quality_ok = features.q_enroll >= config.tau_q_enroll
    activity_state = _check_activity(features, config)
    energy_state = _check_energy(features, config)
    is_misroute = (
        activity_state == "A_PLUS"
        and energy_state == "G_PLUS"
        and quality_ok
        and _check_misroute(features, config)
    )

    # Decide state
    if absent_activity_energy and not gray_reasons:
        state = "EMPTY"
        hard_reject = True
        reason_codes.append("TARGET_ABSENT_ACTIVITY_ENERGY")
    elif is_misroute and not gray_reasons:
        state = "EMPTY"
        hard_reject = True
        reason_codes.append("TARGET_MISROUTED_TO_RESIDUAL")
    elif is_present and not gray_reasons:
        state = "PRESENT"
        reason_codes.append("TARGET_PRESENT_STRONG")
    else:
        state = "GRAY"
        reason_codes.extend(gray_reasons)

    # --- Populate diagnostics ---
    diagnostics["activity_state"] = activity_state
    diagnostics["energy_state"] = energy_state
    diagnostics["quality_ok"] = quality_ok
    diagnostics["identity_positive"] = _check_identity_positive(features, config)
    diagnostics["misroute_detected"] = is_misroute
    diagnostics["absent_activity_energy"] = absent_activity_energy
    diagnostics["gray_reasons"] = gray_reasons

    # Build detailed reason codes
    if activity_state == "A_PLUS":
        reason_codes.append("ACTIVITY_HIGH")
    elif activity_state == "A_MINUS":
        reason_codes.append("ACTIVITY_LOW")
    else:
        reason_codes.append("ACTIVITY_UNCERTAIN")

    if energy_state == "G_PLUS":
        reason_codes.append("ENERGY_VALID")
    elif energy_state == "G_MINUS":
        reason_codes.append("ENERGY_LOW")
    else:
        reason_codes.append("ENERGY_UNCERTAIN")

    if features.valid_tgt_sv:
        reason_codes.append("TGT_SV_VALID")
    else:
        reason_codes.append("TGT_SV_INVALID")

    if features.valid_res_sv:
        reason_codes.append("RES_SV_VALID")
    else:
        reason_codes.append("RES_SV_INVALID")

    if features.window_consistent:
        reason_codes.append("MULTIWINDOW_CONSISTENT")
    else:
        reason_codes.append("MULTIWINDOW_INCONSISTENT")

    if hard_reject:
        reason_codes.append("HARD_REJECT")

    # Compute identity score (simple proxy: p_I before calibration)
    identity_score = _compute_proxy_identity_score(features, config)

    # Map audit_state to final_action (PRESENT → EMIT if text, EMPTY → EMPTY, GRAY → deleg to decide_gray)
    if state == "EMPTY":
        final_action = "EMPTY"
    elif state == "PRESENT":
        final_action = "EMIT"   # P5 runs ASR, EMIT if non-empty
    else:
        final_action = "GRAY"   # P5 runs ASR then calls decide_gray

    return {
        "sample_id": features.sample_id,
        "audit_state": state,
        "final_action": final_action,
        "hard_reject": hard_reject,
        "identity_score": identity_score,
        "identity_log_likelihood_ratio": (
            _safe_logit(identity_score) if identity_score is not None else None
        ),
        "reason_codes": reason_codes,
        "diagnostics": diagnostics,
        "feature_version": features.feature_version,
        "audit_version": config.version,
        "identity_model_version": "proxy_v0",
    }


def _compute_proxy_identity_score(
    features: IdentityFeatures,
    config: AuditConfig,
) -> Optional[float]:
    """Compute a proxy identity score before calibrated model.

    This is a simple heuristic; replaced by calibrated logistic regression in P4-15.
    Returns score in [0, 1] or None.
    """
    if not features.valid_tgt_sv or features.s_tgt is None:
        return None

    score = (features.s_tgt + 1) / 2  # map [-1, 1] → [0, 1]
    score = features.q_enroll * score  # attenuate by quality
    return float(max(0.0, min(1.0, score)))


def _safe_logit(p: Optional[float], eps: float = 1e-8) -> Optional[float]:
    """Safe logit transform with clipping."""
    if p is None:
        return None
    p = max(eps, min(1 - eps, p))
    return float(np.log(p / (1 - p)))


def get_feature_vector(
    features: IdentityFeatures,
) -> dict:
    """Extract the frozen feature vector z_I for identity model training.

    z_I = [q_enroll, p̄_tgt, D_tgt, s_tgt, ρ_s, valid_sv]

    Returns dict with values and missing indicators.
    """
    return {
        "q_enroll": features.q_enroll,
        "activity_mean": features.activity_mean,
        "target_activity_duration": features.target_activity_duration,
        "s_tgt": features.s_tgt,
        "s_tgt_missing": features.s_tgt is None,
        "target_energy_ratio": features.target_energy_ratio,
        "valid_tgt_sv": features.valid_tgt_sv,
        # Candidate (not in v1):
        # "s_res": features.s_res,
        # "s_res_missing": features.s_res is None,
        # "delta_s": features.delta_s,
        # "delta_s_missing": features.delta_s is None,
        # "window_cosine_median": features.window_cosine_median,
        # "window_cosine_iqr": features.window_cosine_iqr,
    }
