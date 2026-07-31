r"""GRAY Decision Strategy (P4-18).

Implements the constrained two-dimensional EMIT/EMPTY decision for GRAY samples.

Core rule (manual section 6 / P4-18):
  EMIT ⟺ ¬I_hard-reject ∧ |ŷ| > 0 ∧ Λ_I ≥ τ_I_min ∧ Λ_I ≥ h(r_asr)

First version uses identity-floor-only rule:
  EMIT ⟺ ¬I_hard-reject ∧ |ŷ| > 0 ∧ Λ_I ≥ τ_I_min

Character-cost plugin (P4-19, optional):
  EMIT ⟺ ¬I_hard-reject ∧ |ŷ| > 0 ∧ b̂ > 0
          ∧ Λ_I > max(τ_I_min, κ_char / b̂)

All thresholds loaded from config file. Equality → EMPTY.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_GRAY_CONFIG_PATH = "configs/gray_policy_v1.yaml"


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
class GrayConfig:
    """GRAY decision configuration loaded from YAML file.

    ALL thresholds come from the config file. No hardcoded defaults.
    Use GrayConfig.from_yaml() to instantiate.
    """
    tau_I_min: float
    use_char_cost: bool
    b_max: float
    kappa_char: float
    policy_version: str = "gray_policy_v1"

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "GrayConfig":
        if config_path is None:
            project_root = _find_project_root()
            config_path = os.path.join(project_root, _GRAY_CONFIG_PATH)

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            tau_I_min=float(raw["identity_floor"]["tau_I_min"]),
            use_char_cost=bool(raw["character_cost"]["use_char_cost"]),
            b_max=float(raw["character_cost"]["b_max"]),
            kappa_char=float(raw["character_cost"]["kappa_char"]),
        )


def decide_gray_v1(
    sample_id: str,
    audit_state: str,
    hard_reject: bool,
    identity_score: Optional[float],
    identity_log_lr: Optional[float],
    asr_text: Optional[str],
    asr_features: Optional[dict],
    config: Optional[GrayConfig] = None,
) -> dict:
    """First-version GRAY decision: identity-floor-only rule.

    For AUDIT == GRAY only. P5 calls this after running ASR.

    Returns dict with final_action ("EMIT" or "EMPTY"), policy_version,
    reason_codes, diagnostics.
    """
    if config is None:
        config = GrayConfig.from_yaml()

    diagnostics = {"identity_floor": config.tau_I_min, "rule": "v1_floor_only"}

    if audit_state != "GRAY":
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["NOT_GRAY_STATE"],
            "diagnostics": diagnostics,
        }

    if hard_reject:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["HARD_REJECT"],
            "diagnostics": diagnostics,
        }

    has_text = asr_text is not None and len(asr_text.strip()) > 0
    if not has_text:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["ASR_EMPTY"],
            "diagnostics": diagnostics,
        }

    if identity_score is None:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_BELOW_FLOOR", "IDENTITY_SCORE_MISSING"],
            "diagnostics": diagnostics,
        }

    if identity_score < config.tau_I_min:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_BELOW_FLOOR"],
            "diagnostics": diagnostics,
        }

    # Equality → EMPTY
    if identity_score == config.tau_I_min:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_AT_FLOOR_EQUALITY"],
            "diagnostics": diagnostics,
        }

    return {
        "sample_id": sample_id,
        "final_action": "EMIT",
        "policy_version": config.policy_version,
        "reason_codes": ["IDENTITY_ABOVE_FLOOR"],
        "diagnostics": diagnostics,
    }


def decide_gray_with_char_cost(
    sample_id: str,
    audit_state: str,
    hard_reject: bool,
    identity_score: Optional[float],
    identity_log_lr: Optional[float],
    asr_text: Optional[str],
    d_hat: Optional[float],
    L_hat: Optional[float],
    config: Optional[GrayConfig] = None,
) -> dict:
    """GRAY decision with constrained character-cost plugin (P4-19).

    b̂ = clip(L̂ - d̂, 0, b_max)
    EMIT ⟺ ¬I_hard-reject ∧ |ŷ| > 0 ∧ b̂ > 0
            ∧ Λ_I > max(τ_I_min, κ_char / b̂)

    Equality or b̂=0 → EMPTY.
    """
    if config is None:
        config = GrayConfig.from_yaml()

    diagnostics = {
        "identity_floor": config.tau_I_min,
        "rule": "char_cost_plugin",
        "use_char_cost": config.use_char_cost,
    }

    if audit_state != "GRAY":
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["NOT_GRAY_STATE"],
            "diagnostics": diagnostics,
        }

    if hard_reject:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["HARD_REJECT"],
            "diagnostics": diagnostics,
        }

    has_text = asr_text is not None and len(asr_text.strip()) > 0
    if not has_text:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["ASR_EMPTY"],
            "diagnostics": diagnostics,
        }

    if identity_score is None or identity_log_lr is None:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_SCORE_MISSING"],
            "diagnostics": diagnostics,
        }

    if identity_score < config.tau_I_min:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_BELOW_FLOOR"],
            "diagnostics": diagnostics,
        }

    if d_hat is None or L_hat is None:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["CHARACTER_MODEL_UNAVAILABLE"],
            "diagnostics": diagnostics,
        }

    b_hat = max(0.0, min(L_hat - d_hat, config.b_max))
    diagnostics["b_hat"] = b_hat

    if b_hat <= 0:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["CHARACTER_BENEFIT_NONPOSITIVE"],
            "diagnostics": diagnostics,
        }

    threshold = max(config.tau_I_min, config.kappa_char / b_hat)
    diagnostics["char_cost_threshold"] = threshold

    if identity_score > threshold:
        return {
            "sample_id": sample_id,
            "final_action": "EMIT",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_ABOVE_CHAR_COST_THRESHOLD"],
            "diagnostics": diagnostics,
        }
    else:
        return {
            "sample_id": sample_id,
            "final_action": "EMPTY",
            "policy_version": config.policy_version,
            "reason_codes": ["IDENTITY_BELOW_CHAR_COST_THRESHOLD"],
            "diagnostics": diagnostics,
        }
