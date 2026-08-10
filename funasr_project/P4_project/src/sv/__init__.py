"""SV 包公共符号导出（无副作用，import 时不加载模型）。"""

from .types import (
    BootstrapConfig,
    CampplusBackendConfig,
    EmbedMode,
    EnrollConfig,
    EnrollUtterance,
    EnrollmentSession,
    SpeakerEmbedding,
    VerificationPair,
    VerificationResult,
)
from .campplus_backend import CampplusBackend
from .encode_enrollment import (
    EnrollmentEncoder,
    encode_bootstrap,
    encode_session,
)

__all__ = [
    "BootstrapConfig",
    "CampplusBackend",
    "CampplusBackendConfig",
    "EmbedMode",
    "EnrollConfig",
    "EnrollUtterance",
    "EnrollmentEncoder",
    "EnrollmentSession",
    "SpeakerEmbedding",
    "VerificationPair",
    "VerificationResult",
    "encode_bootstrap",
    "encode_session",
]