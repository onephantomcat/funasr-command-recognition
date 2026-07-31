"""CAM++ Model Backend.

Frozen model: iic/speech_campplus_sv_zh-cn_16k-common v1.0.0
SHA256: 3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8
Embedding dim: 192, FBank 80-dim with mean_nor, no L2 norm on output.

This module wraps the frozen 3D-Speaker CAM++ model with a fixed frontend.
The model is loaded once and reused. All configuration is versioned.
"""

import os
import sys
import hashlib
from typing import Optional, Tuple

import numpy as np
import torch

# Frozen source path
_SPEAKERLAB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "artifacts", "models", "speakerlab_source"
)
if _SPEAKERLAB_PATH not in sys.path:
    sys.path.insert(0, _SPEAKERLAB_PATH)

from speakerlab.models.campplus.DTDNN import CAMPPlus
from speakerlab.process.processor import FBank


# Frozen configuration
CONFIG = {
    "model_id": "iic/speech_campplus_sv_zh-cn_16k-common",
    "revision": "v1.0.0",
    "checkpoint_filename": "campplus_cn_common.bin",
    "checkpoint_sha256": "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8",
    "feat_dim": 80,
    "embedding_size": 192,
    "sample_rate": 16000,
    "mean_nor": True,
    "dtype": "float32",
    "l2_normalized": False,
    "preprocess_version": "sv_preprocess_v1",
}

# Global singleton
_model: Optional[CAMPPlus] = None
_frontend: Optional[FBank] = None
_model_sha256_verified: Optional[str] = None


def get_model_path() -> str:
    """Return path to frozen CAM++ checkpoint."""
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(
        project_root, "artifacts", "models", "campplus_frozen",
        CONFIG["checkpoint_filename"]
    )


def verify_model_sha256() -> Tuple[bool, str]:
    """Verify the checkpoint SHA256 matches the frozen value."""
    model_path = get_model_path()
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    actual = h.hexdigest()
    expected = CONFIG["checkpoint_sha256"]
    return actual == expected, actual


def load_model(device: str = "cpu") -> Tuple[CAMPPlus, FBank]:
    """Load and return CAM++ model and FBank frontend (singleton).

    The model is loaded once. Subsequent calls return the cached instance.
    Returns (model, frontend).
    """
    global _model, _frontend, _model_sha256_verified

    if _model is not None and _frontend is not None:
        return _model, _frontend

    # Verify checkpoint hash
    ok, actual_hash = verify_model_sha256()
    if not ok:
        raise RuntimeError(
            f"Model SHA256 mismatch! Expected {CONFIG['checkpoint_sha256']}, "
            f"got {actual_hash}"
        )
    _model_sha256_verified = actual_hash

    # Load model
    model = CAMPPlus(
        feat_dim=CONFIG["feat_dim"],
        embedding_size=CONFIG["embedding_size"],
    )
    state = torch.load(get_model_path(), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Create frontend
    frontend = FBank(
        CONFIG["feat_dim"],
        sample_rate=CONFIG["sample_rate"],
        mean_nor=CONFIG["mean_nor"],
    )

    _model = model
    _frontend = frontend
    return model, frontend


def compute_embedding(
    waveform: torch.Tensor,
    model: Optional[CAMPPlus] = None,
    frontend: Optional[FBank] = None,
) -> np.ndarray:
    """Compute CAM++ embedding from a preprocessed waveform.

    Args:
        waveform: [1, T] float32 tensor at 16kHz mono.
        model: Optional model instance (loads singleton if None).
        frontend: Optional frontend instance.

    Returns:
        numpy array of shape [D] (192 for this model), float32, NOT L2-normalized.
    """
    if model is None or frontend is None:
        model, frontend = load_model()

    feat = frontend(waveform)  # [T, 80]
    feat = feat.unsqueeze(0)  # [1, T, 80]
    with torch.no_grad():
        embedding = model(feat).squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def get_model_sha256() -> Optional[str]:
    """Return the verified model SHA256, or None if not yet loaded."""
    return _model_sha256_verified
