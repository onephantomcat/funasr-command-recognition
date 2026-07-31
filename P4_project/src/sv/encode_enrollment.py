"""Encode Enrollment API.

Frozen interface for encoding speaker enrollment audio into embeddings.
Schema: p4_enrollment.v1

Key contract (manual section 5.3):
  - Invalid enrollment returns embedding=None (NEVER all-zeros as normal).
  - Multi-channel is explicitly downmixed and logged.
  - Non-16kHz uses single frozen resampler.
  - quality is null until P4-07; never fake 1.0.
"""

import hashlib
import logging
from typing import Optional, Tuple, List

import numpy as np
import soundfile as sf
import torch

from .campplus_backend import (
    load_model,
    compute_embedding,
    cosine_similarity,
    get_model_sha256,
    CONFIG,
)
from .types import EnrollmentOutput

logger = logging.getLogger(__name__)

# Frozen preprocessing constants
TARGET_SR = 16000
MIN_DURATION = 0.3    # seconds
MAX_DURATION = 30.0   # seconds
CLIP_THRESHOLD = 0.99  # fraction of max amplitude for clipping detection
MIN_RMS = 1e-6


def _validate_and_preprocess(
    audio: np.ndarray,
    sample_rate: int,
) -> Tuple[torch.Tensor, float, dict]:
    """Validate and preprocess audio to frozen 16kHz mono float32.

    Returns (waveform_tensor, speech_duration, diagnostics).
    Raises ValueError with specific error code on invalid input.
    """
    diagnostics = {
        "reason_codes": [],
        "original_channels": 1,
        "original_sr": sample_rate,
        "was_downmixed": False,
        "was_resampled": False,
    }

    # Check empty
    if audio is None or audio.size == 0:
        raise ValueError("SV_EMPTY_INPUT")

    # Check for NaN/Inf
    if not np.all(np.isfinite(audio)):
        raise ValueError("SV_NONFINITE_AUDIO")

    # Handle multi-channel
    if audio.ndim > 1:
        if audio.shape[1] > 1:
            diagnostics["original_channels"] = audio.shape[1]
            audio = audio.mean(axis=1)
            diagnostics["was_downmixed"] = True
            diagnostics["reason_codes"].append("SV_MULTICHANNEL_DOWNMIXED")
        else:
            audio = audio.squeeze(1)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Check sample rate
    if sample_rate != TARGET_SR:
        diagnostics["was_resampled"] = True
        # Use simple linear interpolation resampling for offline use
        import scipy.signal
        new_len = int(len(audio) * TARGET_SR / sample_rate)
        audio = scipy.signal.resample(audio, new_len).astype(np.float32)

    duration = len(audio) / TARGET_SR

    # Check duration
    if duration < MIN_DURATION:
        raise ValueError("SV_TOO_SHORT")
    if duration > MAX_DURATION:
        raise ValueError("SV_TOO_LONG")

    # Check for valid speech (non-silence)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < MIN_RMS:
        raise ValueError("SV_NO_VALID_SPEECH")

    # Convert to tensor
    waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, T]

    return waveform, duration, diagnostics


def _compute_embedding_hash(embedding: np.ndarray) -> str:
    """Compute SHA256 of embedding bytes."""
    return hashlib.sha256(embedding.tobytes()).hexdigest()


def encode_enrollment(
    wav_path: str,
    sample_rate: Optional[int] = None,
) -> EnrollmentOutput:
    """Encode a speaker enrollment audio into an embedding.

    Args:
        wav_path: Path to enrollment audio file (WAV/FLAC, mono preferred).
        sample_rate: Explicit sample rate. If not provided and not in file
            header, raises error.

    Returns:
        EnrollmentOutput with embedding, quality, validity, and diagnostics.
        Invalid enrollment returns embedding=None, valid=False.
    """
    sample_id = hashlib.sha256(wav_path.encode()).hexdigest()[:16]

    try:
        # Load audio
        audio, file_sr = sf.read(wav_path, dtype="float32")
        if sample_rate is None:
            sample_rate = file_sr

        # Validate and preprocess
        waveform, duration, diagnostics = _validate_and_preprocess(audio, sample_rate)

        # Load model (singleton)
        model, frontend = load_model()

        # Compute embedding
        embedding = compute_embedding(waveform, model, frontend)

        # Get actual embedding dimension from output
        embedding_dim = int(embedding.shape[0])

        # Verify dimension matches config
        if embedding_dim != CONFIG["embedding_size"]:
            logger.warning(
                f"Embedding dim {embedding_dim} != config {CONFIG['embedding_size']}"
            )

        emb_hash = _compute_embedding_hash(embedding)

        diagnostics.update({
            "embedding_dim": embedding_dim,
            "embedding_norm_before": float(np.linalg.norm(embedding)),
            "preprocess_version": CONFIG["preprocess_version"],
            "model_sha256": get_model_sha256() or "",
            "model_id": CONFIG["model_id"],
            "model_revision": CONFIG["revision"],
        })

        return EnrollmentOutput(
            sample_id=sample_id,
            embedding=embedding,
            embedding_dim=embedding_dim,
            embedding_sha256=emb_hash,
            embedding_l2_normalized=CONFIG["l2_normalized"],
            quality=None,  # P4-07 fills this in
            speech_duration=duration,
            valid=True,
            diagnostics=diagnostics,
        )

    except ValueError as e:
        error_code = str(e)
        return EnrollmentOutput(
            sample_id=sample_id,
            embedding=None,
            embedding_dim=0,
            embedding_sha256=None,
            embedding_l2_normalized=False,
            quality=None,
            speech_duration=0.0,
            valid=False,
            diagnostics={
                "reason_codes": [error_code],
                "error": error_code,
            },
        )
    except Exception as e:
        logger.error(f"Enrollment failed for {wav_path}: {e}")
        return EnrollmentOutput(
            sample_id=sample_id,
            embedding=None,
            embedding_dim=0,
            embedding_sha256=None,
            embedding_l2_normalized=False,
            quality=None,
            speech_duration=0.0,
            valid=False,
            diagnostics={
                "reason_codes": ["SV_INFERENCE_FAILED"],
                "error": str(e),
            },
        )


def encode_enrollment_from_array(
    audio: np.ndarray,
    sample_rate: int,
    sample_id: str = "unknown",
) -> EnrollmentOutput:
    """Encode enrollment from an in-memory audio array.

    Same contract as encode_enrollment() but takes numpy array input.
    """
    try:
        waveform, duration, diagnostics = _validate_and_preprocess(audio, sample_rate)
        model, frontend = load_model()
        embedding = compute_embedding(waveform, model, frontend)
        embedding_dim = int(embedding.shape[0])
        emb_hash = _compute_embedding_hash(embedding)

        diagnostics.update({
            "embedding_dim": embedding_dim,
            "embedding_norm_before": float(np.linalg.norm(embedding)),
            "preprocess_version": CONFIG["preprocess_version"],
            "model_sha256": get_model_sha256() or "",
        })

        return EnrollmentOutput(
            sample_id=sample_id,
            embedding=embedding,
            embedding_dim=embedding_dim,
            embedding_sha256=emb_hash,
            embedding_l2_normalized=CONFIG["l2_normalized"],
            quality=None,
            speech_duration=duration,
            valid=True,
            diagnostics=diagnostics,
        )
    except ValueError as e:
        return EnrollmentOutput(
            sample_id=sample_id,
            embedding=None,
            embedding_dim=0,
            embedding_sha256=None,
            embedding_l2_normalized=False,
            quality=None,
            speech_duration=0.0,
            valid=False,
            diagnostics={"reason_codes": [str(e)], "error": str(e)},
        )
    except Exception as e:
        return EnrollmentOutput(
            sample_id=sample_id,
            embedding=None,
            embedding_dim=0,
            embedding_sha256=None,
            embedding_l2_normalized=False,
            quality=None,
            speech_duration=0.0,
            valid=False,
            diagnostics={"reason_codes": ["SV_INFERENCE_FAILED"], "error": str(e)},
        )
