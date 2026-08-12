"""P3-side runtime adapter for the frozen P2 ``extract_target()`` API."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from P2_project.src.tse import DualOutputTSE, extract_target


NEAR_SILENT_RMS_RATIO = 1e-4


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("P2 checkpoint must be a mapping")
    for key in ("model", "model_state_dict", "model_state", "state_dict"):
        state = checkpoint.get(key)
        if isinstance(state, dict):
            return state
    raise KeyError("P2 checkpoint has no model/model_state_dict/model_state/state_dict")


class P2TSERuntime:
    """Load one immutable P2 checkpoint and extract target-speaker WAVs."""

    def __init__(self, checkpoint_path, device=None, expected_sha256=None):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"P2 checkpoint not found: {self.checkpoint_path}")

        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        if expected_sha256:
            expected = expected_sha256.strip().lower()
            if self.checkpoint_sha256.lower() != expected:
                raise ValueError(
                    "P2 checkpoint SHA256 mismatch: "
                    f"expected {expected}, got {self.checkpoint_sha256}"
                )

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if resolved_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("P2 TSE requested CUDA but torch.cuda.is_available() is false")
        self.device = torch.device(resolved_device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.cfg = checkpoint.get("cfg") if isinstance(checkpoint, dict) else None
        if not isinstance(self.cfg, dict):
            raise KeyError("P2 checkpoint must contain cfg mapping")
        self.sample_rate = int(self.cfg.get("sample_rate", 16000))
        self.emb_dim = int(self.cfg["emb_dim"])
        self.step = checkpoint.get("step")
        config_json = json.dumps(self.cfg, ensure_ascii=False, sort_keys=True, default=str)
        self.config_sha256 = hashlib.sha256(config_json.encode("utf-8")).hexdigest()

        self.model = DualOutputTSE(self.cfg).to(self.device)
        self.model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        self.model.eval()

    def metadata(self):
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "step": self.step,
            "sample_rate": self.sample_rate,
            "emb_dim": self.emb_dim,
            "device": str(self.device),
            "api": "P2_project.src.tse.extract_target",
        }

    def _read_command(self, path):
        waveform, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=False,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError(f"P2 command audio must be non-empty mono waveform: {path}")
        if not np.isfinite(waveform).all():
            raise ValueError(f"P2 command audio contains NaN/Inf: {path}")

        tensor = torch.from_numpy(waveform)
        if sample_rate != self.sample_rate:
            tensor = torchaudio.functional.resample(
                tensor,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
        return tensor.to(dtype=torch.float32)

    def _prepare_embedding(self, embedding):
        tensor = torch.as_tensor(embedding, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape != (1, self.emb_dim):
            raise ValueError(
                f"P2 enrollment embedding must be [1,{self.emb_dim}], got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError("P2 enrollment embedding contains NaN/Inf")
        norm = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
        if torch.any(norm <= 1e-8):
            raise ValueError("P2 enrollment embedding has zero norm")
        return (tensor / norm).to(self.device)

    def extract_file(self, command_path, enrollment_embedding, output_path):
        """Extract target speech and atomically write a float32 WAV."""
        command_path = Path(command_path)
        output_path = Path(output_path)
        command = self._read_command(command_path)
        embedding = self._prepare_embedding(enrollment_embedding)
        command_batch = command.unsqueeze(0).to(self.device)

        started_at = time.perf_counter()
        target = extract_target(command_batch, embedding, self.model, self.cfg)
        elapsed = time.perf_counter() - started_at
        target = target.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if target.shape != (command.shape[-1],):
            raise RuntimeError(
                f"P2 output length mismatch: {target.shape} vs {(command.shape[-1],)}"
            )
        if not np.isfinite(target).all():
            raise RuntimeError("P2 output contains NaN/Inf")

        command_array = command.detach().cpu().numpy().astype(np.float32, copy=False)
        input_peak = float(np.max(np.abs(command_array)))
        input_rms = float(np.sqrt(np.mean(np.square(command_array), dtype=np.float64)))
        output_peak = float(np.max(np.abs(target)))
        output_rms = float(np.sqrt(np.mean(np.square(target), dtype=np.float64)))
        output_to_input_rms_ratio = (
            output_rms / input_rms if input_rms > 1e-12 else None
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f"{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}"
        )
        sf.write(temporary_path, target, self.sample_rate, subtype="FLOAT")
        os.replace(temporary_path, output_path)

        duration = len(target) / self.sample_rate
        return {
            **self.metadata(),
            "input": str(command_path.resolve()),
            "output": str(output_path.resolve()),
            "samples": len(target),
            "duration_sec": round(duration, 6),
            "inference_sec": round(elapsed, 6),
            "rtf": round(elapsed / duration, 6) if duration else None,
            "input_peak": input_peak,
            "input_rms": input_rms,
            "output_peak": output_peak,
            "output_rms": output_rms,
            "output_to_input_rms_ratio": output_to_input_rms_ratio,
            "output_near_silent": bool(
                output_to_input_rms_ratio is not None
                and output_to_input_rms_ratio < NEAR_SILENT_RMS_RATIO
            ),
            "near_silent_rms_ratio_threshold": NEAR_SILENT_RMS_RATIO,
            "cached": False,
        }
