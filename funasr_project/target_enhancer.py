"""Target-conditioned speech enhancement before Paraformer ASR.

The model learns a spectral mask from an external clean target waveform and a
synthetic noisy/reverberant mixture.  A summary of the wake audio conditions
the mask, so the same network can suppress both MUSAN noise and competing
speech without consulting any DatasetA label.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torch.nn.functional as F


SR = 16000


def read_audio(path, sample_rate=SR):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr == sample_rate:
        return audio
    target_len = max(1, round(len(audio) * sample_rate / sr))
    positions = np.linspace(0, max(0, len(audio) - 1), target_len)
    return np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)


def write_audio(path, audio, sample_rate=SR):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio * (0.98 / peak)
    sf.write(path, audio, sample_rate)


def apply_spectral_denoise(audio, n_fft=512, hop_length=128):
    """Lightweight stationary spectral subtraction noise reduction."""
    if len(audio) < n_fft:
        return np.asarray(audio, dtype=np.float32)
    audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    window = torch.hann_window(n_fft)
    spec = torch.stft(
        audio_tensor, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    ).squeeze(0)
    mag = spec.abs()
    phase = spec.angle()
    noise_est = torch.quantile(mag, 0.10, dim=-1, keepdim=True)
    clean_mag = torch.clamp(mag - 1.2 * noise_est, min=0.05 * mag)
    clean_spec = torch.polar(clean_mag, phase)
    restored = torch.istft(
        clean_spec.unsqueeze(0), n_fft=n_fft, hop_length=hop_length, window=window, length=len(audio)
    ).squeeze(0).numpy().astype(np.float32)
    return restored


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        y = self.depthwise(x)
        y = F.silu(self.norm(self.pointwise(y)))
        return x + y


class TargetEnhancer(nn.Module):
    """Small wake-conditioned spectral mask estimator for 16 kHz audio."""

    def __init__(self, n_fft=512, hop_length=128, channels=64, blocks=6):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_freq = self.n_fft // 2 + 1
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)
        self.input_proj = nn.Conv1d(self.n_freq, self.channels, kernel_size=1)
        self.condition_proj = nn.Sequential(
            nn.Linear(self.n_freq, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels),
        )
        self.residual = nn.ModuleList(
            ResidualDilatedBlock(self.channels, 2 ** (index % 4))
            for index in range(self.blocks)
        )
        self.mask_head = nn.Conv1d(self.channels, self.n_freq, kernel_size=1)

    def config(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "channels": self.channels,
            "blocks": self.blocks,
        }

    def _stft(self, audio):
        window = self.window.to(device=audio.device, dtype=audio.dtype)
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

    def forward(self, mixture, wake):
        if mixture.ndim == 1:
            mixture = mixture.unsqueeze(0)
        if wake.ndim == 1:
            wake = wake.unsqueeze(0)
        mixture_spec = self._stft(mixture)
        wake_spec = self._stft(wake)
        mixture_features = torch.log1p(mixture_spec.abs())
        wake_summary = torch.log1p(wake_spec.abs()).mean(dim=-1)
        hidden = self.input_proj(mixture_features)
        hidden = hidden + self.condition_proj(wake_summary).unsqueeze(-1)
        for block in self.residual:
            hidden = block(hidden)
        mask = torch.sigmoid(self.mask_head(hidden))
        enhanced_spec = mixture_spec * mask
        window = self.window.to(device=mixture.device, dtype=mixture.dtype)
        enhanced = torch.istft(
            enhanced_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            length=mixture.shape[-1],
        )
        return enhanced, mask


def enhancement_loss(enhanced, clean, n_fft=512, hop_length=128):
    """Waveform + log-magnitude loss stable on short command utterances."""
    window = torch.hann_window(n_fft, device=enhanced.device, dtype=enhanced.dtype)
    enhanced_spec = torch.stft(
        enhanced, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
    )
    clean_spec = torch.stft(
        clean, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
    )
    waveform = F.l1_loss(enhanced, clean)
    magnitude = F.l1_loss(torch.log1p(enhanced_spec.abs()), torch.log1p(clean_spec.abs()))
    return magnitude + 0.2 * waveform, {"waveform_l1": waveform.detach(), "logmag_l1": magnitude.detach()}


def load_target_enhancer(checkpoint_path, device=None):
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    model = TargetEnhancer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(resolved_device).eval()
    return model


@torch.inference_mode()
def enhance_audio(model, wake_audio, mixture_audio, chunk_sec=8.0, overlap_sec=1.0):
    """Enhance arbitrarily long audio with overlap-add chunking."""
    if not len(mixture_audio):
        return np.asarray(mixture_audio, dtype=np.float32)
    device = next(model.parameters()).device
    chunk = max(int(chunk_sec * SR), model.n_fft * 4)
    overlap = min(max(int(overlap_sec * SR), 0), chunk // 2)
    hop = max(1, chunk - overlap)
    output = np.zeros(len(mixture_audio), dtype=np.float32)
    weights = np.zeros(len(mixture_audio), dtype=np.float32)
    wake = torch.from_numpy(np.asarray(wake_audio, dtype=np.float32)).unsqueeze(0).to(device)
    for start in range(0, len(mixture_audio), hop):
        end = min(len(mixture_audio), start + chunk)
        segment = mixture_audio[start:end]
        original_len = len(segment)
        if original_len < model.n_fft:
            segment = np.pad(segment, (0, model.n_fft - original_len))
        mixture = torch.from_numpy(np.asarray(segment, dtype=np.float32)).unsqueeze(0).to(device)
        enhanced, _ = model(mixture, wake)
        restored = enhanced.squeeze(0).detach().cpu().numpy()[:original_len]
        taper = np.ones(original_len, dtype=np.float32)
        if overlap and start:
            taper[:min(overlap, original_len)] = np.linspace(0.0, 1.0, min(overlap, original_len), dtype=np.float32)
        if overlap and end < len(mixture_audio):
            taper[-min(overlap, original_len):] *= np.linspace(1.0, 0.0, min(overlap, original_len), dtype=np.float32)
        output[start:end] += restored * taper
        weights[start:end] += taper
        if end == len(mixture_audio):
            break
    return output / np.maximum(weights, 1e-6)


def enhance_file(model, wake_path, mixture_path, out_path, chunk_sec=8.0, overlap_sec=1.0):
    enhanced = enhance_audio(
        model,
        read_audio(wake_path),
        read_audio(mixture_path),
        chunk_sec=chunk_sec,
        overlap_sec=overlap_sec,
    )
    write_audio(out_path, enhanced)
    return {"samples": int(len(enhanced)), "path": str(out_path)}
