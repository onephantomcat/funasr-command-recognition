"""TSE Dual-Output (target + residual) Extraction MVP Module.

Provides joint target and residual speech signal extraction. Inputs mixture audio and wake enrollment audio,
outputting both purified target audio and residual background/interference audio for transparent auditing.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 16000


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = F.silu(self.norm(self.pointwise(y)))
        return x + y


class TSEDualOutputNet(nn.Module):
    """Dual-output Target Signal Extraction network.
    
    Produces complementary masks: target_mask (0~1) and residual_mask (1 - target_mask).
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 128, channels: int = 64, blocks: int = 6):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_freq = self.n_fft // 2 + 1
        self.channels = int(channels)
        self.blocks = int(blocks)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

        self.input_proj = nn.Conv1d(self.n_freq, self.channels, kernel_size=1)
        self.enroll_proj = nn.Sequential(
            nn.Linear(self.n_freq, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels),
        )
        self.backbone = nn.ModuleList([
            ResidualConvBlock(self.channels, 2 ** (i % 4))
            for i in range(self.blocks)
        ])
        self.mask_target = nn.Conv1d(self.channels, self.n_freq, kernel_size=1)

    def forward_stft(self, mix_mag: torch.Tensor, enroll_mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """STFT domain forward pass.
        
        mix_mag: (B, N_freq, T)
        enroll_mag: (B, N_freq, T_enroll)
        Returns: (target_mask, residual_mask)
        """
        x = self.input_proj(mix_mag)
        c = self.enroll_proj(enroll_mag.mean(dim=-1)).unsqueeze(-1)
        x = x + c
        for block in self.backbone:
            x = block(x)
        target_mask = torch.sigmoid(self.mask_target(x))
        residual_mask = torch.clamp(1.0 - target_mask, min=0.0, max=1.0)
        return target_mask, residual_mask

    def forward_audio(self, mix_audio: torch.Tensor, enroll_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Waveform domain forward pass.
        
        mix_audio: (B, L_mix)
        enroll_audio: (B, L_enroll)
        Returns: (target_audio, residual_audio)
        """
        device = mix_audio.device
        window = self.window.to(device)
        
        mix_spec = torch.stft(
            mix_audio, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
        )
        enroll_spec = torch.stft(
            enroll_audio, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True
        )
        
        mix_mag = mix_spec.abs()
        mix_angle = mix_spec.angle()
        enroll_mag = enroll_spec.abs()

        target_mask, residual_mask = self.forward_stft(mix_mag, enroll_mag)

        target_spec = torch.polar(mix_mag * target_mask, mix_angle)
        residual_spec = torch.polar(mix_mag * residual_mask, mix_angle)

        target_audio = torch.istft(
            target_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=window, length=mix_audio.shape[-1]
        )
        residual_audio = torch.istft(
            residual_spec, n_fft=self.n_fft, hop_length=self.hop_length, window=window, length=mix_audio.shape[-1]
        )
        return target_audio, residual_audio


def smoke_test():
    """Module smoke test verifying shapes and non-nan outputs."""
    net = TSEDualOutputNet()
    mix = torch.randn(2, 48000)
    enroll = torch.randn(2, 32000)
    target, residual = net.forward_audio(mix, enroll)
    assert target.shape == mix.shape, f"Target shape mismatch: {target.shape} vs {mix.shape}"
    assert residual.shape == mix.shape, f"Residual shape mismatch: {residual.shape} vs {mix.shape}"
    assert not torch.isnan(target).any(), "NaN in target audio"
    assert not torch.isnan(residual).any(), "NaN in residual audio"
    print("TSEDualOutputNet smoke test passed successfully! Target shape:", target.shape, "Residual shape:", residual.shape)


if __name__ == "__main__":
    smoke_test()
