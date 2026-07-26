"""Train TSE Dual-Output (target + residual) Extraction Network on GPU.

Uses AISHELL-1 target speech + MUSAN noise/interferer speech + RIRS_NOISES reverberation.
Trains TSEDualOutputNet to simultaneously extract target speech and residual interference speech.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from tse_dual_output import TSEDualOutputNet, SR


def read_audio(path: str | Path, target_sr: int = SR) -> np.ndarray:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr == target_sr:
        return data
    target_len = max(1, round(len(data) * target_sr / sr))
    positions = np.linspace(0, max(0, len(data) - 1), target_len)
    return np.interp(positions, np.arange(len(data)), data).astype(np.float32)


def make_fixed_segment(audio: np.ndarray, num_samples: int) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros((num_samples,), dtype=np.float32)
    if len(audio) < num_samples:
        repeats = (num_samples // len(audio)) + 1
        audio = np.tile(audio, repeats)
    start = random.randint(0, len(audio) - num_samples)
    return audio[start:start + num_samples]


class SyntheticTSEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        speaker_wavs: dict[str, list[Path]],
        noise_files: list[Path],
        rir_files: list[Path],
        steps: int = 250,
        segment_sec: float = 3.0,
    ):
        self.speaker_ids = sorted(list(speaker_wavs.keys()))
        self.speaker_wavs = speaker_wavs
        self.noise_files = noise_files
        self.rir_files = rir_files
        self.steps = steps
        self.num_samples = int(round(segment_sec * SR))

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        target_spk = random.choice(self.speaker_ids)
        target_files = self.speaker_wavs[target_spk]
        enroll_path = random.choice(target_files)
        target_path = random.choice(target_files)

        enroll_audio = make_fixed_segment(read_audio(enroll_path), self.num_samples)
        target_audio = make_fixed_segment(read_audio(target_path), self.num_samples)

        # Scale target audio
        t_peak = max(1e-4, float(np.max(np.abs(target_audio))))
        target_audio = target_audio / t_peak * random.uniform(0.3, 0.9)

        # Interferer audio (other speaker or noise)
        interferer_spks = [s for s in self.speaker_ids if s != target_spk]
        if interferer_spks and random.random() < 0.6:
            interferer_spk = random.choice(interferer_spks)
            interferer_path = random.choice(self.speaker_wavs[interferer_spk])
            interferer_audio = make_fixed_segment(read_audio(interferer_path), self.num_samples)
        elif self.noise_files:
            noise_path = random.choice(self.noise_files)
            interferer_audio = make_fixed_segment(read_audio(noise_path), self.num_samples)
        else:
            interferer_audio = np.random.randn(self.num_samples).astype(np.float32) * 0.05

        i_peak = max(1e-4, float(np.max(np.abs(interferer_audio))))
        interferer_audio = interferer_audio / i_peak * random.uniform(0.2, 0.7)

        # Mix mixture = target + interferer
        mix_audio = target_audio + interferer_audio

        return {
            "mix": torch.from_numpy(mix_audio),
            "enroll": torch.from_numpy(enroll_audio),
            "target": torch.from_numpy(target_audio),
            "residual": torch.from_numpy(interferer_audio),
        }


def dual_loss_fn(
    target_est: torch.Tensor,
    residual_est: torch.Tensor,
    target_gt: torch.Tensor,
    residual_gt: torch.Tensor,
) -> torch.Tensor:
    l_target = F.l1_loss(target_est, target_gt)
    l_residual = F.l1_loss(residual_est, residual_gt)
    return l_target + 0.5 * l_residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-root", required=True)
    parser.add_argument("--noise-root", required=True)
    parser.add_argument("--rir-root", required=True)
    parser.add_argument("--out", default="models/tse_dual_output_mvp.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--dev-steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wav_root = Path(args.wav_root)
    noise_root = Path(args.noise_root)
    rir_root = Path(args.rir_root)

    print(f"Scanning speech datasets from {wav_root}...")
    spk_files: dict[str, list[Path]] = {}
    for p in wav_root.rglob("*.wav"):
        spk = p.parent.name
        spk_files.setdefault(spk, []).append(p)

    all_spks = sorted(list(spk_files.keys()))
    if len(all_spks) < 2:
        raise ValueError(f"Need at least 2 speakers in wav_root, found {len(all_spks)}")

    train_spks = all_spks[:-3] if len(all_spks) > 3 else all_spks[:-1]
    dev_spks = all_spks[-3:] if len(all_spks) > 3 else all_spks[-1:]

    train_wavs = {s: spk_files[s] for s in train_spks}
    dev_wavs = {s: spk_files[s] for s in dev_spks}

    noise_files = list(noise_root.rglob("*.wav"))
    rir_files = list(rir_root.rglob("*.wav"))

    train_ds = SyntheticTSEDataset(train_wavs, noise_files, rir_files, steps=args.steps_per_epoch * args.batch_size)
    dev_ds = SyntheticTSEDataset(dev_wavs, noise_files, rir_files, steps=args.dev_steps * args.batch_size)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TSE Dual-Output Training device: {device}")

    model = TSEDualOutputNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            mix = batch["mix"].to(device)
            enroll = batch["enroll"].to(device)
            target = batch["target"].to(device)
            residual = batch["residual"].to(device)

            target_est, residual_est = model.forward_audio(mix, enroll)
            loss = dual_loss_fn(target_est, residual_est, target, residual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        dev_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                mix = batch["mix"].to(device)
                enroll = batch["enroll"].to(device)
                target = batch["target"].to(device)
                residual = batch["residual"].to(device)

                target_est, residual_est = model.forward_audio(mix, enroll)
                loss = dual_loss_fn(target_est, residual_est, target, residual)
                dev_loss += loss.item()

        dev_loss /= len(dev_loader)

        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.5f} - dev_loss: {dev_loss:.5f}")

        if dev_loss < best_loss:
            best_loss = dev_loss
            torch.save({
                "model_config": {"n_fft": 512, "hop_length": 128, "channels": 64, "blocks": 6},
                "model_state": model.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
            }, out_path)
            json_path = out_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "model_config": {"n_fft": 512, "hop_length": 128, "channels": 64, "blocks": 6},
                    "epoch": epoch,
                    "dev_loss": dev_loss,
                    "out": str(out_path),
                }, f, indent=2)
            print(f"Saved best TSE model checkpoint to {out_path}")

    print("TSE Dual-Output MVP training completed successfully!")


if __name__ == "__main__":
    main()
