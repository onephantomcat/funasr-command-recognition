"""Train the CER-oriented target speech enhancer on external data only.

Each sample keeps an AISHELL utterance as the clean target, uses a second
utterance from the same speaker as wake audio, and synthesizes a noisy mixture
with MUSAN, another AISHELL speaker, and optionally RIRS_NOISES.  DatasetA is
never read by this script.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from build_external_trainset import apply_rir, discover_audio_files, mix_at_ratio, scan_wavs
from target_enhancer import SR, TargetEnhancer, apply_spectral_denoise, enhancement_loss, read_audio


def crop_or_pad(audio, length, rng):
    if len(audio) >= length:
        start = int(rng.integers(0, len(audio) - length + 1))
        return audio[start:start + length]
    return np.pad(audio, (0, length - len(audio)))


class ExternalMixtureDataset(Dataset):
    def __init__(self, speakers, noise_paths, rir_paths, items, segment_sec, seed, mix_denoised_ratio=0.0):
        self.speakers = {speaker: list(paths) for speaker, paths in speakers.items() if len(paths) >= 2}
        self.speaker_ids = sorted(self.speakers)
        self.noise_paths = list(noise_paths)
        self.rir_paths = list(rir_paths)
        self.items = int(items)
        self.segment_len = int(segment_sec * SR)
        self.seed = int(seed)
        self.mix_denoised_ratio = float(mix_denoised_ratio)
        if len(self.speaker_ids) < 2:
            raise ValueError("Need at least two speakers with two utterances each")

    def __len__(self):
        return self.items

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + index + random.randint(0, 2**20))
        speaker = self.speaker_ids[int(rng.integers(len(self.speaker_ids)))]
        paths = self.speakers[speaker]
        target_idx = int(rng.integers(len(paths)))
        wake_idx = int(rng.integers(len(paths) - 1))
        if wake_idx >= target_idx:
            wake_idx += 1
        clean = crop_or_pad(read_audio(paths[target_idx]), self.segment_len, rng)
        wake = crop_or_pad(read_audio(paths[wake_idx]), self.segment_len, rng)
        mixture = clean.copy()

        # Speech interference is essential for the target-speaker part of CER.
        other_ids = [item for item in self.speaker_ids if item != speaker]
        other = self.speakers[other_ids[int(rng.integers(len(other_ids)))]]
        interferer = crop_or_pad(read_audio(other[int(rng.integers(len(other)))]), self.segment_len, rng)
        mixture = mix_at_ratio(mixture, interferer, float(rng.uniform(-2.0, 8.0)))

        if self.noise_paths:
            noise = crop_or_pad(
                read_audio(self.noise_paths[int(rng.integers(len(self.noise_paths)))]),
                self.segment_len,
                rng,
            )
            mixture = mix_at_ratio(mixture, noise, float(rng.uniform(-5.0, 12.0)))
        if self.rir_paths and rng.random() < 0.7:
            rir = read_audio(self.rir_paths[int(rng.integers(len(self.rir_paths)))])
            mixture = apply_rir(mixture, rir)

        # Apply noise reduction preprocessing to a portion of synthetic samples if requested
        if self.mix_denoised_ratio > 0 and rng.random() < self.mix_denoised_ratio:
            mixture = apply_spectral_denoise(mixture)

        peak = max(float(np.max(np.abs(mixture))), 1e-6)
        if peak > 0.98:
            mixture = mixture * (0.98 / peak)
        return (
            torch.from_numpy(mixture.astype(np.float32)),
            torch.from_numpy(wake.astype(np.float32)),
            torch.from_numpy(clean.astype(np.float32)),
        )


def speaker_split(wav_root, dev_speakers):
    speakers = scan_wavs(wav_root)
    usable = {speaker: paths for speaker, paths in speakers.items() if len(paths) >= 2}
    ids = sorted(usable)
    if len(ids) <= dev_speakers:
        raise ValueError("Need more usable speakers than --dev-speakers")
    dev_ids = ids[-dev_speakers:]
    train_ids = ids[:-dev_speakers]
    return (
        {speaker: usable[speaker] for speaker in train_ids},
        {speaker: usable[speaker] for speaker in dev_ids},
        train_ids,
        dev_ids,
    )


def run_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total = 0.0
    for mixture, wake, clean in loader:
        mixture, wake, clean = mixture.to(device), wake.to(device), clean.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            enhanced, _ = model(mixture, wake)
            loss, _ = enhancement_loss(enhanced, clean, model.n_fft, model.hop_length)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for mixture, wake, clean in loader:
        mixture, wake, clean = mixture.to(device), wake.to(device), clean.to(device)
        enhanced, _ = model(mixture, wake)
        loss, _ = enhancement_loss(enhanced, clean, model.n_fft, model.hop_length)
        total += float(loss.detach().cpu())
    return total / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-root", default="data/public/aishell1/extracted/data_aishell/wav_expanded/train")
    parser.add_argument("--noise-root", required=True, help="Extracted MUSAN root")
    parser.add_argument("--rir-root", default=None, help="Optional extracted RIRS_NOISES root")
    parser.add_argument("--out", default="models/target_enhancer_musan_rirs.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--dev-steps", type=int, default=40)
    parser.add_argument("--segment-sec", type=float, default=3.0)
    parser.add_argument("--dev-speakers", type=int, default=3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mix-denoised-ratio", type=float, default=0.0,
                        help="Ratio of synthetic samples preprocessed with spectral denoise (0.0 to 1.0).")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_speakers, dev_speakers, train_ids, dev_ids = speaker_split(args.wav_root, args.dev_speakers)
    noise_paths = discover_audio_files(args.noise_root)
    rir_paths = discover_audio_files(args.rir_root)
    if not noise_paths:
        raise SystemExit("No MUSAN audio found. Complete MUSAN extraction before training.")
    print(f"device={device} train_speakers={len(train_ids)} dev_speakers={len(dev_ids)}")
    print(f"noise_assets={len(noise_paths)} rir_assets={len(rir_paths)}")
    print(f"mix_denoised_ratio={args.mix_denoised_ratio}")
    train_set = ExternalMixtureDataset(
        train_speakers, noise_paths, rir_paths,
        items=args.steps_per_epoch * args.batch_size,
        segment_sec=args.segment_sec, seed=args.seed,
        mix_denoised_ratio=args.mix_denoised_ratio,
    )
    dev_set = ExternalMixtureDataset(
        dev_speakers, noise_paths, rir_paths,
        items=args.dev_steps * args.batch_size,
        segment_sec=args.segment_sec, seed=args.seed + 100000,
        mix_denoised_ratio=args.mix_denoised_ratio,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, num_workers=args.num_workers)
    model = TargetEnhancer(channels=args.channels, blocks=args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_dev = float("inf")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device)
        dev_loss = evaluate(model, dev_loader, device)
        print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.5f} dev_loss={dev_loss:.5f}")
        if dev_loss < best_dev:
            best_dev = dev_loss
            payload = {
                "model_config": model.config(),
                "model_state": model.state_dict(),
                "epoch": epoch,
                "dev_loss": dev_loss,
                "source": "AISHELL-1 + MUSAN + RIRS_NOISES external data only",
                "train_speakers": train_ids,
                "dev_speakers": dev_ids,
                "args": vars(args),
            }
            torch.save(payload, out)
            out.with_suffix(".json").write_text(json.dumps({
                key: value for key, value in payload.items() if key != "model_state"
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"saved={out}")


if __name__ == "__main__":
    main()
