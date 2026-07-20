# -*- coding: utf-8 -*-
"""
Build an external train/tuning set in the same shape as datasetA.

The generated files are for model/threshold tuning only. Do not build a phrase
bank or train a gate from datasetA labels when datasetA is the test set.

Default input is the local AISHELL-1 test subset already present in this
workspace. Output:
  data/external_train/
    pos/*.wav, neg/*.wav
    pos.jsonl, neg.jsonl
    phrase_bank.txt
"""
import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


SR = 16000

FIELD_ID = "id"
FIELD_WAKE_AUDIO = "唤醒音频"
FIELD_WAKE_TEXT = "唤醒文本"
FIELD_CMD_AUDIO = "识别音频"
FIELD_CMD_TEXT = "识别文本"


def read_transcripts(csv_path):
    texts = {}
    path = Path(csv_path)
    with open(path, encoding="utf-8") as f:
        if path.suffix.lower() == ".csv":
            for row in csv.reader(f):
                if not row or not row[0].endswith(".wav"):
                    continue
                texts[Path(row[0]).stem] = row[1]
        else:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    texts[parts[0]] = parts[1].replace(" ", "")
    return texts


def scan_wavs(wav_root):
    speakers = defaultdict(list)
    for path in Path(wav_root).rglob("*.wav"):
        match = re.search(r"(S\d{4})", path.stem)
        if match:
            speakers[match.group(1)].append(path)
    for items in speakers.values():
        items.sort()
    return speakers


def read_wav(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float32)
    if sr == SR:
        return x
    # Noise/RIR assets can use a different sample rate. Linear resampling is
    # sufficient for augmentation and avoids another runtime dependency.
    target_len = max(1, round(len(x) * SR / sr))
    source_positions = np.linspace(0, len(x) - 1, target_len)
    return np.interp(source_positions, np.arange(len(x)), x).astype(np.float32)


def write_wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak * 0.98
    sf.write(path, x.astype(np.float32), SR)


def mix_at_ratio(target, interferer, ratio_db):
    if len(interferer) < len(target):
        interferer = np.tile(interferer, len(target) // len(interferer) + 1)
    interferer = interferer[:len(target)]
    pt = np.mean(target ** 2) + 1e-12
    pi = np.mean(interferer ** 2) + 1e-12
    scale = np.sqrt(pt / (pi * 10 ** (ratio_db / 10.0)))
    return target + scale * interferer


def discover_audio_files(root):
    if not root:
        return []
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"Augmentation root does not exist: {root}")
    suffixes = {".wav", ".flac", ".ogg", ".mp3"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)


def apply_rir(audio, rir):
    """Convolve audio with an aligned, energy-normalized room impulse response."""
    if not len(rir):
        return audio
    direct = int(np.argmax(np.abs(rir)))
    rir = rir[direct:direct + int(0.8 * SR)]
    energy = float(np.sqrt(np.sum(rir ** 2)) + 1e-12)
    rir = rir / energy
    return np.convolve(audio, rir, mode="full")[:len(audio)].astype(np.float32)


def copy_wav(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args):
    out = Path(args.out)
    pos_dir = out / "pos"
    neg_dir = out / "neg"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    texts = read_transcripts(args.csv)
    speakers = scan_wavs(args.wav_root)
    noise_paths = discover_audio_files(getattr(args, "noise_root", None))
    rir_paths = discover_audio_files(getattr(args, "rir_root", None))
    source_blocks = 4 + int(bool(noise_paths)) + int(bool(rir_paths))
    usable = [
        spk for spk, wavs in sorted(speakers.items())
        if len(wavs) >= args.enroll_count + args.trials_per_speaker * source_blocks
    ]
    need = args.target_speakers + args.interferer_speakers
    if len(usable) < need:
        raise SystemExit(
            f"Need at least {need} usable speakers with "
            f"{args.enroll_count + args.trials_per_speaker * source_blocks} wavs each; "
            f"found {len(usable)}"
        )

    targets = usable[:args.target_speakers]
    interferers = usable[args.target_speakers:need]
    rng = np.random.default_rng(args.seed)

    pos_rows = []
    neg_rows = []
    phrase_bank = []
    row_id = 0

    for idx, target_spk in enumerate(targets):
        target_wavs = speakers[target_spk]
        interferer_wavs = speakers[interferers[idx % len(interferers)]]
        enroll_audio = np.concatenate([read_wav(p) for p in target_wavs[:args.enroll_count]])

        def make_wake(split_dir, current_id):
            wake_rel = f"{split_dir.name}/kws_{current_id}.wav"
            write_wav(out / wake_rel, enroll_audio)
            return wake_rel

        pool = target_wavs[args.enroll_count:]
        for j in range(args.trials_per_speaker):
            src = pool[j]
            cmd_rel = f"pos/cmd_{row_id}.wav"
            copy_wav(src, out / cmd_rel)
            ref = texts.get(src.stem, "")
            if ref and ref not in phrase_bank:
                phrase_bank.append(ref)
            pos_rows.append({
                FIELD_ID: row_id,
                FIELD_WAKE_AUDIO: make_wake(pos_dir, row_id),
                FIELD_WAKE_TEXT: args.wake_text,
                FIELD_CMD_AUDIO: cmd_rel,
                FIELD_CMD_TEXT: ref,
            })
            row_id += 1

        for ratio_db, name in [(5, "overlap5"), (0, "overlap0")]:
            for j in range(args.trials_per_speaker):
                src = pool[args.trials_per_speaker + j]
                inter = interferer_wavs[j]
                cmd_rel = f"pos/cmd_{row_id}_{name}.wav"
                mixed = mix_at_ratio(read_wav(src), read_wav(inter), ratio_db)
                write_wav(out / cmd_rel, mixed)
                ref = texts.get(src.stem, "")
                if ref and ref not in phrase_bank:
                    phrase_bank.append(ref)
                pos_rows.append({
                    FIELD_ID: row_id,
                    FIELD_WAKE_AUDIO: make_wake(pos_dir, row_id),
                    FIELD_WAKE_TEXT: args.wake_text,
                    FIELD_CMD_AUDIO: cmd_rel,
                    FIELD_CMD_TEXT: ref,
                })
                row_id += 1

        for j in range(args.trials_per_speaker):
            src = pool[args.trials_per_speaker * 3 + j]
            target = read_wav(src)
            babble = np.zeros_like(target)
            picks = rng.choice(len(interferer_wavs), size=min(4, len(interferer_wavs)), replace=False)
            for pick in picks:
                noise = read_wav(interferer_wavs[int(pick)])
                if len(noise) < len(target):
                    noise = np.tile(noise, len(target) // len(noise) + 1)
                babble += noise[:len(target)]
            cmd_rel = f"pos/cmd_{row_id}_babble5.wav"
            write_wav(out / cmd_rel, mix_at_ratio(target, babble, 5))
            ref = texts.get(src.stem, "")
            if ref and ref not in phrase_bank:
                phrase_bank.append(ref)
            pos_rows.append({
                FIELD_ID: row_id,
                FIELD_WAKE_AUDIO: make_wake(pos_dir, row_id),
                FIELD_WAKE_TEXT: args.wake_text,
                FIELD_CMD_AUDIO: cmd_rel,
                FIELD_CMD_TEXT: ref,
            })
            row_id += 1

        next_block = 4
        if noise_paths:
            for j in range(args.trials_per_speaker):
                src = pool[args.trials_per_speaker * next_block + j]
                noise = read_wav(noise_paths[(idx * args.trials_per_speaker + j) % len(noise_paths)])
                cmd_rel = f"pos/cmd_{row_id}_noise{args.noise_snr_db:g}.wav"
                write_wav(out / cmd_rel, mix_at_ratio(read_wav(src), noise, args.noise_snr_db))
                ref = texts.get(src.stem, "")
                if ref and ref not in phrase_bank:
                    phrase_bank.append(ref)
                pos_rows.append({
                    FIELD_ID: row_id,
                    FIELD_WAKE_AUDIO: make_wake(pos_dir, row_id),
                    FIELD_WAKE_TEXT: args.wake_text,
                    FIELD_CMD_AUDIO: cmd_rel,
                    FIELD_CMD_TEXT: ref,
                })
                row_id += 1
            next_block += 1

        if rir_paths:
            for j in range(args.trials_per_speaker):
                src = pool[args.trials_per_speaker * next_block + j]
                target = apply_rir(
                    read_wav(src),
                    read_wav(rir_paths[(idx * args.trials_per_speaker + j) % len(rir_paths)]),
                )
                if noise_paths:
                    noise = read_wav(noise_paths[(row_id + j) % len(noise_paths)])
                    target = mix_at_ratio(target, noise, args.reverb_noise_snr_db)
                    suffix = f"reverb_noise{args.reverb_noise_snr_db:g}"
                else:
                    suffix = "reverb"
                cmd_rel = f"pos/cmd_{row_id}_{suffix}.wav"
                write_wav(out / cmd_rel, target)
                ref = texts.get(src.stem, "")
                if ref and ref not in phrase_bank:
                    phrase_bank.append(ref)
                pos_rows.append({
                    FIELD_ID: row_id,
                    FIELD_WAKE_AUDIO: make_wake(pos_dir, row_id),
                    FIELD_WAKE_TEXT: args.wake_text,
                    FIELD_CMD_AUDIO: cmd_rel,
                    FIELD_CMD_TEXT: ref,
                })
                row_id += 1

        for j in range(args.trials_per_speaker * 2):
            src = interferer_wavs[args.enroll_count + j]
            cmd_rel = f"neg/cmd_{row_id}.wav"
            copy_wav(src, out / cmd_rel)
            neg_rows.append({
                FIELD_ID: row_id,
                FIELD_WAKE_AUDIO: make_wake(neg_dir, row_id),
                FIELD_WAKE_TEXT: args.wake_text,
                FIELD_CMD_AUDIO: cmd_rel,
                FIELD_CMD_TEXT: None,
            })
            row_id += 1

    write_jsonl(out / "pos.jsonl", pos_rows)
    write_jsonl(out / "neg.jsonl", neg_rows)
    with open(out / "phrase_bank.txt", "w", encoding="utf-8") as f:
        for phrase in phrase_bank:
            f.write(phrase + "\n")

    print(f"targets={targets}")
    print(f"interferers={interferers}")
    print(f"noise assets={len(noise_paths)}  RIR assets={len(rir_paths)}")
    print(f"wrote {len(pos_rows)} pos and {len(neg_rows)} neg rows to {out}")
    print(f"phrase bank: {out / 'phrase_bank.txt'} ({len(phrase_bank)} phrases)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-root", default="data/aishell_test")
    parser.add_argument("--csv", default="data/aishell1_test.csv")
    parser.add_argument("--out", default="data/external_train")
    parser.add_argument("--target-speakers", type=int, default=8)
    parser.add_argument("--interferer-speakers", type=int, default=8)
    parser.add_argument("--enroll-count", type=int, default=3)
    parser.add_argument("--trials-per-speaker", type=int, default=4)
    parser.add_argument("--wake-text", default="hi colmo")
    parser.add_argument("--noise-root", default=None,
                        help="Optional MUSAN audio root for noise/babble augmentation.")
    parser.add_argument("--rir-root", default=None,
                        help="Optional RIRS_NOISES root for far-field augmentation.")
    parser.add_argument("--noise-snr-db", type=float, default=5.0)
    parser.add_argument("--reverb-noise-snr-db", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
