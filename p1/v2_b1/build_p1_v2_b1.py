#!/usr/bin/env python3
"""Build the frozen P1 -> P2 v2_b1 handoff.

The builder is deterministic, restartable, and non-destructive to source data.
It creates 100k train, 10k dev, 2k D_single, and 4k D_overlap samples
from already speaker-disjoint P1 split manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


SR = 16000
N = 57600
ENROLL_N = 48000
GLOBAL_SEED = 20260723
SCHEMA_VERSION = "p1_to_p2.v1"
GENERATOR_VERSION = "p1_v2_b1_builder.v1.0.4"
VERSION = "v2_b1"
EXPECTED = {"train": 100000, "dev": 10000, "D_single": 2000, "D_overlap": 4000}
SOURCE_ROOT: Path | None = None
STAGE_ROOT: Path | None = None
SPEECH: dict[str, list[dict[str, Any]]] = {}
SPEAKERS: dict[str, dict[str, list[int]]] = {}
NOISES: dict[str, list[dict[str, Any]]] = {}
RIRS: dict[str, list[dict[str, Any]]] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def derived_seed(sample_id: str) -> int:
    text = f"sample:{sample_id}:{GLOBAL_SEED}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"invalid JSON {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row is not an object {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def safe_relative(value: str) -> None:
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"unsafe relative path: {value}")


def source_path(row: dict[str, Any], kind: str) -> Path:
    assert SOURCE_ROOT is not None
    if kind == "speech":
        value = str(row["audio_path"])
        parts = PureWindowsPath(value).parts
        lower = [x.lower() for x in parts]
        try:
            pos = lower.index("data_aishell")
        except ValueError as exc:
            raise ValueError(f"cannot map AISHELL path: {value}") from exc
        return SOURCE_ROOT / "data" / "aishell1" / Path(*parts[pos:])
    rel = str(row["relative_path"]).replace("\\", "/").lstrip("/")
    if kind == "noise":
        return SOURCE_ROOT / "data" / "musan" / "musan" / rel
    if kind == "rir":
        return SOURCE_ROOT / "data" / "rirs_noises" / "RIRS_NOISES" / rel
    raise ValueError(f"unknown source kind={kind}")


@lru_cache(maxsize=96)
def read_audio_cached(path_text: str) -> tuple[np.ndarray, int]:
    path = Path(path_text)
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1).astype(np.float32, copy=False)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        audio = resample_poly(audio, SR // g, int(sr) // g).astype(np.float32)
        sr = SR
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"invalid audio: {path}")
    return audio, sr


@lru_cache(maxsize=512)
def source_hash_cached(path_text: str) -> str:
    return sha256_file(Path(path_text))


def crop_or_pad(audio: np.ndarray, length: int, rng: random.Random) -> tuple[np.ndarray, int, int]:
    if len(audio) >= length:
        start = rng.randint(0, len(audio) - length)
        return audio[start : start + length].copy(), start, length
    out = np.zeros(length, dtype=np.float32)
    out[: len(audio)] = audio
    return out, 0, len(audio)


def frame_vad(audio: np.ndarray) -> np.ndarray:
    """Exact P1 clean-source VAD: 25 ms, 10 ms, -40 dB, run cleanup."""
    frame, hop = 400, 160
    starts = np.arange(0, len(audio), hop, dtype=np.int64)
    rms_values = np.empty(len(starts), dtype=np.float64)
    for i, start in enumerate(starts):
        chunk = audio[start : min(start + frame, len(audio))]
        rms_values[i] = math.sqrt(float(np.mean(np.square(chunk, dtype=np.float64))) + 1e-20)
    maximum = float(rms_values.max(initial=0.0))
    if maximum <= 1e-10:
        return np.zeros(len(audio), dtype=np.float32)
    active = rms_values >= maximum * 0.01
    # Remove active runs shorter than 3 frames.
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i + 1
        while j < len(active) and active[j]:
            j += 1
        if j - i < 3:
            active[i:j] = False
        i = j
    # Fill internal inactive gaps of at most 2 frames.
    i = 0
    while i < len(active):
        if active[i]:
            i += 1
            continue
        j = i + 1
        while j < len(active) and not active[j]:
            j += 1
        if i > 0 and j < len(active) and j - i <= 2:
            active[i:j] = True
        i = j
    mask = np.zeros(len(audio), dtype=np.float32)
    for flag, start in zip(active, starts):
        if flag:
            mask[start : min(start + frame, len(audio))] = 1.0
    return mask


def align_interferer(raw: np.ndarray, target_mask: np.ndarray, requested: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    raw_mask = frame_vad(raw)
    if not np.any(raw_mask):
        raise ValueError("interferer has empty clean-source VAD")
    target_active = int(np.count_nonzero(target_mask > 0.5))
    if target_active == 0:
        raise ValueError("target has empty clean-source VAD")
    correlation = fftconvolve(
        (target_mask > 0.5).astype(np.float32),
        (raw_mask > 0.5).astype(np.float32)[::-1],
        mode="full",
    )
    ratios = np.clip(correlation / target_active, 0.0, 1.0)
    best_index = int(np.argmin(np.abs(ratios - requested)))
    measured = float(ratios[best_index])
    if abs(measured - requested) > 0.01:
        raise ValueError(f"cannot attain requested overlap: requested={requested:.6f}, best={measured:.6f}")
    lag = best_index - (len(raw) - 1)
    output = np.zeros(N, dtype=np.float32)
    output_mask = np.zeros(N, dtype=np.float32)
    out_start = max(0, lag)
    raw_start = max(0, -lag)
    length = min(N - out_start, len(raw) - raw_start)
    if length <= 0:
        raise ValueError("invalid interferer alignment")
    output[out_start : out_start + length] = raw[raw_start : raw_start + length]
    output_mask[out_start : out_start + length] = raw_mask[raw_start : raw_start + length]
    both = np.count_nonzero((target_mask > 0.5) & (output_mask > 0.5))
    exact = float(both / target_active)
    if abs(exact - requested) > 0.01:
        raise ValueError(f"overlap reconstruction mismatch: requested={requested}, measured={exact}")
    return output, output_mask, exact, lag


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> tuple[np.ndarray, int]:
    norm = float(np.linalg.norm(rir.astype(np.float64)))
    if norm <= 1e-8:
        raise ValueError("zero-energy RIR")
    normalized = (rir / (norm + 1e-8)).astype(np.float32)
    peak = int(np.argmax(np.abs(normalized)))
    full = fftconvolve(audio, normalized, mode="full").astype(np.float32)
    aligned = full[peak : peak + N]
    if len(aligned) < N:
        aligned = np.pad(aligned, (0, N - len(aligned)))
    return aligned.astype(np.float32, copy=False), peak


def power_on_mask(audio: np.ndarray, mask: np.ndarray) -> float:
    selected = audio[mask > 0.5]
    if len(selected) == 0:
        return 0.0
    return float(np.mean(np.square(selected, dtype=np.float64)))


def select_sir(rng: random.Random) -> float:
    u = rng.random()
    ranges = [(-5.0, -3.0, 0.25), (-3.0, -1.0, 0.20), (-1.0, 1.0, 0.20), (1.0, 3.0, 0.20), (3.0, 5.0, 0.15)]
    cumulative = 0.0
    for low, high, weight in ranges:
        cumulative += weight
        if u <= cumulative:
            return round(rng.uniform(low, high), 4)
    return 5.0


def select_snr(rng: random.Random) -> float | None:
    u = rng.random()
    ranges: list[tuple[float | None, float | None, float]] = [
        (None, None, 0.15), (-5.0, -3.0, 0.15), (-3.0, -1.0, 0.20),
        (-1.0, 1.0, 0.20), (1.0, 3.0, 0.15), (3.0, 5.0, 0.15),
    ]
    cumulative = 0.0
    for low, high, weight in ranges:
        cumulative += weight
        if u <= cumulative:
            if low is None:
                return None
            return round(rng.uniform(low, high), 4)
    return 5.0


def task_spec(split: str, index: int) -> tuple[str, float, float | None]:
    rng = random.Random(derived_seed(f"{split}:{index}"))
    if split == "D_single":
        return "single", 0.0, None
    if split == "D_overlap":
        bucket = index % 5
        if bucket == 0:
            return "overlap_0_25", rng.uniform(0.05, 0.25), None
        if bucket == 1:
            return "overlap_25_75", rng.uniform(0.25, 0.75), None
        if bucket == 2:
            return "overlap_75_100", rng.uniform(0.75, 0.95), None
        if bucket == 3:
            return "overlap_100", 1.0, None
        return "overlap_100_sir_minus5", 1.0, -5.0
    total = EXPECTED[split]
    position = index % total
    if position < int(total * 0.35):
        return "single", 0.0, None
    if position < int(total * 0.70):
        return "overlap_partial", rng.uniform(0.25, 0.75), None
    if position < int(total * 0.90):
        return "overlap_high", rng.uniform(0.75, 0.95), None
    return "overlap_100", 1.0, None


def split_assets(split: str) -> str:
    if split == "train":
        return "train"
    if split == "dev":
        return "dev"
    return "holdout"


def choose_sources(split: str, index: int, scenario: str, requested_overlap: float, rng: random.Random) -> dict[str, Any]:
    asset_split = split_assets(split)
    speech = SPEECH[asset_split]
    by_speaker = SPEAKERS[asset_split]
    speaker_names = sorted(by_speaker)
    target_speaker = speaker_names[(index + rng.randrange(len(speaker_names))) % len(speaker_names)]
    candidates = by_speaker[target_speaker]
    if len(candidates) < 2:
        raise ValueError(f"speaker has fewer than two utterances: {target_speaker}")
    target_idx = candidates[rng.randrange(len(candidates))]
    enroll_choices = [x for x in candidates if x != target_idx]
    enroll_idx = enroll_choices[rng.randrange(len(enroll_choices))]
    result: dict[str, Any] = {
        "target_row": speech[target_idx],
        "enroll_row": speech[enroll_idx],
        "interferer_row": None,
    }
    if scenario != "single":
        other_speakers = [x for x in speaker_names if x != target_speaker]
        # Deterministic retry selection is performed later if overlap is unattainable.
        result["interferer_speakers"] = other_speakers
    result["noise_row"] = NOISES[asset_split][rng.randrange(len(NOISES[asset_split]))]
    result["target_rir"] = RIRS[asset_split][rng.randrange(len(RIRS[asset_split]))]
    result["interferer_rir"] = RIRS[asset_split][rng.randrange(len(RIRS[asset_split]))]
    result["noise_rir"] = RIRS[asset_split][rng.randrange(len(RIRS[asset_split]))]
    return result


def atomic_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    sf.write(str(temp), audio.astype(np.float32), SR, format="WAV", subtype="FLOAT")
    os.replace(temp, path)


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    np.save(temp, array.astype(np.float32), allow_pickle=False)
    os.replace(temp, path)


def build_one(task: tuple[str, int]) -> dict[str, Any]:
    assert STAGE_ROOT is not None
    split, index = task
    sample_id = f"tse_{split.lower()}_{index + 1:08d}"
    seed = derived_seed(sample_id)
    rng = random.Random(seed)
    scenario, requested_overlap, forced_sir = task_spec(split, index)
    requested_overlap = round(float(requested_overlap), 6)
    selected = choose_sources(split, index, scenario, requested_overlap, rng)

    target_path = source_path(selected["target_row"], "speech")
    enroll_path = source_path(selected["enroll_row"], "speech")
    if target_path == enroll_path:
        raise ValueError("enrollment equals target")
    target_raw, _ = read_audio_cached(str(target_path))
    enroll_raw, _ = read_audio_cached(str(enroll_path))
    clean_target, target_crop, valid_samples = crop_or_pad(target_raw, N, rng)
    target_mask = frame_vad(clean_target)
    if not np.any(target_mask):
        raise ValueError("target has empty activity mask")
    enroll, enroll_crop, _ = crop_or_pad(enroll_raw, ENROLL_N, rng)

    target_rir_path = source_path(selected["target_rir"], "rir")
    target_rir, _ = read_audio_cached(str(target_rir_path))
    target, target_rir_peak = apply_rir(clean_target, target_rir)

    clean_interferer = np.zeros(N, dtype=np.float32)
    interferer_mask = np.zeros(N, dtype=np.float32)
    interferer = np.zeros(N, dtype=np.float32)
    interferer_row: dict[str, Any] | None = None
    interferer_path: Path | None = None
    interferer_rir_path: Path | None = None
    interferer_rir_peak: int | None = None
    interferer_lag: int | None = None
    measured_overlap = 0.0
    requested_sir: float | None = None
    measured_sir: float | None = None
    interferer_gain: float | None = None

    if scenario != "single":
        asset_split = split_assets(split)
        speech = SPEECH[asset_split]
        speaker_choices = selected["interferer_speakers"]
        last_error: Exception | None = None
        for _attempt in range(32):
            speaker = speaker_choices[rng.randrange(len(speaker_choices))]
            row_index = SPEAKERS[asset_split][speaker][rng.randrange(len(SPEAKERS[asset_split][speaker]))]
            candidate_row = speech[row_index]
            candidate_path = source_path(candidate_row, "speech")
            try:
                candidate_raw, _ = read_audio_cached(str(candidate_path))
                candidate, candidate_mask, overlap, lag = align_interferer(candidate_raw, target_mask, requested_overlap)
                clean_interferer = candidate
                interferer_mask = candidate_mask
                measured_overlap = overlap
                interferer_lag = lag
                interferer_row = candidate_row
                interferer_path = candidate_path
                break
            except Exception as exc:
                last_error = exc
        if interferer_row is None or interferer_path is None:
            raise ValueError(f"failed to find overlap candidate after 32 attempts: {last_error}")
        interferer_rir_path = source_path(selected["interferer_rir"], "rir")
        interferer_rir, _ = read_audio_cached(str(interferer_rir_path))
        interferer, interferer_rir_peak = apply_rir(clean_interferer, interferer_rir)
        omega = (target_mask > 0.5) & (interferer_mask > 0.5)
        pt = power_on_mask(target, omega.astype(np.float32))
        pi = power_on_mask(interferer, omega.astype(np.float32))
        if pt <= 1e-12 or pi <= 1e-12:
            raise ValueError("invalid simultaneous activity power")
        requested_sir = float(forced_sir if forced_sir is not None else select_sir(rng))
        interferer_gain = math.sqrt(pt / (pi * (10.0 ** (requested_sir / 10.0))))
        interferer = (interferer * interferer_gain).astype(np.float32)
        measured_sir = 10.0 * math.log10(pt / (power_on_mask(interferer, omega.astype(np.float32)) + 1e-20))

    requested_snr = select_snr(rng)
    noise_path: Path | None = None
    noise_rir_path: Path | None = None
    noise_rir_peak: int | None = None
    noise_gain: float | None = None
    measured_snr: float | None = None
    noise = np.zeros(N, dtype=np.float32)
    if requested_snr is not None:
        pt_noise = power_on_mask(target, target_mask)
        if pt_noise <= 1e-12:
            raise ValueError("invalid target activity power for noise scaling")
        asset_split = split_assets(split)
        last_noise_error: Exception | None = None
        for noise_attempt in range(32):
            noise_row = selected["noise_row"] if noise_attempt == 0 else NOISES[asset_split][rng.randrange(len(NOISES[asset_split]))]
            noise_rir_row = selected["noise_rir"] if noise_attempt == 0 else RIRS[asset_split][rng.randrange(len(RIRS[asset_split]))]
            try:
                candidate_noise_path = source_path(noise_row, "noise")
                noise_raw, _ = read_audio_cached(str(candidate_noise_path))
                clean_noise, _, _ = crop_or_pad(noise_raw, N, rng)
                candidate_noise_rir_path = source_path(noise_rir_row, "rir")
                noise_rir, _ = read_audio_cached(str(candidate_noise_rir_path))
                candidate_noise, candidate_noise_rir_peak = apply_rir(clean_noise, noise_rir)
                pn = power_on_mask(candidate_noise, target_mask)
                if pn <= 1e-12:
                    raise ValueError("noise crop has zero activity power")
                noise_path = candidate_noise_path
                noise_rir_path = candidate_noise_rir_path
                noise = candidate_noise
                noise_rir_peak = candidate_noise_rir_peak
                break
            except Exception as exc:
                last_noise_error = exc
        if noise_path is None or noise_rir_path is None:
            raise ValueError(f"failed to find usable noise after 32 attempts: {last_noise_error}")
        pn = power_on_mask(noise, target_mask)
        noise_gain = math.sqrt(pt_noise / (pn * (10.0 ** (requested_snr / 10.0))))
        noise = (noise * noise_gain).astype(np.float32)
        measured_snr = 10.0 * math.log10(pt_noise / (power_on_mask(noise, target_mask) + 1e-20))

    mixture = (target + interferer + noise).astype(np.float32)
    peak = max(float(np.max(np.abs(x))) for x in (mixture, target, interferer, noise))
    common_scale = min(1.0, 0.99 / max(peak, 1e-20))
    mixture *= common_scale
    target *= common_scale
    interferer *= common_scale
    noise *= common_scale
    if not all(np.isfinite(x).all() for x in (mixture, target, interferer, noise, enroll)):
        raise ValueError("non-finite output")
    if float(np.max(np.abs(mixture))) > 0.990001:
        raise ValueError("mixture clipping")

    mix_rel = f"audio/mixture/{sample_id}.wav"
    enroll_rel = f"audio/enroll/{sample_id}.wav"
    target_rel = f"audio/target/{sample_id}.wav"
    int_rel = None if scenario == "single" else f"audio/interferer/{sample_id}.wav"
    mask_rel = f"masks/{sample_id}.npy"
    for rel in [mix_rel, enroll_rel, target_rel, mask_rel] + ([] if int_rel is None else [int_rel]):
        safe_relative(rel)
    atomic_wav(STAGE_ROOT / mix_rel, mixture)
    atomic_wav(STAGE_ROOT / enroll_rel, enroll)
    atomic_wav(STAGE_ROOT / target_rel, target)
    if int_rel is not None:
        atomic_wav(STAGE_ROOT / int_rel, interferer)
    atomic_npy(STAGE_ROOT / mask_rel, target_mask)

    paths = {
        "target": target_path,
        "enroll": enroll_path,
        "interferer": interferer_path,
        "noise": noise_path,
        "target_rir": target_rir_path,
        "interferer_rir": interferer_rir_path,
        "noise_rir": noise_rir_path,
    }
    source_hashes = {key: (None if value is None else source_hash_cached(str(value))) for key, value in paths.items()}
    output_hashes = {
        "mixture_wav": sha256_file(STAGE_ROOT / mix_rel),
        "enroll_wav": sha256_file(STAGE_ROOT / enroll_rel),
        "target_wav": sha256_file(STAGE_ROOT / target_rel),
        "interferer_wav": None if int_rel is None else sha256_file(STAGE_ROOT / int_rel),
        "activity_mask": sha256_file(STAGE_ROOT / mask_rel),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "split": split,
        "scenario": scenario,
        "target_present": True,
        "sample_rate": SR,
        "num_samples": N,
        "valid_samples": int(valid_samples),
        "mixture_wav": mix_rel,
        "enroll_wav": enroll_rel,
        "target_wav": target_rel,
        "interferer_wav": int_rel,
        "activity_mask": mask_rel,
        "target_speaker": str(selected["target_row"]["speaker_id"]),
        "interferer_speaker": None if interferer_row is None else str(interferer_row["speaker_id"]),
        "enroll_utt": str(selected["enroll_row"]["utterance_id"]),
        "target_utt": str(selected["target_row"]["utterance_id"]),
        "interferer_utt": None if interferer_row is None else str(interferer_row["utterance_id"]),
        "requested_sir_db": None if requested_sir is None else round(requested_sir, 4),
        "measured_sir_db": None if measured_sir is None else round(measured_sir, 4),
        "requested_snr_db": None if requested_snr is None else round(float(requested_snr), 4),
        "measured_snr_db": None if measured_snr is None else round(measured_snr, 4),
        "requested_overlap": requested_overlap,
        "measured_overlap": round(measured_overlap, 6),
        "target_gain": 1.0,
        "interferer_gain": None if interferer_gain is None else round(interferer_gain, 8),
        "noise_gain": None if noise_gain is None else round(noise_gain, 8),
        "common_scale": round(common_scale, 8),
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
        "source_paths": {key: (None if value is None else value.relative_to(SOURCE_ROOT).as_posix()) for key, value in paths.items()},
        "processing": {
            "target_crop_start": target_crop,
            "enroll_crop_start": enroll_crop,
            "interferer_lag": interferer_lag,
            "target_rir_peak": target_rir_peak,
            "interferer_rir_peak": interferer_rir_peak,
            "noise_rir_peak": noise_rir_peak,
        },
    }


def schema_document() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "type": "object",
        "required": [
            "schema_version", "sample_id", "split", "scenario", "target_present",
            "sample_rate", "num_samples", "valid_samples", "mixture_wav", "enroll_wav",
            "target_wav", "interferer_wav", "activity_mask", "target_speaker",
            "interferer_speaker", "requested_sir_db", "measured_sir_db",
            "requested_snr_db", "measured_snr_db", "requested_overlap",
            "measured_overlap", "common_scale", "generator_version", "seed",
            "source_hashes", "output_hashes",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "sample_id": {"type": "string"},
            "split": {"enum": list(EXPECTED)},
            "scenario": {"type": "string"},
            "target_present": {"const": True},
            "sample_rate": {"const": SR},
            "num_samples": {"const": N},
            "valid_samples": {"type": "integer", "minimum": 1, "maximum": N},
            "mixture_wav": {"type": "string"},
            "enroll_wav": {"type": "string"},
            "target_wav": {"type": "string"},
            "interferer_wav": {"type": ["string", "null"]},
            "activity_mask": {"type": "string"},
            "target_speaker": {"type": "string"},
            "interferer_speaker": {"type": ["string", "null"]},
            "requested_sir_db": {"type": ["number", "null"]},
            "measured_sir_db": {"type": ["number", "null"]},
            "requested_snr_db": {"type": ["number", "null"]},
            "measured_snr_db": {"type": ["number", "null"]},
            "requested_overlap": {"type": "number", "minimum": 0, "maximum": 1},
            "measured_overlap": {"type": "number", "minimum": 0, "maximum": 1},
            "common_scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "generator_version": {"const": GENERATOR_VERSION},
            "seed": {"type": "integer"},
            "source_hashes": {"type": "object"},
            "output_hashes": {"type": "object"},
        },
        "additionalProperties": True,
    }


def load_assets(inputs: Path) -> None:
    global SPEECH, SPEAKERS, NOISES, RIRS
    split_dir = inputs / "manifests" / "splits"
    SPEECH = {name: read_jsonl(split_dir / f"speech_{name}.jsonl") for name in ("train", "dev", "holdout")}
    NOISES = {name: read_jsonl(split_dir / f"noise_{name}.jsonl") for name in ("train", "dev", "holdout")}
    RIRS = {name: read_jsonl(split_dir / f"rir_{name}.jsonl") for name in ("train", "dev", "holdout")}
    SPEAKERS = {}
    for split, rows in SPEECH.items():
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            grouped[str(row["speaker_id"])].append(index)
        SPEAKERS[split] = dict(grouped)


def init_worker(source_root: str, stage_root: str) -> None:
    global SOURCE_ROOT, STAGE_ROOT
    SOURCE_ROOT = Path(source_root)
    STAGE_ROOT = Path(stage_root)


def write_readme(path: Path) -> None:
    path.write_text(
        f"""# P1 to P2 v2_b1

- Schema: `{SCHEMA_VERSION}`
- Generator: `{GENERATOR_VERSION}`
- Train: 100,000 PRESENT samples
- Dev: 10,000 PRESENT samples
- Frozen confirmation: D_single=2,000, D_overlap=4,000
- Audio: 16 kHz, mono, float32 WAV; primary window 3.6 seconds (57,600 samples)
- Enrollment: 3 seconds
- Activity masks are derived only from clean target audio using the frozen P1 VAD.
- Train/dev/holdout speakers are disjoint. Noise and RIR assets are path-hash partitioned.
- Paths are relative to this directory. Verify SHA256SUMS.txt before use.
- This handoff contains no ABSENT samples; ABSENT belongs to v3_absent_swap.
""",
        encoding="utf-8",
    )


def finalize(stage: Path, partial: Path, input_hashes: dict[str, str]) -> None:
    records = read_jsonl(partial)
    records.sort(key=lambda row: (list(EXPECTED).index(row["split"]), row["sample_id"]))
    counts = Counter(str(row["split"]) for row in records)
    if dict(counts) != EXPECTED:
        raise ValueError(f"row counts mismatch: {dict(counts)} != {EXPECTED}")
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("duplicate sample_id")

    manifests = stage / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    combined = stage / "manifest.jsonl"
    handles = {
        "train": (manifests / "tse_train_present.jsonl").open("w", encoding="utf-8", newline="\n"),
        "dev": (manifests / "tse_dev.jsonl").open("w", encoding="utf-8", newline="\n"),
        "D_single": (manifests / "D_single.jsonl").open("w", encoding="utf-8", newline="\n"),
        "D_overlap": (manifests / "D_overlap.jsonl").open("w", encoding="utf-8", newline="\n"),
    }
    speaker_sets: dict[str, set[str]] = defaultdict(set)
    source_sets: dict[str, set[str]] = defaultdict(set)
    noise_sets: dict[str, set[str]] = defaultdict(set)
    rir_sets: dict[str, set[str]] = defaultdict(set)
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    overlap_failures = 0
    ratio_failures = 0
    with combined.open("w", encoding="utf-8", newline="\n") as all_handle:
        for row in records:
            raw = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            all_handle.write(raw)
            handles[row["split"]].write(raw)
            asset_split = split_assets(row["split"])
            speaker_sets[asset_split].add(row["target_speaker"])
            if row["interferer_speaker"]:
                speaker_sets[asset_split].add(row["interferer_speaker"])
            for role, digest in row["source_hashes"].items():
                if not digest:
                    continue
                if role == "noise":
                    noise_sets[asset_split].add(digest)
                elif role.endswith("rir"):
                    rir_sets[asset_split].add(digest)
                else:
                    source_sets[asset_split].add(digest)
            scenario_counts[row["split"]][row["scenario"]] += 1
            if abs(float(row["requested_overlap"]) - float(row["measured_overlap"])) > 0.01:
                overlap_failures += 1
            if row["requested_sir_db"] is not None and abs(float(row["requested_sir_db"]) - float(row["measured_sir_db"])) > 0.1:
                ratio_failures += 1
            if row["requested_snr_db"] is not None and abs(float(row["requested_snr_db"]) - float(row["measured_snr_db"])) > 0.1:
                ratio_failures += 1
    for handle in handles.values():
        handle.close()

    def intersections(values: dict[str, set[str]]) -> dict[str, int]:
        return {
            "train_dev": len(values["train"] & values["dev"]),
            "train_holdout": len(values["train"] & values["holdout"]),
            "dev_holdout": len(values["dev"] & values["holdout"]),
        }

    leakage = {
        "schema_version": "p1_v2_b1_leakage.v1",
        "speaker_intersections": intersections(speaker_sets),
        "audio_hash_intersections": intersections(source_sets),
        "noise_hash_intersections": intersections(noise_sets),
        "rir_hash_intersections": intersections(rir_sets),
        "enroll_equals_target": sum(row["enroll_utt"] == row["target_utt"] for row in records),
        "duplicate_sample_id": len(records) - len({row["sample_id"] for row in records}),
    }
    leakage["status"] = "PASS" if not any(
        value for key, value in leakage.items() if isinstance(value, dict) for value in value.values()
    ) and leakage["enroll_equals_target"] == 0 and leakage["duplicate_sample_id"] == 0 else "FAIL"
    write_json(stage / "reports" / "p1_v2_b1_leakage.json", leakage)
    acceptance = {
        "schema_version": "p1_v2_b1_acceptance.v1",
        "status": "PASS" if leakage["status"] == "PASS" and overlap_failures == 0 and ratio_failures == 0 else "FAIL",
        "version": VERSION,
        "generator_version": GENERATOR_VERSION,
        "rows": dict(counts),
        "scenario_counts": {key: dict(value) for key, value in scenario_counts.items()},
        "overlap_tolerance_failures": overlap_failures,
        "sir_snr_tolerance_failures": ratio_failures,
        "leakage_status": leakage["status"],
        "input_manifest_sha256": input_hashes,
        "generated_at": utc_now(),
    }
    write_json(stage / "reports" / "p1_v2_b1_acceptance.json", acceptance)
    if acceptance["status"] != "PASS":
        raise RuntimeError(f"formal acceptance failed: {acceptance}")
    write_readme(stage / "README.md")
    write_json(stage / "SCHEMA.json", schema_document())
    (stage / "VERSION").write_text(f"{VERSION}\nschema={SCHEMA_VERSION}\ngenerator={GENERATOR_VERSION}\n", encoding="utf-8")
    frozen = {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "frozen_at": utc_now(),
        "manifest_sha256": sha256_file(combined),
        "read_only": True,
    }
    write_json(stage / "FROZEN", frozen)

    # Build SHA256SUMS without re-reading hundreds of thousands of outputs.
    known: dict[str, str] = {}
    for row in records:
        for key, digest in row["output_hashes"].items():
            rel = row.get(key)
            if rel is not None and digest is not None:
                known[str(rel)] = str(digest)
    for path in stage.rglob("*"):
        if path.is_file() and path.name not in {"SHA256SUMS.txt", partial.name}:
            rel = path.relative_to(stage).as_posix()
            known.setdefault(rel, sha256_file(path))
    (stage / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {rel}\n" for rel, digest in sorted(known.items())),
        encoding="utf-8",
    )
    partial.rename(stage / "reports" / "manifest_generation_journal.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=max(1, min(18, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--preflight", action="store_true", help="build 2 records per formal split")
    args = parser.parse_args()
    output = args.output.resolve()
    stage = output if args.preflight else output.parent / f"{output.name}.building"
    error_report = output.parent / f"{output.name}_error.json"
    try:
        if output.exists() and not args.preflight:
            raise FileExistsError(f"refusing to overwrite frozen output: {output}")
        stage.mkdir(parents=True, exist_ok=True)
        load_assets(args.inputs.resolve())
        global SOURCE_ROOT, STAGE_ROOT
        SOURCE_ROOT = args.source_root.resolve()
        STAGE_ROOT = stage
        input_hashes = {
            path.name: sha256_file(path)
            for path in sorted((args.inputs / "manifests" / "splits").glob("*.jsonl"))
        }
        partial = stage / "manifest.partial.jsonl"
        completed: set[str] = set()
        if partial.exists():
            completed = {str(row["sample_id"]) for row in read_jsonl(partial)}
        tasks: list[tuple[str, int]] = []
        if args.preflight:
            # Exercise every formal scenario and all three asset partitions.
            planned = {
                "train": [0, 35000, 70000, 90000],
                "dev": [0, 3500, 7000, 9000],
                "D_single": [0],
                "D_overlap": [0, 1, 2, 3, 4],
            }
        else:
            planned = {split: list(range(total)) for split, total in EXPECTED.items()}
        for split, indices in planned.items():
            for index in indices:
                sample_id = f"tse_{split.lower()}_{index + 1:08d}"
                if sample_id not in completed:
                    tasks.append((split, index))
        print(f"P1_V2_B1_BUILD_BEGIN TASKS={len(tasks)} COMPLETED={len(completed)} WORKERS={args.workers}", flush=True)
        mode = "a" if partial.exists() else "w"
        context = mp.get_context("fork")
        with partial.open(mode, encoding="utf-8", newline="\n", buffering=1) as journal:
            with context.Pool(
                processes=max(1, args.workers),
                initializer=init_worker,
                initargs=(str(SOURCE_ROOT), str(STAGE_ROOT)),
            ) as pool:
                done = len(completed)
                for row in pool.imap_unordered(build_one, tasks, chunksize=1):
                    journal.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    done += 1
                    if done % 100 == 0 or args.preflight:
                        print(f"BUILD_PROGRESS={done}/{sum(len(x) for x in planned.values()) if args.preflight else sum(EXPECTED.values())}", flush=True)
        if args.preflight:
            print(f"P1_V2_B1_PREFLIGHT_STATUS=PASS ROWS={len(read_jsonl(partial))} OUTPUT={stage}")
            return 0
        finalize(stage, partial, input_hashes)
        stage.rename(output)
        print("P1_V2_B1_BUILD_STATUS=PASS")
        print(f"OUTPUT={output}")
        print(f"ROWS={sum(EXPECTED.values())}")
        print(f"MANIFEST_SHA256={sha256_file(output / 'manifest.jsonl')}")
        print(f"ACCEPTANCE_REPORT={output / 'reports' / 'p1_v2_b1_acceptance.json'}")
        return 0
    except Exception as exc:
        write_json(error_report, {
            "status": "FAIL", "time": utc_now(), "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(), "stage": str(stage),
        })
        print("P1_V2_B1_BUILD_STATUS=FAIL")
        print(f"ERROR={exc}")
        print(f"ERROR_REPORT={error_report}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
