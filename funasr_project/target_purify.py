# -*- coding: utf-8 -*-
"""
Lightweight target-speaker purification.

This is a training-free approximation of target speaker extraction:
  1. Extract the target speaker embedding from wake audio.
  2. Slide over the command audio with short overlapping windows.
  3. Keep/suppress windows by similarity to the target embedding.
  4. Write a softly masked waveform for downstream ASR.

It is intentionally optional. Use it for small-sample experiments before
enabling it in larger evaluation runs.
"""
import contextlib
import io
import os
import tempfile

import numpy as np
import soundfile as sf

from speaker_verify import cosine_sim, extract_embedding


SR = 16000


def extract_embedding_quiet(sv_model, wav_path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_embedding(sv_model, wav_path)


def read_wav(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr} != {SR}")
    return x.astype(np.float32)


def write_wav(path, x):
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 1.0:
        x = x / peak * 0.98
    sf.write(path, x, SR)


def _window_scores(sv_model, target_emb, x, win, hop, tmp_dir):
    scores = []
    windows = []
    if len(x) <= win:
        windows = [(0, len(x))]
    else:
        starts = list(range(0, max(1, len(x) - win + 1), hop))
        if starts[-1] + win < len(x):
            starts.append(max(0, len(x) - win))
        windows = [(s, min(len(x), s + win)) for s in starts]

    for idx, (s, e) in enumerate(windows):
        seg = x[s:e]
        if len(seg) < int(0.25 * SR) or float(np.sqrt(np.mean(seg ** 2) + 1e-9)) < 1e-4:
            scores.append(-1.0)
            continue
        seg_path = os.path.join(tmp_dir, f"seg_{idx:04d}.wav")
        write_wav(seg_path, seg)
        emb = extract_embedding_quiet(sv_model, seg_path)
        scores.append(cosine_sim(target_emb, emb))
    return windows, np.asarray(scores, dtype=np.float32)


def purify_audio(
    sv_model,
    wake_wav,
    cmd_wav,
    out_wav,
    win_sec=1.2,
    hop_sec=0.3,
    keep_ratio=0.45,
    min_sim=-0.05,
    floor_gain=0.03,
):
    """Write a target-speaker-enhanced waveform and return diagnostics.

    The method keeps the highest-similarity windows while strongly attenuating
    the rest. `floor_gain` avoids hard discontinuities and preserves a little
    acoustic context for VAD.
    """
    x = read_wav(cmd_wav)
    target_emb = extract_embedding_quiet(sv_model, wake_wav)
    win = max(int(win_sec * SR), int(0.4 * SR))
    hop = max(int(hop_sec * SR), int(0.1 * SR))

    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="purify_") as tmp_dir:
        windows, scores = _window_scores(sv_model, target_emb, x, win, hop, tmp_dir)

    valid = scores[scores > -0.5]
    if len(valid) == 0:
        write_wav(out_wav, np.zeros_like(x))
        return {"kept_ratio": 0.0, "threshold": None, "scores": scores.tolist()}

    quantile = max(0.0, min(1.0, 1.0 - keep_ratio))
    threshold = max(float(np.quantile(valid, quantile)), float(min_sim))
    keep = scores >= threshold

    # Soft overlap-add mask. Kept windows get full weight; rejected windows get
    # a small floor so VAD still sees a continuous recording without loud clutter.
    weight_sum = np.zeros(len(x), dtype=np.float32)
    mask_sum = np.zeros(len(x), dtype=np.float32)
    for (s, e), is_kept in zip(windows, keep):
        n = e - s
        if n <= 0:
            continue
        taper = np.hanning(n).astype(np.float32)
        if not np.any(taper):
            taper = np.ones(n, dtype=np.float32)
        gain = 1.0 if is_kept else floor_gain
        weight_sum[s:e] += taper
        mask_sum[s:e] += taper * gain
    mask = np.divide(mask_sum, weight_sum + 1e-8)
    y = x * mask
    write_wav(out_wav, y)
    return {
        "kept_ratio": float(np.mean(keep)),
        "threshold": threshold,
        "scores": [round(float(s), 4) for s in scores],
        "max_score": round(float(np.max(valid)), 4),
        "min_score": round(float(np.min(valid)), 4),
    }
