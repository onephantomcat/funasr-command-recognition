"""Evaluate DatasetA with the legacy dual-output TSE and three-state audit.

This runner is retained to re-evaluate the historical ``tse_dual_output_mvp``
checkpoint after the P3 ASR tuple-contract bug was discovered. It is not the
frozen P2 ``extract_target()`` integration; that path is handled by
``p2_tse_runtime.py`` and ``eval_datasetA.py --p2-tse-checkpoint``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from asr_demo import ASR_DIR, VAD_DIR, build_model, recognize
from cer import cer_stats, corpus_cer_stats
from p3_eval_contracts import (
    RecognitionOutcome,
    negative_is_rejected,
    recognize_safely,
)
from speaker_verify import build_sv_model, cosine_sim, extract_embedding
from three_state_audit import ThreeStateAudit
from tse_dual_output import SR, TSEDualOutputNet


EVALUATOR_CONTRACT = "p3_text_eval_v1-rc1"
TSE_BACKEND = "legacy_tse_dual_output_mvp"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_audio(path: str | Path) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float32)
    if not data.size:
        raise ValueError(f"empty audio: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"audio contains NaN/Inf: {path}")
    if sr == SR:
        return data
    target_len = max(1, round(len(data) * SR / sr))
    positions = np.linspace(0, max(0, len(data) - 1), target_len)
    return np.interp(positions, np.arange(len(data)), data).astype(np.float32)


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield row


def _model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("TSE checkpoint must be a state dict or checkpoint mapping")
    for key in ("model_state", "model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(isinstance(key, str) for key in checkpoint):
        return checkpoint
    raise KeyError("TSE checkpoint has no model_state/model/state_dict")


def load_tse_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"TSE checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    allowed = {key: config[key] for key in ("n_fft", "hop_length", "channels", "blocks") if key in config}
    model = TSEDualOutputNet(**allowed).to(device)
    model.load_state_dict(_model_state(checkpoint), strict=True)
    model.eval()
    return model, allowed


def _safe_name(value) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _validate_output(waveform: np.ndarray, expected_length: int, name: str):
    if waveform.ndim != 1 or len(waveform) != expected_length:
        raise RuntimeError(
            f"{name} shape contract failed: {waveform.shape}, expected ({expected_length},)"
        )
    if not np.isfinite(waveform).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def evaluate_row(
    *,
    row,
    split,
    index,
    root,
    cache_dir,
    tse_net,
    auditor,
    sv_model,
    asr_model,
    device,
):
    wake_path = root / row["唤醒音频"]
    mix_path = root / row["识别音频"]
    mix_wav = read_audio(mix_path)
    enroll_wav = read_audio(wake_path)

    mix_tensor = torch.from_numpy(mix_wav).unsqueeze(0).to(device)
    enroll_tensor = torch.from_numpy(enroll_wav).unsqueeze(0).to(device)
    with torch.no_grad():
        target_tensor, residual_tensor = tse_net.forward_audio(mix_tensor, enroll_tensor)

    target_wav = target_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    residual_wav = residual_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    _validate_output(target_wav, len(mix_wav), "target")
    _validate_output(residual_wav, len(mix_wav), "residual")

    sample_key = f"{split}_{index:05d}_{_safe_name(row.get('id', index))}"
    target_path = cache_dir / f"{sample_key}_target.wav"
    residual_path = cache_dir / f"{sample_key}_residual.wav"
    sf.write(target_path, target_wav, SR, subtype="FLOAT")
    sf.write(residual_path, residual_wav, SR, subtype="FLOAT")

    wake_embedding = extract_embedding(sv_model, str(wake_path))
    target_embedding = extract_embedding(sv_model, str(target_path))
    residual_embedding = extract_embedding(sv_model, str(residual_path))
    target_similarity = cosine_sim(wake_embedding, target_embedding)
    residual_similarity = cosine_sim(wake_embedding, residual_embedding)
    audit = auditor.audit(
        target_sim=target_similarity,
        residual_sim=residual_similarity,
    )

    if audit["emit_allowed"]:
        outcome = recognize_safely(recognize, asr_model, str(target_path))
    else:
        outcome = RecognitionOutcome(
            text="", elapsed_sec=0.0, status="SKIPPED_AUDIT"
        )

    return {
        "id": row.get("id", index),
        "sample_key": sample_key,
        "wake_audio": str(wake_path),
        "mixture_audio": str(mix_path),
        "target_audio": str(target_path),
        "residual_audio": str(residual_path),
        "hyp": outcome.text,
        "audit_state": audit["state"],
        "reason_code": audit["reason_code"],
        "emit_allowed": bool(audit["emit_allowed"]),
        "target_sim": round(float(target_similarity), 6),
        "residual_sim": round(float(residual_similarity), 6),
        "target_peak": round(float(np.max(np.abs(target_wav))), 6),
        "residual_peak": round(float(np.max(np.abs(residual_wav))), 6),
        "asr_status": outcome.status,
        "asr_error": outcome.error,
        "asr_latency_sec": round(outcome.elapsed_sec, 6),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/datasetA")
    parser.add_argument("--tse-model", default="models/tse_dual_output_mvp.pt")
    parser.add_argument("--present-thresh", type=float, default=0.30)
    parser.add_argument("--empty-thresh", type=float, default=0.18)
    parser.add_argument("--out", default="data/datasetA/eval_tse_dual_output_fixed.json")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    root = Path(args.root)
    pos_manifest = root / "pos.jsonl"
    neg_manifest = root / "neg.jsonl"
    pos_rows = list(read_jsonl(pos_manifest))
    neg_rows = list(read_jsonl(neg_manifest))
    if args.offset:
        pos_rows = pos_rows[args.offset:]
        neg_rows = neg_rows[args.offset:]
    if args.limit is not None:
        pos_rows = pos_rows[:args.limit]
        neg_rows = neg_rows[:args.limit]

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    checkpoint_path = Path(args.tse_model)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    print(f"Loading legacy TSE model {checkpoint_path} on {device}...")
    tse_net, tse_config = load_tse_model(checkpoint_path, device)

    auditor = ThreeStateAudit(
        present_threshold=args.present_thresh,
        empty_threshold=args.empty_thresh,
    )
    print("Loading Speaker Verification model...")
    sv_model = build_sv_model(device=device_name)
    print("Loading Paraformer ASR model...")
    asr_model = build_model(with_punc=False, device=device_name)

    cache_dir = Path(args.cache_dir) if args.cache_dir else root / "tse_dual_cache_fixed"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos_details = []
    neg_details = []
    pairs = []
    pos_accepted = 0
    neg_rejected = 0
    asr_errors = 0
    asr_latencies = []
    started_at = time.time()

    print(f"\nRunning fixed legacy-TSE evaluation on {len(pos_rows)} positive samples...")
    for index, row in enumerate(pos_rows, 1):
        detail = evaluate_row(
            row=row,
            split="pos",
            index=index,
            root=root,
            cache_dir=cache_dir,
            tse_net=tse_net,
            auditor=auditor,
            sv_model=sv_model,
            asr_model=asr_model,
            device=device,
        )
        ref_text = row.get("识别文本")
        stats = cer_stats(ref_text, detail["hyp"])
        detail.update({"ref": ref_text, **stats.to_dict()})
        pos_details.append(detail)
        pairs.append((ref_text, detail["hyp"]))
        pos_accepted += int(detail["emit_allowed"])
        asr_errors += int(detail["asr_status"] == "ERROR")
        if detail["asr_status"] == "OK":
            asr_latencies.append(detail["asr_latency_sec"])
        if index % 100 == 0 or index == len(pos_rows):
            print(f"  Pos {index}/{len(pos_rows)}: emit={pos_accepted}/{index}")

    corpus = corpus_cer_stats(pairs)
    pos_sentence_avg_cer = float(np.mean([item["cer"] for item in pos_details]))

    print(f"\nRunning fixed legacy-TSE evaluation on {len(neg_rows)} negative samples...")
    for index, row in enumerate(neg_rows, 1):
        detail = evaluate_row(
            row=row,
            split="neg",
            index=index,
            root=root,
            cache_dir=cache_dir,
            tse_net=tse_net,
            auditor=auditor,
            sv_model=sv_model,
            asr_model=asr_model,
            device=device,
        )
        outcome = RecognitionOutcome(
            text=detail["hyp"],
            elapsed_sec=detail["asr_latency_sec"],
            status=detail["asr_status"],
            error=detail["asr_error"],
        )
        rejected = negative_is_rejected(
            emit_allowed=detail["emit_allowed"], outcome=outcome
        )
        detail["rejected"] = rejected
        neg_details.append(detail)
        neg_rejected += int(rejected)
        asr_errors += int(detail["asr_status"] == "ERROR")
        if detail["asr_status"] == "OK":
            asr_latencies.append(detail["asr_latency_sec"])
        if index % 100 == 0 or index == len(neg_rows):
            print(f"  Neg {index}/{len(neg_rows)}: rejected={neg_rejected}/{index}")

    elapsed = time.time() - started_at
    pos_accept_rate = pos_accepted / max(1, len(pos_rows))
    neg_rr = neg_rejected / max(1, len(neg_rows))
    score_80 = 80.0 * (0.6 * max(0.0, 1.0 - corpus.value) + 0.4 * neg_rr)
    latency_p50 = float(np.percentile(asr_latencies, 50)) if asr_latencies else None
    latency_p95 = float(np.percentile(asr_latencies, 95)) if asr_latencies else None

    summary = {
        "evaluator_contract": EVALUATOR_CONTRACT,
        "result_valid": asr_errors == 0,
        "invalid_reason": None if asr_errors == 0 else "ASR_RUNTIME_ERROR",
        "tse_backend": TSE_BACKEND,
        "tse_checkpoint": str(checkpoint_path),
        "tse_checkpoint_sha256": checkpoint_sha256,
        "tse_model_config": tse_config,
        "dataset_hashes": {
            "pos_jsonl_sha256": sha256_file(pos_manifest),
            "neg_jsonl_sha256": sha256_file(neg_manifest),
        },
        "asr_model_id": ASR_DIR,
        "asr_vad_model_id": VAD_DIR,
        "asr_with_punctuation": False,
        "device": device_name,
        "thresholds": {
            "present": args.present_thresh,
            "empty": args.empty_thresh,
        },
        "positive_accept_rate": round(pos_accept_rate, 6),
        "positive_sentence_avg_cer": round(pos_sentence_avg_cer, 6),
        "positive_corpus_cer": round(corpus.value, 6),
        "substitutions": corpus.substitutions,
        "deletions": corpus.deletions,
        "insertions": corpus.insertions,
        "errors": corpus.errors,
        "positive_ref_chars": corpus.reference_chars,
        "negative_rr": round(neg_rr, 6),
        "score_80": round(score_80, 6),
        "asr_errors": asr_errors,
        "asr_latency_sec_p50": None if latency_p50 is None else round(latency_p50, 6),
        "asr_latency_sec_p95": None if latency_p95 is None else round(latency_p95, 6),
        "elapsed_sec": round(elapsed, 3),
        "pos_count": len(pos_rows),
        "neg_count": len(neg_rows),
    }

    output = {
        "summary": summary,
        "pos_details": pos_details,
        "neg_details": neg_details,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    os.replace(temporary_path, output_path)

    print("\n================ Fixed TSE Evaluation Summary ================")
    print(f"  Result valid:              {summary['result_valid']}")
    print(f"  Positive Corpus CER:       {corpus.value * 100:.2f}%")
    print(f"  S/D/I/N:                   {corpus.substitutions}/{corpus.deletions}/{corpus.insertions}/{corpus.reference_chars}")
    print(f"  Positive Accept Rate:      {pos_accept_rate * 100:.2f}%")
    print(f"  Negative Rejection Rate:   {neg_rr * 100:.2f}%")
    print(f"  ASR errors:                {asr_errors}")
    print(f"  Report:                    {output_path}")


if __name__ == "__main__":
    main()
