"""Evaluate DatasetA with TSE Dual-Output (target + residual) Separation & Three-State Audit.

Pipeline:
1. Extract target_audio and residual_audio using TSEDualOutputNet.
2. Compute target_sim (target vs wake) and residual_sim (residual vs wake).
3. Conduct Three-State Audit (PRESENT / EMPTY / GRAY).
4. Run Paraformer ASR on purified target_audio.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from asr_demo import build_model, recognize
from cer import cer, corpus_cer
from speaker_verify import build_sv_model, cosine_sim, extract_embedding
from three_state_audit import ThreeStateAudit
from tse_dual_output import TSEDualOutputNet, SR


def read_audio(path: str | Path) -> np.ndarray:
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr == SR:
        return data
    target_len = max(1, round(len(data) * SR / sr))
    positions = np.linspace(0, max(0, len(data) - 1), target_len)
    return np.interp(positions, np.arange(len(data)), data).astype(np.float32)


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/datasetA")
    parser.add_argument("--tse-model", default="models/tse_dual_output_mvp.pt")
    parser.add_argument("--present-thresh", type=float, default=0.30)
    parser.add_argument("--empty-thresh", type=float, default=0.18)
    parser.add_argument("--out", default="data/datasetA/eval_tse_dual_output.json")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    pos_rows = list(read_jsonl(root / "pos.jsonl"))
    neg_rows = list(read_jsonl(root / "neg.jsonl"))

    if args.limit:
        pos_rows = pos_rows[:args.limit]
        neg_rows = neg_rows[:args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading TSE Dual-Output model from {args.tse_model} on {device}...")

    tse_net = TSEDualOutputNet().to(device)
    if os.path.exists(args.tse_model):
        ckpt = torch.load(args.tse_model, map_location=device)
        if "model_state" in ckpt:
            tse_net.load_state_dict(ckpt["model_state"])
        else:
            tse_net.load_state_dict(ckpt)
    tse_net.eval()

    auditor = ThreeStateAudit(
        present_threshold=args.present_thresh,
        empty_threshold=args.empty_thresh,
    )

    print("Loading Speaker Verification model...")
    sv_model = build_sv_model()

    print("Loading Paraformer ASR model...")
    asr_model = build_model(with_punc=False)

    cache_dir = root / "tse_dual_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    pos_accepted = 0
    pos_details = []
    t0 = time.time()

    print(f"\nRunning TSE Dual-Output evaluation on {len(pos_rows)} positive samples...")
    for i, row in enumerate(pos_rows, 1):
        wake_p = root / row["唤醒音频"]
        mix_p = root / row["识别音频"]
        ref_text = row.get("识别文本", "")

        mix_wav = read_audio(mix_p)
        enroll_wav = read_audio(wake_p)

        mix_t = torch.from_numpy(mix_wav).unsqueeze(0).to(device)
        enroll_t = torch.from_numpy(enroll_wav).unsqueeze(0).to(device)

        with torch.no_grad():
            target_t, residual_t = tse_net.forward_audio(mix_t, enroll_t)

        target_wav = target_t.squeeze(0).cpu().numpy()
        residual_wav = residual_t.squeeze(0).cpu().numpy()

        # Save temporary wave for feature extraction / ASR
        target_path = cache_dir / f"pos_{i}_target.wav"
        residual_path = cache_dir / f"pos_{i}_residual.wav"
        sf.write(target_path, target_wav, SR)
        sf.write(residual_path, residual_wav, SR)

        # Extract embeddings
        wake_emb = extract_embedding(sv_model, str(wake_p))
        target_emb = extract_embedding(sv_model, str(target_path))
        residual_emb = extract_embedding(sv_model, str(residual_path))

        target_sim = cosine_sim(wake_emb, target_emb)
        residual_sim = cosine_sim(wake_emb, residual_emb)

        audit_res = auditor.audit(target_sim=target_sim, residual_sim=residual_sim)

        if audit_res["emit_allowed"]:
            hyp_text = recognize(asr_model, str(target_path))
            pos_accepted += 1
        else:
            hyp_text = ""

        pairs.append((ref_text, hyp_text))
        pos_details.append({
            "id": row.get("id", i),
            "ref": ref_text,
            "hyp": hyp_text,
            "audit_state": audit_res["state"],
            "reason_code": audit_res["reason_code"],
            "target_sim": round(float(target_sim), 4),
            "residual_sim": round(float(residual_sim), 4),
        })

        if i % 100 == 0 or i == len(pos_rows):
            print(f"  Pos {i}/{len(pos_rows)}: Accepted={pos_accepted}/{i}")

    pos_cer, total_ref_chars = corpus_cer(pairs)
    pos_accept_rate = pos_accepted / max(1, len(pos_rows))

    neg_rejected = 0
    neg_details = []
    print(f"\nRunning TSE Dual-Output evaluation on {len(neg_rows)} negative samples...")
    for i, row in enumerate(neg_rows, 1):
        wake_p = root / row["唤醒音频"]
        mix_p = root / row["识别音频"]

        mix_wav = read_audio(mix_p)
        enroll_wav = read_audio(wake_p)

        mix_t = torch.from_numpy(mix_wav).unsqueeze(0).to(device)
        enroll_t = torch.from_numpy(enroll_wav).unsqueeze(0).to(device)

        with torch.no_grad():
            target_t, residual_t = tse_net.forward_audio(mix_t, enroll_t)

        target_wav = target_t.squeeze(0).cpu().numpy()
        residual_wav = residual_t.squeeze(0).cpu().numpy()

        target_path = cache_dir / f"neg_{i}_target.wav"
        residual_path = cache_dir / f"neg_{i}_residual.wav"
        sf.write(target_path, target_wav, SR)
        sf.write(residual_path, residual_wav, SR)

        wake_emb = extract_embedding(sv_model, str(wake_p))
        target_emb = extract_embedding(sv_model, str(target_path))
        residual_emb = extract_embedding(sv_model, str(residual_path))

        target_sim = cosine_sim(wake_emb, target_emb)
        residual_sim = cosine_sim(wake_emb, residual_emb)

        audit_res = auditor.audit(target_sim=target_sim, residual_sim=residual_sim)

        if audit_res["emit_allowed"]:
            hyp_text = recognize(asr_model, str(target_path))
        else:
            hyp_text = ""

        if not hyp_text:
            neg_rejected += 1

        neg_details.append({
            "id": row.get("id", i),
            "hyp": hyp_text,
            "audit_state": audit_res["state"],
            "reason_code": audit_res["reason_code"],
            "target_sim": round(float(target_sim), 4),
            "residual_sim": round(float(residual_sim), 4),
        })

        if i % 100 == 0 or i == len(neg_rows):
            print(f"  Neg {i}/{len(neg_rows)}: Rejected={neg_rejected}/{i}")

    neg_rr = neg_rejected / max(1, len(neg_rows))
    elapsed = time.time() - t0

    # Score 80 formula: 80 * (0.6 * (1 - CER) + 0.4 * RR)
    score_80 = 80.0 * (0.6 * max(0.0, 1.0 - pos_cer) + 0.4 * neg_rr)

    summary = {
        "positive_accept_rate": round(pos_accept_rate, 4),
        "positive_corpus_cer": round(pos_cer, 4),
        "negative_rr": round(neg_rr, 4),
        "score_80": round(score_80, 4),
        "elapsed_sec": round(elapsed, 2),
        "pos_count": len(pos_rows),
        "neg_count": len(neg_rows),
    }

    print("\n================ TSE Dual-Output Evaluation Summary ================")
    print(f"  Positive Corpus CER: {pos_cer * 100:.2f}%")
    print(f"  Positive Accept Rate: {pos_accept_rate * 100:.2f}% ({pos_accepted}/{len(pos_rows)})")
    print(f"  Negative Rejection Rate (RR): {neg_rr * 100:.2f}% ({neg_rejected}/{len(neg_rows)})")
    print(f"  80-Point Total Score: {score_80:.2f}")
    print(f"  Elapsed Time: {elapsed:.1f}s")

    out_data = {
        "summary": summary,
        "pos_details": pos_details,
        "neg_details": neg_details,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    print(f"TSE Dual-Output evaluation report saved to {args.out}")


if __name__ == "__main__":
    main()
