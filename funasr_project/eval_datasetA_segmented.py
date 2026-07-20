# -*- coding: utf-8 -*-
"""
DatasetA segmented target-speaker pipeline.

This keeps the first stable system simple:
  1. Enroll the target speaker from the wake audio.
  2. Run VAD on the command audio.
  3. Cut VAD speech segments.
  4. Compare each segment embedding with the wake embedding.
  5. ASR accepted target-speaker segments only.
  6. Concatenate target text and write the contest JSON.

Start with --limit 20, then expand to 100, 300, and full data.
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from funasr import AutoModel

from asr_demo import VAD_DIR, build_model, recognize
from cer import cer, corpus_cer
from speaker_verify import build_sv_model, cosine_sim, extract_embedding
from text_norm import normalize


ROOT = "data/datasetA"
SR = 16000
DEFAULT_SEGMENT_THRESHOLD = 0.30


class EmbeddingCache:
    def __init__(self, path=None):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    def get(self, model, path):
        key = os.path.abspath(path)
        if key not in self.data:
            self.data[key] = extract_embedding_quiet(model, path)
        return self.data[key]

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)


class AsrCache:
    def __init__(self, path=None):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    def recognize(self, model, path):
        key = os.path.abspath(path)
        if key in self.data:
            return self.data[key], 0.0, True
        text, elapsed = recognize_quiet(model, path)
        self.data[key] = text
        return text, elapsed, False

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)


def quiet_call(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def extract_embedding_quiet(model, path):
    return quiet_call(extract_embedding, model, path)


def recognize_quiet(model, path):
    return quiet_call(recognize, model, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wav_path(root, rel_path):
    return os.path.join(root, rel_path)


def build_vad_model():
    return AutoModel(model=VAD_DIR, device="cpu", disable_update=True)


def vad_segments(vad_model, wav_path, min_ms=300, pad_ms=120):
    result = quiet_call(vad_model.generate, input=wav_path)
    raw = result[0].get("value", []) if result else []
    segments = []
    for start_ms, end_ms in raw:
        start_ms = max(0, int(start_ms) - pad_ms)
        end_ms = int(end_ms) + pad_ms
        if end_ms - start_ms >= min_ms:
            segments.append((start_ms, end_ms))
    return merge_segments(segments)


def merge_segments(segments, gap_ms=160):
    if not segments:
        return []
    segments = sorted(segments)
    merged = [segments[0]]
    for start_ms, end_ms in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start_ms - prev_end <= gap_ms:
            merged[-1] = (prev_start, max(prev_end, end_ms))
        else:
            merged.append((start_ms, end_ms))
    return merged


def read_wav(path):
    x, sr = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: sample rate {sr} != {SR}")
    return x.astype(np.float32)


def write_segment(src_wav, start_ms, end_ms, out_wav):
    x = read_wav(src_wav)
    start = max(0, int(start_ms * SR / 1000))
    end = min(len(x), int(end_ms * SR / 1000))
    seg = x[start:end]
    if len(seg) == 0:
        seg = np.zeros(int(0.1 * SR), dtype=np.float32)
    sf.write(out_wav, seg, SR)


def run_one(row, split, root, models, caches, args, tmp_dir):
    vad_model, sv_model, asr_model = models
    emb_cache, asr_cache = caches
    wake = wav_path(root, row["唤醒音频"])
    cmd = wav_path(root, row["识别音频"])
    target_emb = emb_cache.get(sv_model, wake)
    segments = vad_segments(vad_model, cmd, min_ms=args.min_segment_ms, pad_ms=args.pad_ms)

    accepted_parts = []
    segment_details = []
    asr_elapsed_total = 0.0
    for idx, (start_ms, end_ms) in enumerate(segments):
        seg_path = os.path.join(tmp_dir, f"{split}_{row.get('id', 'x')}_{Path(cmd).stem}_{idx:03d}.wav")
        write_segment(cmd, start_ms, end_ms, seg_path)
        seg_emb = emb_cache.get(sv_model, seg_path)
        sim = cosine_sim(target_emb, seg_emb)
        accepted = sim >= args.segment_threshold
        text = ""
        asr_cached = False
        if accepted:
            text, asr_elapsed, asr_cached = asr_cache.recognize(asr_model, seg_path)
            asr_elapsed_total += asr_elapsed
            text = normalize(text)
            accepted_parts.append(text)
        segment_details.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "similarity": round(sim, 4),
            "accepted": accepted,
            "text": text,
            "asr_cached": asr_cached,
        })

    hyp = normalize("".join(accepted_parts))
    return {
        "split": split,
        "id": row.get("id"),
        "wake_audio": wake,
        "audio": cmd,
        "ref": row.get("识别文本") or "",
        "hyp": hyp,
        "segments_total": len(segments),
        "segments_accepted": sum(1 for s in segment_details if s["accepted"]),
        "segment_details": segment_details,
        "asr_latency_sec": round(asr_elapsed_total, 3),
    }


def build_report(args, details, started_at):
    pairs = [(d["ref"], d["hyp"]) for d in details if d["split"] == "pos"]
    pos_details = [d for d in details if d["split"] == "pos"]
    neg_details = [d for d in details if d["split"] == "neg"]
    pos_cers = []
    for d in pos_details:
        c, ref_len = cer(d["ref"], d["hyp"])
        d["cer"] = round(c, 4)
        d["ref_len"] = ref_len
        pos_cers.append(c)
    total_cer, total_chars = corpus_cer(pairs)
    rejected = sum(1 for d in neg_details if not d["hyp"].strip())
    return {
        "dataset": "datasetA",
        "mode": "segmented_vad_speaker_gate_asr",
        "root": args.root,
        "segment_threshold": args.segment_threshold,
        "min_segment_ms": args.min_segment_ms,
        "pad_ms": args.pad_ms,
        "pos_n": len(pos_details),
        "neg_n": len(neg_details),
        "positive_sentence_avg_cer": round(sum(pos_cers) / max(1, len(pos_cers)), 4),
        "positive_corpus_cer": round(total_cer, 4),
        "positive_ref_chars": total_chars,
        "negative_rejection_rate_rr": round(rejected / max(1, len(neg_details)), 4),
        "negative_rejected": rejected,
        "elapsed_sec": round(time.time() - started_at, 2),
        "details": details,
    }


def build_submission(report):
    results = []
    for item in report["details"]:
        sample_id = item.get("id")
        if sample_id is None or sample_id == "":
            sample_id = Path(item["audio"]).stem
        results.append({
            "id": sample_id,
            "content": item.get("hyp", ""),
            "label": item.get("ref", "") or "",
            "cer": item.get("cer", 0.0),
        })
    return {
        "result": {
            "results": results,
            "avg_cer": report["positive_corpus_cer"],
            "avg_rr": report["negative_rejection_rate_rr"],
            "duration": round(report["elapsed_sec"] * 1000, 2),
        }
    }


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--segment-threshold", type=float, default=DEFAULT_SEGMENT_THRESHOLD)
    parser.add_argument("--min-segment-ms", type=int, default=300)
    parser.add_argument("--pad-ms", type=int, default=120)
    parser.add_argument("--out", default=None)
    parser.add_argument("--submission-out", default=None)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--asr-cache", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=100,
                        help="Save a partial report every N processed rows. Set 0 to disable.")
    args = parser.parse_args()

    out = args.out or os.path.join(args.root, f"eval_report_segmented_{args.limit}.json")
    pos_rows = list(read_jsonl(os.path.join(args.root, "pos.jsonl")))
    neg_rows = list(read_jsonl(os.path.join(args.root, "neg.jsonl")))
    if args.offset:
        pos_rows = pos_rows[args.offset:]
        neg_rows = neg_rows[args.offset:]
    if args.limit is not None and args.limit >= 0:
        pos_rows = pos_rows[:args.limit]
        neg_rows = neg_rows[:args.limit]

    print(f"datasetA segmented: pos={len(pos_rows)} neg={len(neg_rows)}")
    print(f"segment threshold={args.segment_threshold}")
    print("Loading VAD model...")
    vad_model = build_vad_model()
    print("Loading speaker model...")
    sv_model = build_sv_model()
    print("Loading ASR model...")
    asr_model = build_model(with_punc=False)
    emb_cache = EmbeddingCache(args.embedding_cache)
    asr_cache = AsrCache(args.asr_cache)

    details = []
    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="datasetA_segments_") as tmp_dir:
        print("\nRunning positive samples...")
        for i, row in enumerate(pos_rows, 1):
            details.append(run_one(row, "pos", args.root, (vad_model, sv_model, asr_model),
                                   (emb_cache, asr_cache), args, tmp_dir))
            if i % 20 == 0 or i == len(pos_rows):
                print(f"  pos {i}/{len(pos_rows)}")
            if args.checkpoint_every and i % args.checkpoint_every == 0:
                save_json(out, build_report(args, details, started_at))

        print("\nRunning negative samples...")
        for i, row in enumerate(neg_rows, 1):
            details.append(run_one(row, "neg", args.root, (vad_model, sv_model, asr_model),
                                   (emb_cache, asr_cache), args, tmp_dir))
            if i % 20 == 0 or i == len(neg_rows):
                print(f"  neg {i}/{len(neg_rows)}")
            if args.checkpoint_every and i % args.checkpoint_every == 0:
                save_json(out, build_report(args, details, started_at))

    report = build_report(args, details, started_at)
    save_json(out, report)
    emb_cache.save()
    asr_cache.save()
    if args.submission_out:
        save_json(args.submission_out, build_submission(report))

    print("\nSummary")
    print(f"  positive sentence avg CER: {report['positive_sentence_avg_cer']:.4f}")
    print(f"  positive corpus CER:       {report['positive_corpus_cer']:.4f} "
          f"({report['positive_ref_chars']} chars)")
    print(f"  negative RR:               {report['negative_rejection_rate_rr']:.4f} "
          f"({report['negative_rejected']}/{report['neg_n']})")
    print(f"  elapsed:                   {report['elapsed_sec']:.1f}s")
    print(f"Report saved: {out}")
    if args.submission_out:
        print(f"Submission saved: {args.submission_out}")


if __name__ == "__main__":
    main()
