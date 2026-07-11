# -*- coding: utf-8 -*-
"""
Small-sample speaker-gate sweep for datasetA.

This script does not run ASR. It only compares each row's wake audio against
its command audio, then reports positive accept rate and negative rejection
rate for candidate thresholds.

Usage:
  ./venv/bin/python sweep_datasetA_gate.py --limit 50
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import time

from speaker_verify import build_sv_model, cosine_sim, extract_embedding


ROOT = "data/datasetA"


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


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wav_path(root, rel_path):
    return os.path.join(root, rel_path)


def extract_embedding_quiet(model, path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_embedding(model, path)


def collect_scores(model, root, rows, split, cache):
    scores = []
    for i, row in enumerate(rows, 1):
        wake = wav_path(root, row["唤醒音频"])
        cmd = wav_path(root, row["识别音频"])
        sim = cosine_sim(cache.get(model, wake), cache.get(model, cmd))
        scores.append({"split": split, "id": row.get("id"), "sim": sim,
                       "wake_audio": wake, "audio": cmd})
        if i % 50 == 0 or i == len(rows):
            print(f"  {split} {i}/{len(rows)}")
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--thresholds", default="0.25,0.30,0.35,0.40,0.45,0.50,0.55")
    parser.add_argument("--out", default=None)
    parser.add_argument("--embedding-cache", default=None,
                        help="Optional pickle cache for speaker embeddings during tuning runs.")
    args = parser.parse_args()

    pos = list(read_jsonl(os.path.join(args.root, "pos.jsonl")))[:args.limit]
    neg = list(read_jsonl(os.path.join(args.root, "neg.jsonl")))[:args.limit]
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    out = args.out or os.path.join(args.root, f"gate_sweep_{args.limit}.json")

    print(f"datasetA gate sweep: pos={len(pos)} neg={len(neg)}")
    print("Loading speaker model...")
    model = build_sv_model()
    cache = EmbeddingCache(args.embedding_cache)

    t0 = time.time()
    scores = []
    print("\nScoring positive samples...")
    scores.extend(collect_scores(model, args.root, pos, "pos", cache))
    print("\nScoring negative samples...")
    scores.extend(collect_scores(model, args.root, neg, "neg", cache))
    cache.save()

    pos_scores = [x["sim"] for x in scores if x["split"] == "pos"]
    neg_scores = [x["sim"] for x in scores if x["split"] == "neg"]
    rows = []
    for threshold in thresholds:
        pos_accept = sum(s >= threshold for s in pos_scores) / max(1, len(pos_scores))
        neg_rr = sum(s < threshold for s in neg_scores) / max(1, len(neg_scores))
        rows.append({
            "threshold": threshold,
            "positive_accept_rate": round(pos_accept, 4),
            "negative_rejection_rate_rr": round(neg_rr, 4),
        })

    report = {
        "dataset": "datasetA",
        "root": args.root,
        "pos_n": len(pos),
        "neg_n": len(neg),
        "elapsed_sec": round(time.time() - t0, 2),
        "thresholds": rows,
        "scores": scores,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nThreshold sweep")
    for row in rows:
        print(f"  t={row['threshold']:.2f}  pos_accept={row['positive_accept_rate']:.3f}  "
              f"neg_RR={row['negative_rejection_rate_rr']:.3f}")
    print(f"Report saved: {out}")


if __name__ == "__main__":
    main()
