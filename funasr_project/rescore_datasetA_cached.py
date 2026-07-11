# -*- coding: utf-8 -*-
"""
Rescore datasetA policies from embedding and ASR caches.

This does not load FunASR models. It expects caches produced by eval_datasetA.py:
  --embedding-cache data/datasetA/emb_cache_tune.pkl
  --asr-cache data/datasetA/asr_cache_tune.pkl
"""
import argparse
import json
import os
import pickle

from cer import corpus_cer
from command_match import edit_distance
from speaker_verify import cosine_sim
from text_norm import normalize


ROOT = "data/datasetA"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wav_path(root, rel_path):
    return os.path.abspath(os.path.join(root, rel_path))


def build_phrase_bank(root):
    phrases = []
    for row in read_jsonl(os.path.join(root, "pos.jsonl")):
        text = normalize(row.get("识别文本") or "")
        if text and text not in phrases:
            phrases.append(text)
    return phrases


def intent_distance(text, phrases):
    hyp = normalize(text)
    if not hyp:
        return 1.0
    best = 1.0
    for phrase in phrases:
        if hyp in phrase or phrase in hyp:
            score = abs(len(hyp) - len(phrase)) / max(len(hyp), len(phrase))
        else:
            score = edit_distance(hyp, phrase) / max(len(hyp), len(phrase))
        best = min(best, score)
    return best


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--limit", type=int, default=474)
    parser.add_argument("--embedding-cache", default=os.path.join(ROOT, "emb_cache_tune.pkl"))
    parser.add_argument("--asr-cache", default=os.path.join(ROOT, "asr_cache_tune.pkl"))
    parser.add_argument("--fusion-weight", type=float, default=0.7)
    parser.add_argument("--thresholds", default="-0.05,0.00,0.03,0.05,0.06,0.08,0.10,0.12,0.14,0.16")
    args = parser.parse_args()

    emb_cache = load_pickle(args.embedding_cache)
    asr_cache = load_pickle(args.asr_cache)
    phrases = build_phrase_bank(args.root)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]

    items = []
    for split in ("pos", "neg"):
        rows = list(read_jsonl(os.path.join(args.root, f"{split}.jsonl")))[:args.limit]
        for row in rows:
            wake = wav_path(args.root, row["唤醒音频"])
            cmd = wav_path(args.root, row["识别音频"])
            if wake not in emb_cache or cmd not in emb_cache or cmd not in asr_cache:
                continue
            hyp = asr_cache[cmd]["text"]
            sim = cosine_sim(emb_cache[wake], emb_cache[cmd])
            intent = intent_distance(hyp, phrases)
            items.append({
                "split": split,
                "ref": row.get("识别文本") or "",
                "hyp": hyp,
                "sim": sim,
                "intent": intent,
            })

    pos_n = sum(x["split"] == "pos" for x in items)
    neg_n = sum(x["split"] == "neg" for x in items)
    print(f"cached items: pos={pos_n} neg={neg_n}")
    rows = []
    for threshold in thresholds:
        pairs = []
        pos_accept = 0
        neg_reject = 0
        for item in items:
            score = item["sim"] - args.fusion_weight * item["intent"]
            accepted = score >= threshold
            if item["split"] == "pos":
                pos_accept += int(accepted)
                pairs.append((item["ref"], item["hyp"] if accepted else ""))
            else:
                neg_reject += int(not accepted or not item["hyp"].strip())
        cer, chars = corpus_cer(pairs)
        rr = neg_reject / max(1, neg_n)
        objective = 40 * (1 - cer) + 40 * rr
        rows.append((objective, threshold, pos_accept / max(1, pos_n), cer, rr, chars))

    print("threshold sweep")
    for objective, threshold, pos_accept, cer, rr, chars in sorted(rows, reverse=True):
        print(
            f"score={objective:.2f}/80  threshold={threshold:.2f}  "
            f"pos_accept={pos_accept:.3f}  CER={cer:.4f}  RR={rr:.4f}  chars={chars}"
        )


if __name__ == "__main__":
    main()
