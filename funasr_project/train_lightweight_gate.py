# -*- coding: utf-8 -*-
"""
Train a tiny datasetA accept/reject gate from cached speaker and ASR outputs.

Example:
  ./venv/bin/python eval_datasetA.py --limit 474 \
    --embedding-cache data/datasetA/emb_cache_tune.pkl \
    --asr-cache data/datasetA/asr_cache_tune.pkl

  ./venv/bin/python train_lightweight_gate.py --train-limit 300 --dev-limit 174

  ./venv/bin/python eval_datasetA.py --gate-model models/lightweight_gate.json
"""
import argparse
import json
import os
import pickle
import random

import numpy as np

from cer import corpus_cer
from command_match import edit_distance, to_pinyin
from lightweight_gate import (
    DEFAULT_FEATURE_NAMES, EXTENDED_FEATURE_NAMES, make_features, save_gate_model, vectorize,
)
from text_norm import normalize


ROOT = "data/datasetA"


def cosine_sim(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    if not np.isfinite(sim):
        return 0.0
    return sim


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wav_path(root, rel_path):
    return os.path.abspath(os.path.join(root, rel_path))


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_phrase_bank(root):
    phrases = []
    for row in read_jsonl(os.path.join(root, "pos.jsonl")):
        text = normalize(row.get("识别文本") or "")
        if text and text not in phrases:
            phrases.append(text)
    return phrases


def nearest_intent(text, phrases, phrases_py=None):
    """Pinyin edit distance, SAME metric as eval_datasetA.nearest_intent.

    The gate must be trained on the exact feature distribution used at
    inference time; the old char-level distance caused a train/serve skew.
    """
    hyp = normalize(text)
    if not hyp:
        return 1.0, ""
    if phrases_py is None:
        phrases_py = [to_pinyin(p) for p in phrases]
    hyp_py = to_pinyin(hyp)
    best_score, best_phrase = 1.0, ""
    lh = len(hyp_py)
    for phrase, phrase_py in zip(phrases, phrases_py):
        lp = len(phrase_py)
        if abs(lh - lp) / max(lh, lp) >= best_score:  # length lower bound prune
            continue
        dist = edit_distance(hyp_py, phrase_py)
        score = dist / max(lh, lp)
        if score < best_score:
            best_score, best_phrase = score, phrase
    return best_score, best_phrase


def slice_rows(root, split, offset, limit):
    rows = list(read_jsonl(os.path.join(root, f"{split}.jsonl")))
    if offset:
        rows = rows[offset:]
    if limit is not None:
        rows = rows[:limit]
    return rows


def split_rows(rows, train_ratio, seed):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_train = max(1, min(len(rows) - 1, int(round(len(rows) * train_ratio))))
    return rows[:n_train], rows[n_train:]


def build_items(root, split, rows, emb_cache, asr_cache, phrases, fusion_weight, phrases_py=None):
    items = []
    skipped = 0
    missing_asr = 0
    if phrases_py is None:
        phrases_py = [to_pinyin(p) for p in phrases]
    for row in rows:
        wake = wav_path(root, row["唤醒音频"])
        cmd = wav_path(root, row["识别音频"])
        if wake not in emb_cache or cmd not in emb_cache:
            skipped += 1
            continue
        hyp = ""
        if cmd in asr_cache:
            hyp = asr_cache[cmd].get("text", "")
        else:
            missing_asr += 1
        sim = cosine_sim(emb_cache[wake], emb_cache[cmd])
        intent, nearest_phrase = nearest_intent(hyp, phrases, phrases_py)
        features = make_features(sim, hyp, intent, fusion_weight=fusion_weight)
        items.append({
            "split": split,
            "label": 1 if split == "pos" else 0,
            "ref": row.get("识别文本") or "",
            "hyp": hyp,
            "nearest_phrase": nearest_phrase,
            "intent": intent,
            "features": features,
        })
    return items, skipped, missing_asr


def standardize_matrix(items, feature_names):
    x = np.asarray([vectorize(item["features"], feature_names) for item in items], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    z = (x - mean) / scale
    return np.clip(z, -10.0, 10.0), mean, scale


def train_logistic(x, y, epochs, lr, l2, seed, pos_cost=1.0, neg_cost=1.0):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=x.shape[1]).astype(np.float64)
    b = 0.0
    y = y.astype(np.float64)
    pos_weight = pos_cost * 0.5 / max(float(y.mean()), 1e-6)
    neg_weight = neg_cost * 0.5 / max(float(1.0 - y.mean()), 1e-6)
    sample_weight = np.where(y > 0.5, pos_weight, neg_weight).astype(np.float32)

    for _ in range(epochs):
        w = np.clip(np.nan_to_num(w, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
        b = float(np.clip(np.nan_to_num(b, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0))
        logits = np.sum(x * w.reshape(1, -1), axis=1) + b
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        err = (probs - y) * sample_weight
        grad_w = np.sum(x * err.reshape(-1, 1), axis=0) / len(x) + l2 * w
        grad_b = err.mean()
        grad_w = np.nan_to_num(grad_w, nan=0.0, posinf=1.0, neginf=-1.0)
        grad_b = float(np.nan_to_num(grad_b, nan=0.0, posinf=1.0, neginf=-1.0))
        grad_w = np.clip(grad_w, -5.0, 5.0)
        grad_b = float(np.clip(grad_b, -5.0, 5.0))
        w -= lr * grad_w
        b -= lr * grad_b
    w = np.clip(np.nan_to_num(w, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0)
    b = float(np.clip(np.nan_to_num(b, nan=0.0, posinf=20.0, neginf=-20.0), -20.0, 20.0))
    return w, float(b)


def probability(x, w, b):
    logits = np.sum(x * w.reshape(1, -1), axis=1) + b
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def apply_phrase_correction(item, phrase_threshold):
    if item["intent"] <= phrase_threshold and item["nearest_phrase"]:
        return item["nearest_phrase"]
    return item["hyp"]


def evaluate(items, probs, threshold, phrase_threshold):
    pairs = []
    neg_rejected = 0
    pos_accepted = 0
    neg_n = 0
    pos_n = 0
    for item, prob in zip(items, probs):
        accepted = float(prob) >= threshold
        if item["split"] == "pos":
            pos_n += 1
            pos_accepted += int(accepted)
            hyp = apply_phrase_correction(item, phrase_threshold) if accepted else ""
            pairs.append((item["ref"], hyp))
        else:
            neg_n += 1
            neg_rejected += int(not accepted or not normalize(item["hyp"]))
    cer, chars = corpus_cer(pairs)
    rr = neg_rejected / max(1, neg_n)
    objective = 40.0 * (1.0 - cer) + 40.0 * rr
    return {
        "score_80": round(objective, 4),
        "positive_corpus_cer": round(cer, 4),
        "positive_ref_chars": chars,
        "positive_accept_rate": round(pos_accepted / max(1, pos_n), 4),
        "negative_rejection_rate_rr": round(rr, 4),
        "threshold": round(float(threshold), 4),
        "pos_n": pos_n,
        "neg_n": neg_n,
    }


def choose_threshold(items, probs, phrase_threshold, min_rr, max_frr=1.0):
    candidates = np.linspace(0.05, 0.95, 91)
    metrics = [evaluate(items, probs, float(t), phrase_threshold) for t in candidates]
    feasible = [m for m in metrics
                if m["negative_rejection_rate_rr"] >= min_rr
                and (1.0 - m["positive_accept_rate"]) <= max_frr]
    if feasible:
        metrics = feasible
    return max(metrics, key=lambda x: (x["score_80"], x["negative_rejection_rate_rr"], -x["positive_corpus_cer"]))


def threshold_table(items, probs, phrase_threshold, top=10):
    """All candidate thresholds ranked by score_80, for trade-off inspection."""
    candidates = np.linspace(0.05, 0.95, 91)
    metrics = [evaluate(items, probs, float(t), phrase_threshold) for t in candidates]
    metrics.sort(key=lambda x: (x["score_80"], x["negative_rejection_rate_rr"]), reverse=True)
    return metrics[:top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--embedding-cache", default=os.path.join(ROOT, "emb_cache_tune.pkl"))
    parser.add_argument("--asr-cache", default=os.path.join(ROOT, "asr_cache_tune.pkl"))
    parser.add_argument("--model-out", default="models/lightweight_gate.json")
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--dev-offset", type=int, default=300)
    parser.add_argument("--dev-limit", type=int, default=174)
    parser.add_argument("--split-mode", choices=("sequential", "random"), default="sequential",
                        help="sequential uses offset/limit; random does stratified random train/dev split.")
    parser.add_argument("--train-ratio", type=float, default=0.80,
                        help="Train ratio for --split-mode random.")
    parser.add_argument("--max-per-split", type=int, default=None,
                        help="Optional per-class cap before random splitting.")
    parser.add_argument("--fusion-weight", type=float, default=0.70)
    parser.add_argument("--phrase-threshold", type=float, default=0.50)
    parser.add_argument("--min-dev-rr", type=float, default=0.0,
                        help="Optional safety constraint when selecting the probability threshold.")
    parser.add_argument("--max-dev-frr", type=float, default=1.0,
                        help="Optional FRR constraint when selecting the probability threshold.")
    parser.add_argument("--feature-set", choices=("basic", "extended"), default="extended",
                        help="basic=legacy 7 features; extended=+interaction/non-linear features.")
    parser.add_argument("--pos-cost", type=float, default=1.0,
                        help=">1 penalizes false rejects more (pushes FRR down).")
    parser.add_argument("--neg-cost", type=float, default=1.0,
                        help=">1 penalizes false accepts more (pushes RR up).")
    parser.add_argument("--items-cache", default=None,
                        help="Optional precomputed per-row items pickle (sim/intent/hyp/ref); "
                             "skips embedding/ASR cache lookups.")
    parser.add_argument("--select-on", choices=("dev", "full"), default="dev",
                        help="dev=select threshold on the dev split (default); "
                             "full=select on train+dev under the RR/FRR constraints "
                             "(mirrors the strategy used for the deployed model).")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    emb_cache = load_pickle(args.embedding_cache) if not args.items_cache else None
    asr_cache = load_pickle(args.asr_cache) if not args.items_cache else None
    phrases = build_phrase_bank(args.root)
    phrases_py = [to_pinyin(p) for p in phrases]
    precomputed = load_pickle(args.items_cache) if args.items_cache else None

    train_rows = []
    dev_rows = []
    for split in ("pos", "neg"):
        if args.split_mode == "random":
            rows = slice_rows(args.root, split, 0, args.max_per_split)
            train_part, dev_part = split_rows(rows, args.train_ratio, args.seed + (0 if split == "pos" else 10000))
        else:
            train_part = slice_rows(args.root, split, args.train_offset, args.train_limit)
            dev_part = slice_rows(args.root, split, args.dev_offset, args.dev_limit)
        train_rows.extend((split, row) for row in train_part)
        dev_rows.extend((split, row) for row in dev_part)

    def items_for(split, row):
        """One item per row, either from --items-cache or rebuilt from raw caches."""
        if precomputed is not None:
            key = (split, row.get("id"))
            hit = pre_index.get(key)
            if hit is None or hit["sim"] is None:
                return [], 1, 0
            features = make_features(hit["sim"], hit["hyp"], hit["intent"],
                                     fusion_weight=args.fusion_weight)
            return [{
                "split": split,
                "label": 1 if split == "pos" else 0,
                "ref": row.get("识别文本") or "",
                "hyp": hit["hyp"],
                "nearest_phrase": hit.get("nearest_phrase") or "",
                "intent": hit["intent"],
                "features": features,
            }], 0, 0
        return build_items(args.root, split, [row], emb_cache, asr_cache, phrases,
                           args.fusion_weight, phrases_py)

    pre_index = {}
    if precomputed is not None:
        pre_index = {(x["split"], x["id"]): x for x in precomputed}

    train_items = []
    dev_items = []
    skipped = {"train": 0, "dev": 0}
    missing_asr = {"train": 0, "dev": 0}
    for split, row in train_rows:
        items, s, m = items_for(split, row)
        train_items.extend(items)
        skipped["train"] += s
        missing_asr["train"] += m
    for split, row in dev_rows:
        items, s, m = items_for(split, row)
        dev_items.extend(items)
        skipped["dev"] += s
        missing_asr["dev"] += m

    if not train_items:
        raise SystemExit("No train items found. Run eval_datasetA.py with embedding/asr caches first.")
    if not dev_items:
        dev_items = train_items

    feature_names = DEFAULT_FEATURE_NAMES if args.feature_set == "basic" else EXTENDED_FEATURE_NAMES
    x_train, mean, scale = standardize_matrix(train_items, feature_names)
    y_train = np.asarray([item["label"] for item in train_items], dtype=np.float64)
    w, b = train_logistic(x_train, y_train, args.epochs, args.lr, args.l2, args.seed,
                          pos_cost=args.pos_cost, neg_cost=args.neg_cost)

    x_dev_raw = np.asarray([vectorize(item["features"], feature_names) for item in dev_items], dtype=np.float64)
    x_dev_raw = np.nan_to_num(x_dev_raw, nan=0.0, posinf=10.0, neginf=-10.0)
    x_dev = np.clip((x_dev_raw - mean) / scale, -10.0, 10.0)
    dev_probs = probability(x_dev, w, b)
    train_probs = probability(x_train, w, b)
    full_items = train_items + dev_items
    full_probs = np.concatenate([train_probs, dev_probs])

    if args.select_on == "full":
        sel_items, sel_probs = full_items, full_probs
    else:
        sel_items, sel_probs = dev_items, dev_probs
    sel_metrics = choose_threshold(sel_items, sel_probs, args.phrase_threshold,
                                   args.min_dev_rr, args.max_dev_frr)
    threshold = sel_metrics["threshold"]
    dev_metrics = evaluate(dev_items, dev_probs, threshold, args.phrase_threshold)
    train_metrics = evaluate(train_items, train_probs, threshold, args.phrase_threshold)

    model = {
        "type": "logistic_gate",
        "dataset": "datasetA",
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "feature_mean": [round(float(x), 8) for x in mean.tolist()],
        "feature_scale": [round(float(x), 8) for x in scale.tolist()],
        "weights": [round(float(x), 8) for x in w.tolist()],
        "bias": round(float(b), 8),
        "threshold": dev_metrics["threshold"],
        "fusion_weight_for_features": args.fusion_weight,
        "phrase_threshold": args.phrase_threshold,
        "min_dev_rr": args.min_dev_rr,
        "max_dev_frr": args.max_dev_frr,
        "pos_cost": args.pos_cost,
        "neg_cost": args.neg_cost,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio if args.split_mode == "random" else None,
        "max_per_split": args.max_per_split,
        "train": train_metrics,
        "dev": dev_metrics,
        "cache_warnings": {
            "skipped_missing_embedding": skipped,
            "missing_asr_treated_as_empty": missing_asr,
        },
    }

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    save_gate_model(args.model_out, model)

    print(f"train items: {len(train_items)}  dev items: {len(dev_items)}")
    print(f"missing embeddings skipped: {skipped}")
    print(f"missing ASR treated as empty: {missing_asr}")
    print("train:", train_metrics)
    print("dev:  ", dev_metrics)
    print(f"top {args.select_on} threshold candidates:")
    for m in threshold_table(sel_items, sel_probs, args.phrase_threshold, top=8):
        print(f"  t={m['threshold']:.2f} score={m['score_80']:.2f} "
              f"acc={m['positive_accept_rate']:.3f} cer={m['positive_corpus_cer']:.4f} "
              f"rr={m['negative_rejection_rate_rr']:.4f}")
    print(f"model saved: {args.model_out}")


if __name__ == "__main__":
    main()
