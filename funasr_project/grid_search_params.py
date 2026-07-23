# -*- coding: utf-8 -*-
"""
High-speed grid search hyperparameter tuning over DatasetA.

Pre-computes and caches speaker similarities and ASR raw outputs, then
evaluates hundreds of hyperparameter combinations (sv_threshold, phrase_threshold,
phrase_correct, etc.) in seconds to find the Pareto optimal parameter sets.

Usage:
  python grid_search_params.py [--root data/datasetA] [--out data/datasetA/grid_search_results.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure funasr_project directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cer import cer, corpus_cer
from eval_datasetA import (
    AsrCache, EmbeddingCache, build_intent_phrases, build_model, build_sv_model,
    cosine_sim, nearest_intent, read_jsonl, seed_everything, wav_path
)


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        total = len(iterable) if hasattr(iterable, "__len__") else None
        print(f"--> {desc}...")
        for i, item in enumerate(iterable, 1):
            if total and (i % max(1, total // 10) == 0 or i == total):
                print(f"    [{desc}] {i}/{total} ({i/total*100:.1f}%)")
            yield item


def run_grid_search(args):
    seed_everything(42)
    candidates = [args.root, "../datasetA", "datasetA", "data/datasetA", "../data/datasetA"]
    root = None
    for cand in candidates:
        if cand and os.path.exists(os.path.join(cand, "pos.jsonl")):
            root = cand
            break
    if not root:
        raise FileNotFoundError(f"datasetA pos.jsonl not found in any candidate path: {candidates}")

    pos_jsonl = os.path.join(root, "pos.jsonl")
    neg_jsonl = os.path.join(root, "neg.jsonl")
    if not os.path.exists(pos_jsonl) or not os.path.exists(neg_jsonl):
        raise FileNotFoundError(f"datasetA files not found in {root}")

    pos_rows = list(read_jsonl(pos_jsonl))
    neg_rows = list(read_jsonl(neg_jsonl))
    if args.limit:
        pos_rows = pos_rows[:args.limit]
        neg_rows = neg_rows[:args.limit]

    print(f"Loaded {len(pos_rows)} pos and {len(neg_rows)} neg samples from {root}.")

    emb_cache_path = args.embedding_cache or os.path.join(root, "embedding_cache.pkl")
    if not os.path.exists(emb_cache_path) and os.path.exists("data/datasetA/embedding_cache.pkl"):
        emb_cache_path = "data/datasetA/embedding_cache.pkl"

    asr_cache_path = args.asr_cache or os.path.join(root, "asr_cache.pkl")
    if not os.path.exists(asr_cache_path) and os.path.exists("data/datasetA/asr_cache.pkl"):
        asr_cache_path = "data/datasetA/asr_cache.pkl"

    emb_cache = EmbeddingCache(emb_cache_path)
    asr_cache = AsrCache(asr_cache_path)

    # Populate cache if missing entries
    sv_model = None
    asr_model = None

    def get_emb(wpath):
        nonlocal sv_model
        if emb_cache.key(wpath) not in emb_cache.data:
            if sv_model is None:
                print("\nLoading CAM++ SV model to populate missing embedding cache...")
                sv_model = build_sv_model()
            emb_cache.get(sv_model, wpath)
        return emb_cache.data[emb_cache.key(wpath)]

    def get_asr(wpath):
        nonlocal asr_model
        if asr_cache.key(wpath) not in asr_cache.data:
            if asr_model is None:
                print("\nLoading Paraformer ASR model to populate missing ASR cache...")
                asr_model = build_model(with_punc=False)
            asr_cache.recognize(asr_model, wpath)
        return asr_cache.data[asr_cache.key(wpath)]["text"]

    print("Pre-loading and verifying feature caches...")
    t_start_cache = time.time()
    pos_data = []
    for row in tqdm(pos_rows, desc="Caching POS Audio Features"):
        wake_p = wav_path(root, row["唤醒音频"])
        cmd_p = wav_path(root, row["识别音频"])
        w_emb = get_emb(wake_p)
        c_emb = get_emb(cmd_p)
        sim = cosine_sim(w_emb, c_emb)
        raw_text = get_asr(cmd_p)
        pos_data.append({
            "id": row.get("id"),
            "sim": sim,
            "raw_text": raw_text,
            "ref": row.get("识别文本", ""),
        })

    neg_data = []
    for row in tqdm(neg_rows, desc="Caching NEG Audio Features"):
        wake_p = wav_path(root, row["唤醒音频"])
        cmd_p = wav_path(root, row["识别音频"])
        w_emb = get_emb(wake_p)
        c_emb = get_emb(cmd_p)
        sim = cosine_sim(w_emb, c_emb)
        raw_text = get_asr(cmd_p)
        neg_data.append({
            "id": row.get("id"),
            "sim": sim,
            "raw_text": raw_text,
        })

    emb_cache.save()
    asr_cache.save()
    print(f"Feature caching ready in {time.time() - t_start_cache:.2f}s.")

    phrase_bank_path = args.phrase_bank or "data/public_train/aishell1/phrase_bank.txt"
    phrases = []
    intent_cache = {}
    if os.path.exists(phrase_bank_path):
        phrases = build_intent_phrases(root, phrase_bank=phrase_bank_path)
        print(f"Loaded {len(phrases)} command phrases from {phrase_bank_path}.")
        unique_texts = set(d["raw_text"] for d in pos_data + neg_data)
        for txt in tqdm(unique_texts, desc="Pre-computing Pinyin Distances"):
            intent_cache[txt] = nearest_intent(txt, phrases)

    # Define hyperparameter search grids
    sv_thresholds = [0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40]
    phrase_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    phrase_correct_options = [True, False] if phrases else [False]
    intent_filter_options = [False, True] if phrases else [False]
    intent_thresholds = [0.30, 0.40]

    grid = []
    for sv_t in sv_thresholds:
        for p_correct in phrase_correct_options:
            p_thresh_list = phrase_thresholds if p_correct else [None]
            for p_t in p_thresh_list:
                for i_filter in intent_filter_options:
                    i_thresh_list = intent_thresholds if i_filter else [None]
                    for i_t in i_thresh_list:
                        grid.append({
                            "sv_threshold": sv_t,
                            "phrase_correct": p_correct,
                            "phrase_threshold": p_t,
                            "intent_filter": i_filter,
                            "intent_threshold": i_t,
                        })

    print(f"\nStarting Grid Search over {len(grid)} parameter combinations...")
    t_search_start = time.time()
    results = []

    for combo in tqdm(grid, desc="Grid Searching Parameters"):
        sv_t = combo["sv_threshold"]
        p_correct = combo["phrase_correct"]
        p_t = combo["phrase_threshold"]
        i_filter = combo["intent_filter"]
        i_t = combo["intent_threshold"]

        # Evaluate positive samples
        pos_accepted = 0
        pairs = []
        for item in pos_data:
            sim = item["sim"]
            accepted = sim >= sv_t
            if accepted and i_filter:
                dist, _ = intent_cache.get(item["raw_text"], (1.0, ""))
                if dist > i_t:
                    accepted = False
            if accepted:
                pos_accepted += 1
                hyp = item["raw_text"]
                if p_correct and hyp:
                    dist, match = intent_cache.get(hyp, (1.0, ""))
                    if dist <= p_t:
                        hyp = match
            else:
                hyp = ""
            pairs.append((item["ref"], hyp))

        valid_pairs = [(ref, hyp) for ref, hyp in pairs if ref]
        if not valid_pairs:
            continue
        corpus_cer_val, total_chars = corpus_cer(valid_pairs, do_norm=False)
        pos_accept_rate = pos_accepted / max(1, len(pos_data))

        # Evaluate negative samples
        neg_rejected = 0
        for item in neg_data:
            sim = item["sim"]
            accepted = sim >= sv_t
            if accepted and i_filter:
                dist, _ = intent_cache.get(item["raw_text"], (1.0, ""))
                if dist > i_t:
                    accepted = False
            if not accepted:
                neg_rejected += 1

        rr = neg_rejected / max(1, len(neg_data))
        balanced_score = (1.0 - corpus_cer_val) * rr

        res_item = {
            "sv_threshold": sv_t,
            "phrase_correct": p_correct,
            "phrase_threshold": p_t,
            "intent_filter": i_filter,
            "intent_threshold": i_t,
            "pos_corpus_cer": round(corpus_cer_val, 4),
            "pos_accept_rate": round(pos_accept_rate, 4),
            "neg_rr": round(rr, 4),
            "balanced_score": round(balanced_score, 4),
        }
        results.append(res_item)

    search_elapsed = time.time() - t_search_start
    print(f"Grid search completed in {search_elapsed:.2f}s ({len(grid)} combos evaluated).\n")

    # Sort results
    # 1. Balanced Score Rank
    results_balanced = sorted(results, key=lambda x: x["balanced_score"], reverse=True)
    # 2. Minimum CER Rank
    results_cer = sorted(results, key=lambda x: x["pos_corpus_cer"])
    # 3. Constrained High RR Rank (RR >= 0.85)
    results_rr85 = sorted([r for r in results if r["neg_rr"] >= 0.85], key=lambda x: x["pos_corpus_cer"])

    print("=" * 80)
    print(" TOP 5 GOLDEN BALANCED COMBINATIONS (Max S = (1 - CER) * RR)")
    print("=" * 80)
    for idx, r in enumerate(results_balanced[:5], 1):
        print(f"#{idx}: sv_threshold={r['sv_threshold']} phrase_correct={r['phrase_correct']} "
              f"p_thresh={r['phrase_threshold']} intent_filter={r['intent_filter']} -> "
              f"CER={r['pos_corpus_cer']*100:.2f}% | RR={r['neg_rr']*100:.2f}% | Score={r['balanced_score']:.4f}")

    print("\n" + "=" * 80)
    print(" TOP 5 CONSTRAINED COMBINATIONS (Rejection Rate RR >= 85.0%)")
    print("=" * 80)
    for idx, r in enumerate(results_rr85[:5], 1):
        print(f"#{idx}: sv_threshold={r['sv_threshold']} phrase_correct={r['phrase_correct']} "
              f"p_thresh={r['phrase_threshold']} intent_filter={r['intent_filter']} -> "
              f"CER={r['pos_corpus_cer']*100:.2f}% | RR={r['neg_rr']*100:.2f}% | Score={r['balanced_score']:.4f}")

    print("\n" + "=" * 80)
    print(" TOP 5 LOWEST CER COMBINATIONS (Unconstrained Minimum CER)")
    print("=" * 80)
    for idx, r in enumerate(results_cer[:5], 1):
        print(f"#{idx}: sv_threshold={r['sv_threshold']} phrase_correct={r['phrase_correct']} "
              f"p_thresh={r['phrase_threshold']} intent_filter={r['intent_filter']} -> "
              f"CER={r['pos_corpus_cer']*100:.2f}% | RR={r['neg_rr']*100:.2f}% | Score={r['balanced_score']:.4f}")
    print("=" * 80)

    out_file = args.out or os.path.join(root, "grid_search_results.json")
    if os.path.dirname(out_file):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_combos": len(grid),
            "search_elapsed_sec": round(search_elapsed, 2),
            "best_balanced": results_balanced[0] if results_balanced else None,
            "best_rr85": results_rr85[0] if results_rr85 else None,
            "best_min_cer": results_cer[0] if results_cer else None,
            "top_balanced": results_balanced[:10],
            "top_rr85": results_rr85[:10],
            "top_min_cer": results_cer[:10],
            "all_results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nFull grid search report saved to {out_file}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/datasetA")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--asr-cache", default=None)
    parser.add_argument("--phrase-bank", default="data/public_train/aishell1/phrase_bank.txt")
    parser.add_argument("--out", default="data/datasetA/grid_search_results.json")
    run_grid_search(parser.parse_args())


if __name__ == "__main__":
    main()
