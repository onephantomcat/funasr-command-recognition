# -*- coding: utf-8 -*-
"""Precompute per-row gate-training items (sim/intent/hyp/ref) from datasetA caches.

intent is computed on normalize(hyp) -> pinyin edit distance, EXACTLY matching
eval_datasetA.nearest_intent, so gate training and inference see identical
features (this removes the train/serve skew of the old space-only pipeline).

The output pickle feeds train_lightweight_gate.py --items-cache.

Usage:
  # build items from the base (whole-file) embedding cache + ASR cache:
  python precompute_items.py --out data/datasetA/items_precomputed.pkl

  # optionally rebuild the embedding cache with VAD-trimmed, chunk-averaged
  # embeddings, then build items from it:
  python precompute_items.py --rebuild-emb-vad --emb-cache data/datasetA/emb_cache_vad_local.pkl
  python precompute_items.py --emb-cache data/datasetA/emb_cache_vad_local.pkl --out data/datasetA/items_vad.pkl
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import time

from command_match import edit_distance, to_pinyin
from text_norm import normalize
from train_lightweight_gate import build_phrase_bank, cosine_sim

ROOT = "data/datasetA"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_rows(root):
    for split in ("pos", "neg"):
        for row in read_jsonl(os.path.join(root, f"{split}.jsonl")):
            yield split, row


def best_intent(hyp_py, phrases, phrases_py):
    """Min normalized pinyin edit distance to the phrase bank (+ nearest phrase)."""
    if not hyp_py:
        return 1.0, ""
    best = 1.0
    best_phrase = ""
    lh = len(hyp_py)
    for phrase, py in zip(phrases, phrases_py):
        lp = len(py)
        if abs(lh - lp) / max(lh, lp) >= best:  # length lower bound cannot beat best
            continue
        d = edit_distance(hyp_py, py)
        score = d / max(lh, lp)
        if score < best:
            best, best_phrase = score, phrase
        if best == 0.0:
            break
    return best, best_phrase


def build_items(root, emb_cache_path, asr_cache_path, out_path):
    """Build items with normalize-based intent (matches inference exactly)."""
    t0 = time.time()
    emb = pickle.load(open(emb_cache_path, "rb"))
    asr = pickle.load(open(asr_cache_path, "rb"))
    phrases = build_phrase_bank(root)
    phrases_py = [to_pinyin(p) for p in phrases]
    print(f"phrases={len(phrases)}  emb={len(emb)}  asr={len(asr)}")

    rows = list(iter_rows(root))
    items = []
    for i, (split, row) in enumerate(rows, 1):
        wake = os.path.abspath(os.path.join(root, row["唤醒音频"]))
        cmd = os.path.abspath(os.path.join(root, row["识别音频"]))
        hyp = asr.get(cmd, {}).get("text", "")
        sim = cosine_sim(emb[wake], emb[cmd]) if wake in emb and cmd in emb else None
        hyp_norm = normalize(hyp or "")
        intent, nearest = best_intent(to_pinyin(hyp_norm) if hyp_norm else "",
                                      phrases, phrases_py)
        items.append({
            "split": split, "id": row.get("id"),
            "wake": wake, "cmd": cmd,
            "sim": sim, "intent": intent, "hyp": hyp,
            "nearest_phrase": nearest,
            "ref": row.get("识别文本") or "",
        })
        if i % 200 == 0 or i == len(rows):
            print(f"{i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
            pickle.dump(items, open(out_path, "wb"))
    pickle.dump(items, open(out_path, "wb"))
    print(f"saved {len(items)} items -> {out_path}  ({time.time()-t0:.0f}s)")


def rebuild_embeddings_vad(root, out_path):
    """Rebuild the embedding cache with VAD-trimmed, chunk-averaged embeddings."""
    from speaker_verify import build_sv_model, build_vad_model, extract_embedding

    paths = []
    for _split, row in iter_rows(root):
        for key in ("唤醒音频", "识别音频"):
            p = os.path.abspath(os.path.join(root, row[key]))
            if p not in paths:
                paths.append(p)
    print(f"unique audio files: {len(paths)}")

    done = {}
    if os.path.exists(out_path):
        done = pickle.load(open(out_path, "rb"))
        print(f"resume: {len(done)} already done")

    sv = build_sv_model()
    vad = build_vad_model()
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        if p in done:
            continue
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            done[p] = extract_embedding(sv, p, vad_model=vad)
        if i % 100 == 0 or i == len(paths):
            print(f"{i}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
            pickle.dump(done, open(out_path, "wb"))
    pickle.dump(done, open(out_path, "wb"))
    print(f"saved {len(done)} embeddings -> {out_path}  ({time.time()-t0:.0f}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--emb-cache", default=os.path.join(ROOT, "emb_cache_tune_local.pkl"),
                        help="embedding cache to read (build items) or write (--rebuild-emb-vad).")
    parser.add_argument("--asr-cache", default=os.path.join(ROOT, "asr_cache_tune_local.pkl"))
    parser.add_argument("--out", default=os.path.join(ROOT, "items_precomputed.pkl"))
    parser.add_argument("--rebuild-emb-vad", action="store_true",
                        help="rebuild --emb-cache with VAD-trimmed embeddings instead of building items.")
    args = parser.parse_args()

    if args.rebuild_emb_vad:
        rebuild_embeddings_vad(args.root, args.emb_cache)
    else:
        build_items(args.root, args.emb_cache, args.asr_cache, args.out)


if __name__ == "__main__":
    main()
