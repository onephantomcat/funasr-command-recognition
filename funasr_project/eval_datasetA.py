# -*- coding: utf-8 -*-
"""
Evaluate datasetA:
  - enroll target speaker from "唤醒音频"
  - run ASR only if "识别音频" passes speaker verification
  - pos: compute CER against "识别文本"; rejected pos is scored as empty output
  - neg: count rejection rate, where empty output means rejected

Usage:
  ./venv/bin/python eval_datasetA.py [--limit N]
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from asr_demo import build_model, recognize
from cer import cer, corpus_cer
from command_match import edit_distance
from lightweight_gate import accept as gate_accept
from lightweight_gate import load_gate_model, make_features as make_gate_features
from speaker_verify import build_sv_model, cosine_sim, extract_embedding
from target_purify import purify_audio
from text_norm import normalize


ROOT = "data/datasetA"
DATASETA_DEFAULT_HARD_SV_THRESHOLD = 0.30
DATASETA_DEFAULT_FUSION_PRE_SV_THRESHOLD = 0.0
DATASETA_DEFAULT_INTENT_THRESHOLD = 0.45
DATASETA_DEFAULT_DECISION_POLICY = "fusion"
DATASETA_DEFAULT_FUSION_WEIGHT = 0.70
DATASETA_DEFAULT_FUSION_THRESHOLD = 0.03
DATASETA_DEFAULT_ALLOWED_WAKE_TEXTS = "hi colmo,hicolmo,你好科慕"
DATASETA_DEFAULT_PHRASE_THRESHOLD = 0.30  # Lowered for pinyin distance


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EmbeddingCache:
    def __init__(self, path=None):
        self.path = path
        self.data = {}
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                self.data = pickle.load(f)

    def key(self, path):
        return os.path.abspath(path)

    def get(self, model, path):
        key = self.key(path)
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

    def key(self, path):
        return os.path.abspath(path)

    def recognize(self, model, path):
        key = self.key(path)
        if key in self.data:
            return self.data[key]["text"], 0.0, True
        text, elapsed = recognize_quiet(model, path)
        self.data[key] = {"text": text, "elapsed": elapsed}
        return text, elapsed, False

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f"{self.path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)


def recognize_quiet(model, path):
    """FunASR prints per-file progress bars; hide them so long runs stay readable."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return recognize(model, path)


def extract_embedding_quiet(sv_model, path):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return extract_embedding(sv_model, path)


def should_purify(args, speaker_similarity):
    if not args.purify or args.asr_only:
        return False
    if args.purify_sim_trigger is None:
        return True
    return speaker_similarity is not None and speaker_similarity <= args.purify_sim_trigger


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def wav_path(root, rel_path):
    return os.path.join(root, rel_path)


def normalize_wake_text(text):
    return (text or "").replace(" ", "").lower()


def build_report(args, pos_rows, neg_rows, details, pairs, rejected, started_at, complete):
    pos_done = sum(1 for d in details if d["split"] == "pos")
    neg_done = sum(1 for d in details if d["split"] == "neg")
    pos_cers = [d["cer"] for d in details if d["split"] == "pos"]
    pos_accepted = sum(1 for d in details if d["split"] == "pos" and d.get("accepted", True))
    sent_cer = sum(pos_cers) / max(1, pos_done)
    total_cer, total_chars = corpus_cer(pairs)
    rr = rejected / max(1, neg_done)
    return {
        "dataset": "datasetA",
        "root": args.root,
        "complete": complete,
        "mode": "asr_only" if args.asr_only else "speaker_gate_asr",
        "speaker_threshold": None if args.asr_only else args.sv_threshold,
        "intent_filter": False if args.asr_only else args.intent_filter,
        "intent_threshold": None if args.asr_only or not args.intent_filter else args.intent_threshold,
        "decision_policy": "asr_only" if args.asr_only else args.decision_policy,
        "gate_model": None if args.asr_only else args.gate_model,
        "fusion_weight": None if args.asr_only or args.decision_policy != "fusion" else args.fusion_weight,
        "fusion_threshold": None if args.asr_only or args.decision_policy != "fusion" else args.fusion_threshold,
        "wake_guard": False if args.asr_only else args.wake_guard,
        "allowed_wake_texts": [] if args.asr_only or not args.wake_guard else args.allowed_wake_texts,
        "phrase_correct": False if args.asr_only else args.phrase_correct,
        "phrase_threshold": None if args.asr_only or not args.phrase_correct else args.phrase_threshold,
        "phrase_bank": None if args.asr_only else args.phrase_bank,
        "use_test_label_phrase_bank": False if args.asr_only else args.use_test_label_phrase_bank,
        "purify": False if args.asr_only else args.purify,
        "purify_sim_trigger": (
            None if args.asr_only or not args.purify else args.purify_sim_trigger
        ),
        "embedding_cache": bool(args.embedding_cache),
        "asr_cache": bool(args.asr_cache),
        "pos_n": len(pos_rows),
        "neg_n": len(neg_rows),
        "pos_processed": pos_done,
        "neg_processed": neg_done,
        "positive_accept_rate": round(pos_accepted / max(1, pos_done), 4),
        "positive_sentence_avg_cer": round(sent_cer, 4),
        "positive_corpus_cer": round(total_cer, 4),
        "positive_ref_chars": total_chars,
        "negative_rejection_rate_rr": round(rr, 4),
        "negative_rejected": rejected,
        "elapsed_sec": round(time.time() - started_at, 2),
        "details": details,
    }


def save_report(out, report):
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out)


def build_submission(report):
    results = []
    for item in report["details"]:
        label = item.get("ref") or ""
        cer_value = item.get("cer", "")
        if isinstance(cer_value, float):
            cer_value = f"{cer_value:.4f}"
        results.append({
            "id": Path(item["audio"]).stem,
            "content": normalize(item.get("hyp", "")),
            "label": label,
            "cer": cer_value,
        })
    return {
        "result": {
            "results": results,
            "final_cer": f"{report['positive_corpus_cer']:.4f}",
            "duration": f"{report['elapsed_sec']:.2f}",
        }
    }


def build_intent_phrases(root, phrase_bank=None, use_test_label_phrase_bank=False):
    phrases = []
    if phrase_bank:
        path = Path(phrase_bank)
        if not path.exists():
            raise FileNotFoundError(f"phrase bank not found: {phrase_bank}")
        if path.suffix.lower() == ".jsonl":
            for row in read_jsonl(path):
                text = normalize(
                    row.get("识别文本")
                    or row.get("识别文本标签")
                    or row.get("text")
                    or row.get("label")
                    or row.get("command")
                    or ""
                )
                if text and text not in phrases:
                    phrases.append(text)
        elif path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = data.get("phrases") or data.get("commands") or data.get("labels") or []
            for item in data:
                text = normalize(item if isinstance(item, str) else item.get("text", ""))
                if text and text not in phrases:
                    phrases.append(text)
        else:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    text = normalize(line.strip())
                    if text and text not in phrases:
                        phrases.append(text)
        return phrases

    if not use_test_label_phrase_bank:
        return phrases

    pos_jsonl = os.path.join(root, "pos.jsonl")
    if not os.path.exists(pos_jsonl):
        return phrases
    for row in read_jsonl(pos_jsonl):
        text = normalize(row.get("识别文本") or "")
        if text and text not in phrases:
            phrases.append(text)
    return phrases


def nearest_intent(text, phrases):
    from command_match import to_pinyin
    hyp = normalize(text)
    if not hyp:
        return 1.0, ""
    best_score, best_phrase = 1.0, ""
    
    # Use sliding window pinyin match to simulate command_match behavior
    hyp_py = to_pinyin(hyp)
    for phrase in phrases:
        phrase_py = to_pinyin(phrase)
        # Use pinyin edit distance to handle homophones
        dist = edit_distance(hyp_py, phrase_py)
        score = dist / max(len(hyp_py), len(phrase_py))
        if score < best_score:
            best_score, best_phrase = score, phrase
    return best_score, best_phrase


def main():
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit rows from each split for a quick smoke test.")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip this many rows from each split before applying --limit.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--submission-out", default=None,
                        help="Optional contest-format JSON output path.")
    parser.add_argument("--asr-only", action="store_true",
                        help="Disable speaker rejection and run the old pure-ASR baseline.")
    parser.add_argument("--sv-threshold", type=float, default=None,
                        help="Speaker pre-gate threshold. Defaults: 0.0 for fusion, 0.30 for hard.")
    parser.add_argument("--intent-filter", action=argparse.BooleanOptionalAction, default=True,
                        help="Reject accepted audio when ASR text is far from the datasetA command phrase bank.")
    parser.add_argument("--intent-threshold", type=float, default=DATASETA_DEFAULT_INTENT_THRESHOLD,
                        help="Maximum normalized distance to a known command phrase when --intent-filter is enabled.")
    parser.add_argument("--decision-policy", choices=("hard", "fusion"), default=DATASETA_DEFAULT_DECISION_POLICY,
                        help="hard: speaker threshold + intent threshold; fusion: sim - w*intent joint score.")
    parser.add_argument("--gate-model", default=None,
                        help="Optional tiny trained gate JSON from train_lightweight_gate.py.")
    parser.add_argument("--fusion-weight", type=float, default=DATASETA_DEFAULT_FUSION_WEIGHT,
                        help="Intent penalty weight for --decision-policy fusion.")
    parser.add_argument("--fusion-threshold", type=float, default=DATASETA_DEFAULT_FUSION_THRESHOLD,
                        help="Minimum joint score for --decision-policy fusion.")
    parser.add_argument("--wake-guard", action=argparse.BooleanOptionalAction, default=False,
                        help="Reject rows whose wake text is not in --allowed-wake-texts.")
    parser.add_argument("--allowed-wake-texts", default=DATASETA_DEFAULT_ALLOWED_WAKE_TEXTS,
                        help="Comma-separated target wake texts for --wake-guard.")
    parser.add_argument("--phrase-correct", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize accepted ASR output to the nearest known datasetA phrase.")
    parser.add_argument("--phrase-threshold", type=float, default=DATASETA_DEFAULT_PHRASE_THRESHOLD,
                        help="Maximum nearest-phrase distance for --phrase-correct.")
    parser.add_argument("--phrase-bank", default=None,
                        help="External train-set phrase bank (.txt/.json/.jsonl) for intent filtering and correction.")
    parser.add_argument("--use-test-label-phrase-bank", action="store_true",
                        help="Legacy tuning mode: build phrase bank from datasetA positive labels. Do not use for fair test evaluation.")
    parser.add_argument("--embedding-cache", default=None,
                        help="Optional pickle cache for speaker embeddings during tuning runs.")
    parser.add_argument("--asr-cache", default=None,
                        help="Optional pickle cache for ASR text during tuning runs; do not use for formal timing.")
    parser.add_argument("--purify", action="store_true",
                        help="Run optional target-speaker purification before ASR.")
    parser.add_argument("--purify-dir", default=os.path.join(ROOT, "purified_cache"),
                        help="Directory for purified wav files.")
    parser.add_argument("--purify-keep-ratio", type=float, default=0.45)
    parser.add_argument("--purify-floor-gain", type=float, default=0.03)
    parser.add_argument(
        "--purify-sim-trigger",
        type=float,
        default=None,
        help="Only purify accepted audio at or below this speaker similarity; default purifies all accepted audio.",
    )
    args = parser.parse_args()
    if args.sv_threshold is None:
        args.sv_threshold = (
            DATASETA_DEFAULT_FUSION_PRE_SV_THRESHOLD
            if args.decision_policy == "fusion"
            else DATASETA_DEFAULT_HARD_SV_THRESHOLD
        )
    args.allowed_wake_texts = list(dict.fromkeys(
        normalize_wake_text(x) for x in args.allowed_wake_texts.split(",") if x.strip()
    ))

    out = args.out or os.path.join(args.root, "eval_report.json")
    pos_rows = list(read_jsonl(os.path.join(args.root, "pos.jsonl")))
    neg_rows = list(read_jsonl(os.path.join(args.root, "neg.jsonl")))
    if args.offset:
        pos_rows = pos_rows[args.offset:]
        neg_rows = neg_rows[args.offset:]
    if args.limit is not None:
        pos_rows = pos_rows[:args.limit]
        neg_rows = neg_rows[:args.limit]

    mode = "ASR only" if args.asr_only else f"speaker gate + ASR (threshold={args.sv_threshold})"
    print(f"datasetA: pos={len(pos_rows)} neg={len(neg_rows)}")
    print(f"Mode: {mode}")
    gate_model = None
    if args.gate_model and not args.asr_only:
        gate_model = load_gate_model(args.gate_model)

    if not args.asr_only:
        if args.decision_policy == "fusion":
            print(f"Decision policy: fusion score = sim - {args.fusion_weight}*intent >= {args.fusion_threshold}")
        else:
            print("Decision policy: hard thresholds")
        if gate_model:
            print(f"Trained gate: {args.gate_model} threshold={gate_model.get('threshold', 0.5)}")
        if args.wake_guard:
            print(f"Wake guard: {args.allowed_wake_texts}")
        if args.phrase_correct:
            print(f"Phrase correction: threshold={args.phrase_threshold}")
        if args.purify:
            print(f"Target purification: keep_ratio={args.purify_keep_ratio}, floor_gain={args.purify_floor_gain}")
    intent_phrases = []
    if (args.intent_filter or gate_model) and not args.asr_only:
        intent_phrases = build_intent_phrases(
            args.root,
            phrase_bank=args.phrase_bank,
            use_test_label_phrase_bank=args.use_test_label_phrase_bank,
        )
        print(f"Intent filter: threshold={args.intent_threshold}, phrases={len(intent_phrases)}")
    sv_model = None
    emb_cache = EmbeddingCache(args.embedding_cache)
    asr_cache = AsrCache(args.asr_cache)
    if not args.asr_only:
        print("Loading speaker model...")
        sv_model = build_sv_model()
    print("Loading ASR model...")
    model = build_model(with_punc=False)

    details = []
    pairs = []
    rejected = 0
    t0 = time.time()

    print("\nRunning positive samples...")
    for i, row in enumerate(pos_rows, 1):
        wake_path = wav_path(args.root, row["唤醒音频"])
        path = wav_path(args.root, row["识别音频"])
        t_item = time.time()
        wake_text = row.get("唤醒文本", "")
        wake_allowed = (
            args.asr_only
            or (not args.wake_guard)
            or normalize_wake_text(wake_text) in args.allowed_wake_texts
        )
        sim = None
        accepted = wake_allowed
        if not args.asr_only:
            if accepted:
                wake_emb = emb_cache.get(sv_model, wake_path)
                cmd_emb = emb_cache.get(sv_model, path)
                sim = cosine_sim(wake_emb, cmd_emb)
                accepted = sim >= args.sv_threshold
        asr_path = path
        purify_info = None
        purify_applied = accepted and should_purify(args, sim)
        if purify_applied:
            purify_name = f"pos_{Path(path).stem}_kr{args.purify_keep_ratio:.2f}_fg{args.purify_floor_gain:.2f}.wav"
            asr_path = os.path.join(args.purify_dir, purify_name)
            if not os.path.exists(asr_path):
                purify_info = purify_audio(
                    sv_model, wake_path, path, asr_path,
                    keep_ratio=args.purify_keep_ratio,
                    floor_gain=args.purify_floor_gain,
                )
        hyp, asr_elapsed, asr_cached = asr_cache.recognize(model, asr_path) if accepted else ("", 0.0, False)
        raw_hyp = hyp
        intent_score = None
        nearest_phrase = ""
        decision_score = None
        gate_probability = None
        speaker_accepted = accepted
        if accepted and intent_phrases:
            intent_score, nearest_phrase = nearest_intent(hyp, intent_phrases)
            if gate_model:
                features = make_gate_features(
                    sim, hyp, intent_score,
                    fusion_weight=gate_model.get("fusion_weight_for_features", args.fusion_weight),
                )
                accepted, gate_probability = gate_accept(gate_model, features)
                decision_score = gate_probability
            elif args.intent_filter and args.decision_policy == "fusion":
                decision_score = sim - args.fusion_weight * intent_score
                accepted = decision_score >= args.fusion_threshold
            elif args.intent_filter:
                accepted = intent_score <= args.intent_threshold
            if (gate_model or args.intent_filter) and not accepted:
                hyp = ""
            elif args.phrase_correct and intent_score <= args.phrase_threshold:
                hyp = nearest_phrase
        elapsed = time.time() - t_item
        ref = row["识别文本"] or ""
        c, ref_len = cer(ref, hyp)
        pairs.append((ref, hyp))
        details.append({
            "split": "pos",
            "id": row.get("id"),
            "wake_text": wake_text,
            "wake_allowed": wake_allowed,
            "wake_audio": wake_path,
            "audio": path,
            "asr_audio": asr_path,
            "ref": ref,
            "hyp": hyp,
            "raw_hyp": raw_hyp,
            "speaker_similarity": round(sim, 4) if sim is not None else None,
            "speaker_accepted": speaker_accepted,
            "intent_score": round(intent_score, 4) if intent_score is not None else None,
            "nearest_phrase": nearest_phrase,
            "decision_score": round(decision_score, 4) if decision_score is not None else None,
            "gate_probability": round(gate_probability, 4) if gate_probability is not None else None,
            "purify_info": purify_info,
            "purify_applied": purify_applied,
            "accepted": accepted,
            "cer": round(c, 4),
            "ref_len": ref_len,
            "asr_latency_sec": round(asr_elapsed, 3),
            "asr_cached": asr_cached,
            "latency_sec": round(elapsed, 3),
        })
        if i % 50 == 0 or i == len(pos_rows):
            print(f"  pos {i}/{len(pos_rows)}")
            save_report(out, build_report(args, pos_rows, neg_rows, details, pairs,
                                          rejected, t0, complete=False))

    print("\nRunning negative samples...")
    for i, row in enumerate(neg_rows, 1):
        wake_path = wav_path(args.root, row["唤醒音频"])
        path = wav_path(args.root, row["识别音频"])
        t_item = time.time()
        wake_text = row.get("唤醒文本", "")
        wake_allowed = (
            args.asr_only
            or (not args.wake_guard)
            or normalize_wake_text(wake_text) in args.allowed_wake_texts
        )
        sim = None
        accepted = wake_allowed
        if not args.asr_only:
            if accepted:
                wake_emb = emb_cache.get(sv_model, wake_path)
                cmd_emb = emb_cache.get(sv_model, path)
                sim = cosine_sim(wake_emb, cmd_emb)
                accepted = sim >= args.sv_threshold
        asr_path = path
        purify_info = None
        purify_applied = accepted and should_purify(args, sim)
        if purify_applied:
            purify_name = f"neg_{Path(path).stem}_kr{args.purify_keep_ratio:.2f}_fg{args.purify_floor_gain:.2f}.wav"
            asr_path = os.path.join(args.purify_dir, purify_name)
            if not os.path.exists(asr_path):
                purify_info = purify_audio(
                    sv_model, wake_path, path, asr_path,
                    keep_ratio=args.purify_keep_ratio,
                    floor_gain=args.purify_floor_gain,
                )
        hyp, asr_elapsed, asr_cached = asr_cache.recognize(model, asr_path) if accepted else ("", 0.0, False)
        raw_hyp = hyp
        intent_score = None
        nearest_phrase = ""
        decision_score = None
        gate_probability = None
        speaker_accepted = accepted
        if accepted and intent_phrases:
            intent_score, nearest_phrase = nearest_intent(hyp, intent_phrases)
            if gate_model:
                features = make_gate_features(
                    sim, hyp, intent_score,
                    fusion_weight=gate_model.get("fusion_weight_for_features", args.fusion_weight),
                )
                accepted, gate_probability = gate_accept(gate_model, features)
                decision_score = gate_probability
            elif args.intent_filter and args.decision_policy == "fusion":
                decision_score = sim - args.fusion_weight * intent_score
                accepted = decision_score >= args.fusion_threshold
            elif args.intent_filter:
                accepted = intent_score <= args.intent_threshold
            if (gate_model or args.intent_filter) and not accepted:
                hyp = ""
            elif args.phrase_correct and intent_score <= args.phrase_threshold:
                hyp = nearest_phrase
        elapsed = time.time() - t_item
        is_rejected = not hyp.strip()
        rejected += int(is_rejected)
        details.append({
            "split": "neg",
            "id": row.get("id"),
            "wake_text": wake_text,
            "wake_allowed": wake_allowed,
            "wake_audio": wake_path,
            "audio": path,
            "asr_audio": asr_path,
            "hyp": hyp,
            "raw_hyp": raw_hyp,
            "speaker_similarity": round(sim, 4) if sim is not None else None,
            "speaker_accepted": speaker_accepted,
            "intent_score": round(intent_score, 4) if intent_score is not None else None,
            "nearest_phrase": nearest_phrase,
            "decision_score": round(decision_score, 4) if decision_score is not None else None,
            "gate_probability": round(gate_probability, 4) if gate_probability is not None else None,
            "purify_info": purify_info,
            "purify_applied": purify_applied,
            "accepted": accepted,
            "rejected": is_rejected,
            "asr_latency_sec": round(asr_elapsed, 3),
            "asr_cached": asr_cached,
            "latency_sec": round(elapsed, 3),
        })
        if i % 50 == 0 or i == len(neg_rows):
            print(f"  neg {i}/{len(neg_rows)}")
            save_report(out, build_report(args, pos_rows, neg_rows, details, pairs,
                                          rejected, t0, complete=False))

    report = build_report(args, pos_rows, neg_rows, details, pairs, rejected, t0, complete=True)
    save_report(out, report)
    emb_cache.save()
    asr_cache.save()
    if args.submission_out:
        save_report(args.submission_out, build_submission(report))

    print("\nSummary")
    print(f"  positive accept rate:      {report['positive_accept_rate']:.4f}")
    print(f"  positive sentence avg CER: {report['positive_sentence_avg_cer']:.4f}")
    print(f"  positive corpus CER:       {report['positive_corpus_cer']:.4f} "
          f"({report['positive_ref_chars']} chars)")
    print(f"  negative RR:               {report['negative_rejection_rate_rr']:.4f} "
          f"({rejected}/{len(neg_rows)})")
    print(f"  elapsed:                   {time.time() - t0:.1f}s")
    print(f"Report saved: {out}")
    if args.submission_out:
        print(f"Submission saved: {args.submission_out}")


if __name__ == "__main__":
    main()
