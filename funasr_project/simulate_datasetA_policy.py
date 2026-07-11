# -*- coding: utf-8 -*-
"""
Simulate datasetA joint rejection policies from cached reports.

This does not run ASR or speaker models. It combines:
  - an ASR report, e.g. data/datasetA/eval_report_100.json
  - a gate sweep report, e.g. data/datasetA/gate_sweep_100_fine.json

Usage:
  ./venv/bin/python simulate_datasetA_policy.py
"""
import argparse
import json
import os

from cer import corpus_cer
from command_match import edit_distance
from text_norm import normalize


ROOT = "data/datasetA"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--asr-report", default=os.path.join(ROOT, "eval_report_100.json"))
    parser.add_argument("--gate-report", default=os.path.join(ROOT, "gate_sweep_100_fine.json"))
    parser.add_argument("--sv-thresholds", default="0.26,0.28,0.30,0.32,0.34,0.36,0.38,0.40")
    parser.add_argument("--intent-thresholds", default="0.25,0.35,0.45,0.55")
    args = parser.parse_args()

    phrases = build_phrase_bank(args.root)
    asr = json.load(open(args.asr_report, encoding="utf-8"))
    gate = json.load(open(args.gate_report, encoding="utf-8"))
    sims = {(d["split"], d["audio"]): d["sim"] for d in gate["scores"]}

    items = []
    for item in asr["details"]:
        key = (item["split"], item["audio"])
        if key not in sims:
            continue
        merged = dict(item)
        merged["sim"] = sims[key]
        merged["intent_score"] = intent_distance(item.get("hyp", ""), phrases)
        items.append(merged)

    sv_thresholds = [float(x) for x in args.sv_thresholds.split(",") if x.strip()]
    intent_thresholds = [float(x) for x in args.intent_thresholds.split(",") if x.strip()]
    rows = []
    for sv_threshold in sv_thresholds:
        for intent_threshold in intent_thresholds:
            pairs = []
            rejected = 0
            accepted_pos = 0
            for item in items:
                accepted = item["sim"] >= sv_threshold and item["intent_score"] <= intent_threshold
                if item["split"] == "pos":
                    accepted_pos += int(accepted)
                    pairs.append((item["ref"], item["hyp"] if accepted else ""))
                else:
                    rejected += int(not accepted or not item.get("hyp", "").strip())
            pos_n = sum(1 for item in items if item["split"] == "pos")
            neg_n = sum(1 for item in items if item["split"] == "neg")
            cer, chars = corpus_cer(pairs)
            rr = rejected / max(1, neg_n)
            objective_80 = 40 * (1 - cer) + 40 * rr
            rows.append({
                "sv_threshold": sv_threshold,
                "intent_threshold": intent_threshold,
                "positive_accept_rate": accepted_pos / max(1, pos_n),
                "positive_corpus_cer": cer,
                "negative_rr": rr,
                "positive_ref_chars": chars,
                "objective_80": objective_80,
            })

    rows.sort(key=lambda x: x["objective_80"], reverse=True)
    print("Top policies")
    for row in rows[:12]:
        print(
            f"score={row['objective_80']:.2f}/80  "
            f"sv={row['sv_threshold']:.2f} intent={row['intent_threshold']:.2f}  "
            f"pos_accept={row['positive_accept_rate']:.3f}  "
            f"CER={row['positive_corpus_cer']:.4f}  RR={row['negative_rr']:.3f}"
        )


if __name__ == "__main__":
    main()
