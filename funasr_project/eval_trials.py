# -*- coding: utf-8 -*-
"""
分桶评测 (申报表 Step 1/5: 基线指标 + 分桶归因分析):
对 trials.jsonl 逐条跑 声纹门控 -> (放行)Paraformer识别, 统计:
  各桶 CER (目标被误拒按全删计 CER=1)
  目标误拒率 FRR / 非目标误受率 FAR
输出 data/trials/eval_report.json
用法: python eval_trials.py [阈值]
"""
import json
import sys
import time
from collections import defaultdict

from asr_demo import build_model, recognize
from cer import cer
from speaker_verify import SpeakerGate, build_sv_model
from text_norm import normalize


def main(threshold=0.6):
    trials = [json.loads(l) for l in open("data/trials/trials.jsonl", encoding="utf-8")]
    print(f"{len(trials)} 条试验, 拒识阈值 {threshold}")
    print("加载模型...")
    gate = SpeakerGate(build_sv_model(), threshold=threshold)
    asr = build_model(with_punc=False)

    enrolled = {}
    buckets = defaultdict(list)
    t0 = time.time()
    for i, tr in enumerate(trials):
        if tr["enroll"] not in enrolled:
            gate.enroll(tr["enroll"])
            enrolled[tr["enroll"]] = gate.target_emb
        gate.target_emb = enrolled[tr["enroll"]]
        ok, sim = gate.verify(tr["wav"])
        hyp = ""
        if ok:
            hyp, _ = recognize(asr, tr["wav"])
        rec = dict(tr, sim=round(sim, 4), accepted=ok, hyp=hyp)
        if tr["accept"]:                       # 目标说话人试验
            c, _ = cer(tr["ref"], hyp if ok else "")
            rec["cer"] = round(c, 4)
        buckets[tr["type"]].append(rec)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(trials)} ({time.time()-t0:.0f}s)")

    report = {"threshold": threshold, "buckets": {}}
    print(f"\n{'桶':<12s}{'条数':<6s}{'放行率':<8s}{'平均CER':<10s}说明")
    print("-" * 64)
    for kind in ("clean", "babble5", "overlap5", "overlap0", "nontarget"):
        rs = buckets.get(kind, [])
        if not rs:
            continue
        acc_rate = sum(r["accepted"] for r in rs) / len(rs)
        cers = [r["cer"] for r in rs if "cer" in r]
        avg_cer = sum(cers) / len(cers) if cers else None
        report["buckets"][kind] = {
            "n": len(rs), "accept_rate": round(acc_rate, 3),
            "cer": round(avg_cer, 4) if avg_cer is not None else None,
        }
        note = "应放行" if (rs[0]["accept"]) else "应拒识"
        cer_s = f"{avg_cer:.3f}" if avg_cer is not None else "-"
        print(f"{kind:<12s}{len(rs):<6d}{acc_rate:<8.2f}{cer_s:<10s}{note}")

    tgt = [r for k in ("clean", "babble5", "overlap5", "overlap0") for r in buckets[k]]
    non = buckets["nontarget"]
    frr = sum(not r["accepted"] for r in tgt) / max(1, len(tgt))
    far = sum(r["accepted"] for r in non) / max(1, len(non))
    report["FRR_target_rejected"] = round(frr, 4)
    report["FAR_nontarget_accepted"] = round(far, 4)
    report["details"] = [r for rs in buckets.values() for r in rs]
    print(f"\n目标误拒率 FRR = {frr:.3f}   非目标误受率 FAR = {far:.3f}")
    print(f"总耗时 {time.time()-t0:.0f}s")

    with open("data/trials/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告已存: data/trials/eval_report.json")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.6)
