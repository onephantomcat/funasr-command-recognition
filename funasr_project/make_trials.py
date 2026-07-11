# -*- coding: utf-8 -*-
"""
从 AISHELL-1 测试集(真实录音, 20说话人)构建评测试验集 (申报表"多维评测桶"):
  clean     目标说话人干净语音            -> 应放行, CER 评测
  overlap5  目标 + 干扰说话人混叠 SIR=5dB -> 应放行并识别目标内容 (多人混叠桶)
  overlap0  目标 + 干扰说话人混叠 SIR=0dB -> 同上, 更难
  babble5   目标 + 4人嘈杂人声 SNR=5dB    -> 应放行 (真实人声噪声桶)
  nontarget 干扰说话人单独说话            -> 应拒识
每个目标说话人用3条音频拼接注册声纹。输出 data/trials/ + trials.jsonl
用法: python make_trials.py [wav根目录] (默认 data/aishell_test)
"""
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

SR = 16000
N_TARGET_SPK = 4        # 目标说话人数
N_INTERF_SPK = 4        # 干扰说话人数(与目标不相交)
N_ENROLL = 3            # 注册用音频条数
N_TRIAL = 3             # 每桶每说话人试验条数
OUT = "data/trials"


def load_transcripts(csv_path="data/aishell1_test.csv"):
    m = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0].endswith(".wav"):
                m[os.path.basename(row[0]).replace(".wav", "")] = row[1]
    return m


def scan_wavs(root):
    """返回 {speaker: [(utt_id, path), ...]}"""
    spk = defaultdict(list)
    for p in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True):
        utt = os.path.basename(p).replace(".wav", "")
        m = re.search(r"(S\d{4})", utt)
        if m:
            spk[m.group(1)].append((utt, p))
    for v in spk.values():
        v.sort()
    return spk


def read_wav(path):
    x, sr = sf.read(path)
    assert sr == SR, f"{path} 采样率 {sr} != {SR}"
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def mix_at_ratio(target, interf, ratio_db):
    """按信干比混叠: 干扰循环/截断到目标长度"""
    if len(interf) < len(target):
        interf = np.tile(interf, len(target) // len(interf) + 1)
    interf = interf[:len(target)]
    pt, pi = np.mean(target ** 2), np.mean(interf ** 2) + 1e-12
    k = np.sqrt(pt / (pi * 10 ** (ratio_db / 10)))
    mixed = target + k * interf
    return (mixed / max(1e-9, np.abs(mixed).max()) * 0.9).astype(np.float32)


def main(root):
    os.makedirs(OUT, exist_ok=True)
    texts = load_transcripts()
    spk = scan_wavs(root)
    spks = sorted(s for s, utts in spk.items() if len(utts) >= N_ENROLL + N_TRIAL * 3)
    assert len(spks) >= N_TARGET_SPK + N_INTERF_SPK, f"说话人不足: {len(spks)}"
    targets, interfs = spks[:N_TARGET_SPK], spks[N_TARGET_SPK:N_TARGET_SPK + N_INTERF_SPK]
    print(f"目标说话人: {targets}\n干扰说话人: {interfs}")

    rng = np.random.default_rng(2026)
    trials = []
    for ti, tspk in enumerate(targets):
        utts = spk[tspk]
        # 注册: 拼接前 N_ENROLL 条
        enroll_wav = f"{OUT}/enroll_{tspk}.wav"
        sf.write(enroll_wav, np.concatenate([read_wav(p) for _, p in utts[:N_ENROLL]]), SR)
        pool = utts[N_ENROLL:]
        ispk = interfs[ti % len(interfs)]
        ipool = spk[ispk]

        def add(idx_base, kind, make_audio, accept, ref_of):
            for j in range(N_TRIAL):
                utt, path = pool[idx_base + j]
                out = f"{OUT}/{kind}_{tspk}_{j}.wav"
                make_audio(path, ipool[j + 1][1], out)
                trials.append({"wav": out, "type": kind, "target_spk": tspk,
                               "enroll": enroll_wav, "accept": accept,
                               "ref": ref_of(utt)})

        cp = lambda src, _i, out: sf.write(out, read_wav(src), SR)
        ov5 = lambda src, i, out: sf.write(out, mix_at_ratio(read_wav(src), read_wav(i), 5), SR)
        ov0 = lambda src, i, out: sf.write(out, mix_at_ratio(read_wav(src), read_wav(i), 0), SR)

        def bb5(src, _i, out):
            t = read_wav(src)
            picks = rng.choice(len(ipool), 4, replace=False)
            babble = np.zeros(len(t), dtype=np.float32)
            for k in picks:
                w = read_wav(ipool[k][1])
                if len(w) < len(t):
                    w = np.tile(w, len(t) // len(w) + 1)
                babble += w[:len(t)]
            sf.write(out, mix_at_ratio(t, babble, 5), SR)

        add(0, "clean", cp, True, lambda u: texts.get(u, ""))
        add(N_TRIAL, "overlap5", ov5, True, lambda u: texts.get(u, ""))
        add(N_TRIAL * 2, "overlap0", ov0, True, lambda u: texts.get(u, ""))
        add(0, "babble5", bb5, True, lambda u: texts.get(u, ""))
        # 非目标: 干扰说话人单独音频, 应拒识
        for j in range(N_TRIAL):
            utt, path = ipool[j + 1 + N_TRIAL]
            out = f"{OUT}/nontarget_{tspk}_{j}.wav"
            sf.write(out, read_wav(path), SR)
            trials.append({"wav": out, "type": "nontarget", "target_spk": tspk,
                           "enroll": enroll_wav, "accept": False, "ref": ""})

    with open(f"{OUT}/trials.jsonl", "w", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"共 {len(trials)} 条试验 -> {OUT}/trials.jsonl")
    from collections import Counter
    print(Counter(t["type"] for t in trials))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/aishell_test")
