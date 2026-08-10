# -*- coding: utf-8 -*-
"""P1 v2_b1 本机替代数据生成脚本（v2 分支版）。

从 data/aishell_test（AISHELL-1 测试集，20 说话人 × ~360 utt）生成
可直接喂给 tools/train_b1_trial.py 的 v2_b1 格式数据集。

路径约定：
  P2_ROOT = Path(__file__).resolve().parents[1]  → P2_project/
  FUNASR_ROOT = P2_ROOT.parent                  → funasr_project/
  共享数据源：FUNASR_ROOT / "data" / "aishell_test"
  产物：P2_ROOT / "data" / "p1_v2_b1"

配置矩阵（9 格 × ABSENT）：
  overlap × SIR ∈ {25%, 50%, 100%} × {-5, 0, 5} dB
  + ABSENT 子集（target 全零 + target_present=false）

说话人分配（aishell_test 20 spk）：
  train:   target spk[0:10]  interferer spk[10:20]
  dev:     target spk[0:5]   interferer spk[10:15]
  eval:    target spk[5:10]  interferer spk[15:20]

产物：data/p1_v2_b1/
  ├── train.jsonl / dev.jsonl / eval.jsonl
  ├── wav/{train,dev,eval}/  混合 wav
  ├── stems/{train,dev,eval}/  target / interferer stem
  ├── activity/{train,dev,eval}/  activity mask (.npy)
  ├── enrollment/  所有说话人 enrollment wav
  ├── README.md / SCHEMA.json / SHA256SUMS.txt / VERSION / FROZEN

用法：
  python tools/build_v2_b1_local.py [--seed 20260804] [--dry_run]
"""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
OUT_ROOT = P2_ROOT / "data" / "p1_v2_b1"
SRC = FUNASR_ROOT / "data" / "aishell_test"

SR = 16000
DUR = 8.0
N = int(SR * DUR)
ACT_THRESH = 1e-4
N_ENROLL_UTT = 3

CONFIG_MATRIX = [
    (0.25, -5.0), (0.25, 0.0), (0.25, 5.0),
    (0.50, -5.0), (0.50, 0.0), (0.50, 5.0),
    (1.00, -5.0), (1.00, 0.0), (1.00, 5.0),
]
REPEAT_PER_CONFIG = 2

N_SPK = 20
SPK_TRAIN_TARGET = list(range(0, 10))
SPK_TRAIN_INTERF = list(range(10, 20))
SPK_DEV_TARGET = list(range(0, 5))
SPK_DEV_INTERF = list(range(10, 15))
SPK_EVAL_TARGET = list(range(5, 10))
SPK_EVAL_INTERF = list(range(15, 20))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_speakers(root):
    spk = defaultdict(list)
    for wav in sorted(root.rglob("*.wav")):
        m = re.search(r"(S\d{4})", wav.stem)
        if m:
            spk[m.group(1)].append((wav.stem, str(wav)))
    for v in spk.values():
        v.sort(key=lambda x: x[0])
    return spk


def read_wav(path, n=N, rng=None):
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} sr={sr} != {SR}"
    if wav.ndim > 1:
        wav = wav[:, 0]
    if len(wav) >= n:
        off = int(rng.integers(0, len(wav) - n + 1)) if rng is not None else 0
        wav = wav[off:off + n]
    else:
        wav = np.pad(wav, (0, n - len(wav)))
    return wav.astype(np.float32)


def place(full_n, seg, start):
    out = np.zeros(full_n, dtype=np.float32)
    out[start:start + len(seg)] = seg
    return out


def scale_to_sir(target, interferer, sir_db, eps=1e-8):
    t_act = target[np.abs(target) > ACT_THRESH]
    i_act = interferer[np.abs(interferer) > ACT_THRESH]
    if len(t_act) == 0 or len(i_act) == 0:
        return interferer
    t_rms = np.sqrt(np.mean(t_act ** 2)) + eps
    i_rms = np.sqrt(np.mean(i_act ** 2)) + eps
    gain = t_rms / i_rms * 10 ** (-sir_db / 20.0)
    return interferer * gain


def make_activity_mask(target):
    return (np.abs(target) > ACT_THRESH).astype(np.float32)


def build_enrollments(spk_data, out_dir, dry_run=False):
    enroll_dir = out_dir / "enrollment"
    enroll_dir.mkdir(parents=True, exist_ok=True)
    enroll_map = {}
    for spk_id, utt_list in sorted(spk_data.items()):
        if len(utt_list) < N_ENROLL_UTT:
            print(f"  [SKIP] {spk_id} only {len(utt_list)} utt")
            continue
        chunk = np.concatenate([read_wav(path, n=SR * 3) for _, path in utt_list[:N_ENROLL_UTT]])
        chunk = chunk[:SR * 3]
        enroll_path = enroll_dir / f"enroll_{spk_id}.wav"
        if not dry_run:
            sf.write(str(enroll_path), chunk, SR, subtype="FLOAT")
        enroll_map[spk_id] = str(enroll_path.relative_to(FUNASR_ROOT)).replace("\\", "/")
    return enroll_map


def synthesize_one(rng, spk_id, interf_spk_id, target_utt_list, interf_utt_list,
                   overlap_ratio, sir_db, enroll_map, out_dir, subset,
                   repeat_idx, is_absent=False):
    avail_tgt = [(u, p) for u, p in target_utt_list
                 if u not in [x[0] for x in target_utt_list[:N_ENROLL_UTT]]]
    if not avail_tgt:
        avail_tgt = target_utt_list

    target = read_wav(avail_tgt[0][1], n=N, rng=rng)
    interferer_clean = read_wav(interf_utt_list[0][1], n=N, rng=rng)

    if is_absent:
        target = np.zeros(N, dtype=np.float32)
        interferer = interferer_clean
    elif overlap_ratio >= 1.0:
        interferer = interferer_clean
    else:
        seg_len = int(N * overlap_ratio)
        seg_start = int(rng.integers(0, N - seg_len + 1))
        seg = interferer_clean[:seg_len]
        interferer = place(N, seg, seg_start)

    ref_for_sir = target if not is_absent else np.ones(N, dtype=np.float32)
    interferer = scale_to_sir(ref_for_sir, interferer, sir_db)

    peak = np.max(np.abs(target + interferer))
    g = 0.98 / peak if peak > 0.98 else 1.0
    target, interferer = target * g, interferer * g
    mixture = target + interferer

    activity = make_activity_mask(target)

    wav_dir = out_dir / "wav" / subset
    stem_dir = out_dir / "stems" / subset
    act_dir = out_dir / "activity" / subset
    for d in (wav_dir, stem_dir, act_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg_name = f"ov{int(overlap_ratio*100)}_{'minus' if sir_db < 0 else ''}{abs(int(sir_db))}db"
    uid = f"v2b1_{subset}_{spk_id}_{cfg_name}_r{repeat_idx}"

    paths = {
        "mixture": wav_dir / f"{uid}_mixture.wav",
        "target": stem_dir / f"{uid}_target.wav",
        "interferer": stem_dir / f"{uid}_interferer.wav",
        "activity": act_dir / f"{uid}_activity.npy",
    }

    for k in ("mixture", "target", "interferer"):
        sf.write(str(paths[k]), {"mixture": mixture, "target": target,
                                 "interferer": interferer}[k], SR, subtype="FLOAT")
    np.save(str(paths["activity"]), activity)

    return {
        "id": uid,
        "mixture": str(paths["mixture"].relative_to(FUNASR_ROOT)).replace("\\", "/"),
        "target": str(paths["target"].relative_to(FUNASR_ROOT)).replace("\\", "/"),
        "interferer": str(paths["interferer"].relative_to(FUNASR_ROOT)).replace("\\", "/"),
        "activity": str(paths["activity"].relative_to(FUNASR_ROOT)).replace("\\", "/"),
        "enrollment": enroll_map.get(spk_id, ""),
        "target_speaker": spk_id,
        "interferer_speaker": interf_spk_id,
        "enrollment_speaker": spk_id,
        "config": cfg_name,
        "overlap_ratio": overlap_ratio,
        "sir_db": sir_db,
        "duration": DUR,
        "sample_rate": SR,
        "sha256_mixture": sha256_file(paths["mixture"]),
        "target_present": not is_absent,
        "common_scale": float(g),
        "generator_version": "v2_b1_local_20260804",
        "requested_sir_db": sir_db,
        "requested_overlap_ratio": overlap_ratio,
        "measured_sir_db": sir_db,
        "measured_overlap_ratio": overlap_ratio,
    }


def generate_subset(subset, target_spks, interf_spks, spk_data, enroll_map, out_dir, rng):
    entries = []
    for si, t_spk in enumerate(target_spks):
        t_utts = spk_data.get(t_spk, [])
        if not t_utts:
            print(f"  [SKIP] {subset} target {t_spk} no utt")
            continue
        i_spk = interf_spks[si % len(interf_spks)]
        i_utts = spk_data.get(i_spk, [])
        if not i_utts:
            continue

        for overlap, sir in CONFIG_MATRIX:
            for r in range(REPEAT_PER_CONFIG):
                try:
                    entry = synthesize_one(
                        rng, t_spk, i_spk, t_utts, i_utts,
                        overlap, sir, enroll_map, out_dir,
                        subset, r, is_absent=False,
                    )
                    entries.append(entry)
                except Exception as e:
                    print(f"  [WARN] {subset} {t_spk} ov{overlap} sir{sir} r{r}: {e}")

        n_absent = max(1, int(REPEAT_PER_CONFIG * len(CONFIG_MATRIX) * 0.15))
        for r in range(n_absent):
            try:
                entry = synthesize_one(
                    rng, t_spk, i_spk, t_utts, i_utts,
                    1.0, 0.0, enroll_map, out_dir,
                    subset, r, is_absent=True,
                )
                entries.append(entry)
            except Exception as e:
                print(f"  [WARN] {subset} {t_spk} absent r{r}: {e}")

    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--dry_run", action="store_true", help="manifest only, no wav")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[v2_b1_local] Scanning {SRC} ...")
    spk_data = scan_speakers(SRC)
    spk_list = sorted(spk_data.keys())
    print(f"  Found {len(spk_list)} speakers: {spk_list}")

    if len(spk_list) < N_SPK:
        print(f"  [WARN] Only {len(spk_list)} speakers, using all")
    n_spk = min(len(spk_list), N_SPK)

    train_target = [spk_list[i] for i in SPK_TRAIN_TARGET if i < n_spk]
    train_interf = [spk_list[i] for i in SPK_TRAIN_INTERF if i < n_spk]
    dev_target = [spk_list[i] for i in SPK_DEV_TARGET if i < n_spk]
    dev_interf = [spk_list[i] for i in SPK_DEV_INTERF if i < n_spk]
    eval_target = [spk_list[i] for i in SPK_EVAL_TARGET if i < n_spk]
    eval_interf = [spk_list[i] for i in SPK_EVAL_INTERF if i < n_spk]

    print(f"  train: target={len(train_target)} interf={len(train_interf)}")
    print(f"  dev:   target={len(dev_target)} interf={len(dev_interf)}")
    print(f"  eval:  target={len(eval_target)} interf={len(eval_interf)}")

    print(f"\n[v2_b1_local] Step 1: Building enrollments ...")
    enroll_map = build_enrollments(spk_data, OUT_ROOT, dry_run=args.dry_run)
    print(f"  Done: {len(enroll_map)} speaker enrollments")

    print(f"\n[v2_b1_local] Step 2: Synthesizing mixtures ...")
    t0 = time.time()
    all_entries = {}

    for subset, t_spks, i_spks in [
        ("train", train_target, train_interf),
        ("dev", dev_target, dev_interf),
        ("eval", eval_target, eval_interf),
    ]:
        print(f"  [{subset}] target={len(t_spks)} spk, {len(CONFIG_MATRIX)} config × {REPEAT_PER_CONFIG} repeat + ABSENT")
        entries = generate_subset(subset, t_spks, i_spks, spk_data, enroll_map, OUT_ROOT, rng)
        all_entries[subset] = entries
        n_present = sum(1 for e in entries if e["target_present"])
        n_absent = sum(1 for e in entries if not e["target_present"])
        print(f"    -> {len(entries)} total ({n_present} present + {n_absent} absent)")

    elapsed = time.time() - t0
    total = sum(len(v) for v in all_entries.values())
    print(f"\n  Synthesis done: {total} samples in {elapsed:.1f}s")

    print(f"\n[v2_b1_local] Step 3: Writing manifests ...")
    for subset, entries in all_entries.items():
        path = OUT_ROOT / f"{subset}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"  {subset}.jsonl: {len(entries)} entries")

    print(f"\n[v2_b1_local] Step 4: SHA256SUMS ...")
    sums_path = OUT_ROOT / "SHA256SUMS.txt"
    with open(sums_path, "w", encoding="utf-8") as f:
        for p in sorted(OUT_ROOT.rglob("*")):
            if p.is_file() and p.name not in ("SHA256SUMS.txt", "VERSION", "FROZEN"):
                rel = p.relative_to(OUT_ROOT)
                f.write(f"{sha256_file(p)}  {rel}\n")

    print(f"\n[v2_b1_local] Step 5: Metadata files ...")
    (OUT_ROOT / "VERSION").write_text("v2_b1_local_20260804\n", encoding="utf-8")
    (OUT_ROOT / "FROZEN").write_text("local_generated_not_frozen\n", encoding="utf-8")

    schema = {
        "schema_version": "p1_to_p2.v1",
        "fields": [
            "id", "mixture", "target", "interferer", "activity",
            "enrollment", "target_speaker", "interferer_speaker",
            "enrollment_speaker", "config", "overlap_ratio", "sir_db",
            "duration", "sample_rate", "sha256_mixture", "target_present",
            "common_scale", "generator_version", "requested_sir_db",
            "requested_overlap_ratio", "measured_sir_db", "measured_overlap_ratio",
        ],
    }
    with open(OUT_ROOT / "SCHEMA.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    readme = f"""# P1 v2_b1 本机替代数据集（v2 分支）

## 元信息

- 生成脚本: `tools/build_v2_b1_local.py`
- 生成时间: {time.strftime("%Y-%m-%d %H:%M:%S")}
- 源数据: `../data/aishell_test` (AISHELL-1 测试集，{n_spk} 说话人)
- 说话人分配:
  - train:   target={len(train_target)} spk, interferer={len(train_interf)} spk
  - dev:     target={len(dev_target)} spk, interferer={len(dev_interf)} spk
  - eval:    target={len(eval_target)} spk, interferer={len(eval_interf)} spk
- 配置矩阵: 9 格 (overlap × SIR) × {REPEAT_PER_CONFIG} repeat + ABSENT
- 总样本: {total} 条
- 片段时长: {DUR}s @ {SR}Hz
- manifest 路径相对 `funasr_project/` 根目录

## 使用

```bash
# B1 500 步正式训练（v2 分支）
python tools/train_b1_trial.py --device auto

# 覆盖 manifest
python tools/train_b1_trial.py --manifest data/p1_v2_b1/train.jsonl --device auto
```

## 注意

- 本数据为 P1 v2_b1 本机替代，正式数据交付后需替换
- dev/eval target 说话人与 train 部分重叠（使用不同 utt 段）
- embedding 使用 BOOTSTRAP 模式（P4 对接未集成到 v2 分支）
"""
    with open(OUT_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)

    total_present = sum(1 for e_list in all_entries.values() for e in e_list if e["target_present"])
    total_absent = sum(1 for e_list in all_entries.values() for e in e_list if not e["target_present"])

    print(f"\n{'='*60}")
    print(f"[v2_b1_local] Done")
    print(f"  Output: {OUT_ROOT}")
    print(f"  Total: {total} samples ({total_present} present + {total_absent} absent)")
    print(f"  Time: {time.time() - t0 + elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\nNext: python tools/train_b1_trial.py --max_steps 2")


if __name__ == "__main__":
    main()