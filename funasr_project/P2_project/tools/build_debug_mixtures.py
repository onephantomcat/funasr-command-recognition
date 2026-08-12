# -*- coding: utf-8 -*-
"""P2-07 DEBUG_ONLY 小语音集构建脚本。

来源：data/trials/ 4 说话人（S0764~S0767）的 clean/enroll。
产物：12 条 4s 混合（4 speakers × 3 配置）+ target/interferer stem +
sample 级 activity_mask + manifest.jsonl + SHA256SUMS.txt + _meta.json。

配置设计（每条 4s @16k = 64000 采样）：
  partial25 : 干扰 1s 落在 1.5s 处（重叠 25%），SIR 5 dB，enrollment=本人
  full100   : 干扰全程 4s（重叠 100%），SIR 0 dB，enrollment=本人   ×4 条
  swap50    : 干扰 2s 落在 1s 处（重叠 50%），SIR 0 dB，enrollment=他人 ×4 组

红线：DEBUG_ONLY，不进任何正式训练配置；不修改 data/ 下任何文件（只读源）。

运行：
  python tools/build_debug_mixtures.py --seed 20260725
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
DEFAULT_SRC = FUNASR_ROOT / "data" / "trials"
DEFAULT_OUT = P2_ROOT / "artifacts" / "debug_mixtures_v0"

SR = 16000
DUR = 4.0
N = int(SR * DUR)
SPEAKERS = ["S0764", "S0765", "S0766", "S0767"]
ACT_THRESH = 1e-4


def read_wav(path, n=N, rng=None):
    wav, sr = sf.read(str(path), dtype="float32")
    assert sr == SR, f"{path} 采样率 {sr} != {SR}"
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
        raise RuntimeError("活动区为空，无法标定 SIR")
    t_rms = np.sqrt(np.mean(t_act ** 2)) + eps
    i_rms = np.sqrt(np.mean(i_act ** 2)) + eps
    gain = t_rms / i_rms * 10 ** (-sir_db / 20.0)
    return interferer * gain


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(path):
    try:
        return str(path.resolve().relative_to(FUNASR_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def build_one(rng, spk_idx, config, source, out):
    spk = SPEAKERS[spk_idx]
    inter = SPEAKERS[(spk_idx + 1) % 4]
    swap_spk = SPEAKERS[(spk_idx + 2) % 4]

    if config == "partial25":
        t_idx, i_idx, sir, overlap = 0, 0, 5.0, 0.25
        target = read_wav(source / f"clean_{spk}_{t_idx}.wav", rng=rng)
        seg = read_wav(source / f"clean_{inter}_{i_idx}.wav", n=SR, rng=rng)
        interferer = place(N, seg, int(1.5 * SR))
        enroll_spk = spk
        scenario = "enroll_swap_target_1"
        target_present = True
    elif config == "full100":
        t_idx, i_idx, sir, overlap = 1, 1, 0.0, 1.0
        target = read_wav(source / f"clean_{spk}_{t_idx}.wav", rng=rng)
        interferer = read_wav(source / f"clean_{inter}_{i_idx}.wav", rng=rng)
        enroll_spk = spk
        scenario = "enroll_swap_target_2"
        target_present = True
    elif config == "swap50":
        t_idx, i_idx, sir, overlap = 2, 2, 0.0, 0.50
        target = read_wav(source / f"clean_{spk}_{t_idx}.wav", rng=rng)
        seg = read_wav(source / f"clean_{inter}_{i_idx}.wav", n=2 * SR, rng=rng)
        interferer = place(N, seg, int(1.0 * SR))
        enroll_spk = swap_spk
        scenario = "enroll_swap_absent"
        target_present = False
    else:
        raise ValueError(config)

    interferer = scale_to_sir(target, interferer, sir)

    peak = np.max(np.abs(target + interferer))
    g = 0.98 / peak if peak > 0.98 else 1.0
    mixture_target, interferer = target * g, interferer * g
    mixture = mixture_target + interferer

    supervised_target = (mixture_target if target_present
                         else np.zeros_like(mixture, dtype=np.float32))
    supervised_residual = mixture - supervised_target
    activity = ((np.abs(supervised_target) > ACT_THRESH).astype(np.float32)
                if target_present else np.zeros(N, dtype=np.float32))

    uid = f"dbg_{spk}_{config}"
    paths = {
        "mixture": out / f"{uid}_mixture.wav",
        "target": out / f"{uid}_target.wav",
        "interferer": out / f"{uid}_interferer.wav",
        "activity": out / f"{uid}_activity.npy",
    }
    sf.write(str(paths["mixture"]), mixture, SR, subtype="FLOAT")
    sf.write(str(paths["target"]), supervised_target, SR, subtype="FLOAT")
    sf.write(str(paths["interferer"]), supervised_residual, SR, subtype="FLOAT")
    np.save(str(paths["activity"]), activity)

    return {
        "id": uid,
        "mixture": _manifest_path(paths["mixture"]),
        "target": _manifest_path(paths["target"]),
        "interferer": _manifest_path(paths["interferer"]),
        "activity": _manifest_path(paths["activity"]),
        "enrollment": _manifest_path(source / f"enroll_{enroll_spk}.wav"),
        "target_speaker": enroll_spk,
        "interferer_speaker": inter,
        "enrollment_speaker": enroll_spk,
        "mixture_speakers": [spk, inter],
        "scenario": scenario,
        "target_present": target_present,
        "is_absent": not target_present,
        "is_swap": True,
        "config": config,
        "overlap_ratio": overlap,
        "sir_db": sir,
        "duration": DUR,
        "sample_rate": SR,
        "sha256_mixture": sha256(paths["mixture"]),
        "debug_only": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--source", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    source = args.source.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    entries = []
    for i in range(4):
        for config in ("partial25", "full100", "swap50"):
            entries.append(build_one(rng, i, config, source, out))

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(out / "SHA256SUMS.txt", "w", encoding="utf-8") as f:
        for p in sorted(out.glob("*")):
            if p.name in ("SHA256SUMS.txt", "_meta.json"):
                continue
            f.write(f"{sha256(p)}  {p.name}\n")

    meta = {
        "stage": "P2-07",
        "debug_only": True,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": f"python tools/build_debug_mixtures.py --seed {args.seed}",
        "seed": args.seed,
        "source": str(source),
        "counts": {
            "total": len(entries),
            "full100": sum(1 for e in entries if e["config"] == "full100"),
            "enrollment_swap": sum(1 for e in entries if e["scenario"] == "enroll_swap_absent"),
        },
    }
    with open(out / "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"built {len(entries)} mixtures -> {out}")
    print(f"full100={meta['counts']['full100']} enrollment_swap={meta['counts']['enrollment_swap']}")


if __name__ == "__main__":
    main()
