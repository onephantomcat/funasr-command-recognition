"""生成 --debug_data 训练用小型 fixture（5 条合成音频 + manifest.jsonl）。

用法：
    python tools/generate_debug_fixture.py

输出到 P2_project/artifacts/debug_mixtures_v0/，包含：
    - mixture_0~4.wav, target_0~4.wav, interferer_0~4.wav, enroll_0~4.wav
    - activity_0~4.npy
    - manifest.jsonl
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DURATION_S = 2.0
N_SAMPLES = int(SR * DURATION_S)
OUT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "debug_mixtures_v0"


def _sine(freq, n, sr, amp=0.3):
    t = np.arange(n) / sr
    return amp * np.sin(2 * np.pi * freq * t).astype("float32")


def _noise(n, amp=0.1):
    return (amp * np.random.randn(n)).astype("float32")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    entries = []
    for i in range(5):
        f_tgt = 200 + i * 50      # 200,250,300,350,400 Hz
        f_itr = 600 + i * 80      # 600,680,760,840,920 Hz

        target = _sine(f_tgt, N_SAMPLES, SR, amp=0.3)
        interferer = _sine(f_itr, N_SAMPLES, SR, amp=0.25)
        mixture = (target + interferer + _noise(N_SAMPLES, amp=0.05)).astype("float32")

        # enrollment: 同说话人不同句（用不同频率的短段模拟）
        enroll = _sine(f_tgt + 20, N_SAMPLES, SR, amp=0.3)

        # activity mask: target 非零区域为 1
        activity = (np.abs(target) > 0.01).astype("float32")

        mix_path = OUT_DIR / f"mixture_{i}.wav"
        tgt_path = OUT_DIR / f"target_{i}.wav"
        itr_path = OUT_DIR / f"interferer_{i}.wav"
        enr_path = OUT_DIR / f"enroll_{i}.wav"
        act_path = OUT_DIR / f"activity_{i}.npy"

        sf.write(str(mix_path), mixture, SR, subtype="FLOAT")
        sf.write(str(tgt_path), target, SR, subtype="FLOAT")
        sf.write(str(itr_path), interferer, SR, subtype="FLOAT")
        sf.write(str(enr_path), enroll, SR, subtype="FLOAT")
        np.save(str(act_path), activity)

        entries.append({
            "id": f"debug_{i:03d}",
            "mixture": str(mix_path.relative_to(Path(__file__).resolve().parents[2])),
            "target": str(tgt_path.relative_to(Path(__file__).resolve().parents[2])),
            "interferer": str(itr_path.relative_to(Path(__file__).resolve().parents[2])),
            "enrollment": str(enr_path.relative_to(Path(__file__).resolve().parents[2])),
            "activity": str(act_path.relative_to(Path(__file__).resolve().parents[2])),
        })

    manifest_path = OUT_DIR / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"生成 {len(entries)} 条 fixture 到 {OUT_DIR}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
