# -*- coding: utf-8 -*-
"""
可信语音交互全链路 demo (对应申报表总体架构图):
  注册音频 -> CAM++ 目标声纹
  测试音频 -> 声纹核验门控 --拒识--> 输出空字符串 ""
                          --放行--> VAD + Paraformer 识别 -> 指令匹配 -> JSON
用法: python trusted_pipeline.py   (先运行 make_test_audio.py 和 make_speaker_audio.py)
"""
import json
import time

from asr_demo import build_model, recognize
from command_match import match_command
from speaker_verify import SpeakerGate, build_sv_model

ENROLL_WAV = "test_audio/target_enroll.wav"
TEST_WAVS = [
    ("test_audio/target_cmd.wav",      "目标说话人-干净"),
    ("test_audio/target_cmd_5db.wav",  "目标说话人-5dB噪声"),
    ("test_audio/stranger_cmd.wav",    "陌生人1-干净"),
    ("test_audio/stranger_cmd_5db.wav", "陌生人1-5dB噪声"),
    ("test_audio/stranger2_cmd.wav",   "陌生人2-干净"),
]


def main():
    print("加载声纹模型(CAM++)...")
    gate = SpeakerGate(build_sv_model())
    print("加载识别模型(Paraformer + VAD, 省内存不加载标点)...")
    asr = build_model(with_punc=False)

    print(f"\n注册目标声纹: {ENROLL_WAV}")
    gate.enroll(ENROLL_WAV)

    results = []
    print(f"\n{'测试条件':<22s}{'声纹相似度':<12s}{'判决':<8s}识别/指令")
    print("-" * 78)
    for wav, label in TEST_WAVS:
        t0 = time.time()
        ok, sim = gate.verify(wav)
        if ok:
            text, _ = recognize(asr, wav)
            cmd, score = match_command(text)
        else:
            text, cmd, score = "", None, 1.0   # 拒识: 输出空字符串
        elapsed = time.time() - t0
        verdict = "放行" if ok else "拒识"
        print(f"{label:<22s}{sim:<12.3f}{verdict:<8s}{text or '(空)'}"
              f"{('  -> ' + cmd) if cmd else ''}")
        results.append({
            "audio": wav, "condition": label,
            "speaker_similarity": round(sim, 4), "accepted": ok,
            "text": text, "command": cmd,
            "latency_sec": round(elapsed, 2),
        })

    out = "results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果已保存: {out}")


if __name__ == "__main__":
    main()
