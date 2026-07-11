# -*- coding: utf-8 -*-
"""
完整管线: 音频 -> VAD -> Paraformer 识别 -> 拼音模糊指令匹配
对比干净语音和不同信噪比带噪语音的识别效果(抗干扰实验)
用法: python pipeline.py
"""
import glob
import time

from asr_demo import build_model, recognize
from command_match import match_command

if __name__ == "__main__":
    print("加载模型中...")
    t0 = time.time()
    model = build_model()
    print(f"模型加载完成 ({time.time()-t0:.1f}s)\n")

    wavs = sorted(glob.glob("test_audio/*.wav"))
    if not wavs:
        print("没有测试音频, 先运行: python make_test_audio.py")
        raise SystemExit(1)

    print(f"{'音频':<28s}{'耗时':<8s}识别结果 / 命中指令")
    print("-" * 80)
    for wav in wavs:
        text, elapsed = recognize(model, wav)
        cmd, score = match_command(text)
        print(f"{wav:<28s}{elapsed:<8.2f}{text}")
        print(f"{'':<36s}-> 指令: {cmd} (距离 {score:.2f})")
