# -*- coding: utf-8 -*-
"""
生成声纹鉴权测试音频:
- 目标说话人(Tingting): 注册音频 + 干净指令 + 带噪指令
- 陌生人(Meijia/Sinji): 同样的指令, 应被拒识
"""
import os
import subprocess

import numpy as np
import soundfile as sf

from make_test_audio import add_noise, SR

OUT = "test_audio"
ENROLL_TEXT = "你好小助手，我是这台设备的主人，请记住我的声音。"
CMD_TEXT = "请帮我打开空调，温度调到二十六度。"


def tts(voice, text, wav_path):
    aiff = wav_path.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    aiff, wav_path], check=True)
    os.remove(aiff)
    print(f"生成: {wav_path} ({voice})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    tts("Tingting", ENROLL_TEXT, f"{OUT}/target_enroll.wav")
    tts("Tingting", CMD_TEXT, f"{OUT}/target_cmd.wav")
    tts("Meijia", CMD_TEXT, f"{OUT}/stranger_cmd.wav")
    tts("Sinji", CMD_TEXT, f"{OUT}/stranger2_cmd.wav")

    rng = np.random.default_rng(7)
    for name in ("target_cmd", "stranger_cmd"):
        clean, _ = sf.read(f"{OUT}/{name}.wav")
        sf.write(f"{OUT}/{name}_5db.wav", add_noise(clean, 5, rng), SR)
        print(f"生成: {OUT}/{name}_5db.wav (SNR=5dB)")
