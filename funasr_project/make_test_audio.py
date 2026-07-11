# -*- coding: utf-8 -*-
"""
生成测试音频:
1. 用 macOS 自带 TTS(say) 合成干净指令语音 -> test_audio/clean.wav (16k 单声道)
2. 叠加不同信噪比的噪声 -> test_audio/noisy_10db.wav / noisy_5db.wav / noisy_0db.wav
   噪声用白噪声+嘈杂人声混合模拟"复杂交互场景"
"""
import os
import subprocess

import numpy as np
import soundfile as sf

OUT = "test_audio"
TEXT = "你好，请帮我打开空调，温度调到二十六度，然后导航回家。"
SR = 16000


def tts_to_wav(text, wav_path):
    aiff = wav_path.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", "Tingting", "-o", aiff, text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    aiff, wav_path], check=True)
    os.remove(aiff)


def add_noise(clean, snr_db, rng):
    # 白噪声 + 低频嗡嗡声(模拟车噪/空调噪声)
    noise = rng.standard_normal(len(clean))
    t = np.arange(len(clean)) / SR
    hum = 0.5 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 65 * t)
    noise = noise + hum
    # 按目标信噪比缩放噪声
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2)
    k = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    mixed = clean + k * noise
    return mixed / max(1e-9, np.abs(mixed).max()) * 0.9


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    clean_path = os.path.join(OUT, "clean.wav")
    tts_to_wav(TEXT, clean_path)
    clean, sr = sf.read(clean_path)
    print(f"干净语音: {clean_path} ({len(clean)/sr:.1f}s)")

    rng = np.random.default_rng(42)
    for snr in (10, 5, 0):
        out = os.path.join(OUT, f"noisy_{snr}db.wav")
        sf.write(out, add_noise(clean, snr, rng), SR)
        print(f"带噪语音: {out} (SNR={snr}dB)")
