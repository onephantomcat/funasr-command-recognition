# -*- coding: utf-8 -*-
"""
实时麦克风语音指令识别 (流式 Paraformer)
用法: python mic_realtime.py   (Ctrl+C 退出)
首次运行 macOS 会弹出麦克风权限授权, 点允许
"""
import queue

import numpy as np
import sounddevice as sd
from funasr import AutoModel

from command_match import match_command

SR = 16000
CHUNK_SIZE = [0, 10, 5]            # 600ms 一帧: 10*60ms
CHUNK_STRIDE = CHUNK_SIZE[1] * 960  # 9600 采样点


def main():
    print("加载流式模型中...")
    model = AutoModel(model="paraformer-zh-streaming", device="cpu",
                      disable_update=True)
    audio_q = queue.Queue()

    def callback(indata, frames, t, status):
        audio_q.put(indata[:, 0].copy())

    cache = {}
    buf = np.zeros(0, dtype=np.float32)
    sentence = ""
    print("开始说话吧 (Ctrl+C 退出)...")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=CHUNK_STRIDE, callback=callback):
        try:
            while True:
                buf = np.concatenate([buf, audio_q.get()])
                while len(buf) >= CHUNK_STRIDE:
                    chunk, buf = buf[:CHUNK_STRIDE], buf[CHUNK_STRIDE:]
                    res = model.generate(input=chunk, cache=cache,
                                         is_final=False,
                                         chunk_size=CHUNK_SIZE,
                                         encoder_chunk_look_back=4,
                                         decoder_chunk_look_back=1)
                    if res and res[0]["text"]:
                        sentence += res[0]["text"]
                        print(f"\r>> {sentence}", end="", flush=True)
                        cmd, score = match_command(sentence)
                        if cmd:
                            print(f"\n[命中指令] {cmd} (距离 {score:.2f})")
                            sentence = ""
        except KeyboardInterrupt:
            print("\n退出")


if __name__ == "__main__":
    main()
