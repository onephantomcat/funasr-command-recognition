# -*- coding: utf-8 -*-
"""
抗干扰语音指令识别 - 基础识别 demo
管线: fsmn-vad 切分 -> Paraformer 识别 -> ct-punc 加标点
用法: python asr_demo.py <音频文件.wav>
首次运行会自动从 ModelScope 下载模型(约1GB), 之后离线可用
"""
import os
import sys
import time

from funasr import AutoModel
import torch

# 模型加载支持自动下载并缓存
ASR_DIR = "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_DIR = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_DIR = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"


def resolve_device(device=None):
    """Use an explicit device when given, otherwise prefer CUDA when present."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def compact_asr_text(text):
    """Remove formatting whitespace that Paraformer inserts between Han chars.

    This is output serialization, not label-based correction: it never changes
    lexical characters and is safe for unknown test-set transcripts.
    """
    return "".join(ch for ch in str(text or "") if not ch.isspace())


def build_model(with_punc=True, device=None):
    """with_punc=False 时不加载标点模型, 省约0.5GB内存(指令匹配场景用不到标点)"""
    kwargs = dict(
        model=ASR_DIR,                  # 非流式 Paraformer-large 中文模型
        vad_model=VAD_DIR,              # 语音端点检测, 过滤静音和长音频切分
        vad_kwargs={"max_single_segment_time": 30000},
        device=resolve_device(device),
        disable_update=True,
    )
    if with_punc:
        kwargs["punc_model"] = PUNC_DIR    # 标点恢复
    return AutoModel(**kwargs)


def recognize(model, wav_path):
    t0 = time.time()
    res = model.generate(input=wav_path, batch_size_s=60)
    elapsed = time.time() - t0
    text = res[0]["text"] if res else ""
    return compact_asr_text(text), elapsed


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "test_audio/clean.wav"
    print(f"加载模型中(首次运行需下载, 请耐心等待)...")
    model = build_model()
    print(f"识别: {wav}")
    text, elapsed = recognize(model, wav)
    print(f"耗时 {elapsed:.2f}s")
    print(f"结果: {text}")
