# -*- coding: utf-8 -*-
"""
声纹鉴权模块 (申报表 Step 2/3: 目标声纹提取 + 动态拒识门控)
模型: CAM++ (iic/speech_campplus_sv_zh-cn_16k-common), FunASR 中文声纹基座
流程: 注册音频 -> 声纹向量 -> 与测试音频声纹算余弦相似度 -> 阈值判决
"""
import os

import numpy as np
import soundfile as sf
import torchaudio
from funasr import AutoModel
import torch

# 声纹模型支持自动下载并缓存
SV_DIR = "iic/speech_campplus_sv_zh-cn_16k-common"
VAD_DIR = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# 拒识阈值需按数据标定(用 search_threshold()):
#   TTS合成测试集: 最优≈0.56~0.6 (合成女声声纹相近, 需高阈值)
#   AISHELL-1真实60条试验: 最优0.34~0.40 (真人声纹区分度高, FRR 2%/FAR 0%)
# 默认取偏高值, 安全优先(宁误拒不误受, 对应申报表"重拒识"设计)
DEFAULT_THRESHOLD = 0.6


def search_threshold(scores, labels, step=0.01):
    """在验证集上搜索拒识阈值 (申报表 Step 3)
    scores: 声纹相似度列表; labels: 1=目标说话人, 0=非目标
    返回 (最优阈值, 该阈值下的错误数): 平衡误拒(FR)与误受(FA)
    """
    best_t, best_err = 0.5, len(scores) + 1
    t = min(scores)
    while t <= max(scores) + step:
        fr = sum(1 for s, l in zip(scores, labels) if l == 1 and s < t)
        fa = sum(1 for s, l in zip(scores, labels) if l == 0 and s >= t)
        if fr + fa < best_err:
            best_t, best_err = round(t, 3), fr + fa
        t += step
    return best_t, best_err


def build_sv_model(device=None):
    """Prefer CUDA when it is available, while keeping CPU compatibility."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return AutoModel(model=SV_DIR, device=resolved_device, disable_update=True)


def build_vad_model(device=None):
    """FSMN-VAD, used to trim non-speech before embedding extraction."""
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return AutoModel(model=VAD_DIR, device=resolved_device, disable_update=True)


def load_wav_16k(wav_path):
    """Load audio as mono 16kHz float32 numpy array."""
    wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), orig_freq=sr, new_freq=16000
        ).numpy()
    return wav


def vad_segments(vad_model, wav_path):
    """Return [(start_ms, end_ms), ...] speech segments from FSMN-VAD."""
    res = vad_model.generate(input=wav_path)
    if not res:
        return []
    return [(int(s[0]), int(s[1])) for s in res[0].get("value", [])]


def _norm(emb):
    return emb / (np.linalg.norm(emb) + 1e-9)


def _embed_array(sv_model, wav):
    res = sv_model.generate(input=wav)
    emb = res[0]["spk_embedding"]
    if hasattr(emb, "numpy"):
        emb = emb.detach().cpu().numpy()
    return _norm(np.asarray(emb, dtype=np.float32).reshape(-1))


def extract_embedding(sv_model, wav_path, vad_model=None, chunk_sec=30.0, min_seg_sec=0.15):
    """Extract L2-normalized speaker embedding.

    vad_model=None: legacy whole-file behavior.
    vad_model given: trim non-speech with VAD, split speech into <=chunk_sec
    chunks, embed each chunk and average (robust to silence/noise dilution).
    """
    if vad_model is None:
        res = sv_model.generate(input=wav_path)
        emb = res[0]["spk_embedding"]
        if hasattr(emb, "numpy"):
            emb = emb.detach().cpu().numpy()
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return _norm(emb)

    wav = load_wav_16k(wav_path)
    segments = vad_segments(vad_model, wav_path)
    embs = []
    chunk_len = int(chunk_sec * 16000)
    for start_ms, end_ms in segments:
        seg = wav[int(start_ms / 1000 * 16000):int(end_ms / 1000 * 16000)]
        if len(seg) < int(min_seg_sec * 16000):
            continue
        for off in range(0, len(seg), chunk_len):
            chunk = seg[off:off + chunk_len]
            if len(chunk) >= int(min_seg_sec * 16000):
                embs.append(_embed_array(sv_model, chunk))
    if not embs:  # VAD found nothing usable; fall back to whole audio
        return _embed_array(sv_model, wav)
    return _norm(np.mean(np.stack(embs), axis=0))


def cosine_sim(a, b):
    return float(np.dot(a, b))


class SpeakerGate:
    """拒识门控: 注册目标声纹, 对每段输入音频做通过/拒识判决

    vad_model: 可选, 给定时提取声纹前先做 VAD 裁剪(抗静音/噪声稀释)
    """

    def __init__(self, sv_model, threshold=DEFAULT_THRESHOLD, vad_model=None):
        self.sv_model = sv_model
        self.threshold = threshold
        self.vad_model = vad_model
        self.target_emb = None

    def enroll(self, wav_paths):
        """注册目标声纹; 支持单个路径或多条注册音频(embedding 取平均更稳)"""
        if isinstance(wav_paths, (str, os.PathLike)):
            wav_paths = [wav_paths]
        embs = [extract_embedding(self.sv_model, p, vad_model=self.vad_model)
                for p in wav_paths]
        self.target_emb = _norm(np.mean(np.stack(embs), axis=0))

    def verify(self, wav_path):
        """返回 (是否目标说话人, 相似度)"""
        assert self.target_emb is not None, "先调用 enroll() 注册目标声纹"
        emb = extract_embedding(self.sv_model, wav_path, vad_model=self.vad_model)
        sim = cosine_sim(self.target_emb, emb)
        return sim >= self.threshold, sim


if __name__ == "__main__":
    import sys
    enroll_wav = sys.argv[1] if len(sys.argv) > 1 else "test_audio/target_enroll.wav"
    test_wav = sys.argv[2] if len(sys.argv) > 2 else "test_audio/stranger_cmd.wav"
    print("加载声纹模型...")
    gate = SpeakerGate(build_sv_model())
    gate.enroll(enroll_wav)
    ok, sim = gate.verify(test_wav)
    print(f"注册: {enroll_wav}")
    print(f"测试: {test_wav} -> 相似度 {sim:.3f}, "
          f"{'目标说话人(放行)' if ok else '非目标说话人(拒识)'}")
