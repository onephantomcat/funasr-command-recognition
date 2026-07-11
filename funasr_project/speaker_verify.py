# -*- coding: utf-8 -*-
"""
声纹鉴权模块 (申报表 Step 2/3: 目标声纹提取 + 动态拒识门控)
模型: CAM++ (iic/speech_campplus_sv_zh-cn_16k-common), FunASR 中文声纹基座
流程: 注册音频 -> 声纹向量 -> 与测试音频声纹算余弦相似度 -> 阈值判决
"""
import os

import numpy as np
from funasr import AutoModel

# 声纹模型支持自动下载并缓存
SV_DIR = "iic/speech_campplus_sv_zh-cn_16k-common"

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


def build_sv_model():
    return AutoModel(model=SV_DIR, device="cpu", disable_update=True)


def extract_embedding(sv_model, wav_path):
    res = sv_model.generate(input=wav_path)
    emb = res[0]["spk_embedding"]
    if hasattr(emb, "numpy"):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-9)


def cosine_sim(a, b):
    return float(np.dot(a, b))


class SpeakerGate:
    """拒识门控: 注册目标声纹, 对每段输入音频做通过/拒识判决"""

    def __init__(self, sv_model, threshold=DEFAULT_THRESHOLD):
        self.sv_model = sv_model
        self.threshold = threshold
        self.target_emb = None

    def enroll(self, wav_path):
        self.target_emb = extract_embedding(self.sv_model, wav_path)

    def verify(self, wav_path):
        """返回 (是否目标说话人, 相似度)"""
        assert self.target_emb is not None, "先调用 enroll() 注册目标声纹"
        emb = extract_embedding(self.sv_model, wav_path)
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
