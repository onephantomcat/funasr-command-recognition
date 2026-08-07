"""P4→P2 sv_contract_v1 接口。

冻结的跨项目契约：P2 通过此接口调用 P4 说话人验证能力。
P2 侧只需依赖此文件的函数签名，不直接 import P4 内部模块。

接口：
- encode_enrollment(wav_path, model_id, device) -> np.ndarray [D] float32
- cosine_similarity(emb_a, emb_b) -> float
- get_model_info(model_id) -> dict

红线：
- 本文件是 P4 对 P2 的唯一公开接口；
- 内部实现可自由变更，但函数签名和返回格式必须冻结；
- 纯函数，不写文件、不打印日志。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def _ensure_p4_paths() -> None:
    """确保 P4 src/sv 与 speakerlab 在 sys.path 中。"""
    if hasattr(_ensure_p4_paths, "_done"):
        return
    _ensure_p4_paths._done = True

    _here = Path(__file__).resolve()
    _p4_root = _here.parents[1]                              # P4_project/
    _p4_src = _p4_root / "src"
    _p4_speakerlab = _p4_root / "artifacts" / "models" / "speakerlab_source"

    for _p in [str(_p4_src), str(_p4_speakerlab)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)


def encode_enrollment(
    wav_path: str,
    model_id: str = "",
    device: str = "cpu",
    embedding_size: int = 192,
) -> np.ndarray:
    """编码注册音频为说话人嵌入（P4→P2 契约 v1）。

    Args:
        wav_path: 注册音频文件路径（WAV/FLAC，16kHz 或其他采样率）
        model_id: ModelScope 模型 ID 或本地权重路径（空=默认 Campplus）
        device: 推理设备（"cpu" / "cuda"）
        embedding_size: 输出嵌入维度（必须与 P2 emb_dim 对齐）

    Returns:
        embedding: [D] float32 numpy 数组，L2 归一化

    Raises:
        RuntimeError: 后端加载失败
        FileNotFoundError: wav_path 不存在
    """
    _ensure_p4_paths()

    from sv.campplus_backend import CampplusBackend
    from sv.types import CampplusBackendConfig
    from sv.encode_enrollment import _load_wav, _segment_wav

    cfg = CampplusBackendConfig(
        model_id=model_id or "iic/speech_campplus_sv_zh-cn_16k-common",
        embedding_size=embedding_size,
        device=device,
    )
    backend = CampplusBackend(cfg)
    backend.load()

    wav = _load_wav(wav_path, cfg.sample_rate)
    segments = _segment_wav(wav, 3000)

    seg_embs = []
    for seg in segments:
        seg = seg.to(backend.device)
        emb = backend.embed(seg.unsqueeze(0))
        seg_embs.append(emb)

    stacked = torch.cat(seg_embs, dim=0)
    mean_emb = stacked.mean(dim=0, keepdim=True)
    mean_emb = torch.nn.functional.normalize(mean_emb, p=2, dim=-1)

    result = mean_emb.squeeze(0).cpu().numpy().astype(np.float32)
    return result


def cosine_similarity(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """计算两个嵌入的余弦相似度。

    Args:
        emb_a: [D] float32 嵌入
        emb_b: [D] float32 嵌入

    Returns:
        similarity: [-1, 1] 余弦相似度
    """
    a = emb_a.astype(np.float64)
    b = emb_b.astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def get_model_info(model_id: str = "") -> dict:
    """获取 Campplus 模型元信息。

    Args:
        model_id: ModelScope 模型 ID（空=默认）

    Returns:
        info: {model_id, embedding_size, feat_dim, sample_rate}
    """
    return {
        "model_id": model_id or "iic/speech_campplus_sv_zh-cn_16k-common",
        "embedding_size": 192,
        "feat_dim": 80,
        "sample_rate": 16000,
        "backend": "campplus",
        "contract_version": "sv_contract_v1",
    }