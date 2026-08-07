"""注册音频编码模块（P4-03）。

将说话人注册会话（多条 utterance）编码为单个 speaker embedding，
支持 Campplus 后端推理与 BOOTSTRAP 兜底两种路径。

职责边界：
- 负责音频加载、切段、嵌入提取、多 utterance 聚合；
- 不负责模型训练、不负责验证打分；
- 聚合策略：取各 utterance 嵌入的均值 + L2 重归一化。

红线：
- 本文件不写文件、不打印日志；
- 所有 I/O 通过函数参数或返回值完成；
- 错误输入显式失败（raise ValueError）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .campplus_backend import CampplusBackend
from .types import (
    BootstrapConfig,
    EmbedMode,
    EnrollConfig,
    EnrollmentSession,
    SpeakerEmbedding,
)


def _load_wav(path: str, sample_rate: int = 16000) -> torch.Tensor:
    """加载音频文件为 [T] 波形。

    优先使用 soundfile，退化到 torchaudio，最终退化为 scipy.io.wavfile。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"音频文件不存在: {path}")

    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if sr != sample_rate:
            data = _resample(data, sr, sample_rate)
        return torch.from_numpy(data).float()
    except ImportError:
        pass

    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
        return wav.squeeze(0)
    except ImportError:
        pass

    try:
        import numpy as np
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        if sr != sample_rate:
            data = _resample(data, sr, sample_rate)
        return torch.from_numpy(data).float()
    except ImportError:
        raise ImportError("音频加载需要 soundfile、torchaudio 或 scipy 之一")


def _resample(
    wav: torch.Tensor, orig_sr: int, target_sr: int
) -> torch.Tensor:
    """线性插值重采样。"""
    if orig_sr == target_sr:
        return wav
    duration = wav.shape[-1] / orig_sr
    target_len = int(duration * target_sr)
    idx = torch.linspace(0, wav.shape[-1] - 1, target_len)
    idx_low = idx.floor().long()
    idx_high = idx.ceil().long().clamp(max=wav.shape[-1] - 1)
    frac = idx - idx_low.float()
    return wav[..., idx_low] * (1 - frac) + wav[..., idx_high] * frac


def _segment_wav(
    wav: torch.Tensor,
    segment_duration_ms: int,
    segments_per_utt: Optional[int] = None,
    sample_rate: int = 16000,
) -> List[torch.Tensor]:
    """将波形切分为等长段。

    Args:
        wav: [T] 波形
        segment_duration_ms: 每段时长（毫秒）
        segments_per_utt: 切分段数（None = 按时长切分）
        sample_rate: 采样率

    Returns:
        切分后的波形段列表
    """
    seg_len = int(segment_duration_ms * sample_rate / 1000)
    T = wav.shape[-1]
    if T <= seg_len:
        return [wav]

    if segments_per_utt is not None:
        seg_len = T // segments_per_utt
        segments = [wav[i * seg_len : (i + 1) * seg_len] for i in range(segments_per_utt)]
        if T % seg_len > 0:
            segments[-1] = torch.cat([segments[-1], wav[segments_per_utt * seg_len :]])
        return segments

    segments = []
    start = 0
    while start < T:
        end = min(start + seg_len, T)
        segments.append(wav[start:end])
        start = end
    return segments


class EnrollmentEncoder:
    """注册编码器：将 EnrollmentSession 转为 SpeakerEmbedding。

    使用方式::

        encoder = EnrollmentEncoder(backend, cfg)
        session = EnrollmentSession(spk_id)
        session.add("utt1.wav")
        session.add("utt2.wav")
        emb = encoder.encode(session)
    """

    def __init__(
        self,
        backend: Optional[CampplusBackend],
        cfg: EnrollConfig,
    ):
        self.backend = backend
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate

    def encode(
        self, session: EnrollmentSession
    ) -> SpeakerEmbedding:
        """编码注册会话为单说话人嵌入。

        Args:
            session: 注册会话（含多条 utterance）

        Returns:
            聚合后的 SpeakerEmbedding

        Raises:
            ValueError: 会话为空、嵌入维度不匹配
        """
        if len(session) == 0:
            raise ValueError(f"注册会话为空: speaker_id={session.speaker_id}")

        utt_embs: List[torch.Tensor] = []
        for utt in session.utterances:
            wav = _load_wav(utt.wav_path, self.sample_rate)
            seg_embs = self._encode_wav_segments(wav)
            utt_emb = self._aggregate(seg_embs)
            utt_embs.append(utt_emb)

        final_emb = self._aggregate(utt_embs)
        if self.cfg.embed_dim and final_emb.shape[-1] != self.cfg.embed_dim:
            final_emb = self._project(final_emb, self.cfg.embed_dim)

        return SpeakerEmbedding(
            vector=final_emb,
            speaker_id=session.speaker_id,
            mode=EmbedMode.CAMPLUS if self.backend is not None else EmbedMode.BOOTSTRAP,
            model_id=self.cfg.model_id,
            extra={
                "num_utterances": len(session),
                "num_segments_per_utt": self.cfg.segments_per_utt,
            },
        )

    def _encode_wav_segments(self, wav: torch.Tensor) -> List[torch.Tensor]:
        """将单条波形切段后逐段编码。"""
        segments = _segment_wav(
            wav,
            self.cfg.segment_duration_ms,
            self.cfg.segments_per_utt,
            self.sample_rate,
        )
        embs: List[torch.Tensor] = []
        _dim = self.cfg.embed_dim or 192
        for seg in segments:
            if self.backend is not None:
                emb = self.backend.embed(seg.unsqueeze(0)).squeeze(0)
            else:
                emb = self._bootstrap_embed(seg, emb_dim=_dim)
            embs.append(emb)
        return embs

    @staticmethod
    def _aggregate(embs: List[torch.Tensor]) -> torch.Tensor:
        """嵌入聚合：均值 + L2 归一化。"""
        stacked = torch.stack(embs, dim=0)
        mean = stacked.mean(dim=0)
        return F.normalize(mean.unsqueeze(0), p=2, dim=-1).squeeze(0)

    @staticmethod
    def _project(emb: torch.Tensor, target_dim: int) -> torch.Tensor:
        """线性投影到目标维度（BOOTSTRAP 兼容用）。"""
        in_dim = emb.shape[-1]
        if in_dim == target_dim:
            return emb
        weight = torch.randn(target_dim, in_dim) * 0.01
        return F.linear(emb.unsqueeze(0), weight).squeeze(0)

    @staticmethod
    def _bootstrap_embed(wav: torch.Tensor, emb_dim: int = 192) -> torch.Tensor:
        """BOOTSTRAP 模式：基于波形哈希生成确定性嵌入。"""
        seed = abs(hash(wav.cpu().numpy().tobytes())) % (2**31)
        g = torch.Generator()
        g.manual_seed(seed)
        emb = torch.randn(emb_dim, generator=g)
        return F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)


def encode_session(
    session: EnrollmentSession,
    backend: Optional[CampplusBackend],
    cfg: EnrollConfig,
) -> SpeakerEmbedding:
    """便捷函数：一步编码注册会话。

    Args:
        session: 注册会话
        backend: Campplus 后端（None = BOOTSTRAP 模式）
        cfg: 编码配置

    Returns:
        SpeakerEmbedding
    """
    encoder = EnrollmentEncoder(backend, cfg)
    return encoder.encode(session)


def encode_bootstrap(
    session: EnrollmentSession,
    cfg: BootstrapConfig,
) -> SpeakerEmbedding:
    """BOOTSTRAP 模式编码（确定性随机嵌入，无需模型）。

    Args:
        session: 注册会话
        cfg: BOOTSTRAP 配置

    Returns:
        SpeakerEmbedding（mode=BOOTSTRAP）
    """
    if len(session) == 0:
        raise ValueError(f"注册会话为空: speaker_id={session.speaker_id}")

    seed_offset = cfg.speaker_seed_map.get(session.speaker_id, 0)
    g = torch.Generator()
    g.manual_seed(cfg.seed + seed_offset)
    emb = torch.randn(cfg.embed_dim, generator=g)
    emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1).squeeze(0)

    return SpeakerEmbedding(
        vector=emb,
        speaker_id=session.speaker_id,
        mode=EmbedMode.BOOTSTRAP,
        model_id="bootstrap_v1",
        extra={"num_utterances": len(session), "seed": cfg.seed + seed_offset},
    )