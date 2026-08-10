"""SV 模块类型定义（P4-01 契约）。

定义说话人验证子系统的核心数据结构，覆盖注册（enrollment）、
嵌入（embedding）、验证结果（verification result）全链路。

红线：
- 本文件纯类型 + 轻量工厂，不依赖模型或外部服务；
- 所有张量形状契约在此冻结，P2/P4 以此对齐；
- 不写文件、不打印日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

import torch


class EmbedMode(str, Enum):
    """嵌入来源模式。"""
    BOOTSTRAP = "bootstrap"
    CAMPLUS = "campplus"


@dataclass(frozen=True)
class EnrollUtterance:
    """单条注册音频。

    Attributes:
        wav_path: 音频文件路径
        speaker_id: 说话人唯一标识
        duration_ms: 音频时长（毫秒，None 表示未知）
    """
    wav_path: str
    speaker_id: str
    duration_ms: Optional[int] = None


@dataclass
class EnrollmentSession:
    """一次说话人注册会话（可含多条 utterance）。

    Attributes:
        speaker_id: 目标说话人唯一标识
        utterances: 注册音频列表
        metadata: 会话元信息（采集设备、环境噪声等）
    """
    speaker_id: str
    utterances: List[EnrollUtterance] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add(self, wav_path: str, duration_ms: Optional[int] = None) -> None:
        self.utterances.append(
            EnrollUtterance(
                wav_path=wav_path,
                speaker_id=self.speaker_id,
                duration_ms=duration_ms,
            )
        )

    def __len__(self) -> int:
        return len(self.utterances)


@dataclass
class SpeakerEmbedding:
    """说话人嵌入向量。

    Attributes:
        vector: [D] 归一化嵌入
        speaker_id: 对应说话人 ID
        mode: 生成模式（BOOTSTRAP / CAMPLUS）
        model_id: 使用的模型版本标识符
        extra: 附加元信息（如均值/方差、注册条数等）
    """
    vector: torch.Tensor
    speaker_id: str
    mode: EmbedMode
    model_id: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.vector.shape[-1])

    def to_batch(self) -> torch.Tensor:
        return self.vector.unsqueeze(0)


@dataclass
class VerificationPair:
    """验证对（注册 vs 测试）。

    Attributes:
        enroll: 注册嵌入
        test_wav: 测试音频波形 [T]
        test_speaker_id: 测试音频的真实说话人 ID（仅评测用）
    """
    enroll: SpeakerEmbedding
    test_wav: torch.Tensor
    test_speaker_id: Optional[str] = None


@dataclass
class VerificationResult:
    """验证结果。

    Attributes:
        score: 余弦相似度得分
        is_same: 是否为同一说话人（基于阈值）
        threshold: 使用的判定阈值
    """
    score: float
    is_same: bool
    threshold: float

    def __lt__(self, other: VerificationResult) -> bool:
        return self.score < other.score


@dataclass
class EnrollConfig:
    """注册编码配置。

    Attributes:
        model_id: Campplus 模型 ID 或路径
        embed_dim: 输出嵌入维度
        sample_rate: 期望采样率
        norm_embed: 是否对嵌入做 L2 归一化
        device: 推理设备
        batch_size: 批量编码大小
        segments_per_utt: 每条 utterance 切分段数（None = 不切分）
        segment_duration_ms: 切段时长（毫秒）
    """
    model_id: str = ""
    embed_dim: int = 512
    sample_rate: int = 16000
    norm_embed: bool = True
    device: str = "cpu"
    batch_size: int = 1
    segments_per_utt: Optional[int] = None
    segment_duration_ms: int = 3000


@dataclass
class CampplusBackendConfig:
    """Campplus 后端配置。

    Attributes:
        model_id: ModelScope 模型 ID 或本地权重路径
        feat_dim: 输入 fbank 维度
        embedding_size: 模型输出嵌入维度
        sample_rate: 期望采样率
        mean_norm: fbank 均值归一化
        device: 推理设备
    """
    model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    feat_dim: int = 80
    embedding_size: int = 192
    sample_rate: int = 16000
    mean_norm: bool = True
    device: str = "cpu"


@dataclass
class BootstrapConfig:
    """BOOTSTRAP 模式配置（确定性随机嵌入）。

    Attributes:
        seed: 随机种子
        embed_dim: 输出嵌入维度（需与 TSE 模型 emb_dim 对齐）
        speaker_seed_map: speaker_id → seed 偏移映射
    """
    seed: int = 20260804
    embed_dim: int = 192
    speaker_seed_map: dict = field(default_factory=dict)