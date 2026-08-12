"""P2→P4 注册嵌入桥接适配器（P2-17 / P4-04）。

职责：
- 将 P4 说话人注册编码能力适配到 P2 TSE 的 ``enroll_embedding`` 接口；
- 支持 BOOTSTRAP（确定性随机）与 CAMPLUS（Campplus 推理）双模式无缝切换；
- 对 P2 暴露统一的 ``build_enroll_embedding`` 函数，返回 [B, D] 张量；
- 嵌入维度自动与 P2 cfg["emb_dim"] 对齐。

使用方式::

    # BOOTSTRAP 模式（默认，无需模型权重）
    adapter = EnrollmentAdapter.from_config(cfg)
    emb = adapter.get_embedding("speaker_001")  # [1, 192]

    # CAMPLUS 模式（需要 Campplus 权重 + speakerlab）
    adapter = EnrollmentAdapter.from_config(cfg, mode="campplus")
    adapter.load_backend()
    emb = adapter.encode_file("speaker_001", "enroll.wav")

红线：
- 本模块不写文件、不打印日志；
- 嵌入缓存内存持有，持久化由上层 pipeline 负责；
- BOOTSTRAP 模式不依赖任何外部库；
- CAMPLUS 模式按需延迟导入 speakerlab，import 时不加载。
- CAMPPlus embedding_size 直接设为 P2 emb_dim，无需投影层。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


# ── 跨项目路径注入（P4 sv 模块 + speakerlab 源码） ──────────
def _inject_p4_paths() -> None:
    """将 P4 src/sv 与 speakerlab_source 加入 sys.path。

    P2 与 P4 是独立项目目录，需要显式注入路径才能跨项目导入。
    仅执行一次（幂等）。
    """
    if hasattr(_inject_p4_paths, "_done"):
        return
    _inject_p4_paths._done = True

    _here = Path(__file__).resolve()                              # funasr_project/P2_project/src/tse/enrollment_adapter.py
    _funasr_root = _here.parents[3]                               # funasr_project/
    _p4_src = _funasr_root / "P4_project" / "src"                 # P4_project/src/
    _p4_speakerlab = (
        _funasr_root / "P4_project" / "artifacts" / "models" / "speakerlab_source"
    )

    for _p in [str(_p4_src), str(_p4_speakerlab)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)


MODE_BOOTSTRAP = "bootstrap"
MODE_CAMPLUS = "campplus"

VALID_MODES = {MODE_BOOTSTRAP, MODE_CAMPLUS}


def _validate_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"未知模式 {mode!r}，期望 {VALID_MODES}")


def _validate_emb_dim(emb: torch.Tensor, expected_dim: int) -> None:
    if emb.ndim != 2:
        raise ValueError(f"emb 需为 [B,D]，得到 ndim={emb.ndim}")
    if emb.shape[-1] != expected_dim:
        raise ValueError(
            f"嵌入维度不匹配: 期望 {expected_dim}，得到 {emb.shape[-1]}"
        )
    if not torch.isfinite(emb).all():
        raise ValueError("emb 含 NaN/Inf")


class EnrollmentAdapter:
    """P2↔P4 注册嵌入适配器。

    双模式架构：
    ┌─────────────┐  BOOTSTRAP   ┌──────────────────┐
    │ P2 TSE cfg  │──────────────►│ 确定性随机嵌入    │
    │ emb_dim=D   │              │ [1, D]            │
    └──────┬──────┘              └──────────────────┘
           │ CAMPLUS
           ▼
    ┌──────────────────┐  encode_file  ┌──────────────────┐
    │ CampplusBackend   │──────────────►│ 真实说话人嵌入   │
    │ (P4)              │              │ [1, D] (对齐)    │
    └──────────────────┘              └──────────────────┘
    """

    def __init__(
        self,
        mode: str = MODE_BOOTSTRAP,
        emb_dim: int = 192,
        device: str = "cpu",
        seed: int = 20260804,
        model_id: str = "",
    ):
        _validate_mode(mode)
        self._mode = mode
        self._emb_dim = emb_dim
        self._device = torch.device(device)
        self._seed = seed
        self._model_id = model_id
        self._cache: Dict[str, torch.Tensor] = {}
        self._backend = None

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        mode: Optional[str] = None,
    ) -> "EnrollmentAdapter":
        """从 P2 配置字典构建适配器。

        Args:
            cfg: P2 训练/推理配置（需含 ``emb_dim``、可选 ``device``）
            mode: 强制覆盖模式（None 则从 cfg 读取，默认 BOOTSTRAP）

        Returns:
            EnrollmentAdapter 实例
        """
        resolved_mode = mode or cfg.get("sv_mode", MODE_BOOTSTRAP)
        return cls(
            mode=resolved_mode,
            emb_dim=cfg["emb_dim"],
            device=cfg.get("device", "cpu"),
            seed=cfg.get("sv_bootstrap_seed", 20260804),
            model_id=cfg.get("sv_model_id", ""),
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def emb_dim(self) -> int:
        return self._emb_dim

    # ── BOOTSTRAP 模式 ──────────────────────────────────────────

    def get_embedding(self, speaker_id: str) -> torch.Tensor:
        """获取说话人嵌入（优先缓存）。

        BOOTSTRAP: 确定性随机生成；
        CAMPLUS: 返回之前 encode_file 的缓存结果。

        Args:
            speaker_id: 说话人唯一标识

        Returns:
            emb: [1, D] L2 归一化嵌入
        """
        if speaker_id in self._cache:
            return self._cache[speaker_id]

        if self._mode == MODE_BOOTSTRAP:
            emb = self._bootstrap_embedding(speaker_id)
        elif self._mode == MODE_CAMPLUS:
            raise KeyError(
                f"CAMPLUS 模式下 speaker_id={speaker_id} 未注册，"
                "先调用 encode_file()"
            )
        else:
            raise RuntimeError(f"未知模式: {self._mode}")

        self._cache[speaker_id] = emb
        return emb

    def get_batch_embeddings(self, speaker_ids: List[str]) -> torch.Tensor:
        """批量获取嵌入 [B, D]。

        Args:
            speaker_ids: 说话人 ID 列表

        Returns:
            emb: [B, D] 批量嵌入
        """
        embs = [self.get_embedding(sid) for sid in speaker_ids]
        return torch.cat(embs, dim=0)

    def _bootstrap_embedding(self, speaker_id: str) -> torch.Tensor:
        """BOOTSTRAP：基于 speaker_id 哈希的确定性随机嵌入。"""
        # Python 内置 hash 会受每个进程的随机盐影响，同一个 speaker 在两次
        # 进程启动间可能得到不同向量。调试替代向量必须跨进程可复现。
        seed_material = f"{self._seed}\0{speaker_id}".encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big") % (2**31)
        g = torch.Generator(device=self._device)
        g.manual_seed(seed)
        emb = torch.randn(self._emb_dim, generator=g, device=self._device)
        emb = F.normalize(emb.unsqueeze(0), p=2, dim=-1)
        return emb

    # ── CAMPLUS 模式 ────────────────────────────────────────────

    def load_backend(self) -> None:
        """延迟加载 Campplus 后端（仅 CAMPLUS 模式需要）。

        注入 P4 路径后，通过绝对导入加载 CampplusBackend。
        CAMPPlus 的 embedding_size 直接设为 self._emb_dim，
        输出维度与 P2 TSE 对齐，无需额外投影。

        Raises:
            RuntimeError: speakerlab 不可用或权重加载失败
        """
        if self._mode != MODE_CAMPLUS:
            return
        if self._backend is not None:
            return

        _inject_p4_paths()

        try:
            from sv.campplus_backend import CampplusBackend
            from sv.types import CampplusBackendConfig

            cfg = CampplusBackendConfig(
                model_id=self._model_id or "iic/speech_campplus_sv_zh-cn_16k-common",
                embedding_size=self._emb_dim,
                device=str(self._device),
            )
            self._backend = CampplusBackend(cfg)
            self._backend.load()
        except ImportError as exc:
            raise RuntimeError(
                f"Campplus 后端不可用: {exc}\n"
                "请确认 speakerlab_source 已克隆且依赖已安装。"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Campplus 后端加载失败: {exc}") from exc

    def encode_file(
        self,
        speaker_id: str,
        wav_path: str,
        segment_duration_ms: int = 3000,
    ) -> torch.Tensor:
        """CAMPLUS 模式：从音频文件编码说话人嵌入。

        Args:
            speaker_id: 说话人 ID（用于缓存）
            wav_path: 注册音频文件路径
            segment_duration_ms: 切段时长

        Returns:
            emb: [1, D] L2 归一化嵌入

        Raises:
            RuntimeError: 后端未加载
            ValueError: 嵌入维度不匹配
        """
        if self._mode != MODE_CAMPLUS:
            raise RuntimeError("encode_file() 仅 CAMPLUS 模式可用")
        self.load_backend()
        if self._backend is None:
            raise RuntimeError("Campplus 后端未初始化")

        _inject_p4_paths()
        from sv.encode_enrollment import _load_wav, _segment_wav

        wav = _load_wav(wav_path, self._backend.cfg.sample_rate)
        segments = _segment_wav(wav, segment_duration_ms)

        seg_embs = []
        for seg in segments:
            seg = seg.to(self._device)
            emb = self._backend.embed(seg.unsqueeze(0))
            seg_embs.append(emb)

        stacked = torch.cat(seg_embs, dim=0)
        mean_emb = stacked.mean(dim=0, keepdim=True)
        mean_emb = F.normalize(mean_emb, p=2, dim=-1)

        _validate_emb_dim(mean_emb, self._emb_dim)
        self._cache[speaker_id] = mean_emb
        return mean_emb

    def encode_wav(
        self,
        speaker_id: str,
        wav: torch.Tensor,
    ) -> torch.Tensor:
        """CAMPLUS 模式：从波形张量编码嵌入。

        Args:
            speaker_id: 说话人 ID（用于缓存）
            wav: [T] 或 [1, T] 波形

        Returns:
            emb: [1, D]
        """
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        self.load_backend()
        if self._backend is None:
            raise RuntimeError("Campplus 后端未初始化")

        wav = wav.to(self._device)
        emb = self._backend.embed(wav)

        _validate_emb_dim(emb, self._emb_dim)
        self._cache[speaker_id] = emb
        return emb

    # ── 工具方法 ────────────────────────────────────────────────

    def invalidate(self, speaker_id: Optional[str] = None) -> None:
        """清除嵌入缓存。"""
        if speaker_id is None:
            self._cache.clear()
        else:
            self._cache.pop(speaker_id, None)

    def register_embedding(
        self, speaker_id: str, embedding: torch.Tensor
    ) -> torch.Tensor:
        """手动注册外部计算的嵌入（用于预计算场景）。

        Args:
            speaker_id: 说话人 ID
            embedding: [D] 或 [1, D] 嵌入

        Returns:
            归一化后的 [1, D] 嵌入
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        embedding = F.normalize(embedding, p=2, dim=-1)
        _validate_emb_dim(embedding, self._emb_dim)
        self._cache[speaker_id] = embedding
        return embedding

    def build_batch(
        self,
        speaker_ids: List[str],
        wav_paths: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """构建 P2 所需的 enroll_embedding 批量张量。

        BOOTSTRAP: 根据 speaker_ids 批量生成确定性嵌入；
        CAMPLUS: 若提供 wav_paths 则编码，否则用缓存。

        Args:
            speaker_ids: 说话人 ID 列表（长度 B）
            wav_paths: 对应音频路径列表（CAMPLUS 模式可选）

        Returns:
            enroll_embedding: [B, D] 张量，可直接传入 DualOutputTSE

        Raises:
            ValueError: speaker_ids 与 wav_paths 长度不匹配
        """
        if wav_paths is not None and len(wav_paths) != len(speaker_ids):
            raise ValueError(
                f"speaker_ids 长度 ({len(speaker_ids)}) "
                f"与 wav_paths 长度 ({len(wav_paths)}) 不匹配"
            )

        embs = []
        for i, sid in enumerate(speaker_ids):
            if sid in self._cache:
                embs.append(self._cache[sid])
            elif self._mode == MODE_CAMPLUS and wav_paths is not None:
                embs.append(self.encode_file(sid, wav_paths[i]))
            else:
                embs.append(self.get_embedding(sid))

        batch = torch.cat(embs, dim=0)
        _validate_emb_dim(batch, self._emb_dim)
        return batch

    def __repr__(self) -> str:
        n_cached = len(self._cache)
        return (
            f"EnrollmentAdapter(mode={self._mode!r}, "
            f"emb_dim={self._emb_dim}, "
            f"device={self._device}, "
            f"cached={n_cached})"
        )
