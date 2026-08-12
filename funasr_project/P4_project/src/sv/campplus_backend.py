"""Campplus 说话人嵌入后端（P4-02）。

封装 3D-Speaker Campplus 模型，提供波形 → 嵌入的最小推理链：
    波形 → fbank → Campplus → L2 归一化 → 嵌入 [B, D]

职责边界：
- 只负责嵌入提取，不关心注册会话、验证打分；
- 模型加载支持 ModelScope model_id 与本地权重两种路径；
- 推理全程 @torch.no_grad()，不保留梯度；
- 错误输入显式失败（raise ValueError），不静默通过。

标记：CAMPLUS_BACKEND_READY / MODEL_WEIGHTS_REQUIRED
"""

from __future__ import annotations

from typing import List, Optional, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import CampplusBackendConfig


class CampplusBackend:
    """Campplus 说话人嵌入提取后端。

    使用方式::

        cfg = CampplusBackendConfig(model_id="iic/speech_campplus_sv_zh-cn_16k-common")
        backend = CampplusBackend(cfg)
        backend.load()          # 加载权重（一次性）
        emb = backend.embed(wav)  # wav: [B, T] → [B, D]
    """

    def __init__(self, cfg: CampplusBackendConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self._model: Optional[nn.Module] = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """加载 Campplus 模型结构 + 预训练权重。

        优先尝试从 speakerlab 导入 CAMPPlus 源码构建模型；
        若 speakerlab 不可用则报 RuntimeError，禁止静默 fallback。

        权重加载优先级：
        1. cfg.model_id 指向本地 .bin/.pt/.pth 文件 → 直接加载
        2. cfg.model_id 是 ModelScope ID + 本地缓存目录有权重 → 加载缓存
        3. 都没有 → 明确失败（正式声纹不能用随机初始化冒充）
        """
        if self._loaded:
            return
        try:
            from speakerlab.models.campplus.DTDNN import CAMPPlus
            self._model = CAMPPlus(
                feat_dim=self.cfg.feat_dim,
                embedding_size=self.cfg.embedding_size,
            )
            self._model = self._model.to(self.device)
            self._model.eval()

            weight_path = self._resolve_weight_path()
            if weight_path is None:
                raise FileNotFoundError(
                    "未找到 CAMPPlus 预训练权重；拒绝使用随机初始化声纹模型。"
                    "请提供本地 model_id，或把 campplus_cn_common.bin 放入 "
                    "P4_project/artifacts/models/ / ModelScope 缓存。"
                )
            self._load_weights(weight_path)
            self._loaded = True
        except ImportError as exc:
            raise RuntimeError(
                f"speakerlab 不可用: {exc}\n"
                "请确认 speakerlab_source 已克隆到 P4_project/artifacts/models/speakerlab_source/，"
                "且路径已加入 sys.path。"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Campplus 模型构建失败: {exc}") from exc

    def _resolve_weight_path(self) -> Optional[Path]:
        """解析权重文件路径。

        Returns:
            权重文件路径（.bin/.pt/.pth），或 None 表示未找到。
        """
        model_id = self.cfg.model_id
        if not model_id:
            return None

        # 1) model_id 直接是本地文件路径
        candidate = Path(model_id)
        if candidate.is_file():
            return candidate

        # 2) 在 P4_project/artifacts/models/ 下查找
        p4_root = Path(__file__).resolve().parents[2]
        models_dir = p4_root / "artifacts" / "models"
        for name in ["campplus_cn_common.bin", "campplus_sv_zh-cn_16k-common.pt",
                      "campplus_sv_zh-cn_16k-common.pth"]:
            p = models_dir / name
            if p.is_file():
                return p

        # 3) model_id 是 ModelScope ID，尝试标准与旧版 ModelScope 缓存布局
        try:
            cache_root = Path.home() / ".cache" / "modelscope"
            model_parts = [part for part in model_id.split("/") if part]
            direct_dirs = []
            if len(model_parts) >= 2:
                direct_dirs.extend([
                    cache_root / "hub" / "models" / model_parts[-2] / model_parts[-1],
                    cache_root / "hub" / model_parts[-2] / model_parts[-1],
                ])
            for candidate_dir in direct_dirs:
                for name in ("campplus_cn_common.bin", "campplus_cn_common.pt",
                             "campplus_cn_common.pth"):
                    path = candidate_dir / name
                    if path.is_file():
                        return path
            if cache_root.is_dir():
                for path in cache_root.rglob("campplus_cn_common.bin"):
                    if not model_parts or model_parts[-1] in path.parts:
                        return path
        except Exception:
            pass

        return None

    def _load_weights(self, weight_path: Path) -> None:
        """从 .bin/.pt/.pth 文件加载权重。

        Args:
            weight_path: 权重文件路径
        """
        import logging as _logging
        _log = _logging.getLogger("campplus")

        _log.info(f"加载权重: {weight_path}")
        try:
            state = torch.load(str(weight_path), map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(str(weight_path), map_location=self.device)

        # HuggingFace 格式: {"state_dict": {...}, ...} 或直接 state_dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        if not isinstance(state, dict):
            raise RuntimeError(f"权重格式异常: {type(state)}，期望 state_dict")

        # 键名适配：HuggingFace 可能加前缀
        model_state = self._model.state_dict()
        adapted = {}
        for k, v in state.items():
            if k in model_state:
                adapted[k] = v
            else:
                # 尝试去掉常见前缀
                for prefix in ["model.", "module.", "encoder."]:
                    stripped = k.removeprefix(prefix) if k.startswith(prefix) else k
                    if stripped in model_state:
                        adapted[stripped] = v
                        break

        missing = sorted(set(model_state) - set(adapted))
        if missing:
            raise RuntimeError(
                f"CAMPPlus 权重不完整：仅匹配 {len(adapted)}/{len(model_state)} 个参数，"
                f"缺失 {missing[:5]}"
            )
        self._model.load_state_dict(adapted, strict=True)
        _log.info(f"权重完整加载: {len(adapted)}/{len(model_state)} 个参数")

    def _extract_fbank(
        self, wav: torch.Tensor, sample_rate: int = 16000
    ) -> torch.Tensor:
        """从波形提取 fbank 特征 [B, T, F]。

        使用 torchaudio.compliance.kaldi.fbank（若可用），
        否则退化为 mel 频谱 + 均值归一化。
        """
        try:
            import torchaudio.compliance.kaldi as kaldi
            # kaldi.fbank 期望 [batch, time] 或 [time]，直接传 [B, T]
            fbank = kaldi.fbank(
                wav,
                num_mel_bins=self.cfg.feat_dim,
                sample_frequency=sample_rate,
            )
            if fbank.ndim == 2:
                fbank = fbank.unsqueeze(0)  # 确保 [B, T, F]
            if self.cfg.mean_norm:
                fbank = fbank - fbank.mean(dim=1, keepdim=True)
            return fbank
        except ImportError:
            return self._mel_fallback(wav)

    def _mel_fallback(self, wav: torch.Tensor) -> torch.Tensor:
        """torchaudio 不可用时的 mel 频谱兜底实现。"""
        n_fft = 400
        hop_length = 160
        win_length = 400
        window = torch.hann_window(win_length, device=wav.device)
        spec = torch.stft(
            wav, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, window=window,
            return_complex=True,
        )
        mag = spec.abs().transpose(1, 2)
        n_mels = self.cfg.feat_dim
        mel_bank = torch.zeros(n_fft // 2 + 1, n_mels, device=wav.device)
        for m in range(n_mels):
            f_center = 8000.0 * (m + 1) / n_mels
            for k in range(n_fft // 2 + 1):
                freqs = torch.linspace(0, 8000, n_fft // 2 + 1, device=wav.device)
                mel_bank[k, m] = torch.exp(-0.5 * ((freqs[k] - f_center) / (8000.0 / n_mels)) ** 2)
        mel = mag @ mel_bank
        log_mel = torch.log(mel + 1e-8)
        if self.cfg.mean_norm:
            log_mel = log_mel - log_mel.mean(dim=1, keepdim=True)
        return log_mel

    @torch.no_grad()
    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        """从波形提取嵌入。

        Args:
            wav: [B, T] float 波形（16 kHz）

        Returns:
            emb: [B, D] L2 归一化嵌入

        Raises:
            ValueError: 输入形状错误、含 NaN/Inf
            RuntimeError: 模型未加载
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("模型未加载，先调用 load()")
        if wav.ndim != 2:
            raise ValueError(f"wav 需为 [B,T]，得到 ndim={wav.ndim}")
        if wav.numel() == 0:
            raise ValueError("wav 为空数组")
        if not torch.isfinite(wav).all():
            raise ValueError("wav 含 NaN/Inf")

        wav = wav.to(self.device)
        feats = self._extract_fbank(wav, self.cfg.sample_rate)
        emb = self._model(feats)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb

    @torch.no_grad()
    def embed_batch(
        self, wavs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """批量嵌入提取（逐条处理，避免 fbank batch 问题）。

        Args:
            wavs: 多个 [T] 或 [1, T] 波形

        Returns:
            emb: [B, D] 批量嵌入
        """
        embs = []
        for w in wavs:
            w = w if w.ndim == 2 else w.unsqueeze(0)
            embs.append(self.embed(w))
        return torch.cat(embs, dim=0)

    @staticmethod
    def _pad_and_stack(wavs: Sequence[torch.Tensor]) -> torch.Tensor:
        lengths = [w.shape[-1] for w in wavs]
        max_len = max(lengths)
        batch = []
        for w in wavs:
            w = w.squeeze(0) if w.ndim > 1 else w
            if w.shape[-1] < max_len:
                w = F.pad(w, (0, max_len - w.shape[-1]))
            batch.append(w)
        return torch.stack(batch, dim=0)

    def unload(self) -> None:
        """释放模型资源。"""
        self._model = None
        self._loaded = False

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "unloaded"
        return (
            f"CampplusBackend(model_id={self.cfg.model_id!r}, "
            f"embedding_size={self.cfg.embedding_size}, "
            f"device={self.cfg.device!r}, status={status})"
        )
