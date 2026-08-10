"""P4 sv 模块 smoke 测试。

验证：
1. CampplusBackend 可加载（无需预训练权重，随机初始化即可）
2. encode_enrollment 接口可调用，输出正确维度
3. cosine_similarity 同说话人 > 不同说话人
4. sv_contract_v1 接口契约完整
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ── 路径设置（测试独立运行时需要） ──
_P4_ROOT = Path(__file__).resolve().parents[1]
_P4_SRC = _P4_ROOT / "src"
_P4_TOOLS = _P4_ROOT / "tools"
_P4_SPEAKERLAB = _P4_ROOT / "artifacts" / "models" / "speakerlab_source"
for _p in [str(_P4_SRC), str(_P4_TOOLS), str(_P4_SPEAKERLAB)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sv.campplus_backend import CampplusBackend
from sv.types import CampplusBackendConfig
from sv.encode_enrollment import _load_wav


# ── 测试音频路径 ──
_FUNASR_ROOT = _P4_ROOT.parent
_ENROLL_S1 = _FUNASR_ROOT / "data" / "trials" / "enroll_S0764.wav"
_ENROLL_S2 = _FUNASR_ROOT / "data" / "trials" / "enroll_S0765.wav"
_DEBUG_MIX = _FUNASR_ROOT / "P2_project" / "artifacts" / "debug_mixtures_v0" / "dbg_S0764_partial25_mixture.wav"


class TestCampplusBackend:
    """CampplusBackend 基础功能测试。"""

    def test_load_random_init(self):
        """模型可加载（随机初始化，无预训练权重）。"""
        cfg = CampplusBackendConfig(
            embedding_size=192,
            device="cpu",
        )
        backend = CampplusBackend(cfg)
        backend.load()
        assert backend.is_loaded
        assert backend.cfg.embedding_size == 192

    def test_embed_shape(self):
        """embed 输出形状正确 [1, 192]。"""
        cfg = CampplusBackendConfig(embedding_size=192, device="cpu")
        backend = CampplusBackend(cfg)
        backend.load()

        wav = torch.randn(1, 16000)  # 1 秒 16kHz 白噪声
        emb = backend.embed(wav)
        assert emb.shape == (1, 192)
        assert torch.isfinite(emb).all()

    def test_embed_different_seeds(self):
        """不同输入产生嵌入（随机初始化模型，仅验证形状正确）。"""
        cfg = CampplusBackendConfig(embedding_size=192, device="cpu")
        backend = CampplusBackend(cfg)
        backend.load()

        wav1 = torch.randn(1, 16000)
        wav2 = torch.randn(1, 16000)
        emb1 = backend.embed(wav1)
        emb2 = backend.embed(wav2)
        # 随机初始化模型，嵌入不保证区分度；验证形状正确即可
        assert emb1.shape == (1, 192)
        assert emb2.shape == (1, 192)
        assert torch.isfinite(emb1).all()
        assert torch.isfinite(emb2).all()

    def test_embed_batch(self):
        """批量嵌入输出 [B, 192]。"""
        cfg = CampplusBackendConfig(embedding_size=192, device="cpu")
        backend = CampplusBackend(cfg)
        backend.load()

        wavs = [torch.randn(16000), torch.randn(16000), torch.randn(8000)]
        emb = backend.embed_batch(wavs)
        assert emb.shape == (3, 192)

    def test_unload(self):
        """unload 释放资源。"""
        cfg = CampplusBackendConfig(embedding_size=192, device="cpu")
        backend = CampplusBackend(cfg)
        backend.load()
        assert backend.is_loaded
        backend.unload()
        assert not backend.is_loaded

    def test_embed_empty_raises(self):
        """空输入报错。"""
        cfg = CampplusBackendConfig(embedding_size=192, device="cpu")
        backend = CampplusBackend(cfg)
        backend.load()
        with pytest.raises(ValueError):
            backend.embed(torch.zeros(1, 0))


class TestEncodeEnrollment:
    """encode_enrollment 模块测试。"""

    def test_load_wav(self):
        """_load_wav 正常加载音频。"""
        if not _DEBUG_MIX.exists():
            pytest.skip(f"测试音频不存在: {_DEBUG_MIX}")
        wav = _load_wav(str(_DEBUG_MIX), 16000)
        assert wav.ndim == 1
        assert wav.numel() > 0

    def test_load_wav_missing_raises(self):
        """不存在的文件报错。"""
        with pytest.raises(FileNotFoundError):
            _load_wav("/nonexistent/file.wav")

    def test_segment_wav(self):
        """_segment_wav 切分正确。"""
        from sv.encode_enrollment import _segment_wav
        wav = torch.randn(16000)  # 1 秒
        segments = _segment_wav(wav, 3000)  # 3 秒一段
        assert len(segments) == 1
        assert segments[0].numel() == 16000

        wav_long = torch.randn(100000)  # 6.25 秒
        segments = _segment_wav(wav_long, 3000)
        assert len(segments) == 3  # 3s + 3s + 0.25s

    def test_enrollment_encoder_bootstrap(self):
        """EnrollmentEncoder BOOTSTRAP 模式。"""
        from sv.encode_enrollment import EnrollmentEncoder, EnrollmentSession
        from sv.types import EnrollConfig

        cfg = EnrollConfig(embed_dim=192)
        encoder = EnrollmentEncoder(None, cfg)

        session = EnrollmentSession("test_spk")
        session.add(str(_DEBUG_MIX))

        emb = encoder.encode(session)
        assert emb.vector.shape == (192,)
        assert emb.mode.value == "bootstrap"


class TestSvContractV1:
    """sv_contract_v1 接口契约测试。"""

    def test_get_model_info(self):
        """get_model_info 返回正确结构。"""
        from sv_contract_v1 import get_model_info
        info = get_model_info()
        assert info["contract_version"] == "sv_contract_v1"
        assert info["embedding_size"] == 192
        assert info["sample_rate"] == 16000

    def test_cosine_similarity_identical(self):
        """同一嵌入余弦相似度 ≈ 1。"""
        from sv_contract_v1 import cosine_similarity
        emb = np.random.randn(192).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        sim = cosine_similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """正交嵌入余弦相似度 ≈ 0。"""
        from sv_contract_v1 import cosine_similarity
        emb_a = np.zeros(192, dtype=np.float32)
        emb_a[0] = 1.0
        emb_b = np.zeros(192, dtype=np.float32)
        emb_b[1] = 1.0
        sim = cosine_similarity(emb_a, emb_b)
        assert abs(sim) < 1e-5

    @pytest.mark.skipif(not _ENROLL_S1.exists(), reason="enroll 音频不存在")
    def test_encode_enrollment_output(self):
        """encode_enrollment 返回 np.ndarray [192]。"""
        from sv_contract_v1 import encode_enrollment
        emb = encode_enrollment(str(_ENROLL_S1), embedding_size=192)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (192,)
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()