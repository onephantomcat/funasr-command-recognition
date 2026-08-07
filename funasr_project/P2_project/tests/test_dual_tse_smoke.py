# -*- coding: utf-8 -*-
"""P2-06 双输出 TSE smoke 的 pytest 薄包装（v2 分支版）。

断言口径与 smoke_tse_random.py 一致；P2-08 条件验证见
test_tse_conditioning.py（独立脚本，可直接运行）。

标记：DEBUG_ONLY（随机张量，不构成正式指标）。
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

P2_ROOT = Path(__file__).resolve().parents[1]
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from src.tse import (
    DualOutputTSE,
    absent_zero_loss,
    activity_bce_loss,
    extract_target,
    mix_consistency_loss,
    scale_sensitive_l1,
    si_sdr,
)

CONFIG_PATH = P2_ROOT / "configs" / "tse_smoke.yaml"
LENGTHS = [160, 16000, 57600, 160000]
SEED = 20260725


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(SEED)
    m = DualOutputTSE(cfg)
    m.train()
    return m


def _rand_pair(cfg, length):
    g = torch.Generator().manual_seed(SEED + length)
    x = 0.5 * torch.randn(1, length, generator=g)
    e = torch.randn(1, int(cfg["emb_dim"]), generator=g)
    return x, e


@pytest.mark.parametrize("length", LENGTHS)
def test_forward_shape_finite(cfg, model, length):
    x, e = _rand_pair(cfg, length)
    s, r, p = model(x, e)
    assert s.shape == x.shape and r.shape == x.shape
    assert p.ndim == 2 and p.shape[0] == x.shape[0]
    assert torch.isfinite(s).all() and torch.isfinite(r).all() and torch.isfinite(p).all()


@pytest.mark.parametrize("length", LENGTHS)
def test_consistency_projection(cfg, model, length):
    x, e = _rand_pair(cfg, length)
    s, r, _ = model(x, e)
    assert (s + r - x).abs().max().item() < 1e-5


def test_backward_and_grads(cfg, model):
    x, e = _rand_pair(cfg, 16000)
    s, r, p = model(x, e)
    frame_act = torch.ones_like(p)
    loss = (-float(cfg["lambda_target"]) * si_sdr(s, x)
            + float(cfg["lambda_residual"]) * scale_sensitive_l1(
                r, 0.1 * x, float(cfg["zero_ref_kappa"]))
            + float(cfg["lambda_mix"]) * mix_consistency_loss(s, r, x)
            + activity_bce_loss(p, frame_act))
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"
        assert p.grad.abs().max().item() > 0, f"{name} 梯度全零"


def test_absent_zero_loss_reserved():
    assert absent_zero_loss(torch.zeros(1, 16000)).item() == pytest.approx(0.0)
    noisy = torch.randn(1, 16000, generator=torch.Generator().manual_seed(1))
    val = absent_zero_loss(noisy)
    assert val.item() > 0 and torch.isfinite(val)
    noisy.requires_grad_(True)
    absent_zero_loss(noisy).backward()
    assert torch.isfinite(noisy.grad).all()


def test_api_rejects_bad_input(cfg, model):
    x, e = _rand_pair(cfg, 16000)
    with pytest.raises(ValueError):
        extract_target(torch.full_like(x, float("nan")), e, model, cfg)
    with pytest.raises(ValueError):
        extract_target(x, torch.randn(1, int(cfg["emb_dim"]) + 1), model, cfg)
    with pytest.raises(ValueError):
        extract_target(x[:, :0], e, model, cfg)


def test_api_shape_contract(cfg, model):
    x, e = _rand_pair(cfg, 16000)
    out = extract_target(x, e, model, cfg)
    assert out.shape == x.shape and torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda_unavailable(BLOCKED_EXTERNAL)")
def test_vram_under_6gb(cfg):
    m = DualOutputTSE(cfg).cuda()
    torch.cuda.reset_peak_memory_stats()
    x, e = _rand_pair(cfg, 160000)
    x, e = x.cuda(), e.cuda()
    s, r, _ = m(x, e)
    loss = si_sdr(s, x) + mix_consistency_loss(s, r, x)
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    assert peak < 6.0, f"峰值显存 {peak:.2f} GB 超 6 GB 预算"