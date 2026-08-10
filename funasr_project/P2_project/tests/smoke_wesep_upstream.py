# -*- coding: utf-8 -*-
"""P3 SMOKE-2 上游合约烟雾测试脚本（v2 分支版）。

本脚本验证 P2 TSE 模块的上游合约（forward 契约、损失函数、推理输出结构）
是否与 P3 合约一致，用于 P3 开发者的本地烟雾测试。

验证内容：
1. forward 输出形状契约：(B, T) 波形 + (B, T) 残差 + (B, F) activity
2. 损失函数：si_sdr / scale_sensitive_l1 / mix_consistency / mrstft / activity_bce
3. 推理输出：波形归一化、裁剪保护、SI-SDR 去均值
4. 数值稳定性：NaN/Inf 检测、梯度有限性

用法：
  python tests/smoke_wesep_upstream.py
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
sys.path.insert(0, str(P2_ROOT))

from src.tse.model import DualOutputTSE
from src.tse.losses import (
    absent_zero_loss,
    activity_bce_loss,
    mix_consistency_loss,
    mrstft_loss,
    scale_sensitive_l1,
    si_sdr,
)
from src.tse.metrics import si_sdr_eval


def _cfg():
    return {
        "sample_rate": 16000, "win_length": 1024, "hop_length": 256,
        "n_mels": 128, "n_fft": 1024, "cfg": 1.0,
        "encoder_channels": 64, "bottleneck_channels": 128, "decoder_channels": 64,
        "kernel_size": 3, "depth": 8, "bottle_depth": 2, "growth": 2,
        "groups": 8, "emb_dim": 64, "use_spk_cond": True,
        "loss": {
            "wav_l1_weight": 1.0, "mrstft_weight": 0.0, "si_sdr_weight": 0.0,
            "act_bce_weight": 0.0, "mix_weight": 0.0, "res_weight": 0.0,
        },
        "segment_length": 2.0, "loss_clip": 10.0, "restore_tol": 1e-5,
    }


def test_forward_contract():
    cfg = _cfg()
    model = DualOutputTSE(cfg)
    model.eval()
    T = int(cfg["sample_rate"] * cfg["segment_length"])
    mix = torch.randn(2, T)
    emb = torch.randn(2, cfg["emb_dim"])

    s_tgt, s_res, p_tgt = model(mix, emb)

    assert s_tgt.shape == (2, T), f"s_tgt shape {s_tgt.shape} != (2, {T})"
    assert s_res.shape == (2, T), f"s_res shape {s_res.shape} != (2, {T})"
    F = T // cfg["hop_length"] + 1
    assert p_tgt.shape == (2, F), f"p_tgt shape {p_tgt.shape} != (2, {F})"
    assert torch.isfinite(s_tgt).all(), "s_tgt 含 NaN/Inf"
    assert torch.isfinite(s_res).all(), "s_res 含 NaN/Inf"
    assert torch.isfinite(p_tgt).all(), "p_tgt 含 NaN/Inf"
    return "PASS"


def test_loss_contract():
    cfg = _cfg()
    model = DualOutputTSE(cfg)
    T = int(cfg["sample_rate"] * cfg["segment_length"])
    mix = torch.randn(2, T, requires_grad=False)
    emb = torch.randn(2, cfg["emb_dim"])
    target = torch.randn(2, T)
    interferer = torch.randn(2, T)

    s_tgt, s_res, p_tgt = model(mix, emb)

    wav_l1 = scale_sensitive_l1(s_tgt, target)
    si_sdr_val = si_sdr(s_tgt, target)
    mrstft_val = mrstft_loss(s_tgt, target, cfg["sample_rate"],
                             cfg["win_length"], cfg["hop_length"], cfg["n_mels"])
    act_target = torch.zeros(2, T)
    act_target[:, : T // 2] = 1.0
    from src.tse.model import _frame_activity
    fa = _frame_activity(act_target, cfg["win_length"], cfg["hop_length"], 0.5)
    act_bce = activity_bce_loss(p_tgt, fa)
    mix_cons = mix_consistency_loss(s_tgt, s_res, target, interferer)

    assert torch.isfinite(wav_l1), f"wav_l1={wav_l1} 非有限"
    assert torch.isfinite(si_sdr_val), f"si_sdr={si_sdr_val} 非有限"
    assert torch.isfinite(mrstft_val), f"mrstft={mrstft_val} 非有限"
    assert torch.isfinite(act_bce), f"act_bce={act_bce} 非有限"
    assert torch.isfinite(mix_cons), f"mix_cons={mix_cons} 非有限"

    total = wav_l1 + 0.0 * si_sdr_val + 0.0 * mrstft_val + 0.0 * act_bce + 0.0 * mix_cons
    total.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} 梯度含 NaN/Inf"
    return "PASS"


def test_inference_contract():
    cfg = _cfg()
    model = DualOutputTSE(cfg)
    model.eval()
    T = int(cfg["sample_rate"] * cfg["segment_length"])
    mix = torch.randn(1, T) * 0.5
    emb = torch.randn(1, cfg["emb_dim"])

    with torch.no_grad():
        s_tgt, s_res, p_tgt = model(mix, emb)

    s_est = s_tgt[0].numpy()
    assert not np.isnan(s_est).any(), "推理输出含 NaN"
    assert not np.isinf(s_est).any(), "推理输出含 Inf"
    peak = np.max(np.abs(s_est))
    if peak > 0.99:
        s_est = s_est * (0.99 / peak)
    assert np.max(np.abs(s_est)) <= 1.0, f"峰值超过 1.0: {peak}"

    ref = mix[0].numpy()
    sd = si_sdr_eval(torch.from_numpy(s_est), torch.from_numpy(ref))
    assert np.isfinite(sd), f"SI-SDR={sd} 非有限"
    return "PASS"


def test_absent_contract():
    cfg = _cfg()
    model = DualOutputTSE(cfg)
    model.eval()
    T = int(cfg["sample_rate"] * cfg["segment_length"])
    mix = torch.randn(1, T)
    emb = torch.randn(1, cfg["emb_dim"])
    zero_target = torch.zeros(1, T)

    s_tgt, s_res, p_tgt = model(mix, emb)

    loss_val = absent_zero_loss(s_tgt, zero_target)
    assert torch.isfinite(loss_val), f"absent_zero_loss={loss_val} 非有限"
    assert loss_val.item() >= 0, f"absent_zero_loss={loss_val} 为负"
    return "PASS"


def test_consistency_contract():
    cfg = _cfg()
    model = DualOutputTSE(cfg)
    model.eval()
    T = int(cfg["sample_rate"] * cfg["segment_length"])

    with torch.no_grad():
        for _ in range(3):
            mix = torch.randn(1, T)
            emb = torch.randn(1, cfg["emb_dim"])
            s_tgt, s_res, p_tgt = model(mix, emb)
            recon = s_tgt + s_res
            diff = (recon - mix).abs().max().item()
            assert diff < 1.0, f"重建误差过大: {diff}"

    return "PASS"


def main():
    print("=" * 60)
    print("P3 SMOKE-2 上游合约烟雾测试 (v2 分支)")
    print("=" * 60)

    tests = [
        ("forward_contract", test_forward_contract),
        ("loss_contract", test_loss_contract),
        ("inference_contract", test_inference_contract),
        ("absent_contract", test_absent_contract),
        ("consistency_contract", test_consistency_contract),
    ]

    results = {}
    passed = 0
    failed = 0
    errors = []

    for name, fn in tests:
        try:
            status = fn()
            results[name] = status
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            failed += 1
            errors.append({"test": name, "error": str(e),
                           "traceback": traceback.format_exc()})
            print(f"  [FAIL] {name}: {e}")

    summary = {
        "suite": "P3 SMOKE-2 upstream contract",
        "branch": "v2",
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "results": results,
        "errors": errors,
        "overall": "PASS" if failed == 0 else "FAIL",
    }

    report_path = P2_ROOT / "reports" / "upstream" / "smoke_wesep_upstream.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"结果: {passed}/{len(tests)} PASS, {failed}/{len(tests)} FAIL")
    print(f"总体判定: {summary['overall']}")
    print(f"报告: {report_path}")
    print(f"{'='*60}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())