# -*- coding: utf-8 -*-
"""P2-08 注册条件验证（五条件测试）。

目的：证明 enrollment 条件（FiLM 注入）确实影响 TSE 输出，而不是摆设；
同时验证条件路径不会凭空制造能量（05B §8 防幻觉）。

检查项：
  c_determinism     相同 (x, e) 两次前向输出完全一致（eval 模式确定性基线）
  c_diff_embedding  同一 mixture，正确 e vs 错误 e2：输出显著不同
  c_zero_embedding  同一 mixture，正确 e vs 全零 e：输出显著不同
  c_shuffle         batch 内打乱 embedding：各样本输出与自身条件输出显著不同
  c_zero_mixture    全零 mixture + 任意 e：输出 ≈ 0（条件不凭空造能量）
  c_embedding_grad  损失对 embedding 张量的梯度有限且非零（条件通路在计算图内）

判定：显著性阈值 rel_diff > 1e-3（相对 L2，基线为确定性噪声 ~0）。

运行：
  python tests/p2/test_tse_conditioning.py --device auto
退出码 0 = 全 PASS。报告：reports/smoke/tse_conditioning.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tse import DualOutputTSE

REL_DIFF_THRESHOLD = 1.0e-3
ZERO_OUT_THRESHOLD = 1.0e-6
REPORT_PATH = ROOT / "reports" / "smoke" / "tse_conditioning.json"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rel_diff(a, b, eps=1e-12):
    """相对 L2 差异：||a-b|| / (||a||+eps)，逐样本计算后返回张量。"""
    num = torch.linalg.norm((a - b).flatten(1), dim=1)
    den = torch.linalg.norm(a.flatten(1), dim=1) + eps
    return num / den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "p2" / "tse_smoke.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )

    checks = []
    report = {
        "test": "P2-08_tse_conditioning",
        "date": time.strftime("%Y-%m-%d"),
        "device": str(device),
        "torch": torch.__version__,
        "seed": args.seed,
        "rel_diff_threshold": REL_DIFF_THRESHOLD,
        "architecture_status": "DEBUG_ONLY_scaffold_not_formal",
        "checks": checks,
    }

    model = DualOutputTSE(cfg).to(device)
    model.eval()

    emb_dim = int(cfg["emb_dim"])
    B, T = 4, 16000

    # 固定一条混合与两个不同注册条件
    g = torch.Generator().manual_seed(args.seed)
    x = (0.5 * torch.randn(B, T, generator=g)).to(device)
    e_a = torch.randn(B, emb_dim, generator=g).to(device)
    e_b = torch.randn(B, emb_dim, generator=g).to(device)
    e_zero = torch.zeros(B, emb_dim, device=device)
    # 循环移位保证无不动点（randperm 可能让某样本仍拿到自己的 embedding）
    e_shuffle = e_a[list(range(1, B)) + [0]]

    with torch.no_grad():
        s1, r1, _ = model(x, e_a)
        s1_b, r1_b, _ = model(x, e_a)       # 确定性基线
        s2, r2, _ = model(x, e_b)           # 错误 embedding
        s0, r0, _ = model(x, e_zero)        # 全零 embedding
        s_sh, r_sh, _ = model(x, e_shuffle)  # 打乱 embedding
        x_zero = torch.zeros_like(x)
        sz, rz, _ = model(x_zero, e_a)      # 全零 mixture

    # c_determinism
    d = max((s1 - s1_b).abs().max().item(), (r1 - r1_b).abs().max().item())
    checks.append({"name": "c_determinism", "pass": d < 1e-7,
                   "detail": f"max_abs_diff={d:.3e}"})

    # c_diff_embedding（目标路输出相对差异）
    rd = rel_diff(s1, s2)
    checks.append({"name": "c_diff_embedding",
                   "pass": bool((rd > REL_DIFF_THRESHOLD).all()),
                   "detail": f"rel_diff min={rd.min():.4f} mean={rd.mean():.4f}"})

    # c_zero_embedding
    rd0 = rel_diff(s1, s0)
    checks.append({"name": "c_zero_embedding",
                   "pass": bool((rd0 > REL_DIFF_THRESHOLD).all()),
                   "detail": f"rel_diff min={rd0.min():.4f} mean={rd0.mean():.4f}"})

    # c_shuffle
    rdsh = rel_diff(s1, s_sh)
    checks.append({"name": "c_shuffle",
                   "pass": bool((rdsh > REL_DIFF_THRESHOLD).all()),
                   "detail": f"rel_diff min={rdsh.min():.4f} mean={rdsh.mean():.4f}"})

    # c_zero_mixture（条件不得凭空造能量）
    mz = max(sz.abs().max().item(), rz.abs().max().item())
    checks.append({"name": "c_zero_mixture", "pass": mz < ZERO_OUT_THRESHOLD,
                   "detail": f"max|out|={mz:.3e} (threshold {ZERO_OUT_THRESHOLD})"})

    # c_embedding_grad（条件通路在计算图内，梯度可达 embedding）
    model.train()
    e_grad = e_a.clone().requires_grad_(True)
    s_g, r_g, _ = model(x, e_grad)
    loss = (s_g ** 2).mean() + (r_g ** 2).mean()
    loss.backward()
    g_ok = (e_grad.grad is not None
            and bool(torch.isfinite(e_grad.grad).all())
            and e_grad.grad.abs().max().item() > 0)
    checks.append({"name": "c_embedding_grad", "pass": g_ok,
                   "detail": f"emb_grad max|g|={e_grad.grad.abs().max():.4e}"})
    model.eval()

    all_pass = all(c["pass"] for c in checks)
    report["verdict"] = "PASS" if all_pass else "FAIL"

    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['name']:20s} {c['detail']}")
    print(f"verdict={report['verdict']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"report={REPORT_PATH}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
