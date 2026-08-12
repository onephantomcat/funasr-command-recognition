#!/usr/bin/env python
"""P2-06 双输出 TSE 随机张量 smoke 测试（v2 分支版）。

退出码：0 = 至少一个设备全 PASS 且无 FAIL；1 = 存在 FAIL 或指定设备被阻塞。
产物：logs/smoke_tse_random.log、reports/smoke/tse_random_forward.json
"""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tse import (
    DualOutputTSE,
    activity_bce_loss,
    absent_zero_loss,
    extract_target,
    mix_consistency_loss,
    scale_sensitive_l1,
    si_sdr,
)

DEFAULT_LENGTHS = [160, 16000, 57600, 160000]
CONSISTENCY_TOL = 1e-5
VRAM_BUDGET_BYTES = 6 * 1024 ** 3
TINY_SCALE = 1e-8

log = logging.getLogger("smoke_tse_random")


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.handlers = [sh, fh]


def check(name, ok, detail=""):
    return {"name": name, "pass": bool(ok), "detail": detail}


def run_length_tier(model, cfg, T, device):
    tier = {"length": T, "checks": [], "losses": {}, "output_shapes": {}}
    torch.manual_seed(cfg["seed"] + T)
    x = torch.randn(1, T, device=device)
    e = torch.randn(1, cfg["emb_dim"], device=device)

    s_tgt, s_res, p_tgt = model(x, e)
    tier["output_shapes"] = {
        "x": list(x.shape), "s_tgt": list(s_tgt.shape), "s_res": list(s_res.shape)
    }
    tier["checks"].append(check(
        "c1_shape", s_tgt.shape == x.shape and s_res.shape == x.shape,
        f"s_tgt{tuple(s_tgt.shape)} s_res{tuple(s_res.shape)} x{tuple(x.shape)}"))
    finite = torch.isfinite(s_tgt).all() and torch.isfinite(s_res).all()
    tier["checks"].append(check("c2_finite", finite))

    l_sisdr = si_sdr(s_tgt, x)
    l_res = scale_sensitive_l1(s_res, x, cfg["zero_ref_kappa"])
    l_mix = mix_consistency_loss(s_tgt, s_res, x)
    # Random non-silent mixtures are treated as active in this structural
    # smoke.  Including the activity term ensures act_head participates in
    # the backward pass instead of being reported as a false dead branch.
    l_act = activity_bce_loss(p_tgt, torch.ones_like(p_tgt))
    loss = (-cfg["lambda_target"] * l_sisdr
            + cfg["lambda_residual"] * l_res
            + cfg["lambda_mix"] * l_mix
            + cfg.get("lambda_act", 0.5) * l_act)
    tier["losses"] = {
        "si_sdr_db": round(l_sisdr.item(), 4),
        "res_l1": round(l_res.item(), 6),
        "mix": round(l_mix.item(), 8),
        "act_bce": round(l_act.item(), 6),
        "total": round(loss.item(), 4),
    }
    tier["checks"].append(check("c3_loss_computable", torch.isfinite(loss)))

    model.zero_grad(set_to_none=True)
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is None
           or not torch.isfinite(p.grad).all()
           or p.grad.abs().max().item() <= 0.0]
    tier["checks"].append(check(
        "c4_grad_finite_nonzero", not bad,
        "all 16 param tensors ok" if not bad else f"bad: {bad}"))

    cons = (s_tgt + s_res - x).abs().max().item()
    tier["checks"].append(check(
        "c5_consistency", cons < CONSISTENCY_TOL, f"max_err={cons:.3e}"))
    return tier


def run_anomaly_suite(model, cfg, device):
    results = []
    e = torch.randn(1, cfg["emb_dim"], device=device)
    for tag, x in [("zero_input", torch.zeros(1, 16000, device=device)),
                   ("tiny_input", TINY_SCALE * torch.randn(1, 16000, device=device))]:
        with torch.no_grad():
            s, r, _ = model(x, e)
        ok = bool(torch.isfinite(s).all() and torch.isfinite(r).all())
        results.append(check(f"a_stability_{tag}", ok,
                             f"max|s|={s.abs().max().item():.3e}"))
    model.train()
    x_ok = torch.randn(1, 16000, device=device)
    for tag, x_bad, e_bad in [
        ("nan_input", torch.full((1, 16000), float("nan"), device=device), e),
        ("wrong_emb_dim", x_ok, torch.randn(1, cfg["emb_dim"] + 1, device=device)),
        ("empty_input", torch.empty(1, 0, device=device), e),
    ]:
        try:
            extract_target(x_bad, e_bad, model, cfg)
            results.append(check(f"a_reject_{tag}", False, "未拒绝（应抛 ValueError）"))
        except ValueError as err:
            results.append(check(f"a_reject_{tag}", True, str(err)))
    model.train()
    return results


def run_absent_zero_loss_unit(device):
    out = {"checks": []}
    z = torch.zeros(1, 16000, device=device)
    r = torch.randn(1, 16000, device=device, requires_grad=True)
    lz, lr = absent_zero_loss(z), absent_zero_loss(r)
    out["checks"].append(check("az_zero_is_zero", lz.item() == 0.0, f"{lz.item()}"))
    out["checks"].append(check("az_nonzero_positive", lr.item() > 0.0, f"{lr.item():.4f}"))
    lr.backward()
    out["checks"].append(check(
        "az_grad_finite", bool(torch.isfinite(r.grad).all()),
        f"max|g|={r.grad.abs().max().item():.3e}"))
    out["note"] = "预留损失，未接入总损失"
    return out


def run_device_pass(device, cfg, lengths):
    p = {"device": device, "status": "PASS", "tiers": [], "anomaly": [],
         "peak_vram_bytes": None, "vram_budget_bytes": VRAM_BUDGET_BYTES,
         "vram_check": "SKIPPED_cpu"}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(cfg["seed"])
    model = DualOutputTSE(cfg).to(device)
    model.train()
    p["param_count"] = sum(q.numel() for q in model.parameters())
    log.info("[%s] 参数量 %d", device, p["param_count"])
    try:
        for T in lengths:
            tier = run_length_tier(model, cfg, T, device)
            p["tiers"].append(tier)
            status = "PASS" if all(c["pass"] for c in tier["checks"]) else "FAIL"
            log.info("[%s] T=%d %s losses=%s", device, T, status, tier["losses"])
        p["anomaly"] = run_anomaly_suite(model, cfg, device)
        for c in p["anomaly"]:
            log.info("[%s] %s %s %s", device, c["name"],
                     "PASS" if c["pass"] else "FAIL", c["detail"])
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated()
            p["peak_vram_bytes"] = peak
            ok = peak < VRAM_BUDGET_BYTES
            p["vram_check"] = "PASS" if ok else "FAIL"
            log.info("[cuda] 峰值显存 %.2f GB（预算 6 GB）%s",
                     peak / 1024 ** 3, p["vram_check"])
    except Exception:
        p["status"] = "FAIL"
        p["exception"] = traceback.format_exc()
        log.error("[%s] 设备通道异常:\n%s", device, p["exception"])
    all_checks = ([c for t in p["tiers"] for c in t["checks"]]
                  + p["anomaly"])
    if p["vram_check"] == "FAIL":
        all_checks.append({"name": "c6_vram", "pass": False})
    if any(not c["pass"] for c in all_checks):
        p["status"] = "FAIL"
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--lengths", type=int, nargs="+", default=DEFAULT_LENGTHS)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "tse_smoke.yaml"))
    args = ap.parse_args()

    setup_logging(PROJECT_ROOT / "logs" / "smoke_tse_random.log")
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.seed is not None:
        cfg["seed"] = args.seed
    log.info("config=%s lengths=%s device=%s", args.config, args.lengths, args.device)
    log.info("torch=%s cuda_available=%s", torch.__version__, torch.cuda.is_available())

    report = {
        "test": "tse_random_forward",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "architecture_status": "DEBUG_ONLY_scaffold_not_formal",
        "markers": ["DEBUG_ONLY", "BOOTSTRAP_ENCODER_ONLY", "NOT_FOR_FORMAL_METRICS"],
        "seed": cfg["seed"],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_passes": [],
        "blocked": [],
        "overall": "FAIL",
    }

    devices = {"auto": ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
               "cpu": ["cpu"], "cuda": ["cuda"]}[args.device]
    wants_cuda = args.device in ("auto", "cuda")
    if wants_cuda and not torch.cuda.is_available():
        report["blocked"].append({
            "scope": "cuda_pass", "status": "BLOCKED_EXTERNAL",
            "reason": "cuda_unavailable",
            "note": "本机无 CUDA"})
        devices = [d for d in devices if d != "cuda"]
    for d in devices:
        report["device_passes"].append(run_device_pass(d, cfg, args.lengths))

    if report["device_passes"]:
        report["absent_zero_loss_unit"] = run_absent_zero_loss_unit(
            report["device_passes"][0]["device"])
        az_ok = all(c["pass"] for c in report["absent_zero_loss_unit"]["checks"])
        passes_ok = [p["status"] == "PASS" for p in report["device_passes"]]
        if any(passes_ok) and all(passes_ok) and az_ok:
            report["overall"] = "PASS"
        log.info("absent_zero_loss 单元检查 %s", "PASS" if az_ok else "FAIL")
    elif report["blocked"]:
        report["overall"] = "BLOCKED_EXTERNAL"

    out = PROJECT_ROOT / "reports" / "smoke" / "tse_random_forward.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("报告已写入 %s", out)
    log.info("总体判定: %s", report["overall"])
    sys.exit(0 if report["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
