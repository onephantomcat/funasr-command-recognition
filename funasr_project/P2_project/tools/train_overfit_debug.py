# -*- coding: utf-8 -*-
"""P2-09 固定小批量 100 step 过拟合闭环验证。

目的（手册 STEP P2-09）：烧大量 GPU 前证明 数据→目标→损失→条件→梯度→checkpoint 闭环。

固定设置：8 条 DEBUG_ONLY 混合（不 shuffle）、seed 20260723、AMP 关、增强关。
PRESENT 损失：-λ_sisdr·SI-SDR + λ_wav·L1(尺度敏感) + λ_stft·MR-STFT + λ_act·BCE(p_tgt, m)
              + λ_residual·L1(residual, interferer)   （λ_id=0 硬约束）

产物：artifacts/experiments/DEBUG_OVERFIT_seed<seed>/
  config.yaml / config.sha256 / data.sha256 / train.log / metrics.jsonl /
  checkpoint_step100.pt / checkpoint.sha256 / before_after_audio/ / report.md

运行：
  python tools/train_overfit_debug.py --device auto
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
sys.path.insert(0, str(P2_ROOT))

from src.tse.model import DualOutputTSE
from src.tse.losses import (
    si_sdr,
    scale_sensitive_l1,
    mix_consistency_loss,
    mrstft_loss,
    activity_bce_loss,
    _stft_mag,
)


LOG = logging.getLogger("p2_overfit")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bootstrap_embedding(enrollment_rel_path, emb_dim):
    """BOOTSTRAP_ENCODER_ONLY：由注册路径哈希派生的确定性随机 embedding。

    同一注册音频 → 同一 embedding；不同说话人 → 不同 embedding（保留 swap 语义）。
    """
    seed = int(sha256_text(enrollment_rel_path)[:8], 16)
    g = torch.Generator().manual_seed(seed)
    return torch.randn(emb_dim, generator=g)


def frame_activity(activity, win_length, hop_length, ratio_thresh):
    """采样级 activity [T] → 帧级 [Fr]（与 center=True STFT 对齐）。

    两端各补 win//2 后按 win/hop 平均池化，占比 > ratio_thresh 判活动。
    """
    a = torch.from_numpy(activity).float().unsqueeze(0).unsqueeze(0)
    a = torch.nn.functional.pad(a, (win_length // 2, win_length // 2))
    frame_ratio = torch.nn.functional.avg_pool1d(a, kernel_size=win_length, stride=hop_length)
    return (frame_ratio.squeeze(0).squeeze(0) > ratio_thresh).float()


def load_fixed_batch(cfg, manifest_path, batch_size, device):
    """按 manifest id 排序取前 batch_size 条，固定不 shuffle。"""
    entries = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            entries.append(json.loads(line))
    entries = sorted(entries, key=lambda e: e["id"])[:batch_size]
    assert len(entries) == batch_size, f"样本不足：{len(entries)} < {batch_size}"

    base = FUNASR_ROOT
    mixes, targets, inters, acts, embs, ids = [], [], [], [], [], []
    for e in entries:
        mix, sr = sf.read(str(base / e["mixture"]), dtype="float32")
        tgt, _ = sf.read(str(base / e["target"]), dtype="float32")
        itr, _ = sf.read(str(base / e["interferer"]), dtype="float32")
        act = np.load(str(base / e["activity"]))
        assert sr == cfg["sample_rate"]
        mixes.append(torch.from_numpy(mix))
        targets.append(torch.from_numpy(tgt))
        inters.append(torch.from_numpy(itr))
        acts.append(frame_activity(act, cfg["win_length"], cfg["hop_length"],
                                   float(cfg["act_frame_ratio"])))
        embs.append(bootstrap_embedding(e["enrollment"], int(cfg["emb_dim"])))
        ids.append(e["id"])

    batch = {
        "ids": ids,
        "entries": entries,
        "mix": torch.stack(mixes).to(device),
        "target": torch.stack(targets).to(device),
        "interferer": torch.stack(inters).to(device),
        "frame_act": torch.stack(acts).to(device),
        "emb": torch.stack(embs).to(device),
    }
    return batch


def compute_losses(cfg, model_out, batch):
    """损失组装（支持 absent_loss_scale）。

    当配置中存在 absent_loss_scale 时，对 absent 样本（target 全零）的
    SI-SDR、MR-STFT 和 L1 损失乘以该系数，防止 absent 样本主导训练。
    """
    s_tgt, s_res, p_tgt = model_out
    y, itr, x = batch["target"], batch["interferer"], batch["mix"]
    kappa = float(cfg["zero_ref_kappa"])

    # 检测 absent 样本：target 全零
    is_absent = (y.abs().mean(dim=-1) < 1e-6)  # [B]
    n_absent = is_absent.sum().item()
    n_present = y.shape[0] - n_absent

    terms = {
        "si_sdr_db": si_sdr(s_tgt, y),
        "wav_l1": scale_sensitive_l1(s_tgt, y, kappa),
        "mrstft": mrstft_loss(s_tgt, y, cfg["mrstft_resolutions"]),
        "act_bce": activity_bce_loss(p_tgt, batch["frame_act"]),
        "res_l1": scale_sensitive_l1(s_res, itr, kappa),
        "mix": mix_consistency_loss(s_tgt, s_res, x),
    }

    # absent 样本损失缩放
    absent_scale = float(cfg.get("absent_loss_scale", 1.0))
    if absent_scale != 1.0 and n_absent > 0:
        # 重新计算 per-sample 损失
        si_sdr_per_sample = _si_sdr_per_sample(s_tgt, y)
        mrstft_per_sample = _mrstft_per_sample(s_tgt, y, cfg["mrstft_resolutions"])
        wav_l1_per_sample = _scale_sensitive_l1_per_sample(s_tgt, y, kappa)

        # absent 样本缩放
        si_sdr_per_sample[is_absent] *= absent_scale
        mrstft_per_sample[is_absent] *= absent_scale
        wav_l1_per_sample[is_absent] *= absent_scale

        terms["si_sdr_db"] = si_sdr_per_sample.mean()
        terms["mrstft"] = mrstft_per_sample.mean()
        terms["wav_l1"] = wav_l1_per_sample.mean()
        terms["_absent_count"] = n_absent
        terms["_present_count"] = n_present

    total = (-float(cfg["lambda_sisdr"]) * terms["si_sdr_db"]
             + float(cfg["lambda_wav"]) * terms["wav_l1"]
             + float(cfg["lambda_stft"]) * terms["mrstft"]
             + float(cfg["lambda_act"]) * terms["act_bce"]
             + float(cfg["lambda_residual"]) * terms["res_l1"]
             + float(cfg["lambda_mix"]) * terms["mix"])
    return total, terms


def _si_sdr_per_sample(est, ref, eps=1e-6):
    """计算 per-sample SI-SDR [B]"""
    ref_energy = (ref ** 2).sum(-1)
    scale = (est * ref).sum(-1) / (ref_energy + eps)
    target = scale.unsqueeze(-1) * ref
    noise = est - target
    ratio = (target ** 2).sum(-1) / (noise ** 2).sum(-1) + eps
    return 10.0 * torch.log10(ratio + eps)


def _mrstft_per_sample(est, ref, resolutions, eps=1e-6):
    """计算 per-sample MR-STFT 损失 [B]"""
    total = est.new_zeros(est.shape[0])
    for n_fft, hop, win in resolutions:
        m_est = _stft_mag(est, n_fft, hop, win)
        m_ref = _stft_mag(ref, n_fft, hop, win)
        sc = (m_est - m_ref).flatten(1).norm(dim=1) / (m_ref.flatten(1).norm(dim=1) + eps)
        mag_est = torch.log(m_est.float() + eps)
        mag_ref = torch.log(m_ref.float() + eps)
        mag = (mag_est - mag_ref).abs().mean(dim=(1, 2))
        total = total + (sc + mag)
    return total / len(resolutions)


def _scale_sensitive_l1_per_sample(est, ref, kappa, eps=1e-6):
    """计算 per-sample scale_sensitive_l1 [B]"""
    ref_energy = (ref ** 2).sum(-1)
    alpha = (est * ref).sum(-1) / (ref_energy + eps)
    l1 = (est - alpha.unsqueeze(-1) * ref).abs().mean(-1)
    zero_ref_l1 = est.abs().mean(-1)
    is_zero_ref = ref.abs().mean(-1) < kappa
    return torch.where(is_zero_ref, zero_ref_l1, l1)


def save_audio_triplet(out_dir, tag, idx, batch, s_tgt, sr):
    d = out_dir / f"sample{idx}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "mixture.wav"), batch["mix"][idx].cpu().numpy(), sr)
    sf.write(str(d / "target.wav"), batch["target"][idx].cpu().numpy(), sr)
    sf.write(str(d / "est_target.wav"), s_tgt[idx].detach().cpu().numpy(), sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P2_ROOT / "configs" / "tse_overfit_debug.yaml"))
    ap.add_argument("--manifest", default=str(P2_ROOT / "artifacts" / "debug_mixtures_v0" / "manifest.jsonl"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out) if args.out else P2_ROOT / "artifacts" / "experiments" / f"DEBUG_OVERFIT_seed{seed}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "before_after_audio").mkdir()

    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(ch)

    LOG.info("config=%s device=%s seed=%d", args.config, device, seed)
    LOG.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())

    shutil.copy(args.config, out_dir / "config.yaml")
    (out_dir / "config.sha256").write_text(sha256_file(out_dir / "config.yaml") + "\n", encoding="utf-8")

    batch = load_fixed_batch(cfg, Path(args.manifest), int(cfg["batch_size"]), device)
    base = FUNASR_ROOT
    LOG.info("固定 batch=%d 条（不 shuffle）: %s", len(batch["ids"]), batch["ids"])
    LOG.info("frame_act 形状=%s 活动帧占比=%.3f",
             list(batch["frame_act"].shape), batch["frame_act"].mean().item())

    with open(out_dir / "data.sha256", "w", encoding="utf-8") as f:
        f.write(f"{sha256_file(args.manifest)}  manifest.jsonl\n")
        for e in batch["entries"]:
            f.write(f"{sha256_file(base / e['mixture'])}  {Path(e['mixture']).name}\n")
            f.write(f"{sha256_file(base / e['target'])}  {Path(e['target']).name}\n")

    model = DualOutputTSE(cfg).to(device)
    LOG.info("参数量 %d", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    assert not cfg["amp"], "手册 P2-09：AMP 必须关闭"

    steps = int(cfg["steps"])
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    si_sdr_mix = si_sdr(batch["mix"], batch["target"]).item()
    LOG.info("基线 SI-SDR(mixture)=%.2f dB", si_sdr_mix)

    metrics_f = open(out_dir / "metrics.jsonl", "w", encoding="utf-8")
    nan_steps = 0
    step0_audio_saved = False
    t_train0 = time.time()

    model.train()
    for step in range(steps + 1):
        t0 = time.time()
        if step > 0:
            opt.zero_grad(set_to_none=True)
        out = model(batch["mix"], batch["emb"])
        total, terms = compute_losses(cfg, out, batch)

        grad_norm, clipped = 0.0, False
        if step > 0:
            if not torch.isfinite(total):
                nan_steps += 1
                LOG.warning("step=%d 非有限损失，跳过本步", step)
            else:
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
                clipped = grad_norm > grad_clip
                opt.step()

        rec = {"step": step, "total": float(total.detach().cpu()),
               **{k: float(v.detach().cpu()) for k, v in terms.items()},
               "lr": float(cfg["lr"]), "grad_norm": grad_norm, "clipped": clipped,
               "nan": not bool(torch.isfinite(total)),
               "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0,
               "step_time_ms": (time.time() - t0) * 1000,
               "si_sdri_db": float(terms["si_sdr_db"].detach().cpu()) - si_sdr_mix}
        metrics_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        metrics_f.flush()

        if step == 0 and not step0_audio_saved:
            for idx in (0, 1):
                save_audio_triplet(out_dir / "before_after_audio", "step0", idx, batch, out[0], cfg["sample_rate"])
            step0_audio_saved = True
        if step % log_every == 0 or step == steps:
            LOG.info("step=%d total=%.4f si_sdr=%.2f si_sdri=%.2f wav_l1=%.4f mrstft=%.4f act_bce=%.4f res_l1=%.4f |g|=%.3f%s",
                     step, rec["total"], rec["si_sdr_db"], rec["si_sdri_db"],
                     rec["wav_l1"], rec["mrstft"], rec["act_bce"], rec["res_l1"],
                     grad_norm, " (clip)" if clipped else "")
    metrics_f.close()

    model.eval()
    with torch.no_grad():
        final_out = model(batch["mix"], batch["emb"])
    for idx in (0, 1):
        save_audio_triplet(out_dir / "before_after_audio", "step100", idx, batch, final_out[0], cfg["sample_rate"])

    ckpt_path = out_dir / f"checkpoint_step{steps}.pt"
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "step": steps, "cfg": cfg, "seed": seed}, ckpt_path)
    (out_dir / "checkpoint.sha256").write_text(sha256_file(ckpt_path) + "\n", encoding="utf-8")

    model2 = DualOutputTSE(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt["model"])
    model2.eval()
    with torch.no_grad():
        out2 = model2(batch["mix"], batch["emb"])
    restore_diff = max((final_out[i] - out2[i]).abs().max().item() for i in range(3))
    restore_ok = restore_diff < float(cfg["restore_tol"])
    LOG.info("恢复一致性 max|Δ|=%.3e tol=%.0e → %s", restore_diff, float(cfg["restore_tol"]),
             "PASS" if restore_ok else "FAIL")

    with torch.no_grad():
        wrong_emb = batch["emb"][list(range(1, len(batch["ids"]))) + [0]]
        out_wrong = model(batch["mix"], wrong_emb)
    enroll_diff = (final_out[0] - out_wrong[0]).abs().mean().item()
    si_sdr_wrong = si_sdr(out_wrong[0], batch["target"]).item()
    LOG.info("注册对比：正确 SI-SDR=%.2f dB，错误 SI-SDR=%.2f dB，输出差异 mean|Δ|=%.3e",
             float(terms["si_sdr_db"].detach().cpu()), si_sdr_wrong, enroll_diff)

    first = json.loads(open(out_dir / "metrics.jsonl", encoding="utf-8").readline())
    last = rec
    loss_drop = (first["total"] - last["total"]) / (abs(first["total"]) + 1e-8)
    s_final = final_out[0]
    not_zero = bool((s_final.abs().mean() > 1e-6).item())
    not_copy = bool(((s_final - batch["mix"]).abs().mean() > 1e-6).item())
    verdicts = {
        "loss_drop_ge_70pct": bool(loss_drop >= 0.70),
        "si_sdr_gain_ge_10db": bool(last["si_sdr_db"] - first["si_sdr_db"] >= 10.0),
        "no_nan": nan_steps == 0,
        "grad_finite": all(json.loads(l)["grad_norm"] >= 0 and np.isfinite(json.loads(l)["grad_norm"])
                           for l in open(out_dir / "metrics.jsonl", encoding="utf-8").readlines()[1:]),
        "output_not_zero": not_zero,
        "output_not_copy_mix": not_copy,
        "enroll_condition_effective": bool(enroll_diff > 1e-6),
        "si_sdri_positive": bool(last["si_sdri_db"] > 0),
        "restore_consistent": restore_ok,
    }
    core_pass = (verdicts["no_nan"] and verdicts["grad_finite"] and verdicts["output_not_zero"]
                 and verdicts["output_not_copy_mix"] and verdicts["restore_consistent"]
                 and verdicts["si_sdri_positive"] and verdicts["enroll_condition_effective"])
    suggest_pass = verdicts["loss_drop_ge_70pct"] or verdicts["si_sdr_gain_ge_10db"]
    overall = "PASS" if (core_pass and suggest_pass) else ("CORE_PASS_BUT_TARGET_MISS" if core_pass else "FAIL")

    train_time = time.time() - t_train0
    LOG.info("训练耗时 %.1fs；建议目标: loss 降 %.1f%% / SI-SDR 升 %.2f dB；总体判定: %s",
             train_time, loss_drop * 100, last["si_sdr_db"] - first["si_sdr_db"], overall)

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# P2-09 固定小批量 {steps} step 过拟合报告（DEBUG_ONLY）\n\n")
        f.write(f"- 日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n- 设备: {device}（torch {torch.__version__}）\n")
        f.write(f"- 固定 batch: {len(batch['ids'])} 条 {batch['ids']}\n")
        f.write(f"- 基线 SI-SDR(mixture): {si_sdr_mix:.2f} dB\n\n")
        f.write("## 指标（step0 → step100）\n\n")
        f.write(f"- total: {first['total']:.4f} → {last['total']:.4f}（降 {loss_drop*100:.1f}%）\n")
        f.write(f"- SI-SDR: {first['si_sdr_db']:.2f} → {last['si_sdr_db']:.2f} dB（SI-SDRi {last['si_sdri_db']:.2f} dB）\n")
        f.write(f"- wav_l1: {first['wav_l1']:.4f} → {last['wav_l1']:.4f}\n")
        f.write(f"- mrstft: {first['mrstft']:.4f} → {last['mrstft']:.4f}\n")
        f.write(f"- act_bce: {first['act_bce']:.4f} → {last['act_bce']:.4f}\n")
        f.write(f"- res_l1: {first['res_l1']:.4f} → {last['res_l1']:.4f}\n")
        f.write(f"- 注册条件：正确 {last['si_sdr_db']:.2f} dB vs 错误 {si_sdr_wrong:.2f} dB，mean|Δ|={enroll_diff:.3e}\n")
        f.write(f"- 恢复一致性: max|Δ|={restore_diff:.3e}（tol {float(cfg['restore_tol']):.0e}）\n")
        f.write(f"- NaN step 数: {nan_steps}；训练耗时: {train_time:.1f}s\n\n")
        f.write("## 判定\n\n| 项 | 结果 |\n|---|---|\n")
        for k, v in verdicts.items():
            f.write(f"| {k} | {'PASS' if v else 'FAIL'} |\n")
        f.write(f"\n**总体判定: {overall}**\n")

    LOG.info("产物目录: %s", out_dir)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())