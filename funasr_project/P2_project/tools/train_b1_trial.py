# -*- coding: utf-8 -*-
"""P2-12 B1 500-step 试车脚本（通用训练器，支持 AMP/梯度累积/warmup+cosine/checkpoint 恢复）。

相对 train_overfit_debug.py 的核心差异：
- Dataset + DataLoader（支持 shuffle / num_workers / pin_memory）
- 按 segment_length 随机截取等长片段（mixture/target/interferer/activity 同起点）
- AMP 混合精度（torch.cuda.amp.autocast + GradScaler，CPU 自动禁用）
- 学习率 warmup(线性) → cosine 衰减
- 梯度累积（effective_batch = batch_size × gradient_accumulation）
- 每 save_every 步存 checkpoint（含 optimizer/scaler/step），支持 --resume 续训
- 记录吞吐量（samples/sec）、数据等待占比、step time P50/P95
- 试车判定 trial_verdict.json + 人读报告 report.md

复用 train_overfit_debug 的纯函数：bootstrap_embedding / frame_activity /
sha256_file / sha256_text / compute_losses（不复制，直接 import）。

标记：B1_TRIAL / BOOTSTRAP_ENCODER_ONLY（P4 契约未交付前 embedding 用哈希随机）

运行示例：
  # P1 v2_b1 未交付时用 DEBUG 数据空跑 10 步验证脚本
  python tools/train_b1_trial.py --debug_data --max_steps 10

  # 正式 500 步试车（需 P1 v2_b1 交付）
  python tools/train_b1_trial.py --device auto

  # 从 checkpoint 续训
  python tools/train_b1_trial.py --resume artifacts/experiments/B1_TRIAL_seed20260723/checkpoint_step100.pt
"""

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.utils.data as data
import yaml

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
P1_DATA_ROOT = Path("/root/autodl-tmp/P1_to_P2_v2_b1")
sys.path.insert(0, str(P2_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tse.model import DualOutputTSE
from src.tse.losses import si_sdr
from src.tse.enrollment_adapter import EnrollmentAdapter
from train_overfit_debug import (
    compute_losses,
    frame_activity,
    sha256_file,
)

LOG = logging.getLogger("p2_b1_trial")

DEBUG_MANIFEST = P2_ROOT / "artifacts" / "debug_mixtures_v0" / "manifest.jsonl"


class B1Dataset(data.Dataset):
    """B1 训练数据集。

    按 segment_length 随机截取等长片段；mixture/target/interferer/activity 必须同起点
    （否则破坏对齐）。不足段长则右侧补零。embedding 由 bootstrap_embedding 派生
    （P4 契约交付前用哈希随机，保留 swap 语义）。
    """

    @staticmethod
    def _resolve_path(p):
        """Resolve audio path: try P1 root first, fallback to FUNASR_ROOT."""
        p = str(p)
        if not p or p == "None":
            return None
        for root in [P1_DATA_ROOT, FUNASR_ROOT]:
            candidate = root / p
            if candidate.exists():
                return str(candidate)
        return str(FUNASR_ROOT / p)

    def __init__(self, manifest_path, cfg, seed=None, adapter=None):
        self.entries = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                self.entries.append(json.loads(line))
        self.cfg = cfg
        self.seg_samples = int(cfg["segment_length"] * cfg["sample_rate"])
        self.win_length = int(cfg["win_length"])
        self.hop_length = int(cfg["hop_length"])
        self.act_frame_ratio = float(cfg["act_frame_ratio"])
        self.emb_dim = int(cfg["emb_dim"])
        self.rng = np.random.default_rng(seed)
        self.adapter = adapter or EnrollmentAdapter.from_config(cfg)
        self._camplus_fallback = False
        if self.adapter.mode == "campplus":
            try:
                self.adapter.load_backend()
                LOG.info("CAMPLUS 后端加载成功")
            except Exception as e:
                LOG.warning(f"CAMPLUS 后端加载失败，fallback BOOTSTRAP: {e}")
                self._camplus_fallback = True
                self.adapter = EnrollmentAdapter.from_config(cfg, mode="bootstrap")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]

        # P1 field mapping: mixture_wav→mixture, target_wav→target, etc.
        mix_path = e.get("mixture", e.get("mixture_wav", ""))
        tgt_path = e.get("target", e.get("target_wav", ""))
        itr_path = e.get("interferer", e.get("interferer_wav", ""))
        act_path = e.get("activity", e.get("activity_mask", ""))
        enroll_path = e.get("enrollment", e.get("enroll_wav", ""))
        sample_id = e.get("id", e.get("sample_id", str(idx)))

        mix, sr = sf.read(self._resolve_path(mix_path), dtype="float32")
        tgt, _ = sf.read(self._resolve_path(tgt_path), dtype="float32")

        # Handle null interferer (single speaker case)
        itr_resolved = self._resolve_path(itr_path)
        if itr_resolved and Path(itr_resolved).exists():
            itr, _ = sf.read(itr_resolved, dtype="float32")
        else:
            itr = np.zeros_like(mix, dtype="float32")

        act_resolved = self._resolve_path(act_path)
        act = np.load(act_resolved)
        assert sr == self.cfg["sample_rate"], f"采样率不一致: {sr} vs {self.cfg['sample_rate']}"

        T = len(mix)
        if T >= self.seg_samples:
            start = int(self.rng.integers(0, T - self.seg_samples + 1))
            mix = mix[start:start + self.seg_samples]
            tgt = tgt[start:start + self.seg_samples]
            itr = itr[start:start + self.seg_samples]
            act = act[start:start + self.seg_samples]
        else:
            pad = self.seg_samples - T
            mix = np.pad(mix, (0, pad))
            tgt = np.pad(tgt, (0, pad))
            itr = np.pad(itr, (0, pad))
            act = np.pad(act, (0, pad))

        return {
            "id": sample_id,
            "mix": torch.from_numpy(mix),
            "target": torch.from_numpy(tgt),
            "interferer": torch.from_numpy(itr),
            "frame_act": frame_activity(act, self.win_length, self.hop_length, self.act_frame_ratio),
            "emb": self._get_embedding(e, enroll_path),
        }

    def _get_embedding(self, e, enroll_path=None):
        spk_id = e.get("target_speaker", e.get("enrollment", e.get("sample_id", "unknown")))
        if self.adapter.mode == "campplus" and not self._camplus_fallback:
            enroll = enroll_path or e.get("enrollment", e.get("enroll_wav", ""))
            enroll_resolved = self._resolve_path(enroll) if enroll else None
            if enroll_resolved and Path(enroll_resolved).exists():
                try:
                    self.adapter.encode_file(spk_id, enroll_resolved)
                    return self.adapter.get_embedding(spk_id).squeeze(0)
                except Exception as ex:
                    LOG.warning(f"CAMPLUS encode 失败 ({spk_id}), fallback BOOTSTRAP: {ex}")
        return self.adapter.get_embedding(spk_id).squeeze(0)


def collate_fn(batch):
    """所有样本等长（由 Dataset 保证），直接 stack。"""
    return {
        "ids": [b["id"] for b in batch],
        "mix": torch.stack([b["mix"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "interferer": torch.stack([b["interferer"] for b in batch]),
        "frame_act": torch.stack([b["frame_act"] for b in batch]),
        "emb": torch.stack([b["emb"] for b in batch]),
    }


def get_lr(step, peak_lr, warmup_steps, total_steps, schedule="cosine"):
    """warmup(线性) → cosine/linear 衰减到 0；constant 不衰减。"""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if schedule == "cosine":
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return peak_lr * (1.0 - progress)
    return peak_lr


@torch.no_grad()
def evaluate_dev(model, loader, cfg, device, use_amp):
    """dev 集平均损失 + SI-SDR（不反向，不记录吞吐量）。"""
    model.eval()
    total_loss, total_sisdr, n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch["mix"], batch["emb"])
            loss, terms = compute_losses(cfg, out, batch)
        total_loss += float(loss.detach().cpu())
        total_sisdr += float(terms["si_sdr_db"].detach().cpu())
        n += 1
    model.train()
    return {"dev_loss": total_loss / max(1, n), "dev_sisdr": total_sisdr / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P2_ROOT / "configs" / "tse_b1_trial.yaml"))
    ap.add_argument("--manifest", default=None, help="覆盖 cfg.datasets.train_manifest")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="覆盖 cfg.steps（空跑测试用）")
    ap.add_argument("--debug_data", action="store_true", help="用 P2-07 DEBUG 集空跑（非正式 B1）")
    ap.add_argument("--resume", default=None, help="从 checkpoint 恢复续训")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

    tag = cfg.get("tag", "B1_TRIAL")
    out_dir = Path(args.out) if args.out else P2_ROOT / "artifacts" / "experiments" / tag
    if args.resume:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(ch)

    LOG.info("config=%s device=%s seed=%d amp=%s sv_mode=%s", args.config, device, seed, use_amp, cfg.get("sv_mode", "bootstrap"))
    LOG.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())

    adapter = EnrollmentAdapter.from_config(cfg)
    LOG.info("EnrollmentAdapter mode=%s emb_dim=%d", adapter.mode, adapter.emb_dim)

    shutil.copy(args.config, out_dir / "config.yaml")
    (out_dir / "config.sha256").write_text(sha256_file(out_dir / "config.yaml") + "\n", encoding="utf-8")

    if args.debug_data:
        train_manifest = str(DEBUG_MANIFEST)
        dev_manifest = str(DEBUG_MANIFEST)
        LOG.warning("--debug_data 模式：用 P2-07 DEBUG 集空跑，非正式 B1 训练")
    else:
        train_manifest = args.manifest or cfg["datasets"]["train_manifest"]
        dev_manifest = cfg["datasets"]["dev_manifest"]
        if not Path(train_manifest).is_absolute():
            train_manifest = str(P2_ROOT / train_manifest)
        if not Path(dev_manifest).is_absolute():
            dev_manifest = str(P2_ROOT / dev_manifest)

    batch_size = int(cfg["batch_size"])
    n_train_samples = sum(1 for _ in open(train_manifest, encoding="utf-8"))
    if batch_size > n_train_samples:
        LOG.warning("batch_size=%d > 样本数=%d，降为 %d", batch_size, n_train_samples, n_train_samples)
        batch_size = n_train_samples

    train_ds = B1Dataset(train_manifest, cfg, seed=seed, adapter=adapter)
    dev_ds = B1Dataset(dev_manifest, cfg, seed=seed + 1, adapter=adapter)
    drop_last = len(train_ds) >= batch_size
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=bool(cfg["shuffle"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        prefetch_factor=int(cfg["prefetch_factor"]) if int(cfg["num_workers"]) > 0 else None,
        collate_fn=collate_fn, drop_last=drop_last)
    dev_loader = data.DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    LOG.info("train=%d 条 dev=%d 条 batch=%d seg=%ds drop_last=%s",
             len(train_ds), len(dev_ds), batch_size, int(cfg["segment_length"]), drop_last)

    with open(out_dir / "data.sha256", "w", encoding="utf-8") as f:
        f.write(f"{sha256_file(train_manifest)}  train_manifest.jsonl\n")
        f.write(f"{sha256_file(dev_manifest)}  dev_manifest.jsonl\n")

    model = DualOutputTSE(cfg).to(device)
    LOG.info("参数量 %d", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = int(args.max_steps if args.max_steps else cfg["steps"])
    warmup = int(cfg["lr_warmup_steps"])
    grad_accum = int(cfg["gradient_accumulation"])
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    save_every = int(cfg["save_every"])
    if save_every > total_steps:
        save_every = max(1, total_steps)
        LOG.warning("save_every > total_steps，降为 %d（空跑模式）", save_every)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if use_amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"]) + 1
        LOG.info("从 step %d 恢复（ckpt=%s）", start_step, args.resume)

    metrics_f = open(out_dir / "metrics.jsonl", "a" if args.resume else "w", encoding="utf-8")
    dev_metrics_f = open(out_dir / "dev_metrics.jsonl", "a" if args.resume else "w", encoding="utf-8")
    nan_steps = 0
    si_sdr_mix_baseline = None
    t_train0 = time.time()

    model.train()
    loader_iter = iter(train_loader)

    for step in range(start_step, total_steps + 1):
        t_step0 = time.time()
        t_data0 = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        data_time = time.time() - t_data0

        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}

        if si_sdr_mix_baseline is None:
            si_sdr_mix_baseline = float(si_sdr(batch["mix"], batch["target"]).item())
            LOG.info("基线 SI-SDR(mixture)=%.2f dB", si_sdr_mix_baseline)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch["mix"], batch["emb"])
            total, terms = compute_losses(cfg, out, batch)

        grad_norm, clipped = 0.0, False
        if step > 0:
            if not torch.isfinite(total):
                nan_steps += 1
                LOG.warning("step=%d 非有限损失，跳过本步", step)
            else:
                scaler.scale(total / grad_accum).backward()
                if step % grad_accum == 0:
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip).item()
                    clipped = grad_norm > grad_clip
                    cur_lr = get_lr(step, float(cfg["lr"]), warmup,
                                    total_steps, cfg["lr_schedule"])
                    for pg in opt.param_groups:
                        pg["lr"] = cur_lr
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

        step_time = time.time() - t_step0
        cur_lr = get_lr(step, float(cfg["lr"]), warmup, total_steps, cfg["lr_schedule"])

        rec = {
            "step": step,
            "total": float(total.detach().cpu()),
            "si_sdr_db": float(terms["si_sdr_db"].detach().cpu()),
            "wav_l1": float(terms["wav_l1"].detach().cpu()),
            "mrstft": float(terms["mrstft"].detach().cpu()),
            "act_bce": float(terms["act_bce"].detach().cpu()),
            "res_l1": float(terms["res_l1"].detach().cpu()),
            "mix": float(terms["mix"].detach().cpu()),
            "lr": cur_lr,
            "grad_norm": grad_norm,
            "clipped": clipped,
            "nan": not bool(torch.isfinite(total)),
            "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3)
                          if device.type == "cuda" else 0.0,
            "step_time_ms": step_time * 1000,
            "data_time_ms": data_time * 1000,
            "data_wait_pct": (data_time / step_time * 100) if step_time > 0 else 0,
            "samples_per_sec": (batch_size * grad_accum / step_time) if step_time > 0 else 0,
            "effective_batch": batch_size * grad_accum,
            "si_sdri_db": float(terms["si_sdr_db"].detach().cpu()) - (si_sdr_mix_baseline or 0),
        }
        metrics_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        metrics_f.flush()

        if step % log_every == 0 or step == total_steps:
            LOG.info("step=%d/%d total=%.4f si_sdr=%.2f si_sdri=%.2f lr=%.2e |g|=%.3f "
                     "mem=%.2fGB %.0fms/step data=%.0f%%",
                     step, total_steps, rec["total"], rec["si_sdr_db"], rec["si_sdri_db"],
                     cur_lr, grad_norm, rec["gpu_mem_gb"], rec["step_time_ms"],
                     rec["data_wait_pct"])

        if step > 0 and step % save_every == 0:
            ckpt_path = out_dir / f"checkpoint_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "step": step,
                "cfg": cfg,
                "seed": seed,
            }, ckpt_path)
            (out_dir / f"checkpoint_step{step}.sha256").write_text(
                sha256_file(ckpt_path) + "\n", encoding="utf-8")
            LOG.info("存 checkpoint: %s", ckpt_path.name)

            dev_rec = {"step": step, **evaluate_dev(model, dev_loader, cfg, device, use_amp)}
            dev_metrics_f.write(json.dumps(dev_rec, ensure_ascii=False) + "\n")
            dev_metrics_f.flush()
            LOG.info("  dev_loss=%.4f dev_sisdr=%.2f dB",
                     dev_rec["dev_loss"], dev_rec["dev_sisdr"])

    metrics_f.close()
    dev_metrics_f.close()
    train_time = time.time() - t_train0

    last_ckpt = out_dir / f"checkpoint_step{total_steps}.pt"
    restore_ok = False
    restore_diff = float("inf")
    if last_ckpt.exists():
        model2 = DualOutputTSE(cfg).to(device)
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model2.load_state_dict(ckpt["model"])
        model2.eval()
        with torch.no_grad():
            out1 = model(batch["mix"], batch["emb"])
            out2 = model2(batch["mix"], batch["emb"])
        restore_diff = max((out1[i] - out2[i]).abs().max().item() for i in range(3))
        restore_ok = restore_diff < float(cfg.get("restore_tol", 1e-5))
        LOG.info("恢复一致性 max|Δ|=%.3e → %s", restore_diff,
                 "PASS" if restore_ok else "FAIL")

    all_metrics = [json.loads(l) for l in open(out_dir / "metrics.jsonl", encoding="utf-8")]
    train_metrics = [m for m in all_metrics if m["step"] > 0]

    if total_steps >= 200:
        first_window = [m for m in train_metrics if m["step"] <= 100]
        last_window = [m for m in train_metrics if m["step"] > total_steps - 100]
    else:
        mid = max(1, total_steps // 2)
        first_window = [m for m in train_metrics if m["step"] <= mid]
        last_window = [m for m in train_metrics if m["step"] > mid]
    loss_first = float(np.mean([m["total"] for m in first_window])) if first_window else 0.0
    loss_last = float(np.mean([m["total"] for m in last_window])) if last_window else 0.0
    loss_decreasing = loss_last < loss_first

    peak_mem = max((m["gpu_mem_gb"] for m in all_metrics), default=0.0)
    mean_sps = float(np.mean([m["samples_per_sec"] for m in train_metrics])) if train_metrics else 0.0
    p50_step = float(np.percentile([m["step_time_ms"] for m in train_metrics], 50)) if train_metrics else 0.0
    p95_step = float(np.percentile([m["step_time_ms"] for m in train_metrics], 95)) if train_metrics else 0.0
    mean_data_wait = float(np.mean([m["data_wait_pct"] for m in train_metrics])) if train_metrics else 0.0

    epoch_steps = max(1, len(train_ds) // batch_size)
    est_epoch_time = epoch_steps * p50_step / 1000.0

    verdicts = {
        "no_nan": nan_steps == 0,
        "grad_finite": all(m["grad_norm"] >= 0 and np.isfinite(m["grad_norm"])
                           for m in train_metrics) if train_metrics else False,
        "peak_mem_under_budget": peak_mem < 4.0,
        "throughput_measured": mean_sps > 0,
        "loss_decreasing": bool(loss_decreasing),
        "checkpoint_restore_ok": restore_ok,
        "full_train_time_estimated": train_time > 0,
    }
    must_pass = ["no_nan", "grad_finite", "peak_mem_under_budget",
                 "throughput_measured", "checkpoint_restore_ok", "full_train_time_estimated"]
    overall = "PASS" if all(verdicts[k] for k in must_pass) else "FAIL"

    summary = {
        "config": args.config,
        "device": str(device),
        "torch_version": torch.__version__,
        "amp": use_amp,
        "batch_size": batch_size,
        "segment_length_sec": int(cfg["segment_length"]),
        "total_steps": total_steps,
        "train_time_sec": float(train_time),
        "n_train_samples": len(train_ds),
        "n_nan_steps": nan_steps,
        "peak_gpu_mem_gb": float(peak_mem),
        "mem_budget_gb": 4.0,
        "mem_margin_gb": float(4.0 - peak_mem),
        "samples_per_sec_mean": mean_sps,
        "step_time_ms_p50": p50_step,
        "step_time_ms_p95": p95_step,
        "data_wait_pct_mean": mean_data_wait,
        "loss_first_100_mean": loss_first,
        "loss_last_100_mean": loss_last,
        "loss_decreasing": bool(loss_decreasing),
        "restore_diff": float(restore_diff),
        "restore_ok": restore_ok,
        "est_epoch_time_sec": float(est_epoch_time),
        "est_epoch_steps": int(epoch_steps),
        "verdicts": verdicts,
        "verdict_overall": overall,
    }
    (out_dir / "trial_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# P2-12 B1 {total_steps}-step 试车报告"
                f"（{cfg.get('asset_class', 'B1_TRIAL')}）\n\n")
        f.write(f"- 日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 设备: {device}（torch {torch.__version__}）\n")
        f.write(f"- 配置: `{Path(args.config).name}`\n")
        f.write(f"- 数据: train={len(train_ds)} 条（manifest={Path(train_manifest).name}）\n")
        f.write(f"- AMP: {use_amp}；batch={batch_size}；seg={cfg['segment_length']}s；"
                f"accum={grad_accum}；effective_batch={batch_size * grad_accum}\n\n")
        f.write("## 吞吐量\n\n")
        f.write(f"- samples/sec: {mean_sps:.2f}\n")
        f.write(f"- step time P50: {p50_step:.1f} ms / P95: {p95_step:.1f} ms\n")
        f.write(f"- 数据等待占比: {mean_data_wait:.1f}%\n")
        f.write(f"- 训练总耗时: {train_time:.1f}s（{total_steps} 步）\n\n")
        f.write("## 显存\n\n")
        f.write(f"- 峰值显存: {peak_mem:.3f} GB\n")
        f.write(f"- 预算: 4.0 GB；余量: {4.0 - peak_mem:.3f} GB\n\n")
        f.write("## 收敛\n\n")
        f.write(f"- total loss 首 100 步均值: {loss_first:.4f}\n")
        f.write(f"- total loss 末 100 步均值: {loss_last:.4f}\n")
        f.write(f"- loss 下降: {'是' if loss_decreasing else '否'}\n")
        f.write(f"- NaN step 数: {nan_steps}\n\n")
        f.write("## 恢复\n\n")
        f.write(f"- checkpoint: {last_ckpt.name if last_ckpt.exists() else 'N/A'}\n")
        f.write(f"- 恢复一致性 max|Δ|: {restore_diff:.3e}\n")
        f.write(f"- 判定: {'PASS' if restore_ok else 'FAIL'}\n\n")
        f.write("## 完整训练预估\n\n")
        f.write(f"- 1 epoch 步数: {epoch_steps}\n")
        f.write(f"- 1 epoch 预估时长: {est_epoch_time:.1f}s"
                f"（{est_epoch_time / 60:.1f} 分钟）\n")
        f.write(f"- 100 epoch 预估: {est_epoch_time * 100 / 3600:.2f} 小时\n\n")
        f.write("## 判定\n\n| 项 | 结果 |\n|---|---|\n")
        for k, v in verdicts.items():
            f.write(f"| {k} | {'PASS' if v else 'FAIL'} |\n")
        f.write(f"\n**总体判定: {overall}**\n")

    LOG.info("产物目录: %s", out_dir)
    LOG.info("总体判定: %s", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())