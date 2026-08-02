# -*- coding: utf-8 -*-
"""TSE 评测指标（P2-11）。

与 src/tse/losses.py 的区别：losses 面向训练（可微、按 batch），
本模块面向评测（逐条、口径冻结、错误输入显式失败）。

口径冻结点（手册 P2-11）：SI-SDR 去均值 + 投影 + eps=1e-8；
MR-STFT 默认三分辨率；activity 诊断阈值 tau_debug=0.5（非 P4 最终阈值）。
"""

import torch

EPS = 1e-8
TAU_DEBUG = 0.5
CLIP_THRESH = 0.99
DEFAULT_MRSTFT = [(256, 64, 256), (512, 128, 512), (1024, 256, 1024)]


def _check_pair(est, ref):
    """形状/长度/有限性检查；违规抛 ValueError（手册：错误输入失败）。"""
    if est.shape != ref.shape:
        raise ValueError(f"长度/形状不一致: est{tuple(est.shape)} vs ref{tuple(ref.shape)}")
    if not torch.isfinite(est).all():
        raise ValueError("est 含 NaN/Inf")
    if not torch.isfinite(ref).all():
        raise ValueError("ref 含 NaN/Inf")


def _apply_valid(est, ref, valid):
    if valid is None:
        return est, ref
    if valid.shape != est.shape:
        raise ValueError("valid mask 形状与音频不一致")
    return est[valid], ref[valid]


def si_sdr_components(est, ref, valid=None, eps=EPS):
    """返回 (信号功率, 误差功率, SI-SDR dB)。去均值+投影，eps 固定。"""
    _check_pair(est, ref)
    est, ref = _apply_valid(est.flatten(), ref.flatten(), valid.flatten() if valid is not None else None)
    est = est - est.mean()
    ref = ref - ref.mean()
    proj = torch.dot(est, ref) / (torch.dot(ref, ref) + eps) * ref
    err = est - proj
    sig_pow = proj.pow(2).sum()
    err_pow = err.pow(2).sum()
    db = 10.0 * torch.log10(sig_pow / (err_pow + eps) + eps)
    return sig_pow.item(), err_pow.item(), db.item()


def si_sdr_eval(est, ref, valid=None):
    return si_sdr_components(est, ref, valid)[2]


def si_sdri(est, ref, mix, valid=None):
    """SI-SDR 改善量 = SI-SDR(est) - SI-SDR(mix)，同一 valid 口径。"""
    return si_sdr_eval(est, ref, valid) - si_sdr_eval(mix, ref, valid)


def waveform_l1(est, ref, valid=None):
    _check_pair(est, ref)
    est, ref = _apply_valid(est, ref, valid)
    return (est - ref).abs().mean().item()


def mrstft_eval(est, ref, resolutions=None):
    """评测口径 MR-STFT（复用 losses.mrstft_loss，分辨率显式传入或用默认）。

    losses 面向训练要求 [B, T]；评测逐条为 [T]，此处补 batch 维。
    """
    from src.tse.losses import mrstft_loss
    _check_pair(est, ref)
    if est.dim() == 1:
        est, ref = est.unsqueeze(0), ref.unsqueeze(0)
    return mrstft_loss(est, ref, resolutions or DEFAULT_MRSTFT).item()


def activity_prf(p_tgt, frame_act, tau=TAU_DEBUG):
    """帧级活动度 precision/recall/F1。p_tgt∈[0,1]，frame_act∈{0,1}。"""
    if p_tgt.shape != frame_act.shape:
        raise ValueError("p_tgt 与 frame_act 形状不一致")
    if not torch.isfinite(p_tgt).all():
        raise ValueError("p_tgt 含 NaN/Inf")
    pred = (p_tgt > tau)
    true = frame_act > 0.5
    tp = (pred & true).sum().item()
    fp = (pred & ~true).sum().item()
    fn = (~pred & true).sum().item()
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return {"precision": precision, "recall": recall, "f1": f1}


def energy_ratio(est, mix, eps=EPS):
    """||est||²/(||mix||²+eps)。PRESENT 诊断 + ABSENT 的 rho_abs（手册公式）。"""
    if not torch.isfinite(est).all():
        raise ValueError("est 含 NaN/Inf")
    return (est.pow(2).sum() / (mix.pow(2).sum() + eps)).item()


def activity_ratio(p_tgt, tau=TAU_DEBUG):
    """A_abs：p_tgt>tau 的帧占比（ABSENT 虚假活动诊断）。"""
    if not torch.isfinite(p_tgt).all():
        raise ValueError("p_tgt 含 NaN/Inf")
    return (p_tgt > tau).float().mean().item()


def clipped_ratio(wav, thresh=CLIP_THRESH):
    """|x|>=thresh 的采样占比（削波诊断）。"""
    if not torch.isfinite(wav).all():
        raise ValueError("wav 含 NaN/Inf")
    return (wav.abs() >= thresh).float().mean().item()


def corpus_sisdr(components):
    """语料级聚合：10*log10(Σsig/Σerr)。与逐句平均不同（手册要求区分）。"""
    sig = sum(c[0] for c in components)
    err = sum(c[1] for c in components)
    import math
    return 10.0 * math.log10(sig / (err + EPS) + EPS)


def utterance_sisdr(components):
    """逐句平均聚合：mean(逐条 dB)。"""
    return sum(c[2] for c in components) / len(components)


def swap_quality(out_e, y1, y2):
    """注册交换质量对：某注册条件下的输出分别对 y1/y2 的 SI-SDR。"""
    return {"si_sdr_vs_y1": si_sdr_eval(out_e, y1), "si_sdr_vs_y2": si_sdr_eval(out_e, y2)}
