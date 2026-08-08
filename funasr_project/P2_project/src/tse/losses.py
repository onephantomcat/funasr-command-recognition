"""TSE 损失函数集（纯函数层）。

红线：
- 本文件不知道模型存在，只依赖 torch；
- si_sdr 返回 SI-SDR 正值（度量语义），取负在组装总损失时由上层完成；
- absent_zero_loss 为 ABSENT 预留（05B），不接入主损失。
"""

import torch
import warnings


def _align_frames(src, target_len, mode):
    """将帧级张量 src [B, Fr] 对齐到 target_len（保持 0/1 语义或概率平滑）。

    mode: 'nearest' 适用于 0/1 标签（不制造中间值），'linear' 适用于连续概率。
    """
    cur_len = src.shape[-1]
    if cur_len == target_len:
        return src
    # F.interpolate 需要 [B, 1, Fr]，输出后再 squeeze
    x = src.unsqueeze(1)
    if mode == "nearest":
        aligned = torch.nn.functional.interpolate(x, size=target_len, mode="nearest-exact")
    else:
        aligned = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return aligned.squeeze(1)


def si_sdr(est, ref, eps=1e-6):
    """SI-SDR（dB，越大越好）。含除零保护：零能量参考时退化为有限值。

    注意：eps=1e-6 适配 AMP(float16)，避免 float16 下下溢。
    """
    ref_energy = (ref ** 2).sum(-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / (ref_energy + eps)
    target = scale * ref
    noise = est - target
    ratio = (target ** 2).sum(-1) / (noise ** 2).sum(-1) + eps
    return 10.0 * torch.log10(ratio + eps).mean()


def scale_sensitive_l1(est, ref, kappa):
    """尺度敏感 L1：|est - alpha*ref| 均值，alpha 为逐样本最小二乘增益。

    全零参考保护：ref 平均幅度 < kappa 时退化为 |est| 均值
    （05B 的 L_zero 前身，此时损失等价于抑制目标支路输出能量）。
    """
    ref_energy = (ref ** 2).sum(-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref_energy + 1e-6)
    l1 = (est - alpha * ref).abs().mean(-1)
    zero_ref_l1 = est.abs().mean(-1)
    is_zero_ref = ref.abs().mean(-1) < kappa
    return torch.where(is_zero_ref, zero_ref_l1, l1).mean()


def mix_consistency_loss(s_tgt, s_res, x, eps=1e-8):
    """混合一致性（软约束）：||x - (s_tgt + s_res)||_1 / (||x||_1 + eps)。

    仅度量；投影修正（硬约束）在 model 内完成，职责分离。
    """
    resid = x - (s_tgt + s_res)
    return resid.abs().mean(-1).sum() / (x.abs().mean(-1).sum() + eps)


def absent_zero_loss(s_tgt):
    """ABSENT 零目标损失（05B §6 预留，今日只定义+单测，不进总损失）。"""
    return (s_tgt ** 2).mean(-1).mean()


def _stft_mag(wav, n_fft, hop, win):
    """STFT 幅度谱 [B, F, Fr]（hann 窗，与模型 center 对齐约定一致）。"""
    window = torch.hann_window(win, device=wav.device, dtype=wav.dtype)
    return torch.stft(
        wav, n_fft=n_fft, hop_length=hop, win_length=win,
        window=window, return_complex=True,
    ).abs()


def mrstft_loss(est, ref, resolutions, eps=1e-6):
    """多分辨率 STFT 损失：各分辨率下 谱收敛项 + log 幅度 L1 的均值。

    resolutions: [(n_fft, hop, win), ...]，从配置读，不硬编码。
    谱收敛项 = ‖|S_est|−|S_ref|‖_F / ‖|S_ref|‖_F（逐样本后取均值）。

    注意：log 幅度项使用 float32 精度 + eps=1e-6，
    避免 AMP(float16) 下 eps 下溢导致 log(0)=-inf 爆炸。
    """
    total = est.new_zeros(())
    for n_fft, hop, win in resolutions:
        m_est = _stft_mag(est, n_fft, hop, win)
        m_ref = _stft_mag(ref, n_fft, hop, win)
        sc = (m_est - m_ref).flatten(1).norm(dim=1) / (m_ref.flatten(1).norm(dim=1) + eps)
        mag_est = torch.log(m_est.float() + eps)
        mag_ref = torch.log(m_ref.float() + eps)
        mag = (mag_est - mag_ref).abs().mean(dim=(1, 2))
        total = total + (sc + mag).mean()
    return total / len(resolutions)


def activity_bce_loss(p_tgt, frame_act, eps=1e-7, _warned=[False]):
    """帧级活动度 BCE：p_tgt [B,Fr]（sigmoid 后概率），frame_act [B,Fr]∈{0,1}。

    健壮性：不同 PyTorch 版本对 torch.stft 默认 center/pad 行为可能变化，
    导致 model 输出帧数 (p_tgt 端) ≠ Dataset 端 frame_activity 估算的帧数。
    若尺寸不一致：
      - 首次触发 ERROR 级告警（附 mismatch 尺寸 + 建议检查项）
      - 用最近邻将 frame_act (0/1 标签) 对齐到 p_tgt 帧数，保证损失可算
         (保 0/1 语义，不引入中间 float 值污染 BCE)
    """
    p_len = p_tgt.shape[-1]
    a_len = frame_act.shape[-1]
    if p_len != a_len:
        if not _warned[0]:
            msg = (
                f"[FRAME ALIGN] p_tgt(model)={p_len}帧 ≠ frame_act(label)={a_len}帧 → "
                f"自动用 nearest-exact 对齐 label 到 model 端帧数。"
                f"可能根因：① 云端 PyTorch 版本 torch.stft 弃用后默认值变 (请确认 model.forward 显式传 center=True); "
                f"② Dataset.seg_samples 算错或 P1 wav 实际 sr 非 16kHz (应=cfg.sample_rate={16000}). "
                f"对齐比例≈{p_len / max(1, a_len):.3f}"
            )
            warnings.warn(msg)
            try:
                import logging as _l
                _l.getLogger("funasr.tse").error(msg)
            except Exception:
                print("!! " + msg, flush=True)
            _warned[0] = True
        frame_act = _align_frames(frame_act, p_len, mode="nearest")
    p = p_tgt.clamp(eps, 1.0 - eps)
    return -(frame_act * torch.log(p) + (1.0 - frame_act) * torch.log(1.0 - p)).mean()