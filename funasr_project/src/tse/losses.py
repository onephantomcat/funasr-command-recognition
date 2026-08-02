"""TSE 损失函数集（纯函数层）。

红线：
- 本文件不知道模型存在，只依赖 torch；
- si_sdr 返回 SI-SDR 正值（度量语义），取负在组装总损失时由上层完成；
- absent_zero_loss 为 ABSENT 预留（05B），不接入主损失。
"""

import torch


def si_sdr(est, ref, eps=1e-8):
    """SI-SDR（dB，越大越好）。含除零保护：零能量参考时退化为有限值。"""
    ref_energy = (ref ** 2).sum(-1, keepdim=True)
    scale = (est * ref).sum(-1, keepdim=True) / (ref_energy + eps)
    target = scale * ref
    noise = est - target
    ratio = (target ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps)
    return 10.0 * torch.log10(ratio + eps).mean()


def scale_sensitive_l1(est, ref, kappa):
    """尺度敏感 L1：|est - alpha*ref| 均值，alpha 为逐样本最小二乘增益。

    全零参考保护：ref 平均幅度 < kappa 时退化为 |est| 均值
    （05B 的 L_zero 前身，此时损失等价于抑制目标支路输出能量）。
    """
    ref_energy = (ref ** 2).sum(-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref_energy + 1e-8)
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


def mrstft_loss(est, ref, resolutions, eps=1e-8):
    """多分辨率 STFT 损失：各分辨率下 谱收敛项 + log 幅度 L1 的均值。

    resolutions: [(n_fft, hop, win), ...]，从配置读，不硬编码。
    谱收敛项 = ‖|S_est|−|S_ref|‖_F / ‖|S_ref|‖_F（逐样本后取均值）。
    """
    total = est.new_zeros(())
    for n_fft, hop, win in resolutions:
        m_est = _stft_mag(est, n_fft, hop, win)
        m_ref = _stft_mag(ref, n_fft, hop, win)
        sc = (m_est - m_ref).flatten(1).norm(dim=1) / (m_ref.flatten(1).norm(dim=1) + eps)
        mag = (torch.log(m_est + eps) - torch.log(m_ref + eps)).abs().mean(dim=(1, 2))
        total = total + (sc + mag).mean()
    return total / len(resolutions)


def activity_bce_loss(p_tgt, frame_act, eps=1e-7):
    """帧级活动度 BCE：p_tgt [B,Fr]（sigmoid 后概率），frame_act [B,Fr]∈{0,1}。"""
    p = p_tgt.clamp(eps, 1.0 - eps)
    return -(frame_act * torch.log(p) + (1.0 - frame_act) * torch.log(1.0 - p)).mean()
