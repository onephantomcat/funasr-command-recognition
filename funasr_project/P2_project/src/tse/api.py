"""TSE 推理侧最小封装（P2-06 接口骨架）。

职责边界：
- 今日只锁函数签名与形状契约，不实现真实推理逻辑（加载、VAD、
  滑动窗、归一化、兜底链均不在今日范围）；
- 内部仅一次 model.forward 透传，对外只暴露 s_tgt（s_res 为内部训练信号）；
- 不写文件、不打印日志（归 smoke/正式 pipeline 管）。

标记：DEBUG_ONLY / BOOTSTRAP_ENCODER_ONLY
（enroll_embedding 目前由随机张量代替 CAM++，本接口契约不变，
 正式版替换 embedding 来源与模型权重即可，调用方无需改动）
"""

import torch


def _validate_inputs(command_wav, enroll_embedding, cfg):
    """P2-06 输入验证（最小集）：错误输入必须显式失败，不得静默通过。"""
    if command_wav.ndim != 2:
        raise ValueError(f"command_wav 需为 [B,T]，得到 ndim={command_wav.ndim}")
    if enroll_embedding.ndim != 2:
        raise ValueError(f"enroll_embedding 需为 [B,D]，得到 ndim={enroll_embedding.ndim}")
    if command_wav.numel() == 0:
        raise ValueError("command_wav 为空数组")
    if not torch.isfinite(command_wav).all():
        raise ValueError("command_wav 含 NaN/Inf")
    if not torch.isfinite(enroll_embedding).all():
        raise ValueError("enroll_embedding 含 NaN/Inf")
    emb_dim = cfg.get("emb_dim")
    if emb_dim is not None and enroll_embedding.shape[-1] != emb_dim:
        raise ValueError(
            f"embedding 维度错误: 期望 {emb_dim}，得到 {enroll_embedding.shape[-1]}"
        )


@torch.no_grad()
def extract_target(command_wav, enroll_embedding, model, cfg):
    """从混合命令音频中提取目标说话人波形。

    参数：
        command_wav:      [B, T] float 波形（16 kHz，来自 cfg["sample_rate"]）
        enroll_embedding: [B, D] 注册 embedding（D = cfg["emb_dim"]）
        model:            DualOutputTSE 实例
        cfg:              配置字典（今日仅用于形状契约校验）

    返回：
        s_tgt: [B, T] 目标说话人波形，长度与输入一致。

    形状契约（P2-06）：len(output) == len(command_wav)。
    错误输入（空数组/NaN/Inf/embedding 维度错误）抛出 ValueError。
    """
    _validate_inputs(command_wav, enroll_embedding, cfg)
    was_training = model.training
    model.eval()
    s_tgt, _s_res, _p_tgt = model(command_wav, enroll_embedding)
    if was_training:
        model.train()
    assert s_tgt.shape == command_wav.shape, (
        f"形状契约违反: 输出{tuple(s_tgt.shape)} != 输入{tuple(command_wav.shape)}"
    )
    return s_tgt