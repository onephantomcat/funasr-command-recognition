"""双输出 TSE 最小可运行模型（LSTM-Spex 风格，STFT 域双掩码）。

输入：混合波形 x [B, T] + 注册 embedding e [B, D]
输出：目标波形 s_tgt [B, T]、残余波形 s_res [B, T]、帧级活动度 p_tgt [B, Fr]

架构（方案 A，FiLM 维度对齐后）：
    x ──STFT──► 复数谱 X [B, F, Fr]，幅度 |X|
    |X| [B, Fr, F] ──input_proj(F→H)──► h [B, Fr, H]
    e ──film_proj(D→2H)──► (γ, β)，h ← (1 + s·tanh(γ))⊙h + β   （隐空间 FiLM）
    h ──LSTM×N──► 双掩码头 Linear(H→F)+Sigmoid ──► mask_tgt, mask_res
    Ŝ = mask_tgt ⊙ X，R̂ = mask_res ⊙ X ──ISTFT(length=T)──► s_tgt, s_res
    硬投影：e = x − s_tgt − s_res；s_tgt += e/2；s_res += e/2 （ŝ + r̂ = x）

红线：
- 只做张量变换，不写文件、不打印；
- 混合一致性硬投影在本类内做，软约束损失在 losses.py 做，职责分离；
- emb_dim 等结构参数全部从 cfg 读，不出现魔法数字；
- 无身份反向损失（λ_id=0 硬约束）。

标记：DEBUG_ONLY / BOOTSTRAP_ENCODER_ONLY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualOutputTSE(nn.Module):
    """双输出目标说话人提取网络。"""

    # 防翻转 FiLM 缩放系数（总方案 §5.1 硬约束，非超参）
    FILM_SCALE = 0.1

    def __init__(self, cfg):
        super().__init__()
        self.n_fft = cfg["n_fft"]
        self.hop_length = cfg["hop_length"]
        self.win_length = cfg["win_length"]
        self.freq_dim = self.n_fft // 2 + 1
        hidden = cfg["lstm_hidden"]

        self.register_buffer("window", torch.hann_window(self.win_length))

        # 方案 A：先投影到隐空间，FiLM 在隐空间注入
        self.input_proj = nn.Linear(self.freq_dim, hidden)
        self.film_proj = nn.Linear(cfg["emb_dim"], 2 * hidden)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=cfg["lstm_layers"],
            batch_first=True,
            dropout=cfg.get("dropout", 0.0),
        )
        # 双掩码头（sigmoid 有界，限制凭空生成能量，05B §8）
        self.mask_tgt = nn.Linear(hidden, self.freq_dim)
        self.mask_res = nn.Linear(hidden, self.freq_dim)
        # 帧级目标活动度头（05A PRESENT 损失 BCE 项）
        self.act_head = nn.Linear(hidden, 1)

    def forward(self, mix_wav, enroll_emb):
        """mix_wav [B,T]、enroll_emb [B,D] → (s_tgt, s_res, p_tgt [B,Fr])。"""
        T = mix_wav.shape[-1]
        x_orig = mix_wav  # 投影基准：原始未补零输入
        # 短波形保护：T < n_fft 时 STFT 的 center 反射填充会失败，右侧补零到 n_fft
        if T < self.n_fft:
            mix_wav = F.pad(mix_wav, (0, self.n_fft - T))

        # 1-2. STFT → 复数谱 [B,F,Fr] 与幅度特征
        spec = torch.stft(
            mix_wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        mag = spec.abs().transpose(1, 2)  # [B, Fr, F]

        # 3-4. 隐空间投影 + FiLM 注入（1 + s·tanh(γ) 防翻转）
        h = self.input_proj(mag)  # [B, Fr, H]
        gamma, beta = self.film_proj(enroll_emb).chunk(2, dim=-1)  # [B,H]×2
        scale = 1.0 + self.FILM_SCALE * torch.tanh(gamma)
        h = scale.unsqueeze(1) * h + beta.unsqueeze(1)

        # 5-6. LSTM → 双掩码 [B,F,Fr] + 帧级活动度 [B,Fr]
        h, _ = self.lstm(h)
        mask_tgt = torch.sigmoid(self.mask_tgt(h)).transpose(1, 2)
        mask_res = torch.sigmoid(self.mask_res(h)).transpose(1, 2)
        p_tgt = torch.sigmoid(self.act_head(h)).squeeze(-1)

        # 7. 掩码乘复数谱 + ISTFT（length=T 保证长度对齐）
        s_tgt = self._istft(spec * mask_tgt, T)
        s_res = self._istft(spec * mask_res, T)

        # 8. 混合一致性硬投影（ŝ + r̂ = x，基准为原始输入）
        s_tgt, s_res = self._apply_consistency_projection(s_tgt, s_res, x_orig)
        return s_tgt, s_res, p_tgt

    def _istft(self, spec, length):
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=length,
        )

    @staticmethod
    def _apply_consistency_projection(s_tgt, s_res, x):
        """正交投影回 ŝ+r̂=x 约束面：e=x−ŝ−r̂ 均分两路（最小范数修正）。"""
        e = x - s_tgt - s_res
        return s_tgt + 0.5 * e, s_res + 0.5 * e
