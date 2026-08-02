# P2 TSE 模块阶段总结报告

**报告日期**：2026-07-25　**负责车道**：P2（目标说话人提取）　**分支**：feature/xuanmo-rejection-optimize-v2
**本阶段定位**：双输出 TSE 最小可运行版本 + 本机可完成的全部验证（不等 GPU / 正式数据）

---

## 一、任务书要求 vs 完成情况

| 任务书硬要求 | 完成情况 |
|---|---|
| 输入 mixture + enrollment，输出 target_waveform + residual_waveform（双输出，非旧版单输出掩码） | ✅ 已实现（LSTM-Spex 双掩码头 + 硬投影） |
| 不修改现有 ASR 主体，独立 TSE 模块 | ✅ 全部新代码位于 src/tse/，老代码零改动 |
| 随机张量 forward / loss.backward() smoke | ✅ PASS（pytest 18 passed, 1 skipped） |
| 输入输出长度一致、s_tgt/s_res 形状正确 | ✅ 四档长度（160/16000/57600/160000）全过 |
| 损失预留：目标重构 + 残余重构 + 混合一致性 | ✅ 三类齐备；ABSENT zero-suppression 已实现未启用（λ_id=0 硬约束） |
| 6GB 显存可运行 | ⏸ BLOCKED_EXTERNAL（本机 torch 2.13.0+cpu 无 CUDA，已显式记录非静默；待 GPU 机器重跑同一条命令补验） |

## 二、完成阶段清单（全部 PASS）

| 阶段 | 内容 | 证据 |
|---|---|---|
| P2-00/01 | 目录骨架 + 进度档/机器档/历史档 | state/P2_progress.json、reports/P2_machine_inventory.md、state/history/2026-07-25.md |
| P2-02 | 依赖登记（7 项，各标 blocking + local_stand_in） | state/P2_dependency_register.json |
| P2-06 | 双输出 TSE 最小模型 + 随机张量 smoke | reports/smoke/tse_random_forward.json、logs/p2/smoke_tse_random.log |
| P2-07 | DEBUG_ONLY 小语音集（12 条混合） | artifacts/p2/debug_mixtures_v0/（manifest + SHA256SUMS + _meta） |
| P2-08 | 注册条件验证（五条件测试） | reports/smoke/tse_conditioning.json |

## 三、模型与关键技术决策

**架构**（05A 正式基线同族，标记 DEBUG_ONLY_scaffold_not_formal）：

```
mixture ──STFT(512/128)──► |X| ──Linear(257→256)──► FiLM ──► LSTM×2(256)
enroll embedding ──Linear(192→512)──► (γ,β)，scale=1+0.1·tanh(γ)（防翻转）
──► 双 sigmoid 掩码头 ──► ISTFT(length=T) ──► 硬投影 ŝ+r̂=x（误差实测 ~0）
```

- 参数量 **1,349,634（≈1.35M）**，低于 3M 预期
- 关键设计论证：硬投影不阻断梯度（雅可比特征值 1/0，仅掐掉违反约束的法向分量，可学的"目标/残余分解"方向梯度无损——16 个参数张量梯度全部有限非零，实证）
- 条件有效性（P2-08）：换/零/打乱 embedding 输出差异 0.4~0.6%（显著）；全零 mixture 输出严格为 0（条件不凭空造能量，符合 05B §8 防幻觉）

## 四、测试与数据资产

**测试套**（`python -m pytest tests/p2 -v`，18 passed + 1 skipped）：

| 文件 | 覆盖 |
|---|---|
| tests/p2/test_dual_tse_smoke.py | shape×4 档、投影一致性×4、梯度、ABSENT 损失、API 拒错、显存(skip) |
| tests/p2/test_tse_conditioning.py | 六条件验证（独立脚本可直接跑） |
| tests/p2/smoke_tse_random.py | 原始 smoke 完整版（含日志/json 落盘、退出码） |
| tests/p2/test_debug_mixtures.py | 语音集 6 项验收 |

**DEBUG_ONLY 小语音集**：trials 4 说话人 × 3 配置（partial25 / full100×4 / enroll-swap×4），每条含 mixture+双 stem+sample 级 activity 掩码，混合闭合误差 <1e-6，全程只读源数据。

**增强素材就绪**：MUSAN（11.73 GB）、RIRS_NOISES（3.52 GB）已解压校验；AISHELL-1 train 14.64 GB 在库。

## 五、已记录决策与偏差

1. **D-2026-07-25-01**：P2 结构建于 funasr_project/ 内，不迁移根目录（用户决策，已入档）
2. 当前模型为冒烟脚手架（05A LSTM-Spex 基线同族）；WeSep 主架构迁移待决策点 B（B1 现在静态克隆 / B2 云端一次做）
3. 损失为任务书最小三类；05A 冻结五项公式（MR-STFT / L_pair / 活动 BCE / L_pre-mix 投影前重构）已注册为 P2-09 前置升级项
4. ABSENT zero-suppression 按 05B 死区 κ=1e-3 实现，未启用；红线遵守：TSE 静音不做拒识判断

## 六、阻塞与下一步

| 阻塞 | 解除条件 |
|---|---|
| GPU 显存验收（<6GB） | GPU 机器重跑 `python tests/p2/smoke_tse_random.py --device auto` |
| P2-09 100 步过拟合 / P2-10 训练 | 云 GPU + 损失升级 + P1 正式合成数据（mixgen_v1.0） |

**下一步（本机可做）**：losses.py 升级 05A 冻结公式 + model 暴露投影前输出 → 云端关口（P2-04 环境锁 → P2-09 → P2-10）。

## 七、过程中发现并修复的问题（工程质量记录）

1. 短波形 STFT 反射填充失败（T<n_fft）→ 补零保护
2. 一致性投影基准误用补零后输入 → 改 x_orig
3. 条件测试 randperm 存在不动点（B=4 约 63% 概率误判）→ 循环移位
4. pytest 包装层 scale_sensitive_l1 漏传 kappa → 修复
5. pip 官方源/代理均不通 → 清华镜像安装 pytest 9.1.1
