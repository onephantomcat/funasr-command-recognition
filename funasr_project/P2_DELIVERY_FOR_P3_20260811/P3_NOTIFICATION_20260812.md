# P2 → P3 通知：代码已提交 + B3 坍塌根因定位 + 修复方案

**日期：** 2026-08-12
**P2 分支：** `feature/xuanmo-rejection-optimize-v2`
**最新 commit：** `0bd3a9d`
**P3 报告依据：** `CER_P3_PLAN_REPORT_20260810.md` + `P1_P2_P3_FULL_DELIVERY_REPORT_20260810(3).md`

---

## 1. P3 报告指出的 5 项代码问题——已全部修复并提交

P3 报告指出"微信 v2.2 包中的代码变化尚未进入 Git 分支"。**此问题已解决**，请拉取 `0bd3a9d`：

```bash
git fetch origin
git checkout feature/xuanmo-rejection-optimize-v2
git pull origin feature/xuanmo-rejection-optimize-v2
```

| # | P3 报告原文 | 修复内容 | 修改文件 |
|---|---|---|---|
| 1 | "diagnose_b3.py 不在已合并 Git 分支" | 已新增并提交 | `tools/diagnose_b3.py` |
| 2 | "evaluate_tse.py 仍允许 CAM++→BOOTSTRAP fallback" | 删除死代码 fallback 分支，正式评测 CAMPPlus 失败直接 RuntimeError | `tools/evaluate_tse.py` |
| 3 | "data.sha256 标注语义不正确：同一哈希同时标 train 与 dev" | 重算 B1/B2/B3 train/dev 单文件 SHA256 | `THIRD_PARTY_PINNINGS.txt` |
| 4 | "speakerlab commit 和 CAMPPlus 权重 SHA 未给" | 固定 commit `065629c3` + SHA `3388cf5f` | 同上 |
| 5 | "choice_accuracy 为 null 还是 0.9892？SI-SDR 互相冲突" | eval_summary.json 披露冲突并标记 deprecated | `B3/eval_summary.json` |

---

## 2. B3 输出坍塌根因：`absent_loss_scale` 配置参数从未被代码使用

### 2.1 发现

P3 报告指出 B3 v3 checkpoint（SHA `13bce1b9…96db8b`）在 DatasetA 20+20 门上 20/20 正样本近静音、正样本接收率 0%、CER 100%。B1 同链路对照正常（0/40 近静音，80% 接收率）。

经代码审计，**根因是 B3 config 中的 `absent_loss_scale: 0.05` 参数从未被任何 Python 代码读取**：

```
全局搜索 absent_loss_scale：
  config.yaml          → 定义了 absent_loss_scale: 0.05
  train_commands.txt   → 文档记录了该参数
  *.py                 → 0 处引用 ← 问题根因
```

### 2.2 坍塌机制

1. B3 `scene_mode=b3` 训练混合 PRESENT + ABSENT + SWAP 样本
2. ABSENT 样本的 target 和 activity_mask 被置零（`train_b1_trial.py L261-263`）
3. `compute_losses()`（`train_overfit_debug.py`）对 ABSENT 样本以**全权重**计算损失：
   - `scale_sensitive_l1(s_tgt, 0, kappa)` → `s_tgt.abs().mean()`（lambda_wav=0.5 全权重推 s_tgt 趋零）
   - `activity_bce_loss(p_tgt, 0)` → p_tgt 趋零（lambda_act=1.0 全权重推 activity head 趋零）
   - `mrstft_loss(s_tgt, 0, ...)` → 频谱趋零
4. 配置的 `absent_loss_scale=0.05` 本应将 ABSENT 梯度降至 5%，但代码未实现此缩放
5. 20000 步后，ABSENT 样本的全权重梯度主导训练，模型学到**全局抑制**行为
6. B1 不受影响，因为 `scene_mode=b1` 只训练 PRESENT 样本

### 2.3 修复

已修改两个文件：

**`src/tse/losses.py`**：为 `si_sdr`、`scale_sensitive_l1`、`activity_bce_loss` 添加 `reduction="none"` 参数，支持返回逐样本损失。

**`tools/train_overfit_debug.py`** `compute_losses()`：检测 batch 中的 `is_absent` 标志，对 ABSENT 样本的 `si_sdr`/`wav_l1`/`act_bce` 损失按 `absent_loss_scale` 逐样本降权：

```python
# 逐样本权重：ABSENT ×absent_scale，PRESENT ×1.0
w = torch.where(is_absent, absent_scale, 1.0)
si_sdr_val = (si_sdr_per_sample * w).sum() / w.sum()
wav_l1_val = (wav_l1_per_sample * w).sum() / w.sum()
act_bce_val = (act_bce_per_sample * w).sum() / w.sum()
```

向后兼容：B1（无 ABSENT 样本）训练行为完全不变。

---

## 3. debug 训练 fixture 已生成

P3 报告指出 `--debug_data` 训练在 `P2_project/artifacts/debug_mixtures_v0/manifest.jsonl` 缺失处停止。

**已修复**：
- 新增 `tools/generate_debug_fixture.py` 脚本，生成 5 条合成音频（mixture/target/interferer/enrollment + activity mask）
- fixture 已包含在交付包 `artifacts/debug_mixtures_v0/` 中
- P3 可直接运行 `python tools/generate_debug_fixture.py` 重新生成

验证命令：
```powershell
python tools/train_overfit_debug.py --config configs/tse_overfit_debug.yaml --max_steps 1 --device cpu
```

---

## 4. 下一步计划

| 步骤 | 内容 | 预计 |
|------|------|------|
| 1 | P3 拉取 `0bd3a9d`，确认 5 项代码修复到位 | 即时 |
| 2 | P2 用修复后的 `compute_losses()` 重新训练 B3（seed=20260813） | 需 GPU 训练 |
| 3 | P2 交付新 checkpoint，P3 复跑 20+20 门 | 新 checkpoint SHA 不同 |
| 4 | 通过后 P3 运行 6000 条 B0/ORACLE/B1 全量 | 按计划 |
| 5 | 多种子验证（至少 3 个种子，2/3 通过） | 按计划 |

---

## 5. 关于 P3 报告中的其他问题

| P3 问题 | P2 回应 |
|---------|---------|
| "B2 与 B3 复用同一 checkpoint" | B2 设计上复用 B3 权重，B3 修复后 B2 同步更新 |
| "多种子候选未提供" | 修复 B3 坍塌后，用 seed=20260814、20260815 各训练一个候选 |
| "choice_accuracy 与 SI-SDR 冲突" | 已在 eval_summary.json 中披露并标记 deprecated，P3 以独立复验为准 |
| "P2 debug 训练 fixture 缺失" | 已生成，见 §3 |

---

## 6. 交付包更新清单

本次更新（相对于 P3 报告审查的 v2.0 版本）：

| 文件 | 变化 |
|------|------|
| `src/tse/losses.py` | 新增 `reduction="none"` 支持 |
| `tools/train_overfit_debug.py` | `compute_losses()` 实现 absent_loss_scale |
| `tools/generate_debug_fixture.py` | 新增，生成 --debug_data fixture |
| `artifacts/debug_mixtures_v0/` | 新增，5 条合成音频 + manifest |
| `tools/diagnose_b3.py` | 新增（已在 `0bd3a9d` 提交） |
| `tools/evaluate_tse.py` | 删除死代码 fallback（已在 `0bd3a9d` 提交） |
| `tools/train_b1_trial.py` | 修复 KeyError + namespace（已在 `0bd3a9d` 提交） |
| `THIRD_PARTY_PINNINGS.txt` | 固定 speakerlab commit + CAMPPlus SHA |
| `DELIVERY_INSTRUCTIONS.md` | 更新文件名和 SHA256 |

**注意**：B3 checkpoint（`13bce1b9…96db8b`）仍为旧版本，需用修复后的代码重新训练。新 checkpoint 交付后 SHA 会不同。
