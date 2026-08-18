# P2/P3 最终验证报告（2026-08-14）

## 结论

本轮 P2 数值稳定性修复、B1 正式重训、B3 三个独立训练种子、外部 6000 条配对 CER，以及一次冻结的 DatasetA 公平比较均已完成。

三种子外部配对 CER 的聚合结果为 `ACCEPT_B1_CANDIDATE`，但冻结候选 `seed=20260813` 在 DatasetA 全流程比较中显著劣于基线。因此当前结论是：**保留基线，不推广该 P2 checkpoint，也不在 DatasetA 上调阈值或重选候选。**

本报告记录的是本地原始字符 CER；正式成绩仍以主办方 scorer 为准。

## 已纳入的代码状态

P2 训练稳定性修复已在仓库代码中：

- `04c86f7`：将损失归约移至 FP32、稳定 BCE / SI-SDR / STFT 路径，并补充 AMP 数值诊断。
- `bd6bef2`：兼容 legacy AMP scaler API。
- `77b9dd8`：P3 三个独立训练种子的配对 CER 聚合支持。

相关测试在修复阶段已通过：本地 `106 passed, 4 skipped`，另有 22 个子测试和云端 5 个 CUDA 定向测试通过。模型权重、音频、缓存、原始评测输出均不纳入 Git；本报告只保存可复核的结论、配置约束和摘要指标。

## 验证流程

1. B1 在正式配置下完成 20,000 optimizer updates，`dev_metrics_finite=true`、`n_nan_steps=0`、`optimizer_steps_skipped=0`，strict restore 通过。
2. B3 的 `20260813`、`20260814`、`20260815` 三个种子均从 B1 step 20,000 strict warm-start，各完成 20,000 updates。
3. 每个 B3 均通过真实 20 正 + 20 负预检：结果有效、P2 输出非近静音、正样本不再全部误拒，负样本抑制并非由全静音造成。
4. 三个 checkpoint 在同一 6,000 条外部 paired manifest 上分别运行 B0 mixture、ORACLE target 与 B1/P2 target CER。
5. 聚合通过后，仅对冻结候选 `seed=20260813` 进行一次 DatasetA 基线/候选公平比较。

所有 DatasetA 比较均固定为：`hard`、`--sv-threshold 0.30`、`--no-intent-filter`、`--no-phrase-correct`、无 embedding cache、无 ASR cache、无 phrase bank。

## 外部 6,000 条 paired CER

| 训练种子 | 结果有效 | ASR errors | P2 近静音 | B0 CER | P2 CER | 绝对变化 | 单次结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 20260813 | true | 0 | 0 / 6000 | 64.16% | 60.55% | -3.61pp | PASS |
| 20260814 | true | 0 | 0 / 6000 | 64.16% | 60.77% | -3.39pp | PASS |
| 20260815 | true | 0 | 0 / 6000 | 64.16% | 60.79% | -3.37pp | PASS |

聚合契约为 `p3_paired_cer_training_seed_aggregate_v1`，冻结训练种子为 `20260813`，结论为 `ACCEPT_B1_CANDIDATE`。外部 paired CER 说明该前端在该外部 ASR 场景中有一致改善；它不能替代下游 hard speaker gate 的端到端验证。

## DatasetA 冻结公平比较

候选仅额外使用 P2 checkpoint `seed=20260813` 的 step 20,000 权重（SHA-256：`9c0fe564fbc6340beb045b610a7e8b34cbbf1f72e4da8ccb877e3f1aa51343a5`）。两组均处理完整的 1,364 正样本和 474 负样本，`result_valid=true`、`complete=true`、`asr_errors=0`。

| 指标 | 基线 | P2 候选 | 候选−基线 |
| --- | ---: | ---: | ---: |
| 正样本 corpus CER | 53.43% | 73.66% | +20.23pp |
| 正样本接受率 | 69.35% | 47.21% | -22.14pp |
| 负样本拒绝率 | 91.14% | 91.56% | +0.42pp |
| 负样本拒绝数 | 432 / 474 | 434 / 474 | +2 |
| 端到端耗时 | 221.03 s | 361.21 s | +140.18 s |

候选的 P2 输出共 1,838 条，`near_silent_samples=0`；输出/输入 RMS 比的中位数为 0.2293。因此该回归不是全静音故障。

## 为什么 DatasetA CER 更高

`eval_datasetA.py` 先对命令音频执行 P2，再以 CAM++ 相似度进行硬门控。未通过门控的正样本不会运行 ASR，假设文本为字符串，故其 CER 会以删除错误计入语料级分数。

- 候选使正样本 gate 拒绝从 418 条增至 720 条，即额外误拒 302 条。
- 366 条正样本从“基线接受”变为“候选拒绝”，带来 +1,677 个字符错误；该组平均 CAM++ 相似度由 0.4305 降至 0.2104，候选均低于固定门限 0.30。
- 在两者均通过门控的 580 条正样本中，候选仍比基线多 441 个字符错误（46.30% vs 36.06%）。因此只降低 speaker threshold 既不能解决全部问题，也会把 DatasetA 变成调参集。

代码审查还确认了一个可验证的训练/部署失配：`configs/tse_b3_v3.yaml` 虽有 `lambda_id`，但 `tools/train_overfit_debug.py::compute_losses()` 未读取该项；同时 `P4_project/src/sv/campplus_backend.py::embed()` 是 `@torch.no_grad()` 推理接口。仅调整 YAML 中的 `lambda_id` 不会形成身份保持梯度。

## 后续修复方向

1. 不以 DatasetA 选择门限、checkpoint 或身份损失权重；维持当前基线作为正式可复现结果。
2. 提供训练专用的冻结、可对 P2 输出反传的 CAM++ 前向。注册音频嵌入不反传，P2 输出嵌入可反传；仅对 `target_present=True` 样本加入 `1 - cosine(e_p2, e_enroll)` 身份保持项。
3. 在独立的 P1/dev 或外部 held-out paired 集选择权重，并同时检查固定 0.30 下的正样本接受率、负样本拒绝率、accepted-only CER 和完整 pipeline CER。
4. 在同一外部集做消融：原混合音频做 SV、P2 输出做 SV、P2 输出做 ASR。若只有 P2→SV 路径退化，才评估“P2 只供 ASR、SV 保持原始音频”的架构。
5. 新候选仍按小规模外部 sanity → 6,000 条外部 paired CER → 一次冻结 DatasetA 比较的顺序执行。

## 复现入口

P2/P3 接入命令、paired CER 与聚合参数见 [P2/P3 接入说明](./P2_P3_INTEGRATION.md)。DataSetA 公平评测的命令和数据泄漏约束见根 [README](../README.md)。

> 本报告不附 checkpoint、音频、`outputs/`、缓存或 DatasetA 标签；这些均由 `.gitignore` 排除。

---

# 附录：id_loss 修复轮三闸门评测（2026-08-18，冻结）

本轮按上文"后续修复方向"第 2 条实施：在训练侧真实接入身份保持损失（`lambda_id` 形成可反传梯度），候选代号 `B3_SWAP_v4_idloss_L1.0`。lambda 扫描 4 个 run 全部 PASS 后选定 `lambda_id=1.0`，全量训练 20,000 optimizer updates 通过（训练种子 `20260817`）。

候选 checkpoint：`P2_project/artifacts/B3_SWAP_v4_idloss_L1.0/checkpoint_step20000.pt`，SHA-256：`3b6b9678c25c1d7b37d2f7b37bd41e30c1f0b0ce9407e6e989c08334be8149a0`。

## 三闸门结果

| 闸门 | 数据域 | 结果 | 关键数字 |
| --- | --- | --- | --- |
| 闸门 1：20+20 预检 | DatasetA 小样 | PASS | 正样本接受 12/20；负样本 20/20 全拒；近静音 0/40 |
| 闸门 2：6,000 条配对 CER | P1 合成分布 | PASS | 见下表，内置 verdict `PASS_SINGLE_RUN_THRESHOLDS`（8/8 checks） |
| 闸门 3：DatasetA 全量冻结比较 | 真实竞赛分布 | **FAIL** | 见下表 |

### 闸门 2：6,000 条配对 CER（P1 合成分布）

`result_valid=true`、`asr_errors=0`、近静音 0/6000，predictions SHA-256：`98fac7a324b85bc35374fd6c30e77d673e71a3c82320f129923d8cf2a63bb4d5`。

| bucket | B0 CER | B1 CER | 绝对变化 |
| --- | ---: | ---: | ---: |
| OVERALL | 64.16% | 59.59% | -4.57pp |
| OVERLAP_100（判据 2） | 88.98% | 81.72% | -7.26pp（≥5pp ✓） |
| SINGLE（判据 3） | 47.80% | 44.08% | -3.72pp（不恶化 ✓） |
| OVERLAP_100_SIR_-5DB（判据 4） | 96.02% | 88.17% | -7.86pp（不反转 ✓） |

### 闸门 3：DatasetA 全量冻结比较（第二次、即最后一次全量额度）

与基线完全同参：`hard`、`--sv-threshold 0.30`、`--no-intent-filter`、`--no-phrase-correct`、新建空 `--p2-tse-dir`。1,364 正 + 474 负全部完成，`result_valid=true`、近静音 0/1838。

| 指标 | 冻结基线 | 旧候选 9c0fe564 | 本轮候选 L1.0 |
| --- | ---: | ---: | ---: |
| 正样本 corpus CER | **53.43%** | 73.66% | 65.02% |
| 正样本接受率 | **69.35%** | 47.21% | 59.24%（808/1364） |
| 负样本拒绝率 | 91.14% | 91.56% | **92.41%**（438/474） |
| 端到端耗时 | 221.03 s | 361.21 s | 764.4 s |

失败机制与旧候选同方向、程度约减半：正样本 CAM++ 相似度分布下移（sim 区间 -0.1561 ~ 0.7417），更多正样本跌破 0.30 门限被硬门控误拒；被拒样本以删除错误计入 corpus CER，多拒约 10pp 恰好解释 CER +11.6pp。`near_silent_samples=0`、RMS 比中位 0.223，排除全静音故障。

## 本轮结论（冻结）

**REJECT 候选 `B3_SWAP_v4_idloss_L1.0`，继续保留冻结基线作为 DatasetA 正式提交配置。**

- id_loss 接入本身成立：在 P1 合成分布上全部 bucket 一致改善（高重叠 -7.26pp），训练 bug 修复有效。
- 但收益未迁移到 DatasetA 真实域：正样本接受率与 corpus CER 显著回退，neg RR +1.27pp 的改善不足以抵消。
- 合规记录：DatasetA 全量评测两次额度已全部用完（backup 消融一次、本轮一次），不再对该数据集做任何重跑、调阈或重选。
- 后续若再启动 P2 前端工作，应先解决"合成域改善 → 真实域 SV 路径退化"的泛化缺口（参见上文修复方向第 3、4 条），再按同样三闸门顺序验证。

评测产物存证：闸门 2 `outputs/p3_external_gate/summary.json`、闸门 3 `outputs/gate3_datasetA_final/result.json`（均不入库，仅存摘要于本报告）。
