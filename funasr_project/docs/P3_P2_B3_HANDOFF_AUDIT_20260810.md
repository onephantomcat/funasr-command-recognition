# P3 对 P1/P2 交付物的接入审计（2026-08-10）

> **2026-08-13 后续状态：** 本文保留 2026-08-10 当时的失败证据。此后 P1 正式外部音频包已补齐并通过 6000 条/18000 文件检查；P2 源码与训练链也已修复并完成真实 CAMPPlus 短程运行，但旧 B2/B3 checkpoint 仍未改变、仍被拒收。最新完整结论及 P1/P2/P3 后续步骤见 [P1 / P2 / P3 修复与交付报告](P1_P2_P3_FIX_DELIVERY_20260813.md)。

## 结论

本轮 P3 接线、数据契约和 GPU 推理均已跑通，但当前 B3 checkpoint 被质量闸门拒绝，正式外部配对 CER 又因 P1 音频缺失无法启动。因此没有运行 DatasetA 全量候选评测，也没有生成可宣称的 B1 CER。

裁决：`REJECT_CURRENT_P2_B3_CHECKPOINT`。

## 资产核验

| 资产 | 核验结果 |
|---|---|
| `checkpoint_step20000.pt` | SHA256 `5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d`，与冻结值完全一致 |
| `config.sha256` | `8c8dd51d…` 与 Git 中冻结 `config.yaml` 原始字节一致；YAML 解析后的 45 个字段与 checkpoint 内嵌 `cfg` 全部相等。本地工作树文件因 CRLF 换行转换而具有不同字节哈希，不是配置语义差异 |
| `data.sha256` | 记录 train/dev 共用 manifest SHA256 `03471f98…`，与 B3 目录副本一致；原训练 manifest 未随包交付，因此只能作为来源声明，暂不能独立复算 |
| P1 manifest 压缩包 | SHA256 `99e3d5c3e69c0a411cad4af47f9b8c71e0acda3b46733d278c0d457b671574f1` |
| `D_single.jsonl` | 2000 条、ID 唯一、字段一致、源/输出哈希齐全 |
| `D_overlap.jsonl` | 4000 条；五个场景各 800 条；ID 唯一 |
| AISHELL 参考文本 | 6000/6000 `target_utt` 匹配；无空文本、替换字符或残留空格 |
| 原始源素材 | 12203/12203 文件存在且 SHA256 匹配，约 2.7 GB |
| P1 正式输出音频 | 缺 mixture 6000、target 6000、enrollment 6000，共 18000 个必需 WAV |

P3 规范清单已生成 6000 条，SHA256：

```text
69f1915f15361511b5da768e97183a841b01d683eb944a793f7250867699e629
```

对应齐备度报告 SHA256：

```text
9ad40f8d6e184eac8422db8a962491f153beab4ed53d65d0b4b1f9dce2ab5e1d
```

## P2 接口与嵌入排查

- `verify_p2_local.py` 通过 checkpoint 加载、模型构造（1,349,891 参数）、forward、restore 和 B3 哈希检查。
- 使用同一条 DatasetA 注册音频，对比 FunASR CAM++ 与训练式 `CAMPPlus + Kaldi fbank + mean norm + L2 norm`：嵌入余弦为 `1.0`，最大绝对差为 `0.0`。
- 两种嵌入送入 P2 后得到完全相同的近静音输出，故排除 P3 传错嵌入或归一化方式。

## GPU 质量预检

环境：RTX 4060 Laptop 8 GB、CUDA、冻结硬门控阈值 `0.30`，DatasetA 前 20 pos + 20 neg。该运行只用于失效诊断，不是正式公平全量结果。

| 指标 | 结果 |
|---|---:|
| P2 近静音输出 | 40/40 |
| 输出/输入 RMS 比最小值 | `2.0099948e-7` |
| 输出/输入 RMS 比中位数 | `2.1341932e-7` |
| 输出/输入 RMS 比最大值 | `2.2848837e-7` |
| 最大输出 peak | `5.9606106e-7` |
| 正样本接收率 | `0/20 = 0%` |
| 正样本诊断 corpus CER | `100%` |
| 负样本 RR | `20/20 = 100%`（由近静音造成，不能视为收益） |

诊断报告：`outputs/p3_checkpoint_preflight_20260810/datasetA_p2_limit20.json`，SHA256 `7b719b049102a0daff22ff13fc119edfa1a82c8f4bee3637ce8c1cf1e1a676ab`。报告内已直接保存 `p2_output_quality` 汇总，不依赖人工二次计算。

## 与 P2 自带日志的一致性

B3 `dev_metrics.jsonl` 最后一条为 `dev_sisdr=-19.233210974551262 dB`。`trial_verdict.json` 虽写 `PASS`，其必过条件只有数值有限、显存、吞吐、restore 等工程检查，并未要求 dev SI-SDR 合格或输出非静音。

训练记录还显示包含 ABSENT 的早期 MR-STFT 项达到千万量级，后期模型趋向将 target 分支压到近零。结合损失实现，较强的可能原因是零参考样本上的 MR-STFT 相对谱收敛项被极小分母放大，压倒 PRESENT 学习；该项属于代码证据支持的根因推断，最终应由 P2 在训练数据上做消融确认。

## 恢复执行所需交付

1. P1 提供 manifest 引用的完整 `audio/mixture`、`audio/target`、`audio/enroll`（建议连同 masks、README、SCHEMA）并保证输出 SHA256 一致；或交付精确的 `p1_v2_b1_builder.v1.0.4` 和可复现说明。
2. P2 修复含 ABSENT 样本时的损失/采样策略，重新训练并交付新 checkpoint、SHA256、配置、训练命令、每场景 dev 指标和非静音检查。
3. 新 checkpoint 至少先通过：PRESENT 输出非静音、SINGLE 不退化、OVERLAP 方向改善、ABSENT 抑制有效。
4. 两项资产到位后，P3 依次运行外部 B0/ORACLE/B1 配对 CER；只有外部门通过，才运行一次冻结配置的 DatasetA 全量对比。

## P3 已完成项

- 冻结 P2 `extract_target()` runtime 与 SHA 校验；
- P1→P3 规范清单生成及音频齐备度审计；
- B0/ORACLE/B1 同 sample_id 配对 runner；
- 逐条与语料级 S/D/I/N、错误/RR 契约、输出 RMS 诊断；
- 49 项回归测试全部通过。
