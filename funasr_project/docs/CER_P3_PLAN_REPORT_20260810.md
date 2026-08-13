# P3（ASR 与 CER）计划报告

**初版日期：** 2026-08-10

**状态更新：** 2026-08-12

> **2026-08-13 补充**：P2 新提交 `0bd3a9d`、`dd91f4c` 已拉取并完成语义集成；B3 损失、真实 CAMPPlus、训练/评测/诊断和 P1 复现入口均已修复并通过测试及短程实际执行。该进展解决的是代码与训练链，不会改变旧 `13bce1b9…db8b` checkpoint 的 P3 `REJECT`。P3 的下一步仍是等待新权重后先跑 20+20，再决定是否运行 6000 条外部配对评测。详见 `P1_P2_P3_FIX_DELIVERY_20260813.md`。

**责任角色：** P3，Paraformer 识别、文本规范与 CER 评测

**本轮目标：** 在不使用 DatasetA 标签训练或调参的前提下，建立可复现的 CER 证据链，判断目标增强/TSE 是否真正改善识别，并在 2026-08-15 前冻结 P3 核心。

## 0. 执行更新（2026-08-12）

- P3 实施分支为 `codex/p3-integrate-pr3`。P2 分支 `feature/xuanmo-rejection-optimize-v2` 已合并为 `65f81be`，P1 分支 `p1-reproducibility` 已合并为 `776a872`；均无冲突。
- P3 源码、测试和必要接入文档已提交为 `69da5fa`，未提交模型、音频、大型输出、独立 hash/baseline/gate 登记表。
- 已修复旧 TSE evaluator 的 ASR 元组契约，CER 输入现在拒绝非字符串，并新增逐条/语料级 S/D/I/N。
- 已新增 P2 `extract_target()` 的 P3 runtime、B0/ORACLE/B1 配对 CER runner，以及防止 ASR 异常被误算为正确拒识的运行时校验。
- 当前 P3 回归为 **49 tests PASS**，其中包含 18 类 UTF-8/CER 黄金样例、P1→P3 清单桥接、精确输出能量诊断和 RTX 4060 上的 P2 API CUDA 冒烟。
- P1 正式外部包已到位；实际重建得到 6000 条唯一配对记录，2000 条 `D_single`、4000 条 `D_overlap`，三类音频各 6000 个，缺失 0，P1→P3 数据入口已经可执行。
- 使用 P1 正式包和 P2 B3 v3 在 RTX 4060 上完成了 1 条真实 B0/ORACLE/B1 端到端冒烟：三路 ASR 均完成、无运行错误，证明接线可执行；该单样本结果不作为模型收益结论。
- 同一注册音频经 FunASR CAM++ 与训练式直接 CAMPPlus 得到的嵌入逐元素一致（余弦 `1.0`、最大绝对差 `0.0`），排除了 P3 嵌入接线错误。
- P2 B3 v3 在 P3 的 DatasetA 20 正 + 20 负诊断中仍为 39/40 近静音、正样本接收率 `0%`、诊断 CER `100%`；因此当前权重继续记为 **REJECT_CURRENT_P2_B3_CHECKPOINT**。
- 群聊 v2.2 包中的 `diagnose_b3.py` 及另外三个变化脚本尚无对应 Git 提交，且 B2/B3 仍是上述被拒权重。下一步只等待 P2 提供可合并代码与新候选，不再等待 P1 正式音频。

下文第 1—10 节保留 2026-08-10 的分析背景；发生冲突时，以本节和第 11 节的 2026-08-12 更新为准。

---

## 1. 结论先行

1. **当前最可信的公平 DatasetA 基线仍是 CER 53.43%、RR 91.14%、正样本接收率 69.35%。** 配置为硬 CAM++ 阈值 `0.30`、关闭意图过滤、关闭短语纠错、不使用 ASR/embedding 缓存。
2. **现有 `target_enhancer_musan_rirs.pt` 不能进入默认推理。** 同一全量配置下，CER 从 53.43% 恶化到 61.35%，正样本接收率从 69.35% 降到 61.07%；RR 虽提高到 93.67%，但这是以更多目标语音误拒和识别退化换来的。
3. **现有 `target_enhancer_denoised_mix.pt` 也不能进入默认推理。** CER 为 62.45%，比基线恶化 9.02 个百分点。
4. **现有双输出 TSE 的 346.02% CER 是无效结果，不代表模型真实性能。** `eval_datasetA_tse_dual.py` 没有拆包 `recognize()` 返回的 `(text, elapsed)`，而是把整个元组交给 CER；负样本侧也把非空元组当作“未拒绝”。必须先修复评测器和补契约测试，再重跑。
5. **“13 维门控降到 16%”不能作为公平 CER 成果。** 当前门控元数据明确写着 `dataset: datasetA`，属于用 A 标签训练/选阈；提交记录声称的 16.12%/98.10% 与仓库内模型元数据及全量报告也互相不一致。该结果只能作为历史调试信息，不能进入正式证据链。
6. **GPU 已经安装好，不需要重新安装。** 正确环境是项目上一级的 `..\.venv`：Python 3.9、PyTorch `2.7.1+cu118`、`torch.cuda.is_available() == True`，设备为 RTX 4060 Laptop 8 GB。当前目录内 `.\.venv` 是另一套 Python 3.13 + CPU PyTorch，后续命令必须避免用错。
7. **当前最重要的不是继续扫阈值，而是修复评测可信度。** 先冻结 CER 契约、修复 TSE 评测器、建立 B0/ORACLE/B1 配对实验；只有外部说话人隔离确认集通过门槛后，才允许对 DatasetA 做一次冻结候选的全量阶段验证。
8. **本次收到的 B3 文件“资产有效、模型质量不合格”。** 哈希、结构和 CUDA forward 都正确，但 40/40 DatasetA 预检输出近静音；不能把“checkpoint 能加载”误写成“P2 可接入”。
9. **P1 正式评测包现已完整。** 6000 条配对清单和 18000 个 mixture/oracle/enrollment 音频引用均已到位；另一个独立问题是合并后的 P1 v2 builder 缺少九份 train/dev/holdout split manifest，尚不能从源码复现生成过程。

建议当前决策为：

> **拒绝当前 B3 checkpoint 进入 P3 正式链路，保留公平硬门控基线作为可提交回退。P2 应先修复含 ABSENT 样本时的训练/质量闸门并交付新冻结权重；P1 应补齐与 manifest 输出哈希一致的完整音频包。两项同时满足后，P3 再从外部 B0/ORACLE/B1 配对 CER 恢复，而不是直接跑 DatasetA 全量。**

---

## 2. 证据来源与可信度边界

### 2.1 微信群聊只读核查

2026-08-10 通过微信聊天记录搜索，以只读方式核查了 `CER`、`0.3`、`门控`、`RR`、`TSE`、`训练`、`测试集`、`降噪`、`增强` 等关键词；未发送任何消息。

已直接核实的关键信息：

- 2026-08-03：群内出现“`cer0.3是什么鬼`”，说明 CER 小数/百分比和阈值含义没有统一展示。
- 2026-08-03：有人回忆“加了一个 13 维门控的都降到 16 了”。这与仓库中的 13 维门控提交相呼应，但不能证明其公平性。
- 2026-08-03：本人明确提醒“避免一下数据集泄露就行了”“尤其是测试集别放进去训练”。
- 2026-07-24：群内说明 `denoiser_best.pt` 所属降噪模型“用 MUSAN 训练出来的”。
- 2026-07-22：群内提出用现有噪声与 AISHELL 训练增强模型，并建议加入降噪预处理再验证。
- 2026-07-24：群内提到阈值门控 v2，目标是改善正确识别样本的通过率。
- TSE 搜索结果中存在 `tse_train_00090001.wav`、`tse_train_00070001.wav`、`tse_train_00035001.wav` 等样例；同时存在 P2→P1 接口需求，要求 smoke 数据覆盖 100% overlap、SIR=-5 dB，并要求正式 train/dev/confirm 划分与说话人重叠/hash 去重报告。
- 一段转发记录中可见测试者反馈“确实能降低 CER，但不多，还不知道误差大”，这只能视为定性反馈，不能替代冻结数据、同配置、逐条配对结果。

本报告不把看不清的转发预览、表情或未打开的附件内容作为硬证据。

### 2.2 会议材料

来源：`C:\Users\13238\Desktop\挑战杯1号语音识别\7.25\7月25项目会议_新流程与五人分工_7页.pptx`，SHA256：`578E5A719303FA2589B484E0876217AED4F9AAD43D32C82FCAB86628007A2CC7`。

会议已冻结的关键共识：

- DatasetA 只做阶段开发评测，不得作为干净 stem 训练集。
- 旧路线“增强 + 单声纹阈值 + ASR”缺少目标/残余的可审计证据；提高 RR 会误拒大量 pos。
- 新路线是注册条件化 TSE：`mixture + enrollment -> target + residual`，Paraformer 唯一正式输入为 `target`，再进行 PRESENT/EMPTY/GRAY 三态审计。
- P3 负责 ASR 与 CER；P3/P4 的下一次会议验收包括 5 条 UTF-8 ASR/CER 测试和 8 类三态审计测试。
- 短期目标：证明双输出 TSE 在重叠语音中改善目标提取，同时不破坏 SINGLE。

### 2.3 P3 专项手册和实施协议

依据 `7.25\outputs\01_六人执行手册\P3_交给陌生AI的零基础逐步教学执行手册.md` v1.2：

- P3 主责：冻结 Paraformer、统一 `recognize` 接口、文本规范、语料级 CER、B0/ORACLE/B1 配对识别、错误分桶和发布冻结。
- P3 不负责：训练 P2 TSE、训练 P4 决策器、用 ASR 置信度证明身份、单方面改变官方规则、用热词/同音纠错偷偷降低 CER。
- 正式 CER 必须同数据、同 ASR、同规范配对；负样本不进入 CER，正样本空输出按整句删除计入。
- B1 验收：高重叠 CER 相对下降至少 15%或绝对下降至少 5 pp；SINGLE 恶化不超过 2 pp；100% overlap、SIR=-5 dB 方向不能反转；至少 2/3 种子改善。
- 硬日期：8 月 7 日前 B1 结论、8 月 15 日冻结 ASR 核心、8 月 21 日后不再启动 ASR 蒸馏、9 月 1 日后不再更换模型/规则/推理链。

截至本报告日期，B1 的 8 月 7 日门已经逾期；旧双输出 TSE 数值因评测错误无效，而新收到的冻结 B3 checkpoint 又因近静音质量失败被拒绝。

### 2.4 仓库范围

本报告审查的是本地工作树：

- 本地 HEAD：`2375254294a95bfc33dac8ad9d57e8f4c6d7572f`
- 分支：`codex/p3-integrate-pr3`
- 远端 `main` 当前指向：`2375254294a95bfc33dac8ad9d57e8f4c6d7572f`

P2 PR #3 已在该基线合并，本地未落后于远端；本报告涉及的 P3 实现仍是该分支上的未提交工作树变更，未覆盖原有可复现实验结果。

---

## 3. 先统一“CER=0.3”到底是什么意思

群聊中的歧义很可能来自两种完全不同的 `0.30`：

| 写法 | 正确含义 | 趋势 |
|---|---|---|
| `CER = 0.30` | 字错误率 30.00% | 越低越好 |
| `SV threshold = 0.30` | CAM++ 声纹相似度的接受阈值 | 不是误差率；提高通常增加拒识 |

以后任何汇报禁止只写“0.3”。统一写成：

```text
Corpus CER = 0.3000 (30.00%; errors/ref_chars = 2855/9515)
SV threshold = 0.30
RR = 0.9114 (91.14%; rejected/neg = 432/474)
Positive accept rate = 0.6935 (69.35%; accepted/pos = 946/1364)
```

CER 使用语料级定义：

```text
CER = sum(S + D + I) / sum(reference_chars)
```

必须遵守：

- 不能用“每句 CER 平均”冒充语料级 CER。
- 正样本被拒绝时，输出为空，全部参考字符计作删除。
- 负样本参考为空，不进入 CER；用 RR 单独评估。
- `raw_text`、`normalized_text`、`final_text` 三者不得互相覆盖。
- 本地 CER 只作调试；主办方 scorer 始终是最终权威。

---

## 4. 当前结果台账

### 4.1 可作为公平阶段基线的结果

| 实验 | CER | RR | pos 接收率 | 结论 |
|---|---:|---:|---:|---|
| ASR-only（会议基线） | 39.57% | 0.21% | 近乎全放行 | 说明 Paraformer 本身相对可用，但没有拒识能力 |
| CAM++ hard `t=0.30` | 53.43% | 91.14% | 69.35% | 当前公平回退基线；大量 pos 误拒使 CER 增加 |
| CAM++ 描述性扫描 `t≈0.28` | 51.46% | 89.66% | 未在会议表中冻结 | 仅小幅改善；不能同时守住 CER 与 RR |

ASR-only 到 hard gate 的 CER 增加了 13.86 pp，说明当前 CER 的主要瓶颈之一不是 Paraformer 词错，而是**门控造成的整句删除**。

### 4.2 `--enhancer-model` 同配置全量比较

来源：`eval_baseline.json` 与 `eval_enhancer.json`，均为 1364 pos + 474 neg、硬阈值 0.30、关闭意图过滤和短语纠错。

| 指标 | 无 enhancer | `target_enhancer_musan_rirs.pt` | 变化 |
|---|---:|---:|---:|
| Corpus CER | 53.43% | 61.35% | **+7.92 pp，恶化 14.82%（相对）** |
| RR | 91.14% | 93.67% | +2.53 pp |
| pos 接收率 | 69.35% | 61.07% | **-8.28 pp** |
| 全量耗时 | 233.48 s | 222.83 s | 分次运行波动，不作正式效率结论 |

`target_enhancer_denoised_mix.pt` 的 CER 为 62.45%、RR 93.88%、pos 接收率 61.00%，同样失败。

解释：现有增强前端虽然提高了负样本拒识，但同时改变声纹/语音内容，导致更多正样本被拒和识别失真。对 CER 任务而言，这不是收益。

### 4.3 当前双输出 TSE 结果为什么无效

`data/datasetA/eval_tse_dual_output.json` 表面报告：

- CER 346.02%
- pos 接收率 84.97%
- RR 72.57%
- 耗时 6723.34 s

但代码存在接口错误：

```python
# asr_demo.py
recognize(...) -> (text, elapsed)

# eval_datasetA_tse_dual.py 当前错误用法
hyp_text = recognize(asr_model, str(target_path))
pairs.append((ref_text, hyp_text))
```

`cer.py` 又把非字符串静默 `str()`，导致 `(文本, 0.123)` 整体参与字符编辑距离。负样本侧的 `(空文本, 耗时)` 元组也是 truthy，RR 同样被污染。

因此：

- 346.02% 不能用于评价 TSE；
- 72.57% RR 也不能用于比较；
- 该 JSON 应标记为 `INVALID_EVALUATOR_CONTRACT`，不能继续引用；
- 修复后必须从空输出目录重跑，不能只修改汇总字段。

### 4.4 13 维门控与历史低 CER 为什么不能进入正式报告

当前存在四组互相冲突的信息：

1. 群聊称“13 维门控降到 16”。
2. Git 提交信息称 CER 16.12%、RR 98.10%、80 分制 72.79。
3. `models/lightweight_gate.json` 的 dev 元数据是 CER 29.09%、RR 97.13%，阈值 0.38。
4. 本地全量 `eval_gate_v2*.json` 是 CER 40.33%、RR 21.10%；短语版本则 CER 88.54%、RR 100%。

更关键的是，模型元数据明确标记 `dataset: datasetA`，历史网格搜索和部分短语纠正也直接使用 A 标签/短语。它们违反“DatasetA 只作阶段验证、不进训练和调参”的共识。

处理规则：

- 所有这类结果统一标记为 `DESCRIPTIVE_LEAKED_OR_UNVERIFIED`。
- 不进入提交方案、不用于选阈、不用于宣传最终 CER。
- 只能帮助定位“为什么门控看起来能救低相似度正样本”。
- 如果保留门控思想，必须在外部说话人隔离数据上重新训练/校准，再对 A 做一次冻结验证。

---

## 5. 代码与产物现状审计

### 5.1 已完成

- `cer.py` 已实现标准字符级语料 CER，正样本空输出会形成整句删除，空参考会报错。
- `asr_demo.compact_asr_text()` 只删除 Unicode 格式空白，不依赖标签或短语库。
- `eval_datasetA.py` 已支持 `--enhancer-model`，增强音频会同时送入 CAM++ 和 Paraformer。
- `target_enhancer_musan_rirs.pt` 和 `target_enhancer_denoised_mix.pt` 都有训练元数据；训练/开发说话人隔离。
- 当前 20 项单元测试全部通过；其中 ASR 文本 2 项、CER 3 项、增强器 3 项、TSE/三态 7 项，其余为数据增强相关。
- 上一级 GPU 环境可用：RTX 4060、CUDA 11.8 runtime、PyTorch 2.7.1+cu118。

### 5.2 尚未达到 P3 手册冻结门槛

| 项目 | 当前状态 | 缺口 |
|---|---|---|
| CER 核心 | 部分完成 | 只返回 CER/参考长度，没有逐条和汇总 S/D/I/N；非字符串会被静默转换 |
| 黄金测试 | 部分完成 | 当前 CER+文本仅 5 项；P3 手册要求 18 类黄金样例和 P1 独立复核 |
| 文本规范 | 部分完成 | 有空白清理，但缺官方规则编号、页码、版本和 P1/P3 签字 |
| ASR API | 部分完成 | 当前返回 `(text, elapsed)`，没有结构化 diagnostics/token score；已造成 TSE evaluator 接口事故 |
| ASR 基线配置 | 有冲突 | P3 手册要求核心基线禁用 VAD/punc/spk/hotword/ITN，但当前 `asr_demo.py` 固定加载 FSMN-VAD；需由 P1/P3 明确一套正式配置后冻结 |
| ASR 资产冻结 | 未完成 | ModelScope ID 存在，但 FunASR commit/model revision/SHA256/本地离线加载未完整冻结 |
| B0/ORACLE/B1 | 未完成 | 没有同 sample_id、同 ASR 配置、三种子、逐条 S/D/I/N 的正式配对报告 |
| TSE 评测 | 无效 | evaluator tuple bug；结果不可用 |
| A 集纪律 | 有冲突 | 公平基线存在，但同仓库还保留 A 训练门控/网格搜索/标签短语实验，文档未隔离 |
| 发布包 | 未完成 | 没有 `releases/P3_ASR_CORE_v1`、环境、模型哈希、KNOWN_LIMITATIONS 和干净目录复现记录 |

### 5.3 增强/TSE 模型本身的风险

`target_enhancer_musan_rirs.pt`：

- 最优 epoch 7/8，dev loss 0.04036；
- 只有 13 个训练说话人、3 个开发说话人，覆盖不足；
- 损失是 log-magnitude L1 + waveform L1，不是 Paraformer CER 或 ASR 表征损失；
- 已被全量 A 阶段结果证伪为默认候选。

`tse_dual_output_mvp.pt`：

- 元数据只记录 best epoch 1 和 dev loss 0.06888；缺数据源、说话人列表、参数、种子、训练曲线和哈希；
- 训练脚本接收 `--rir-root`，但当前数据集逻辑没有实际应用 RIR；
- 合成只包含 target+interferer，没有正式的 SINGLE/ABSENT/SWAP 比例和受控 SIR 桶；
- enrollment 与 target 随机抽样，未强制不同录音；
- 只做 L1 target/residual 损失，没有冻结 ASR 下的确认性 CER 验收。

`denoiser_best.pt`：

- SHA256：`716FF7D9FCBAF9B73C3C216418AA83DCE4CD14419EFE8951C8C5A09F00F60357`；
- 是含 128 个张量的裸 `OrderedDict`，键名类似 `enc1.0.weight`；
- 没有 `model_config`、`model_state`、采样率、STFT/归一化、输入输出约定；
- 当前 `load_target_enhancer()` 要求 `checkpoint["model_config"]` 和 `checkpoint["model_state"]`，因此不能直接作为 `--enhancer-model`；
- 在获得原作者网络类和预处理契约前，只能登记为外部降噪资产，不能比较其 CER。

---

## 6. P3 的明确职责与接口

### 6.1 P3 自己必须交付

1. 冻结 Paraformer 模型、FunASR 版本、推理参数和本地模型哈希。
2. 统一 `recognize()` 结构化返回值，并对所有调用者做契约测试。
3. 冻结 `raw_text -> normalized_text` 规则，保留官方规则来源和版本。
4. 实现逐条 S/D/I/N 与语料级汇总，发布唯一 CER 核心。
5. 在同一个 P3 环境集中跑 B0 mixture、ORACLE target stem、B1 TSE target。
6. 输出总体、SINGLE、OVERLAP、100% overlap/SIR=-5 dB、噪声、混响、长度等分桶。
7. 给 P2 提供错误分析，不替 P2 训练 TSE。
8. 给 P4 提供受约束的 ASR 数值诊断，不把 ASR 分数当身份分数。
9. 给 P5 提供稳定 API，不决定最终 EMIT/EMPTY。
10. 给 P6 提供 FP32 基线、FP16/ONNX 对齐样本、速度和显存验收材料。

### 6.2 依赖其他角色的输入

| 来自 | P3 需要的冻结输入 | 接收门槛 |
|---|---|---|
| P1 | manifest、参考文本、scene/SIR/SNR/overlap、speaker split、hash | sample_id 完整唯一；训练/确认/评测说话人隔离；无 A/B 标签泄漏 |
| P2 | target/residual wav、checkpoint hash、source manifest hash、seed | 与 P1 sample_id 一一对应；采样率/长度/通道正确；无 NaN/Inf/削波/静默丢样本 |
| P4 | 三态协议和 reason_codes | 阈值来自外部或内层校准；P3 不参与身份结论 |
| P5 | 最终路由版本 | ASR 只接收 `target_wav`；系统错误与合法空文本分开 |
| P6 | 导出与效率环境 | 相同代表样本；逐条文本回归；计时含完整链路 |

---

## 7. 从 8 月 10 日到 8 月 15 日的抢救计划

### 8 月 10 日：恢复评测可信度（P0）

交付：`p3_text_eval_v1-rc1`、无效结果清单、GPU 环境确认。

1. 在新分支或新目录对比远端 `538e074...` 与本地 `35af1c...`，只移植经过审查的改动。
2. 修复 `eval_datasetA_tse_dual.py`：
   - `hyp_text, asr_elapsed = recognize(...)`；
   - 明确 `hyp_text` 必须是 `str`；
   - 负样本使用 `not hyp_text.strip()`；
   - 保存 ASR 延迟和错误状态；
   - 报告模型/配置/数据 hash。
3. 修改 CER 输入契约：非字符串 reference/hypothesis 必须报错，不能静默 `str()`。
4. 增加至少以下回归测试：
   - ASR 元组拆包；
   - emit 后 ASR 空文本的 RR 处理；
   - missing hypothesis 报错；
   - corpus CER 不等于句均 CER；
   - UTF-8、内部空格、标点、数字；
   - 重复 sample_id；
   - 全负样本 CER 不可定义。
5. 直接在评测报告中记录每个结果的有效性说明；不另建独立的 hash/baseline/gate 登记表。

停止条件：当天不能通过 evaluator 契约测试，就不得运行新的全量 A。

### 8 月 11 日：冻结 B0/ORACLE 与最小 B1 smoke

交付：同 sample_id 的三路预测和逐条 S/D/I/N。

1. 使用正确 GPU 环境：`..\.venv\Scripts\python.exe`。
2. 对 P1 的外部、说话人隔离确认包跑：
   - B0：mixture -> frozen Paraformer；
   - ORACLE：clean target stem -> 同一 Paraformer；
   - B1：P2 target wav -> 同一 Paraformer。
3. 必须包含 SINGLE、OVERLAP、100% overlap/SIR=-5 dB；建议至少每桶 100 条，smoke 可先 10 条但不能作结论。
4. 若 ORACLE 也差，先修 P3 ASR/文本/数据；若 B0 差而 ORACLE 好，才证明 TSE 有改善空间。

停止条件：P2 输出与 P1 manifest 对不上、TSE 输出严重削波/全零或 ASR 配置哈希不同，立即退回，不跑全量。

### 8 月 12 日：当前 TSE 的限时裁决

交付：`ACCEPT_B1_CANDIDATE` 或 `REJECT_CURRENT_TSE`。

1. 对修复后的现有 checkpoint 跑成对 smoke。
2. 若方向改善，P2 再提供至少 3 个正式种子；P3 不参与看 A 调参。
3. 若高重叠 CER 不改善，或 SINGLE 恶化 >2 pp，立即拒绝当前 TSE，不用阈值或文本纠错掩盖。
4. `target_enhancer_musan_rirs.pt` 和 denoised mix 保持默认关闭，不再消耗全量 A 预算。

### 8 月 13 日：三种子确认与错误分桶

交付：B1 配对报告草案。

- 每 seed 和三 seed 均值；
- 总体/SINGLE/OVERLAP/100%-overlap/-5dB；
- S/D/I/N 和 95% 配对 bootstrap 区间；
- 删除激增、数字、英文、短句、长句、削波、近零输出、残余泄漏等桶；
- 至少 2/3 seeds 同方向改善。

### 8 月 14 日：一次冻结候选的 DatasetA 阶段验证

前提：外部 B1 已通过，模型、参数、文本规范、门控均已冻结。

只运行两条公平命令；唯一变量为 `--enhancer-model`（针对 target enhancer）或经过修复的 `--tse-model`（针对双输出 TSE 独立入口）。不得使用 A 标签短语库、意图纠错、ASR/embedding cache 或 A 上训练的 gate。

```powershell
# 基线
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --out outputs\p3_freeze_20260814\datasetA_baseline.json

# target_enhancer 对照；enhancer-dir 必须是新的空目录，正式计时不得复用增强缓存
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --enhancer-model models\<frozen_candidate>.pt `
  --enhancer-dir outputs\p3_freeze_20260814\enhanced_fresh `
  --out outputs\p3_freeze_20260814\datasetA_enhancer.json
```

报告 CER、S/D/I/N、pos 接收率、RR、错误状态、总耗时、单条 P50/P95、CPU 内存、CUDA peak。A 的结果不得再反向修改模型或阈值。

### 8 月 15 日：冻结 P3 ASR 核心

发布 `releases/P3_ASR_CORE_v1/`：

- `src/`、`configs/`、`tests/`；
- FunASR commit、模型 revision 和资产 SHA256；
- `recognize_api.md`；
- `text_normalization_spec.md`；
- B0/ORACLE/B1 逐条预测和报告；
- `environment.txt`；
- `VERSION`、`SHA256SUMS`、`KNOWN_LIMITATIONS.md`；
- 从干净目录运行 10 条 ASR、黄金 CER、B0/B1 小报告的复现日志。

如果 B1 未通过，冻结内容应诚实记录 `REJECT_CURRENT_TSE`，主链回退到公平 baseline，而不是带着无效增强器进入冻结。

---

## 8. 8 月 16 日后的工作（不阻塞核心冻结）

### 8.1 8 月 16–21 日

- 仅在 B1 已通过且有明确 TSE 伪影时考虑 ASR-friendly loss/蒸馏。
- Paraformer 保持冻结；P3 只提供教师或特征，P2 训练。
- 接受条件：高重叠 CER 再下降至少 2 pp，SINGLE 和罕见字符 CER 恶化不超过 1 pp。
- 8 月 21 日后无稳定收益则删除蒸馏。

### 8.2 8 月 16–23 日

- 配合 P1/P4 做 A 三折分组 OOF；P3 资产 hash 跨折不变。
- P3 只生成冻结 raw text/normalized text/ASR features，不看外层结果改 ASR。
- 所有 A 训练门控和标签短语实验与正式 OOF 目录物理隔离。

### 8.3 8 月 24–30 日

- 与 P6 完成 FP32/FP16/ONNX 逐条文本和 CER 回归。
- 固定 100 条代表样本，报告 P50/P95、显存和失败率。
- 导出版本 CER 变化超过冻结容差则回退 FP32。

### 8.4 8 月 31 日–9 月 1 日

- 在干净目录完整复现；
- 9 月 1 日冻结模型、规则、阈值、依赖和推理链；
- 之后只修部署缺陷，不再训练或看 B 调参。

---

## 9. 最终验收表

| 验收项 | 必须通过 |
|---|---|
| CER 正确性 | 逐条和汇总 S/D/I/N；18 类黄金样例；P1 独立复核 |
| ASR 一致性 | B0/ORACLE/B1 使用同模型、同参数、同文本规范、同 sample_id |
| B1 高重叠 | CER 相对下降 ≥15%或绝对下降 ≥5 pp |
| SINGLE 保护 | CER 恶化 ≤2 pp |
| 极端重叠 | 100% overlap、SIR=-5 dB 方向不反转 |
| 随机性 | 至少 2/3 seeds 改善，并报告配对区间 |
| A 公平性 | A 不进训练/短语库/选阈；只对冻结候选做阶段验证或严格 OOF |
| 负样本 | 不进 CER；RR 独立报告；ERROR 不得冒充正确拒识 |
| 效率 | 完整链路 P50/P95、内存、CUDA peak；正式计时不复用缓存 |
| 可复现 | commit、模型、数据、配置、输出 hash 完整；干净目录复现 |

---

## 10. 向团队汇报时建议只说这五句话

1. “CER=0.30 是 30% 字错误率；SV threshold=0.30 是门控阈值，两者不是一个指标。”
2. “当前公平基线是 CER 53.43%、RR 91.14%；ASR-only 的 CER 是 39.57%，门控误拒贡献了大量删除。”
3. “现有 target enhancer 在相同全量配置下把 CER 恶化到约 61.35%，所以默认关闭。”
4. “双输出 TSE 的 346.02% 是评测器元组接口错误导致的无效数值，修复前不评价模型。”
5. “13 维门控的 16% 结果使用了 DatasetA 训练/选阈且记录互相矛盾，不进入正式公平报告；下一步只做外部确认集配对 CER，再一次性验证 A。”

---

## 11. 当前建议的唯一下一步（2026-08-12）

**由 P2 把微信群 v2.2 包中的实际脚本变化提交到可合并分支，并交付一个不同于当前失败权重的新候选；P3 随后先运行小样本真实测量，再决定是否启动 6000 条配对评测。**

P1 的正式评测数据已经完成，不需要重做 18000 个 WAV。若要求从合并后的 P1 v2 builder 复现数据，P1 还需补齐 `speech/noise/rir × train/dev/holdout` 共九份 split manifest，或提供能从公开源生成它们的已测试命令。P1 v3 当前只有 schema/provenance 文档，还没有对应可执行 builder。

P3 侧已经完成：

- 两个队友分支已无冲突合并，P3 源码已提交；
- P1→P3 的 6000 条规范配对 manifest 已实际重建，18000 个必需音频缺失 0；
- B0/ORACLE/B1 配对 runner、逐条 S/D/I/N、错误/RR 校验和能量诊断；
- 1 条真实 P1→P2→P3 GPU 端到端冒烟；
- 49 项回归测试。

在新 P2 候选通过真实小样本前：

- 不继续扫 DatasetA 阈值；
- 不把 13 维门控的 16% 当成果；
- 不把 `denoiser_best.pt` 直接塞给 `--enhancer-model`；
- 不运行昂贵的全量 DatasetA；
- 不启用当前 target enhancer 或双输出 TSE 进入默认提交链；
- 不把当前 B3 的“可加载”解释为识别质量通过。

---

## 12. 执行更新（2026-08-13 18:54）

P2 修复代码已合入 `main`。B1 500-step CUDA 试训为 PASS，但提交 `3a62e7c` 的首轮 20000-step 正式训练于 17:28 判定 FAIL：

- checkpoint step20000 与 strict restore 正常；
- 6 次非有限梯度均正确跳过 optimizer 更新，说明安全修复确实生效；
- AMP scale 从 512 增长到 8192，6 次 backoff 后最终为 2048；
- 训练误用了 P1 v3 中的 enroll-swap target-present 子集，最终 dev SI-SDR 为 `-0.01189 dB`。

提交 `c92f510` 已修正为 P1 v2 普通 PRESENT（100000 train / 10000 dev），并将 B1/B3 的 20k AMP growth interval 设为 100000，使 scale 保持在已通过试训的 512。专项回归为 `10 passed, 1 skipped`。修正版 B1 已于 18:53 在 AutoDL `63d44c988c-0e1a4480` 启动，输出：

```text
/root/autodl-tmp/P2_retrain_20260813/B1_FORMAL_20000_AMP512_P1V2_20260813
```

当前执行顺序不变：修正版 B1 最终验收通过后，才依次训练 B3 三个种子；每个候选先跑 20 正 + 20 负，至少 2/3 通过后运行 6000 条外部配对 CER，最后才决定是否执行一次冻结候选的 DatasetA 全量比较。

22:43 状态检查：修正版 B1 已到 step `16000/20000`，`checkpoint_step16000.pt` 已保存，当前 step 16000 的 10,000 条 dev 全量评估约完成 `6000/10000`。step 2000 至 12000 的 `dev_sisdr` 总体从 `3.134` 提升到 `4.921 dB`，但 step 14000 出现 `dev_loss=NaN`（`dev_sisdr=5.302 dB`）。step `5832` 的第二次非有限梯度已正确跳过 optimizer 更新，GradScaler 从 `256` 回退到 `128`；累计两次梯度异常均被安全处理。另有 21 次非有限训练 loss 被跳过，分布于 step `11573–15987`。检查瞬间 GPU `0%`、显存 `2359 MiB`，三个训练相关进程仍在，处于 dev 数据处理阶段。现有验收规则下已有明确失败证据，B3 不放行；继续 B1 只为取得最终 checkpoint、verdict 与 strict restore，不修改配置或启动重复任务。

00:33 最终状态：B1 已完成 `20000/20000`，最终 `dev_loss=NaN / dev_sisdr=5.6227000603 dB`，`checkpoint_step20000.pt` 已生成，strict restore `max|Δ|=0` 为 `PASS`。完整 verdict 为 `FAIL`：53 个非有限 loss step，2 个非有限 gradient step（均安全跳过），optimizer `19945 applied / 55 skipped`，AMP scale 最终为 `128`，且 `loss_last_100_mean=NaN / loss_decreasing=false`。因此 B3 三种子、P3 20+20、6000 paired CER 和 DatasetA 全量比较全部保持未执行；后续必须先修复 B1 数值稳定性并重新取得总体 PASS。

资源收尾：AutoDL 实例 `63d44c988c-0e1a4480` 已于 00:35 确认进入“已关机”，未操作其他实例；每小时自动跟进 `p2` 已删除。
