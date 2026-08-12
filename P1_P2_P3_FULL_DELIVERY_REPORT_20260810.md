# P1/P2→P3 完整交付与接续报告

**报告日期：** 2026-08-10  
**群聊证据截止：** 2026-08-10 16:47  
**本地分支：** `codex/p3-integrate-pr3`  
**本地基线：** `2375254294a95bfc33dac8ad9d57e8f4c6d7572f`  
**交付形式：** 单一纯 Markdown 文件；本目录原 `artifact.json` 已由本报告取代  
**当前总状态：** `HOLD_PENDING_UPSTREAM_ARTIFACT_VALIDATION`

## 技术结论

P3 的代码、评测契约、P1 清单转换、P2 冻结接口、输出能量诊断和回归测试已经完成；2026-08-10 本地复跑为 **49/49 tests PASS**。因此，P3 当前属于“实现就绪”。

15:34 之后的微信群聊改变了旧报告对上游工作的描述：P2 队友在处理 B3 退化问题后，于 16:13 明确回复“我这边修改完了”；P1 队友确认 18,000 条 WAV 约 2.9 GB，并于 16:47 分享了 `P1_B1_EXTERNAL_AUDIO_FORMAL_20260810.tar.gz`。因此，P1/P2 的修复或打包工作应记为“团队已完成”，不能继续写成“尚未处理”。

但这两项新产物目前都没有进入本工作区，也没有完成 P3 的 SHA256、文件数量、可解码性、CUDA 前向、输出能量和配对 CER 复验。旧的本地 P1 包仍然只有 manifest；旧的 P2 `checkpoint_step20000.pt` 仍然是 40/40 近静音的被拒候选。故当前只能把新交付记为“群聊确认完成、P3 待验收”，不能写成“正式链路已放行”。

正式执行顺序不变：**先验收 P1 完整音频与 P2 新 checkpoint，再跑外部 B0/ORACLE/B1；外部门通过后，才允许用同一公平配置运行一次 DatasetA 全量候选。**

## 1. 更新后的总状态

为避免把“队友做完”与“P3 验收通过”混为一谈，本报告使用三种证据状态：

- **本地已验证：** 当前工作区存在产物，且已运行检查或测试。
- **群聊已完成、待验：** 队友已明确完成或分享，但产物尚未进入当前工作区复核。
- **未放行：** 尚不能生成或宣称正式指标。

| 环节 | 团队执行状态 | P3 验收状态 | 当前结论 |
| --- | --- | --- | --- |
| P1 `D_single.jsonl` / `D_overlap.jsonl` | 已交付 | 本地已验证 | 6000 条、6000 个唯一 `sample_id`；`D_single=2000`、`D_overlap=4000` |
| P1 18,000 条正式 WAV | 已打包并于 16:47 分享 | 待下载、哈希与齐备性验收 | 不再要求 P1 重复生成；先验收新包 |
| P2 旧 B3 checkpoint | 已交付 | 本地已验证但质量门失败 | 保留为 `REJECT_CURRENT_P2_B3_CHECKPOINT` 回归证据 |
| P2 B3 修复 | 队友于 16:13 确认修改完成 | 新 checkpoint、配置和日志尚未收到 | 修复动作完成；接入状态仍待验 |
| P3 CER/文本/RR 契约 | 完成 | 本地已验证 | 可用于正式配对评测 |
| P3 P2 runtime 与能量诊断 | 完成 | 本地已验证 | 可校验 SHA、加载、CUDA、restore、RMS 与近静音 |
| P3 外部 B0/ORACLE/B1 全量 | 等待上游新产物 | 未运行 | 仍未放行 |
| P3 公平 DatasetA 全量候选 | 按计划停止 | 未运行 | 仅在外部门通过后运行 |
| P3 commit / push / PR | 未执行 | 工作树仍有未提交改动 | 评测结果冻结后再提交 |

## 2. 15:34 后群聊记录对旧结论的修正

以下内容来自群聊“挑战杯（无老师版）”，用于确认团队动作是否完成；它是交付状态证据，不代替本地技术验收。

| 时间 | 群聊事实 | 对交付报告的影响 |
| --- | --- | --- |
| 15:40 后 | 队友指出当前近静音结果与其修复前现象一致 | 说明本地使用的仍是旧候选，不能据此否定尚未接收的新修复 |
| 16:01 | P2 队友表示先处理 B3 的退化问题，并询问本地数据环境 | P2 已进入针对性修复，而非尚未开始 |
| 16:13 | P2 队友明确回复“我这边修改完了”，并附 dev SI-SDR 说明截图 | P2 修复动作记为完成；P3 下一步改为接收并复验新候选 |
| 16:21—16:36 | P1 队友询问 18,000 条 WAV 的传输方式，确认体积约 2.9 GB；团队决定使用网盘 | P1 已完成音频生成，剩余工作是传输与验收 |
| 16:47 | P1 队友通过夸克网盘分享 `P1_B1_EXTERNAL_AUDIO_FORMAL_20260810.tar.gz` | P1 正式音频改记为“已分享、待 P3 下载验收” |

相对于原 `artifact.json`，本报告做了四项状态更新：

1. P1 不再是“尚未补齐音频”，而是“正式音频包已分享，尚未被 P3 验收”。
2. P2 不再是“尚未修复”，而是“修复已由队友确认完成，尚未交付可复核的新 checkpoint 证据包”。
3. 旧 B3 的拒绝结论继续有效，但只针对 SHA256 为 `5c351097…afd68d` 的旧 checkpoint。
4. P3 的阻塞位置从“等待上游开始工作”后移到“等待上游产物传输和验收”。

## 3. P3 已经完成的交付

### 3.1 代码、契约与评测入口

| 交付物 | 已完成能力 | 当前验证 |
| --- | --- | --- |
| [`cer.py`](../../cer.py) | 严格字符串输入；逐条及语料级 `S/D/I/N/CER` | 空串、非字符串、UTF-8、聚合计数测试通过 |
| [`asr_demo.py`](../../asr_demo.py) | 统一 `recognize()` 的文本与耗时返回契约 | 防止把返回元组直接交给 CER |
| [`eval_datasetA_tse_dual.py`](../../eval_datasetA_tse_dual.py) | 修复旧双输出 TSE 的 ASR 元组及负样本判断错误 | 旧 `346.02%` CER 已明确作废 |
| [`p3_eval_contracts.py`](../../p3_eval_contracts.py) | 统一结果有效性、错误计数与样本身份检查 | 重复/缺失 ID、异常状态、拒绝样本测试通过 |
| [`p2_tse_runtime.py`](../../p2_tse_runtime.py) | 冻结加载 P2 `extract_target()`、SHA 校验、CUDA 与输出能量诊断 | 旧 B3 构造 1,349,891 参数，CUDA forward/restore 通过 |
| [`build_p3_paired_manifest.py`](../../build_p3_paired_manifest.py) | 将 P1 两份 manifest 转成 P3 规范入口，并检查三类音频 | 6000 条唯一 ID；支持 `--require-audio` 硬门 |
| [`eval_paired_cer.py`](../../eval_paired_cer.py) | 同一 `sample_id` 的 B0/ORACLE/B1 配对 CER | 支持 P2 checkpoint、SHA、CUDA、逐条 JSONL 与分场景汇总 |
| [`eval_datasetA.py`](../../eval_datasetA.py) | 公平 DatasetA 的 P2 或 enhancer 前端对比 | 两种前端互斥；输出 RMS 与近静音统计 |

### 3.2 已生成的清单与诊断证据

| 证据 | 结果 | 解释 |
| --- | --- | --- |
| [`paired_manifest.jsonl`](../../outputs/p3_external_gate_20260810/paired_manifest.jsonl) | 6000 条；SHA256 `69f1915f…69e629` | P1→P3 的规范评测入口已经生成 |
| [`manifest_readiness.json`](../../outputs/p3_external_gate_20260810/manifest_readiness.json) | 旧本地快照：三类 WAV 各缺 6000，共缺 18,000 | 只描述新音频包进入工作区之前的状态 |
| [`datasetA_p2_limit20.json`](../../outputs/p3_checkpoint_preflight_20260810/datasetA_p2_limit20.json) | 20 正 + 20 负；40/40 近静音；正样本接收率 0%；诊断 CER 100% | 只用于拒绝旧 B3，不是正式 DatasetA 成绩 |
| [`P3_METRICS_REGISTRY.json`](../../docs/P3_METRICS_REGISTRY.json) | CER、RR、拒绝样本、配对与有效性契约已冻结 | 群聊中另行发送的 ZIP 是这一结果的传输副本 |
| 单元/契约测试 | 49/49 PASS | 证明代码与契约可运行，不证明新模型质量 |

### 3.3 P3 仍未完成的部分

- 新 P1 音频包的下载、SHA256、归档安全、文件数量、可解码性和 manifest 哈希复核。
- 新 P2 修复 checkpoint 的接收、身份校验、CUDA 前向、输出能量与小样本质量门。
- 外部 6000 条 B0/ORACLE/B1 全量配对评测。
- 外部门通过后的公平 DatasetA 全量比较。
- 官方 scorer 复核（若主办方提供）、最终证据冻结、commit、push 和 PR。

这些未完成项是上游新产物尚未进入 P3 验收造成的，不是 P3 runner 缺失。

## 4. P2 接下来必须交付什么

### 4.1 当前 P2 结论

旧 `checkpoint_step20000.pt` 的文件身份、加载和 CUDA 工程路径均通过，但质量门失败：

- checkpoint SHA256：`5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d`。
- 旧预检的输出/输入 RMS 比为 `2.01e-7`—`2.28e-7`，远低于 `1e-4` 近静音阈值。
- 20 个正样本和 20 个负样本全部输出近静音；正样本接收率 0%，诊断 CER 100%。
- 旧 B3 日志末尾 `dev SI-SDR=-19.2332 dB`；RR 100% 是全静音造成的错误收益。

群聊已经确认 P2 做完修复，但没有新文件时，P3 无法判断修复是否真正消除了 PRESENT 坍塌。因此 P2 当前任务不是再次口头解释旧指标，而是交付一个可复核的新候选包。

### 4.2 P2 最小交付包

P2 应使用新目录和新版本号，禁止覆盖旧 B3 证据。建议至少包含：

| 必交内容 | 最低要求 | P3 用途 |
| --- | --- | --- |
| 新 checkpoint | 文件名含候选版本或训练时间 | 加载、前向和正式 B1 生成 |
| `checkpoint.sha256` | 对新 checkpoint 的独立 SHA256 | 传输后身份复核 |
| `config.yaml` + `config.sha256` | 与 checkpoint 内嵌 `cfg` 逐字段一致 | 防止错误配置加载 |
| `data.sha256` 与训练/dev manifest | 能独立复算，不只给来源声明 | 验证训练数据版本与隔离 |
| 代码 commit | 精确 Git commit，附修复文件清单 | 审计“修改完了”具体对应哪些代码 |
| 训练命令与随机种子 | 参数、环境、种子完整 | 可复现训练 |
| 完整训练日志 | loss、restore、NaN、显存、吞吐 | 工程健康检查 |
| 分场景 dev 指标 | PRESENT、ABSENT、ENROLL-SWAP、SINGLE、重叠桶、100% overlap、SIR=-5 dB | 验证修复没有用 ABSENT 收益掩盖 PRESENT 退化 |
| 小样本质量报告 | 分开给 input/output RMS、peak、near-silent、SI-SDR、正样本接收率与诊断 CER | 在全量 ASR 前拦截再次坍塌 |
| `change_note.md` | 写明根因、修改点、与旧 B3 的对照 | 把群聊说明转成可审计交付 |

### 4.3 P2→P3 接入门槛

| 门槛 | 通过标准 | 当前状态 |
| --- | --- | --- |
| 资产身份 | checkpoint/config/data/commit 全部可追溯，SHA256 一致 | 新候选待交付 |
| 模型加载 | CPU 加载、CUDA forward、restore、形状与有限值通过 | 新候选待验 |
| PRESENT 非静音 | PRESENT 输出不得全部触发 `output_rms / input_rms < 1e-4` | 旧 B3 失败；新候选待验 |
| 基本可用性 | 正样本接收率不得为 0%，诊断 CER 不得为 100% | 旧 B3 失败；新候选待验 |
| ABSENT 抑制 | ABSENT 能量有效压制，同时不能把 PRESENT 一并压成静音 | 新候选待验 |
| SINGLE 护栏 | B1 相比 B0 的 corpus CER 恶化不超过 2 个百分点 | 待外部集 |
| 高重叠收益 | 高重叠桶相对 CER 下降至少 15%，或绝对下降至少 5 个百分点 | 待外部集 |
| 极端场景 | 100% overlap / SIR=-5 dB 方向不得相对 B0 反转 | 待外部集 |
| 多种子复现 | 至少 3 个固定种子中 2 个通过收益与护栏 | 尚未提供 |

`P2_project/verify_p2_local.py` 当前硬编码旧 B3 路径和旧 SHA，只能验证旧资产，不应被当成新候选的唯一验收。新候选必须通过 P3 runner 指定的新路径与新 SHA 复验。

## 5. P1 接下来如何工作

### 5.1 P1 已完成的工作

P1 已经完成两类交付：

1. 两份冻结 manifest 已进入本地，并通过 6000 条 ID、参考文本与 12,203 个源素材的哈希检查。
2. 群聊确认 18,000 条正式 WAV 已生成，约 2.9 GB，并在 16:47 分享正式音频归档。

因此，P1 下一步不应默认重新生成 18,000 条音频。正确流程是等待 P3 验收；只有具体文件缺失、损坏或哈希不符时，再做定点补包或重建。

### 5.2 P1 现在需要补齐的交付元数据

- 为 `P1_B1_EXTERNAL_AUDIO_FORMAL_20260810.tar.gz` 提供归档级 SHA256。
- 提供归档文件清单及 `mixture/target/enroll` 各自数量。
- 保留 `VERSION`、`FROZEN`、`SCHEMA`、`README`、生成器版本、固定随机种子和依赖锁。
- 提供逐文件 `SHA256SUMS.txt`，并确认与两份冻结 manifest 的输出哈希一致。
- 保持网盘链接在 P3 验收完成前可访问；若传输中断，只补传损坏分卷，不重新改变样本内容。
- 保留 train/dev/confirm 与 external holdout 的 speaker/hash 去重证明，确保没有 DatasetA 标签泄漏或同源污染。

### 5.3 P1 验收标准

| 检查 | 必须结果 |
| --- | --- |
| 样本身份 | 6000 个唯一 `sample_id`；与旧冻结 manifest 完全一致 |
| 混合音频 | `audio/mixture/<sample_id>.wav` 共 6000 个，可解码 |
| Oracle 目标音频 | `audio/target/<sample_id>.wav` 共 6000 个，可解码 |
| 注册音频 | `audio/enroll/<sample_id>.wav` 共 6000 个，可解码 |
| 路径与哈希 | 无重复覆盖、路径穿越；逐文件 SHA256 全部匹配 |
| P3 硬门 | `missing_total=0` 且 `ready_for_paired_eval=true` |

特别注意两个名字相近但内容不同的包：

- 当前本地旧包 `P1_B1_external_holdout_manifest_formal_20260810.tar.gz` 约 23.6 MB，SHA256 为 `99e3d5c3…1574f1`，只含 manifest 等安全条目，不含正式 WAV。
- 群聊新包 `P1_B1_EXTERNAL_AUDIO_FORMAL_20260810.tar.gz` 约 2.9 GB，声称包含 18,000 条正式 WAV；尚未进入当前工作区，不能用旧包的 SHA 或齐备报告替代其验收。

## 6. P3 收到新产物后的执行顺序

### 第一步：隔离接收，不覆盖旧证据

- 新 P1 音频解压到新的、明确命名的数据目录。
- 新 P2 checkpoint 放入新的候选目录。
- 先记录归档/checkpoint SHA256、文件大小和接收时间。
- 旧 P1 manifest-only 包、旧 B3 checkpoint 和旧失效报告全部保留，用于回归对照。

### 第二步：P1 音频硬门

```powershell
..\.venv\Scripts\python.exe build_p3_paired_manifest.py `
  --p1-manifest data\p1_b1_external_holdout_formal_20260810\manifests\D_single.jsonl `
  --p1-manifest data\p1_b1_external_holdout_formal_20260810\manifests\D_overlap.jsonl `
  --transcript data\public\aishell1\extracted\data_aishell\transcript\aishell_transcript_v0.8.txt `
  --data-root data\p1_b1_external_holdout_formal_20260810 `
  --output outputs\p3_external_gate_formal\paired_manifest.jsonl `
  --report outputs\p3_external_gate_formal\readiness.json `
  --require-audio
```

停止条件：命令非零退出，或报告不是 `rows=6000`、`unique_sample_ids=6000`、`missing_total=0`、`ready_for_paired_eval=true`。

### 第三步：P2 新候选的 20+20 质量预检

```powershell
$p2CheckpointPath = 'P2_project\artifacts\final\P2_artifacts\B3_FIXED\checkpoint.pt'
$p2CheckpointSha = 'NEW_CHECKPOINT_SHA256'

..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --limit 20 `
  --decision-policy hard `
  --sv-threshold 0.30 `
  --no-intent-filter `
  --no-phrase-correct `
  --no-wake-guard `
  --p2-tse-checkpoint $p2CheckpointPath `
  --p2-tse-sha256 $p2CheckpointSha `
  --p2-tse-device cuda `
  --p2-tse-dir outputs\p3_checkpoint_preflight_fixed\targets `
  --out outputs\p3_checkpoint_preflight_fixed\datasetA_p2_limit20.json
```

停止条件：加载/restore/CUDA 失败、非有限值、PRESENT 全部近静音、正样本接收率为 0%、诊断 CER 为 100%，或 ABSENT 抑制以 PRESENT 坍塌为代价。

### 第四步：外部 B0/ORACLE/B1 全量配对评测

```powershell
..\.venv\Scripts\python.exe eval_paired_cer.py `
  --manifest outputs\p3_external_gate_formal\paired_manifest.jsonl `
  --data-root data\p1_b1_external_holdout_formal_20260810 `
  --out-dir outputs\p3_external_gate_formal\p2_fixed_candidate `
  --p2-tse-checkpoint $p2CheckpointPath `
  --p2-tse-sha256 $p2CheckpointSha `
  --device cuda
```

输出必须包括逐条预测、`S/D/I/N`、B0/ORACLE/B1 corpus CER、场景分桶、RMS/近静音、耗时、配置和哈希。若 SINGLE、高重叠、极端场景、ABSENT 或多种子任一硬门失败，立即停止，不跑 DatasetA 全量。

### 第五步：仅在外部门通过后运行公平 DatasetA

```powershell
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard `
  --sv-threshold 0.30 `
  --no-intent-filter `
  --no-phrase-correct `
  --no-wake-guard `
  --p2-tse-checkpoint $p2CheckpointPath `
  --p2-tse-sha256 $p2CheckpointSha `
  --p2-tse-device cuda `
  --p2-tse-dir outputs\datasetA_p2_fixed_formal\fresh_targets `
  --out outputs\datasetA_p2_fixed_formal\report.json
```

正式计时不得使用 ASR 或 embedding 缓存。`--enhancer-model` 与 `--p2-tse-checkpoint` 互斥；若保留 target enhancer 作为独立候选，必须在另一次相同配置运行中单独比较。

### 第六步：冻结与入库

1. 用官方 scorer 复核（如提供），记录本地/官方差异。
2. 冻结最终报告、命令、环境、哈希、逐条预测和汇总。
3. 审阅当前未提交的 P3 改动。
4. 只提交代码、小型元数据和文档；音频、模型、大型输出不进 Git。
5. 完成 commit、push 和 PR。

## 7. 指标定义与公平口径

- **B0：** 原始 mixture 的 ASR 结果。
- **ORACLE：** P1 target 音频的 ASR 上界参考。
- **B1：** P2 新 checkpoint 输出 target 的 ASR 结果。
- **Corpus CER：** `sum(S + D + I) / sum(reference characters)`，不是逐句 CER 的平均。
- **正样本被拒绝：** 空假设导致整句参考字符计为删除。
- **负样本：** 参考为空，不进入 CER，单独使用 `RR = rejected_negatives / all_negatives`。
- **正样本接收率：** 被门控接受并进入 ASR 的正样本占比。
- **近静音：** 当前诊断阈值为 `output_rms / input_rms < 1e-4`。
- **配对约束：** B0/ORACLE/B1 必须使用同一 `sample_id`、参考文本、ASR、文本规范和评分实现。
- **DatasetA 公平配置：** `hard`、SV threshold `0.30`、关闭 intent filter、phrase correction 和 wake guard；不使用测试标签调参或构造 phrase bank。

任何报告必须同时给出 CER 小数/百分比以及 `errors/ref_chars`，并同时报告 RR 与正样本接收率。不得把近静音造成的 RR 提升写成模型收益。

## 8. 现有历史结果的正确用途

| 候选 | 已有结果 | 证据级别 | 当前决策 |
| --- | --- | --- | --- |
| 硬 CAM++ 公平回退基线 | CER 53.43%；RR 91.14%；正样本接收率 69.35% | 同一全量 DatasetA 冻结配置 | 保留为当前可信回退 |
| `target_enhancer_musan_rirs.pt` | CER 61.35%；RR 93.67%；正样本接收率 61.07% | 同配置全量对比 | CER 恶化 7.92 个百分点，拒绝默认接入 |
| `target_enhancer_denoised_mix.pt` | CER 62.45% | 已有全量报告 | CER 恶化 9.02 个百分点，拒绝 |
| 旧双输出 TSE | CER 346.02% | 旧 evaluator 把 `recognize()` 元组直接交给 CER | 作废，不用于模型判断 |
| 旧 P2 B3 checkpoint | 20+20：CER 100%；RR 100%；40/40 近静音 | GPU 小样本失效诊断 | 拒绝旧候选；不是正式成绩 |
| 群聊所述 P2 修复候选 | 尚无本地数值 | 只有完成声明和截图 | 等待产物与 P3 复验 |

这些结果来自不同阶段或不同有效性等级，不能拼接成一组正式排行榜。当前 53.43% 只承担回退基线作用；新 P2 候选必须按本报告的外部门重新比较。

## 9. 限制、风险与稳健性检查

- **群聊不是可执行产物。** “修改完了”和“已分享”证明团队动作完成，但不能证明 SHA、结构、训练复现或指标通过。
- **本地证据仍是旧快照。** 当前 P1 齐备报告的 18,000 缺失和 P2 的 40/40 近静音都发生在新交付进入工作区之前。
- **P2 修复细节尚不可审计。** 群聊截图说明了 dev SI-SDR 的解释，但缺少新 commit、配置、checkpoint、日志和分场景数值。
- **49 项测试不等于模型通过。** 测试覆盖代码契约、I/O、CUDA smoke 与错误处理，不覆盖新候选的真实语音质量。
- **正式指标仍为空。** 尚无新候选的外部 B0/ORACLE/B1，也没有被允许运行的公平 DatasetA 全量结果。
- **多种子证据未完成。** 单次修复结果不足以判断稳定性，至少需要 3 个固定种子中 2 个通过。
- **仓库尚未冻结。** 当前分支存在未提交 P3 改动；正式结果和文档审阅前不应仓促提交。
- **本报告不绘制趋势图。** 当前证据是一次性状态、哈希和硬门结果，没有连续时间序列；表格比趋势图更不容易把小样本诊断误解为性能走势。

## 10. 责任人、下一动作与完成定义

| 责任方 | 下一动作 | 完成定义 | 失败时处理 |
| --- | --- | --- | --- |
| P1 | 补充新归档 SHA、清单与生成元数据；保持分享有效 | P3 下载后 `missing_total=0`、三类各 6000、全部可解码且哈希匹配 | 只补缺失/损坏文件；若输出哈希整体不一致，再用冻结 builder 重建 |
| P2 | 交付修复后的新 checkpoint 证据包 | 新 SHA/配置/commit 可追溯；CUDA 与 20+20 质量门通过 | 保留旧/失败候选，定位 PRESENT/ABSENT 损失与采样权重，不覆盖证据 |
| P3 | 先验收 P1/P2，再跑外部 B0/ORACLE/B1 | 6000 条配对完整，并给出分场景和多种子 accept/reject | 任一硬门失败即停止，不运行 DatasetA 全量 |
| P3/团队 | 外部门通过后运行公平 DatasetA 并复核 | 关闭缓存、同一配置、结果和哈希冻结 | 若新候选不优于基线，回退硬 CAM++ 基线 |
| 仓库维护 | 审阅、提交、推送、创建 PR | 代码/小型证据入库，大文件留在约定存储 | 发现未追踪大文件或标签泄漏时停止提交 |

## 11. 进一步需要团队确认的问题

1. P2 新修复候选的文件名、存储位置、checkpoint SHA256 和对应 Git commit 是什么？
2. P2 是否已经按 PRESENT/ABSENT/ENROLL-SWAP 和重叠强度输出分场景 dev 指标？若没有，需在交付前补齐。
3. P1 新归档是否自带逐文件 `SHA256SUMS.txt`，其目录名是否与现有 manifest 路径完全一致？
4. P2 是否接受至少 3 个固定种子的外部门；若只交 1 个种子，结果只能记为阶段候选，不能正式放行。
5. target enhancer 是否继续保留为独立候选？若保留，必须与 P2 TSE 分开运行，不能同时启用两个前端。

## 12. 最终交付结论

**P3 当前已完成并可交付：** 评测实现、6000 条规范清单、P2 冻结运行时、GPU 失效诊断、指标注册表、接入文档以及 49 项通过的测试。

**团队在 15:34 后已完成：** P2 B3 修复动作；P1 18,000 条正式 WAV 的生成、打包与网盘分享。

**当前仍不可宣称：** 新 P2 已通过 P3、正式外部 B1 改善、或公平 DatasetA 全量候选成绩。原因不是 P1/P2 没有行动，而是新产物尚未被 P3 接收和复验。

**恢复口令：** `P1_NEW_AUDIO_ACCEPTED + P2_NEW_CHECKPOINT_ACCEPTED` → P3 外部 B0/ORACLE/B1 全量 → 门槛通过 → 一次性公平 DatasetA → 冻结报告 → commit/push/PR。

## 附录 A：当前冻结证据登记

| 对象 | SHA256 / 状态 | 位置 | 含义 |
| --- | --- | --- | --- |
| 旧 P2 B3 checkpoint | `5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d` | `P2_project/artifacts/final/P2_artifacts/B3/checkpoint_step20000.pt` | 身份通过、质量拒绝 |
| 旧 P2 config | `8c8dd51dfc18635c734755d85deeb48674c31f20c0ce392d6b5a0ba0df62e253` | `config.sha256` / B3 config | 原始配置哈希；checkpoint 内嵌配置字段已核对 |
| 旧 P2 data 来源声明 | `03471f980403cc92163b3dbc835b359b939f32b02796586bd5bdaa04b4dac0b7` | `data.sha256` | 原训练 manifest 未随旧包提供，不能独立复算 |
| 旧 P1 manifest-only 归档 | `99e3d5c3e69c0a411cad4af47f9b8c71e0acda3b46733d278c0d457b671574f1` | `P1_B1_external_holdout_manifest_formal_20260810.tar.gz` | 含 manifest，不含正式 WAV |
| P3 paired manifest | `69f1915f15361511b5da768e97183a841b01d683eb944a793f7250867699e629` | `outputs/p3_external_gate_20260810/paired_manifest.jsonl` | 6000 条规范入口 |
| P3 readiness | `9ad40f8d6e184eac8422db8a962491f153beab4ed53d65d0b4b1f9dce2ab5e1d` | `outputs/p3_external_gate_20260810/manifest_readiness.json` | 新音频到位前的缺失快照 |
| 旧 P2 GPU preflight | `7b719b049102a0daff22ff13fc119edfa1a82c8f4bee3637ce8c1cf1e1a676ab` | `outputs/p3_checkpoint_preflight_20260810/datasetA_p2_limit20.json` | 40/40 近静音失效证据 |
| 仓库基线 | `2375254294a95bfc33dac8ad9d57e8f4c6d7572f` | `codex/p3-integrate-pr3` | P2 PR #3 合并后的 P3 实施起点 |
| 新 P1 正式音频包 | 群聊已分享；本地 SHA 待补 | `P1_B1_EXTERNAL_AUDIO_FORMAL_20260810.tar.gz` | 约 2.9 GB，待 P3 验收 |
| 新 P2 修复候选 | 群聊确认修改完成；文件/SHA 待补 | 待 P2 交付 | 不得借用旧 B3 的哈希或 PASS |

## 附录 B：报告依据

- [`P3_P2_B3_HANDOFF_AUDIT_20260810.md`](../../docs/P3_P2_B3_HANDOFF_AUDIT_20260810.md)：旧 P1/P2 本地接入审计与停止裁决。
- [`CER_P3_PLAN_REPORT_20260810.md`](../../docs/CER_P3_PLAN_REPORT_20260810.md)：P3 指标口径、历史结果、门槛与执行计划。
- [`P2_P3_INTEGRATION.md`](../../docs/P2_P3_INTEGRATION.md)：P2→P3 接入命令和参数约束。
- [`P3_METRICS_REGISTRY.json`](../../docs/P3_METRICS_REGISTRY.json)：机器可读的 CER/RR/有效性契约。
- [`manifest_readiness.json`](../../outputs/p3_external_gate_20260810/manifest_readiness.json)：新 P1 音频到位前的 6000 条清单与 18,000 缺失快照。
- [`datasetA_p2_limit20.json`](../../outputs/p3_checkpoint_preflight_20260810/datasetA_p2_limit20.json)：旧 P2 B3 的 20+20 GPU 近静音诊断。
- 同目录上一 Codex 任务“使用增强模型训练并评测”的近期内容：原完整报告 `artifact.json`、P3 计划以及用户追加的纯 Markdown 改写要求。
- 微信群“挑战杯（无老师版）”2026-08-10 15:34—16:47 的只读记录：P2 修复完成确认、P1 音频规模/传输讨论及正式归档分享。

群聊中的网盘链接和提取码未写入本报告；它们只用于接收新包，应以群聊原消息为准。
