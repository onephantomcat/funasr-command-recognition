# P1 / P2 / P3 修复与交付报告

**报告日期：** 2026-08-13

**集成分支：** `codex/p3-integrate-pr3`

**适用范围：** P1 数据、P2 双输出 TSE、P4 CAMPPlus、P3 ASR/CER 接入及 S004 真人录音
**最终裁决：** `CODE_FIXED / EXECUTION_SMOKE_PASS / FORMAL_P2_MODEL_PENDING`

## 技术摘要

- **代码问题已经修复并通过实际执行。** B3 的 ABSENT 损失缩放、零目标 MR-STFT 爆炸、PRESENT 幅度监督失效、残差监督错误、注册音频静默降级、错误 enrollment 猜测、非严格 warm-start、Windows P1 构建兼容和多 worker 随机裁剪等问题均已处理。
- **旧 B2/B3 权重仍然拒收。** `P2_DELIVERY_FOR_P3_20260811.tar.gz` 的文件 SHA256 与声明一致，但其中 B2/B3 仍复用 `13bce1b90a80710c8342fd28f0aaad90f244571be3be4c7469570992fd96db8b`。该权重已在 P3 20 正 + 20 负独立预检中出现 39/40 近静音、正样本接收率 0%、诊断 CER 100%，本轮代码修复不能追溯改变旧 checkpoint。
- **P1 不再阻塞 P3 数据入口。** 现有正式外部包已经提供 6000 条唯一配对记录和 18000 个 mixture/target/enrollment 音频引用，缺失 0；同时已补齐九份 split manifest 的本地生成入口，并在 Windows 双进程下真实生成 14 条跨场景预检音频。
- **P3 的代码与数据链已经就绪，只等待新 P2 候选。** P3 已完成 manifest 桥接、P2 runtime、B0/ORACLE/B1 配对 CER、输出能量诊断和异常语义保护。新候选到位后，应先做 20+20 小样本复验，再运行 6000 条外部配对评测；不能直接跳到 DatasetA 全量。
- **S004 原始录音内容通过自动预检。** `S004.zip` 含 35 个文件，命名正确；32/32 条命令逐字一致，3 条长句去标点后逐字一致。它仍是原始候选包，正式入库前需人工耳检首尾留白、补授权/环境记录，并另存 16 kHz 单声道 PCM16 WAV，原 M4A 不覆盖。

## 当前可交付状态

| 模块 | 当前状态 | 可交付内容 | 仍缺内容 |
|---|---|---|---|
| P1 外部确认集 | **READY** | 6000 条配对记录；2000 `D_single` + 4000 `D_overlap`；18000 个必需音频缺失 0 | 无需重新下载；只有要求从源码完整重建时才需运行 116000 条正式生成 |
| P1 生成复现 | **PREFLIGHT PASS** | 九份 split manifest 生成器、Windows/POSIX 可移植入口、14 条真实生成预检 | 尚未在本机完整生成 100k train + 10k dev + 6k confirmation |
| P2 源码 | **FIXED** | 损失、数据语义、CAMPPlus、训练、评测、诊断和测试修复 | 无代码级阻断 |
| P2 训练链 | **SMOKE PASS** | 真实 CAMPPlus B1 2-step、B1→B3 2-step strict warm-start、1-step DEBUG 完整反向 | 尚未完成正式 B1/B3 多种子训练 |
| P2 旧 B2/B3 checkpoint | **REJECT** | 可用于失败回归对照 | 不得进入 P3 正式链路 |
| P3 接入代码 | **READY** | P2 runtime、配对 CER、文本/CER 契约、能量诊断 | 等待不同于旧 SHA 的新 P2 checkpoint |
| S004 真人录音 | **RAW CANDIDATE PASS** | 35 条原始 M4A，文件名和朗读内容通过 | 人工耳检、授权/设备环境记录、规范 WAV 副本 |

## 接收内容与 Git 合并结论

### P1/P2 附件

- `P1_B1_EXTERNAL_AUDIO_FORMAL_20260810.tar.gz` 已在本地，大小 3,151,054,262 字节；其 6000 条正式外部配对入口已完成重建与齐备度检查。
- `P2_DELIVERY_FOR_P3_20260811.tar.gz` 已在本地，大小 139,160,455 字节，实算 SHA256 为：

```text
967BE7AD7FF889430E58F42185EDD7B941DCB71EDF3BCC4F637AF0D119954E65
```

- 初始 P2 代码提交 `aa46fc0c024aec7c5250eea3f9b0a87d1929b008` 已在集成分支历史中。
- 2026-08-13 拉取后，队友分支新增 `0bd3a9d` 和 `dd91f4c`。其中有效修复意图已经纳入本实现：正式 CAMPPlus 失败显式中止、路径探测、逐样本 `absent_loss_scale`。
- 新提交同时包含主源码配置/工具被移动到交付目录、大量 DEBUG 音频入 Git、CAMPPlus 维度改成 512、移除显式 STFT 参数等不应倒灌主线的变化。因此采用**语义合并**：保留提交祖先关系，主线使用本报告已实测的严格实现，不接受会删源码或改变 192 维契约的内容覆盖。

### 为什么旧 B3 声明不能视为通过

队友文档曾给出 `choice_accuracy=0.989` 和 `energy_ratio≈0.00108`。这两项不能覆盖 P3 的独立实测：

- `choice_accuracy=0.989` 已被交付包自己标为 deprecated；近静音输出也可能产生虚假高选择率。
- B2/B3 的 checkpoint SHA 未变化，仍是已经失败的 `13bce1b9…db8b`。
- P3 同链路 B1 对照为 0/40 近静音、80% 接收率，排除了“整个 P3 环境不可用”的解释。

因此旧权重继续保持：

```text
ASSET_VALID / QUALITY_REJECTED / DO_NOT_RUN_FORMAL_P3
```

## P2 已修复的问题

### ABSENT 不再压垮 PRESENT 学习

旧训练代码虽然在 YAML 中写了 `absent_loss_scale: 0.05`，但没有完整地作用到逐样本损失；同时把零目标 SI-SDR 纳入平均，语义本身也不成立。当前实现：

- 优先使用 manifest/Dataset 的显式 `target_present` / `is_absent`，不再仅靠 target 能量猜测；
- `waveform L1`、`MR-STFT`、`activity BCE` 按逐样本权重汇总，ABSENT 使用 `absent_loss_scale`；
- SI-SDR 只在 PRESENT 样本上计算；
- `absent_loss_scale` 限定在 `[0,1]`，全 ABSENT batch 不会产生除零或 NaN。

### 修复近静音的两个直接损失根因

1. 旧所谓 `scale_sensitive_l1` 先拟合 `alpha * ref`，会消除整体增益误差；PRESENT 输出缩到接近零仍几乎不受惩罚。当前改为直接 `mean(abs(est-ref))`，全零参考继续使用 `mean(abs(est))`。
2. 零参考 MR-STFT 的相对谱收敛项以接近零的参考范数作分母，实跑曾出现约 `7.6e5` 总损失和约 `7.7e4` 梯度范数。当前零目标分支使用有界、幅度敏感的输出谱抑制项；修复后 B3 smoke 总损失约 `-9.79`、梯度范数约 `0.55`，且无 NaN。

### 残差、swap 与注册语义修复

- residual 监督改为 `mixture - target`，覆盖干扰人声与噪声，而不是错误地只读某个 standalone interferer 文件。
- `target_present=True` 的 `enroll_swap_target_1/2` 保留非零 target；只有 `target_present=False` 才强制零目标。
- ABSENT 样本仍使用真实 enrollment embedding；“目标不在 mixture”不等于“没有注册人”。
- wrong enrollment 只接受 manifest 显式字段或同 triplet 的合法配对，不再猜默认文件路径；缺失时跳过 choice 并记录原因。

### 正式 CAMPPlus 不再静默降级

- 正式训练和评测的 CAMPPlus 加载/编码失败会直接报错；BOOTSTRAP 只允许显式 `--debug-data`。
- 所有 CAMPPlus 正式配置已明确写入 `iic/speech_campplus_sv_zh-cn_16k-common`。
- 本机已用官方 3D-Speaker 源码 commit `065629c313eaf1a01c65c640c46d77e61e9607b4` 和 `campplus_cn_common.bin` 实跑；权重与模型参数严格匹配 `937/937`。
- ModelScope 新旧缓存布局都能定位；不存在权重时禁止随机初始化冒充预训练模型。
- bootstrap 调试向量改为 SHA256 派生种子，跨 Python 进程稳定；这不是新增交付 hash/gate，只是消除 Python 内置 `hash()` 的随机盐错误。

### 训练可靠性修复

- B3 `--init-checkpoint` 使用 `strict=True`，并与 `--resume` 互斥。
- 正式 DataLoader 的随机裁剪使用 PyTorch 的 per-worker seed；不再让多个 worker 因复制同一 NumPy RNG 状态而重复裁剪序列。
- CAMPPlus 本地 embedding 缓存读取检查 shape/finite，写入采用临时文件后替换，避免多 worker 读取半写文件。
- 已存在输出目录默认拒绝递归删除；只有显式 `--overwrite-output` 才允许覆盖。这个门禁仅位于不可逆删除边界，符合全局规则。
- DEBUG 训练器新增 `--max-steps`；短跑只验证执行链，不拿 100-step 收敛目标误判 1-step smoke。

## 实际运行与测量结果

### 测试矩阵

- 提交前最终项目全量 pytest：`100 passed, 2 skipped, 22 subtests passed`；两项 skip 均为当前 Python 3.13 环境无 CUDA 的专属项。
- B3 专项回归（含多 worker RNG）为 `8/8 PASS`。
- 随机 TSE smoke：4 种长度 `160 / 16000 / 57600 / 160000` 全部通过 shape、finite、完整分支反向和 mixture consistency。
- 注册条件 smoke：确定性、不同 embedding、shuffle、零 mixture、embedding 梯度 `6/6 PASS`。
- P4 smoke 已真实加载预训练 CAMPPlus，并完成 shape、batch、unload、输入异常和公开接口测试。

同一轮还对新增/改动的 P1、P2、P4 Python 入口执行了 `py_compile`，全部通过。

### P1 真实生成预检

`prepare_split_manifests.py` 从本地 AISHELL/MUSAN/RIRS 发现并生成九份 split manifest：

| 数据 | train | dev | holdout |
|---|---:|---:|---:|
| speech rows | 4201 | 718 | 699 |
| speech speakers | 12 | 2 | 2 |
| noise rows | 1622 | 200 | 194 |
| RIR rows | 48042 | 6076 | 6100 |

随后用 `build_p1_v2_b1_portable.py --workers 2 --preflight` 在 Windows `spawn` 模式真实生成 14 条覆盖 train/dev/D_single/D_overlap 及正式场景的音频，结果：

```text
P1_V2_B1_PREFLIGHT_STATUS=PASS ROWS=14
```

冻结的 `build_p1_v2_b1.py` 未改变；可移植入口只适配本机公开数据目录和 Windows worker 初始化。

### 真实 CAMPPlus 与 B1→B3 训练链

真实注册诊断使用 4 名说话人、12 条三场景样本：

- embedding L2 norm：约 `1.0`；
- 不同说话人余弦：min `0.1934`、mean `0.3355`、max `0.5045`；
- 输入 mixture RMS：`0.00949～0.06353`；
- 8 个 PRESENT target/mix RMS 比：`0.6869～0.9668`；
- 4 个 ABSENT target 均为真零。

B1 真实 CAMPPlus 2-step：

- total：`-4.08998 → -4.09395`；
- 梯度范数约 `0.289`；
- checkpoint restore max diff `0`；
- GPU 峰值约 `0.69 GB`。

B3 从该 B1 checkpoint 严格 warm-start 的真实 CAMPPlus 2-step：

- total：`-9.79326 → -9.79439`；
- 梯度范数约 `0.554`；
- checkpoint restore max diff `0`；
- GPU 峰值约 `1.27 GB`；
- 8 个 PRESENT 输出/input RMS 比 `0.4998～0.5040`，近静音 `0/8`。

限制必须同时说明：2-step 时 4 个 ABSENT 输出/input RMS 比仍约 `0.50`，抑制尚未学成。上述结果只证明真实 embedding、损失、反向、warm-start 和恢复链能运行，**不证明 B3 正式质量达标**。

### S004 录音预检

- ZIP 顶层为 `S004/`，共 35 个 M4A：3 条 E + 16 条 C × 2 take。
- 文件名从 `S004_E01`～`S004_E03`、`S004_C01(1)`～`S004_C16(2)`，无时间戳猜测问题。
- 源格式为 AAC、48 kHz、单声道；命令时长 `3.392～4.501 s`，长句 `10.133～11.669 s`。
- 与 S002 相同的本地 Paraformer 内容预检：32/32 条命令逐字一致；3 条长句去掉标点后逐字一致。
- 带标点原文直接计算的 35 条 corpus CER 为 `5.42%`，错误来自关闭标点模型后的标点差异，不是朗读词汇错误；该数字只用于质检。
- 手机底噪使严格数字静音检测不能证明每条恰好 1.000 秒；正式签收仍需人工耳检开头/结尾，发现说话截断或提示词时整条重录。

## P1 接下来怎么做

### 不需要重做的内容

P3 已经拥有 6000 条外部配对记录和 18000 个必需音频，因此不要重新下载或用 TTS/DatasetA 替换。该资产可以直接等待新 P2 候选做 B0/ORACLE/B1 配对评测。

### 需要完成的内容

1. 若交付目标包含“从公开源完整复现 P1 v2”，用以下入口生成到**新目录**：

```powershell
python ..\p1\v2_b1\prepare_split_manifests.py `
  --aishell-root data\public\aishell1\extracted\data_aishell\wav_expanded `
  --musan-root data\public\augmentations\musan\extracted\musan `
  --rir-root data\public\augmentations\rirs_noises\extracted\RIRS_NOISES `
  --output data\p1_v2_b1_inputs

python ..\p1\v2_b1\build_p1_v2_b1_portable.py `
  --inputs data\p1_v2_b1_inputs `
  --source-root . `
  --output data\P1_to_P2_v2_b1_formal_rebuild `
  --workers 2
```

2. 全量生成结束后，使用冻结 builder 自带 acceptance report 判定；不要增加另一套 baseline/hash/gate 登记表。
3. 真人采集继续沿用 S004 的直接命名方式。每人 35 条，禁止读编号/提示，读错任一 take 就重录对应文件。
4. S004 入库前补：参与者同意、日期、设备/环境、匿名编号；保留原 M4A，并另生成 `16 kHz / mono / PCM16 WAV` 规范副本。
5. 继续采集到建议的 20 人规模。当前已发现 S002 和 S004 候选，其他编号是否正式通过应以各自审计记录为准，不能仅按编号推算人数。

## P2 接下来怎么做

### 正式重训顺序

1. 用修复后的代码、真实 CAMPPlus 和 P1 train/dev 先跑 B1 500-step 正式试车；确认无 NaN、梯度有限、PRESENT SI-SDR/能量未坍塌、checkpoint 严格恢复。
2. 完成与新代码匹配的 B1 正式训练。旧 B1 可以作为对照或初始化来源，但不能把其旧损失日志当成新实现的训练证明。
3. 从新 B1 strict warm-start 训练 B3；至少保留 `20260813 / 20260814 / 20260815` 三个独立种子候选。
4. B2 结构可以与 B3 相同，但“结构相同”不足以证明质量可复用。只有新 B3 在 ABSENT 与 SWAP 分场景均经 P3 实测通过后，B2 才可引用同一权重；否则单独训练/选择 B2。

### P2 下一次交付必须包含

- 与旧 `13bce1b9…db8b` 不同的新 checkpoint；
- 实际使用的 config、训练命令、seed、起始 checkpoint、train/dev manifest 身份；
- train/dev 日志和分场景指标；
- checkpoint 严格 restore 结果；
- PRESENT 与 ABSENT 的输出/input RMS 分布、近静音计数；
- 20+20 接入所需的明确 enrollment/wrong-enrollment 字段；
- 若 B2 复用 B3，随新 B3 同步更新并提供 P3 独立复验证据。

现有 SHA 文件属于已有交付身份校验，继续沿用即可；不要另建 hash、冻结 contract、baseline 或 gate 登记体系。

## P3 已完成什么、接下来做什么

### 已完成

- P1→P3 6000 条规范配对 manifest 已重建，18000 个必需音频引用缺失 0；
- `P2TSERuntime`、P2 `extract_target()` 接入、checkpoint/shape 校验已实现；
- B0 mixture、ORACLE target、B1 P2 target 的同样本配对 CER runner 已实现；
- CER 使用语料级 S/D/I/N，拒绝非字符串、重复/缺失 sample_id，并区分 ASR ERROR 与真正拒识；
- 18 类 UTF-8/CER 黄金样例、输出能量质量汇总、场景分桶和错误保留已实现；
- 旧 B3 20+20 失败已经有独立证据，不需要对相同 SHA 重复消耗 GPU；
- 当前整仓测试覆盖 P1/P2/P3/P4，提交前全量复跑。

### 新权重到位后的唯一执行顺序

1. 对每个新 B3 seed 运行 20 正 + 20 负：先看输出能量、近静音、正样本接收率、CER、负样本 RR 和 B1 对照。
2. 至少 2/3 种子满足既有跨团队质量验收后，固定一个候选和 ASR 配置。
3. 在 6000 条外部确认集运行同样本 B0/ORACLE/B1 配对 CER。
4. 只有外部结果可信且方向合理，才对 DatasetA 做一次冻结候选的全量比较。
5. 若 ORACLE 也差，先排查 P3 ASR/文本/数据；若 B0 差而 ORACLE 好但 B1 无改善，问题回到 P2，不在 P3 扫阈值掩盖。

## 限制、风险和没有宣称的结果

- 本轮没有完成 20000-step 正式 B3 重训，因此没有新正式 B3 checkpoint，也没有可宣称的新 CER/RR。
- 本轮没有完整重建 P1 的 116000 条正式样本；完成的是输入清单生成和 14 条真实跨场景预检。已有 6000 条外部正式包不受此限制。
- 真实 CAMPPlus 2-step 结果不能外推正式收敛；尤其 ABSENT 2-step 仍未被抑制。
- S004 自动内容预检不能替代人工耳检、授权和最终格式转换。
- 全局规则已遵守：没有新增仪式化 hash、冻结 contract、baseline 或普通流程 gate；唯一新增覆盖确认位于递归删除输出目录这一不可逆边界。

## 关键证据索引

- [P2→P3 接入说明](P2_P3_INTEGRATION.md)
- [P3 计划与执行报告](CER_P3_PLAN_REPORT_20260810.md)
- [旧 B3 接入审计](P3_P2_B3_HANDOFF_AUDIT_20260810.md)
- [P2 README](../P2_project/README.md)
- [B3 诊断脚本](../P2_project/tools/diagnose_b3.py)
- [B3 回归测试](../P2_project/tests/test_b3_training_regressions.py)
- [P1 输入准备说明](../../p1/v2_b1/PREPARE_INPUTS.md)
- [P1 可移植构建入口](../../p1/v2_b1/build_p1_v2_b1_portable.py)
