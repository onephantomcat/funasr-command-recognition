# P2 → P3 冻结接入说明

> **2026-08-14 最终状态更新：** 三个 B3 训练种子已完成外部 6,000 条 paired CER，并聚合为 `ACCEPT_B1_CANDIDATE`；随后冻结的 `seed=20260813` 在 DatasetA 全量公平比较中使 CER 从 53.43% 恶化至 73.66%，因此当前候选已 **REJECT**，基线继续保留。本文其余 2026-08-10～13 的段落是历史接入与复现记录，不得覆盖此最终结论。完整指标与后续修复方向见 [P2/P3 最终验证报告](./P2_P3_DATASETA_FINAL_REPORT_20260814.md)。

> **2026-08-13 状态更新**：队友分支新增的 `0bd3a9d`、`dd91f4c` 已拉取并按语义纳入集成；正式 CAMPPlus silent fallback、逐样本 `absent_loss_scale`、零目标 MR-STFT、PRESENT 幅度监督、残差参考、swap/ABSENT 语义、strict warm-start 和诊断链均已修复并实跑。P1 九份 split manifest 的本地生成入口及 Windows 可移植构建入口已补齐，14 条跨场景真实预检通过。当前代码状态为 `FIXED / SMOKE_PASS`，但 B2/B3 旧 checkpoint `13bce1b9…db8b` 仍是 P3 已拒收权重；必须由 P2 用新代码重训并交付不同 checkpoint，P3 才恢复 20+20 和后续 6000 条评测。完整结论见 `P1_P2_P3_FIX_DELIVERY_20260813.md`。

> **2026-08-12 状态更新**：P1 正式外部包已到位，6000 条配对清单和 18000 个 mixture/target/enrollment 音频引用已通过实际重建检查。Git 分支当前只包含已合并的 P2 版本；微信群收到的 v2.2 包还没有对应 Git 提交，其 `diagnose_b3.py` 不在分支内，另外三个工具脚本也与分支版本不同。v2.2 包中的 B2/B3 checkpoint 仍是已在 P3 20+20 实测中出现 39/40 近静音、正样本接收率 0% 的同一权重，因此在 P2 提供新权重或可复现修复前，不运行全量评测。下方 2026-08-10 内容保留为历史接入依据。

## 冻结基线

- GitHub PR：`#3`
- PR 头提交：`6066e9d6d03371657ce7860502c6ee284d8f708e`
- `main` 合并提交：`2375254294a95bfc33dac8ad9d57e8f4c6d7572f`
- P2 公共 API：`P2_project.src.tse.extract_target()`
- P3 适配器：`p2_tse_runtime.py`
- P3 配对 CER：`eval_paired_cer.py`

P3 只消费 P2 的公开 target 波形，不修改 P2 网络、损失、阈值或训练产物。注册向量由现有 CAM++ 声纹模型产生，归一化后按 `[1, 192]` 传给 P2。

## 历史资产状态（2026-08-10 实测）

PR 按仓库策略排除了所有 `.pt`，因此 GitHub 代码中没有正式 B1/B2/B3 checkpoint。本地已通过独立资产渠道取得 B3 权重并放到：

```text
P2_project/artifacts/final/P2_artifacts/B3/checkpoint_step20000.pt
```

冻结 SHA256 必须是：

```text
5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d
```

哈希不匹配时，`P2TSERuntime` 会在加载前中止。不得使用旧的 `models/tse_dual_output_mvp.pt` 冒充三级训练模型。

随包的 `config.sha256` 和 `data.sha256` 有用：前者已确认匹配 Git 冻结配置，且解析值与 checkpoint 内嵌 `cfg` 的 45 个字段完全一致；后者记录训练 manifest 哈希，但由于原训练 manifest 未交付，P3 目前不能独立复算该数据哈希。Windows checkout 对 YAML 做 CRLF 转换会改变文件字节哈希，校验冻结配置时应对 Git 原始 blob 计算。

本次权重的文件/结构/CUDA 接口检查通过，但质量预检失败：20 pos + 20 neg 的 P2 输出全部近静音，输出/输入 RMS 比为 `2.01e-7～2.28e-7`，20 个正样本全部被误拒。B3 自带开发日志末尾 `dev SI-SDR=-19.23 dB`。因此当前 checkpoint 状态为：

```text
ASSET_VALID / QUALITY_REJECTED / DO_NOT_RUN_FULL_DATASETA
```

## 1. 本地验收

```powershell
Get-FileHash `
  P2_project\artifacts\final\P2_artifacts\B3\checkpoint_step20000.pt `
  -Algorithm SHA256

..\.venv\Scripts\python.exe P2_project\verify_p2_local.py

..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`verify_p2_local.py` 必须显示 B3 SHA 与云端一致；P3 测试必须包含 CER 类型契约、ASR 元组拆包、ERROR/RR、P2 runtime 和配对分桶。

## 2. 外部 B0/ORACLE/B1 配对 CER

先准备说话人隔离的 manifest，每行至少包含：

```json
{"sample_id":"confirm_0001","ref_text":"打开空调","mixture":"mix/0001.wav","target":"target/0001.wav","enrollment":"enroll/spk001.wav","scene":"OVERLAP","overlap_ratio":1.0,"sir_db":-5,"seed":20260723}
```

若收到的是 P1 原始 `D_single.jsonl`/`D_overlap.jsonl`，先用 AISHELL transcript 生成 P3 规范清单与音频齐备度报告：

```powershell
..\.venv\Scripts\python.exe build_p3_paired_manifest.py `
  --p1-manifest data\p1_b1_external_audio_formal_20260810_incoming\P1_to_P2_v2_b1_formal\manifests\D_single.jsonl `
  --p1-manifest data\p1_b1_external_audio_formal_20260810_incoming\P1_to_P2_v2_b1_formal\manifests\D_overlap.jsonl `
  --transcript data\public\aishell1\extracted\data_aishell\transcript\aishell_transcript_v0.8.txt `
  --data-root data\p1_b1_external_audio_formal_20260810_incoming\P1_to_P2_v2_b1_formal `
  --output outputs\p3_external_gate_formal_20260810\paired_manifest.jsonl `
  --report outputs\p3_external_gate_formal_20260810\readiness.json `
  --require-audio
```

当前正式 P1 包已包含 `audio/`；2026-08-12 实际重建结果为 `ready_for_paired_eval=true`、6000 条唯一样本、18000 个必需音频引用、缺失 0。旧的“只含 manifest、缺 18000 音频”结论已经失效。

运行：

```powershell
..\.venv\Scripts\python.exe eval_paired_cer.py `
  --manifest <P1_confirm_manifest.jsonl> `
  --data-root <P1_confirm_root> `
  --p2-tse-checkpoint P2_project\artifacts\final\P2_artifacts\B3\checkpoint_step20000.pt `
  --p2-tse-sha256 5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d `
  --training-seed 20260813 `
  --device cuda `
  --out-dir outputs\p3_paired_b1_rc2
```

输出：

- `paired_predictions.jsonl`：同一 `sample_id` 的 `B0_MIXTURE`、`ORACLE_TARGET`、`B1_P2_TARGET` 三条记录，含 raw hypothesis、S/D/I/N、场景和 ASR 状态；
- `summary.json`：总体、SINGLE、OVERLAP、100% overlap、100% overlap/SIR=-5 dB 分桶及 B1 对 B0 的变化。

单次训练种子条件：结果有效且 ASR 错误为 0、P2 输出诊断完整且无近静音样本、高重叠 CER 相对下降至少 15%或绝对下降至少 5 pp、SINGLE 恶化不超过 2 pp、100% overlap/SIR=-5 dB 方向不反转。

正式结论必须对三个独立训练 checkpoint 分别运行上述命令，再聚合三个 `summary.json`。manifest 每行的 `seed` 只用于构造混合音频，不能代替模型训练种子：

```powershell
..\.venv\Scripts\python.exe aggregate_paired_cer_seeds.py `
  --summary outputs\p3_paired_b3_seed1\summary.json `
  --summary outputs\p3_paired_b3_seed2\summary.json `
  --summary outputs\p3_paired_b3_seed3\summary.json `
  --frozen-training-seed 20260813 `
  --out outputs\p3_paired_b3_three_seed\aggregate.json
```

只有冻结候选自身通过且至少 2/3 个独立训练种子通过时，聚合 verdict 才是 `ACCEPT_B1_CANDIDATE`。

## 3. DatasetA 冻结候选验证

只有外部 B1 通过后才运行。两条命令必须保持相同硬门控配置，不能使用 DatasetA phrase bank、A 上训练的 gate、ASR cache 或 embedding cache。

```powershell
# 基线
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --out outputs\p3_freeze\datasetA_baseline.json

# P2 候选；目录必须是新建的空目录
..\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --p2-tse-checkpoint P2_project\artifacts\final\P2_artifacts\B3\checkpoint_step20000.pt `
  --p2-tse-sha256 5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d `
  --p2-tse-device cuda `
  --p2-tse-dir outputs\p3_freeze\p2_targets_fresh `
  --out outputs\p3_freeze\datasetA_p2.json
```

`--enhancer-model` 与 `--p2-tse-checkpoint` 被强制设为互斥，防止一次实验改变两个前端变量。

## 已知限制

- P2 冻结 API 只公开 target，不公开 residual；P3 CER 不绕过该契约读取内部 residual。
- `eval_datasetA_tse_dual.py` 仅用于修复和复核旧 MVP 的无效历史结果，不代表 PR #3 的 P2 模型。
- 只有 manifest 而没有其引用且哈希冻结的 P1 音频时，只能完成清单/接口/测试验收，不能形成正式 CER 结论。
- `trial_verdict.json` 的工程 PASS 只覆盖 NaN、显存、吞吐和 checkpoint restore；它没有以合格 dev SI-SDR 或非静音输出作为必过项。
- 当前 B3 权重已被 P3 质量预检拒绝；获得新权重后必须先重跑外部配对闸门，不能直接跳到 DatasetA 全量。
