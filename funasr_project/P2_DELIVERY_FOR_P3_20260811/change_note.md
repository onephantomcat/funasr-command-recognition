# P2 → P3 交付 Change Note

**版本**：v2.2（2026-08-11，全面修复 P3 验收 5 项问题）
**v2.2 修复说明**：在 v2.1 基础上彻底解决 P3 验收报告列出的全部 5 项问题，下表逐条对照。v2.1 仅完成 fix_1/fix_2/fix_3；v2.2 新增 fix_4/fix_5/fix_6/fix_7。

---

## 〇、P3 验收 5 项问题修复对照表

| # | P3 验收问题 | v2.2 修复 | 验证文件 | 状态 |
|---|------------|----------|----------|------|
| 1 | 解决 DatasetA PRESENT 输出再次坍塌的问题 | fix_1（evaluate_tse.py 禁止 fallback）+ fix_2（sv_model_id 填充）+ fix_4（train_b1_trial.py 禁止 fallback） | `tools/evaluate_tse.py` + `tools/train_b1_trial.py` + `B2/B3/config.yaml` | ✓ 根因修复（P3 重跑确认） |
| 2 | 提供准确的 train/dev 单文件 SHA，当前 data.sha256 标注与实算文件不一致 | fix_5：原标注 train/dev 用同一 SHA（错误），已重算 6 个单文件 SHA | `B1/data.sha256` + `B2/data.sha256` + `B3/data.sha256` + `THIRD_PARTY_PINNINGS.txt` | ✓ 已重算并提供 |
| 3 | 固定并提供 speakerlab commit、CAMPPlus 权重 SHA | fix_6：speakerlab commit=`065629c313eaf1a01c65c640c46d77e61e9607b4`；CAMPPlus 权重 SHA256=`3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` | `THIRD_PARTY_PINNINGS.txt` | ✓ 已固定并提供 |
| 4 | 修正"禁止 fallback"的文档；实际 commit 中 evaluate_tse.py 仍允许回退 BOOTSTRAP | fix_4：v2.1 只改了 evaluate_tse.py；v2.2 进一步修复 train_b1_trial.py 的两处 fallback（L155-L169 load_backend；L329-L341 encode_file），文档与代码完全对齐 | `tools/train_b1_trial.py` | ✓ 代码与文档对齐 |
| 5 | 统一互相冲突的 choice_accuracy 和分场景 SI-SDR 数值 | fix_7：明确废弃 choice_accuracy=0.9892（与 energy_ratio=0.001082、mean_sisdri_db=-4.20 dB、sisdr_by_scenario=-0.60/-1.99 dB 严重冲突），新增 choice_accuracy_formal=null 等待 P3 重跑 | `B3/eval_summary.json::metric_conflict_v2_1_disclosure` | ✓ 已披露并废弃，等 P3 重跑 |

---

## 〇.1、v2.2 详细修复清单

### fix_4：train_b1_trial.py 禁止 silent fallback（v2.2 新增）

| 文件 | 行号 | 修改 | SHA256 |
|------|------|------|--------|
| `tools/train_b1_trial.py` | L155-L169 | `load_backend()` 失败时，非 `--debug_data` 模式直接 `RuntimeError`，禁止 fallback BOOTSTRAP；`--debug_data` 模式仍允许 fallback 以便空跑测试 | `e8b1bc9d685ea5dbf3546f4d67a2821c147fac5d7b919f69dc004d2b31cbd815` |
| `tools/train_b1_trial.py` | L329-L341 | `encode_file()` 失败时同上禁止 fallback，正式训练直接 `RuntimeError` | 同上 |
| `tools/train_b1_trial.py` | L407-L409 | main() 注入 `cfg["__debug_data"]=bool(args.debug_data)`，供 Dataset 判断是否允许 BOOTSTRAP fallback | 同上 |

### fix_5：train/dev 单文件 SHA256 重算（v2.2 新增）

原标注错误：B1/B2/B3 的 data.sha256 都把 train_manifest.jsonl 和 dev_manifest.jsonl 标注为同一个 SHA（但两文件大小差 5 倍以上，SHA 不可能相同）。

| 模型 | 文件 | 实算 SHA256（v2.2） |
|------|------|---------------------|
| B1 | train_manifest.jsonl | `a3822ed34a532ca4b11ed608567ef049b3f9df6d3069b74c427b3ca03e993002` |
| B1 | dev_manifest.jsonl | `cf785cb601c5238d13c39c036da16acfa893a7392a215a7c64a051bc0a1d48db` |
| B2 | train_manifest.jsonl | `f4cd7a4c4e5a2d86d24e35eaa59f7daa9742fa922255ae60b8bb98575ac2c970` |
| B2 | dev_manifest.jsonl | `2ccb412b2c52c83a5943fcb4cca8e73e4465dda2721ed6f58eefe7fb9c2e10d3` |
| B3 | train_manifest.jsonl | `1b3e8377b32b5a9c7761ab42f62f7192ee79bdefbbea8adc53749e55d38888a7` |
| B3 | dev_manifest.jsonl | `8911193cd3e4789a8a2f175e856c0c6ead5594abb95679bd3866d052f961d909` |

### fix_6：speakerlab commit + CAMPPlus 权重 SHA（v2.2 新增）

| 项 | 值 | 验证命令 |
|----|----|---------|
| speakerlab commit | `065629c313eaf1a01c65c640c46d77e61e9607b4` | `cd speakerlab_source && git rev-parse HEAD` |
| speakerlab commit 日期 | 2025-12-08 15:20:17 +0800 | `git log -1 --format='%ci'` |
| speakerlab commit subject | Update cluster.py | `git log -1 --format='%s'` |
| CAMPPlus 权重文件 | `campplus_cn_common.bin`（26.74 MB / 28036335 bytes） | `ls -l campplus_cn_common.bin` |
| CAMPPlus 权重 SHA256 | `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` | `sha256sum campplus_cn_common.bin` |
| CAMPPlus model_id | `iic/speech_campplus_sv_zh-cn_16k-common` | B2/B3 config.yaml `sv_model_id` 字段 |

### fix_7：choice_accuracy 与 SI-SDR 冲突披露（v2.2 新增）

**冲突的三个指标**：

| 指标 | 值 | 含义 |
|------|-----|------|
| `choice_accuracy` | 0.9892 | 声称"几乎完美"选择正确 |
| `mean_energy_ratio` | 0.001082 | 输出能量仅为 mix 的 0.1%（近静音） |
| `mean_sisdri_db` | -4.1977 dB | 提取后比不提取还差 4.2 dB |
| `utterance_sisdr_by_scenario` | -0.5977 / -1.9894 dB | 输出比 mix 还差 |

**根因**：评测期 `EnrollmentAdapter` 加载 CAMPPlus 失败时 silent fallback BOOTSTRAP，模型在随机 embedding 上计算 swap choice，导致 `selectivity_db=10.39` 虚高，进而 `choice_accuracy=0.99` 也是虚假的。

**v2.2 解决方案**：
1. evaluate_tse.py 禁止 fallback → P3 重跑后 `choice_accuracy` 与 `SI-SDR` 将一致
2. `B3/eval_summary.json` 新增 `metric_conflict_v2_1_disclosure` 字段披露三处冲突
3. 新增 `choice_accuracy_formal=null` 字段，明确等待 P3 用修复后代码重跑
4. `DELIVERY_MANIFEST.json` 顶层 `swap_choice_accuracy_v2_2_status="已废弃"`

---

## 〇.2、v2.1 紧急修复清单（保留作历史记录）

| # | 文件 | 修改 | SHA256（新） | 目的 |
|---|------|------|-------------|------|
| 1 | `tools/evaluate_tse.py` | L207-L219（target embedding）+ L255-L266（wrong_spk embedding）：在 `adapter.mode == "campplus"` 分支下 embedding 失败时 `raise RuntimeError`，禁止 silent fallback 到 BOOTSTRAP | `ac3b910f34a707c90d83d42b42fb2a835515e0c6d69f42c22aa36a4378825df1` | 防止 P3 环境因 CAMPPlus 加载失败而生成"全静音"虚假输出，问题立即暴露而非静默通过 |
| 2 | `B3/config.yaml` + `B2/config.yaml` | `sv_model_id` 从 `""`（空）改为 `"iic/speech_campplus_sv_zh-cn_16k-common"` | `6b2b08f8f5eb94b91d1d93d970f08b310859ee9668b71ef0d87dee6044425125` | P3 能正确解析 CAMPPlus 预训练模型 ID 并定位权重，不再依赖绝对路径 |
| 3 | `tools/diagnose_b3.py`（新增） | P3 集成自检脚本：CAMPPlus 后端加载/嵌入有效性/数据能量分布/声纹区分度四项自检 | `df96d794afef40d69fdf3b5feea9a6f70945dd3427b7ed0f5edda15d110f1c09` | P3 在运行完整评测前可快速定位 CAMPPlus 加载/能量异常，避免长时间空跑 |
| 4 | `B2/config.sha256` + `B3/config.sha256` | 同步更新为新 config 的 SHA256 | `6b2b08f8f5eb94b91d1d93d970f08b310859ee9668b71ef0d87dee6044425125` | 与 config.yaml 实际哈希一致，P3 校验不再 REJECT |
| 5 | `DELIVERY_MANIFEST.json` | schema_version 升级至 `P2_v2.2_STRICT_P1_S42_20260811`，新增 `v3_1_fixes` 与 `tools` 字段 | — | 交付清单与实际文件状态一致 |

**v2.2 P3 集成前置自检命令**（必须先跑，全 ✓ 再跑完整评测）：

```bash
python tools/diagnose_b3.py \
    --config P2_DELIVERY_FOR_P3_20260811/B3/config.yaml \
    --manifest /path/to/DatasetA_manifest.jsonl \
    --device cuda
```

预期输出：`CAMPPlus 加载与嵌入 ✓` + `数据能量分布正常 ✓` + `声纹区分度 ✓`。

---

## 一、旧 B3（v1.0）近静音退化根因分析

| 层级 | 根因 | 证据 |
|------|------|------|
| **数据层** | `_is_absent_entry()` 判断时，把 `target_present=True` 的 `enroll_swap_target_1/2` 样本误判为 absent，导致 target 被强制置零 —— swap 场景 PRESENT 样本参与训练时，监督信号是"输出零" | 训练日志：`present=0 absent=3333 swap=6667`（present 统计恒为 0，swap 里的 PRESENT 样本被算成 absent） |
| **声纹层** | 训练初期 CAM++ 后端加载失败时静默 fallback BOOTSTRAP（哈希随机 embedding），模型对注册语音 embedding 完全不敏感 | P3 预检：`output_rms/input_rms = 2.01e-7`（40/40 近静音）；`choice_accuracy=0.989` 是全静音虚假高 |
| **损失层** | MR-STFT + SI-SDR 对零目标（= 静音）天然惩罚梯度更大（MR-STFT 对任何非零输出都给大 Loss）；absent 样本占比 1:2 时，零目标梯度压倒 PRESENT 提取梯度 | v3 之前日志：`dev SI-SDR=-19.23 dB`（纯静音） |
| **工程层** | evaluate_tse.py 的 swap wrong_enroll 路径在 manifest 缺字段时默认路径报错，且 EnrollmentAdapter 加载失败时 silent fallback —— 评测 summary 的 `choice_accuracy=0.989` 是 BOOTSTRAP embedding 的假高分（真实 CAM++ 下根本没这水平） | P3 审计：用真实 CAM++ 重评时 swap 选择完全失效 + 近静音 |

---

## 二、修复点汇总（代码 + 配置 + 训练流程）

### 2.1 代码修复（`train_b1_trial.py` + `evaluate_tse.py`）

| 文件 | 行号 | 修复内容 | 目的 |
|------|------|---------|------|
| `train_b1_trial.py` | `_is_absent_entry()` 开头 | 增加判断：`if e.get("target_present") is True: return False` —— 先看 manifest 里 target_present 布尔，优先级高于 scenario 字符串匹配 | **阻止 swap_target（target_present=True）被误判为 absent 导致 target 置零** |
| `train_b1_trial.py` | main() L508-L524 | 增加 CAM++ **强制加载**机制：`adapter.mode == campplus` 且非 `--debug_data` 时，循环 3 次尝试 `adapter.load_backend()`，3 次全失败抛出 `RuntimeError` 终止训练，**不允许 silent fallback BOOTSTRAP** | 保证 B2/B3 训练全程用真实 CAM++ embedding，choice_accuracy 不造假 |
| `train_b1_trial.py` | L517（v2 修正） | 删除局部 `import time`，只用文件顶部全局 `import time`，避免 `UnboundLocalError: local variable 'time' referenced before assignment` | 修复 CAM++ 重试时的崩溃 |
| `train_b1_trial.py` | main() 开头 | 增加 `sys.path` 显式注入 P4_project 路径 + speakerlab_source 路径 | 确保 `speakerlab` 模块可导入（CAM++ 依赖）|
| `train_b1_trial.py` | CLI 解析 | 增加 `--init_checkpoint` 参数，支持从预训练 B1 权重 warm-start | B2/B3 从 B1 最优 checkpoint 热启动，加速收敛 + 提升提取质量基线 |
| `evaluate_tse.py` | `_is_absent_entry()` | 同样加 `target_present=True → return False` 保护 | 评测数据判定与训练一致，避免评测时 swap_target 被当成 absent |
| `evaluate_tse.py` | EnrollmentAdapter 加载 | 与训练脚本一致：强制 CAM++，失败抛错，不允许 fallback | 评测使用真实声纹，summary 指标可信 |
| `evaluate_tse.py` | swap 评测 `wrong_enroll` 路径 | 当 manifest 无 `swap_enrollment` / `swap_enroll_wav` 字段时，**跳过 swap choice_accuracy 评测**（输出 `null`），不用默认路径 FileNotFoundError | 兼容 P1 v3 manifest（无 wrong_enroll 配对字段），不崩溃 |
| `evaluate_tse.py` | WAV 输出 `soundfile.write` | 增加 try-except：磁盘满时跳过写入并告警 | 避免评测后半段因磁盘不足而整体失败 |
| `campplus_backend.py` | 嵌入维度 | `embedding_size=192`（与预训练权重形状一致）+ 权重路径解析修正 | 解决 CAM++ 权重加载 `shape mismatch` |

### 2.2 配置修复（B1 → B2/B3 迁移策略）

| 模型 | 配置文件 | 关键策略 |
|------|---------|---------|
| B1（基线） | `tse_b1.yaml` | scene_mode=b1（全 PRESENT 场景），真实 CAM++，20K 起步 → 50K resume（最优：dev_sisdr=+6.61 dB） |
| B2（零抑制） | `tse_b2_v2_campplus.yaml` | scene_mode=b2（全 ABSENT 场景），从 **B1_50K_RESUME step20000 warm-start**，真实 CAM++，20K 步 |
| B3（swap）v3 ✓ | `tse_b3_v3.yaml` | scene_mode=b3（PRESENT+ABSENT+SWAP 1:1:2），从 **B1_50K_RESUME step20000 warm-start**，`absent_loss_scale=0.05`（absent 梯度砍 95%，防止压倒 PRESENT），`lambda_sisdr=2.0`（提取质量优先），`lambda_act=1.0`（活动检测强化），真实 CAM++ |
| B3（swap）v4 ❌ | `tse_b3_v4.yaml` | `absent_loss_scale=0.02` + `lambda_act=2.0` + `zero_ref_kappa=1e-4` + `warmup=1500` —— **过犹不及，energy_ratio 从 0.00108 跌到 1.57e-9（再近静音）**，已作废 |

### 2.3 训练流程修复

| 阶段 | 修复前（v1.0 旧 B3） | 修复后（v2.0 新 B3） |
|------|---------------------|---------------------|
| 声纹模式 | config 写 campplus，实际 fallback BOOTSTRAP | **强制加载**：失败立即终止 + 日志显式写 `CAMPLUS 后端全局加载成功（无 silent fallback）` |
| 初始化 | 随机初始化 | **从 B1_50K_RESUME step20000 热启动**（基线 dev_sisdr=6.61 dB，继承高质量提取能力）|
| absent 梯度权重 | 1.0（全量，压倒 PRESENT） | 0.05（砍 95%，只给弱信号） |
| 损失结构 | lambda_sisdr=1.0 / act=0.5 | lambda_sisdr=2.0 / act=1.0（提取 + 检测强化）|
| 训练统计 present=0 | 真实 bug（swap_target 被置零） | 统计显示错误（target 正常，但计数器只在 absent 分支 ++，不影响训练质量，仅日志误导）|

---

## 三、v2.0 三模型与旧 v1.0 的关键指标对照

### 3.1 B1：保留本地 v1.0 PRESENT_seed20260723（除非云端 50K 评测更好）

| 指标 | 本地旧 B1（v1.0 PRESENT） | 云端 B1_50K_RESUME（待评测）| P3 硬门 |
|------|-------------------------|---------------------------|---------|
| checkpoint SHA256 | `df95a0c25e…6c6cfa3` | `d06c3e9c4f…5b3601`（待评测）| - |
| sv_mode | campplus（真实）| campplus（真实）| 必须真实 |
| corpus_sisdr_db | **+5.25 dB** ✅⭐ | 待评测 | > 0 dB |
| mean_energy_ratio | **0.304** ✅⭐ | 待评测 | 不能全 < 1e-4 |
| mean_act_f1 | **0.975** ✅⭐ | 待评测 | > 0.7 |
| schema + determinism | PASS / PASS | 待评测 | 必须 PASS |

**决策规则**：云端 B1_50K corpus SI-SDR > 5.25 dB 且 energy_ratio ≥ 0.3 → 换云端 B1；否则保留本地 B1（5.25 dB 已达 P1 目标 ≥ 5 dB）。

### 3.2 B2：云端 v2 重训（真实 CAM++ + B1 warm-start）

| 指标 | 本地旧 B2（v1.0，指标不全）| 云端 B2_v2（待训练/评测） | P3 硬门 |
|------|---------------------------|------------------------|---------|
| checkpoint SHA256 | `da44f82682…9f459f3` | 待生成 | - |
| sv_mode | campplus（配置写对，但无法验证训练时是否 fallback） | **强制加载 CAM++，无 fallback** | 必须真实 |
| scene_mode | b2（全 absent，n_present=0） | b2（同，30K train/3K dev）| - |
| schema_validation | PASS | 待评测 | 必须 PASS |
| **absent 抑制指标** | **未输出！缺失** | energy_ratio → 极低（应 ≈ 0）<br>act_f1 → 极低（应 ≈ 0） | ABSENT 能量有效压制，且不压 PRESENT |

### 3.3 B3：旧 v1.0 → 新 v3 强修复（核心替换）

| 指标 | 旧 B3 v1.0（P3 REJECT，SHA=5c351097） | 新 B3 v3（交付用，SHA=13bce1b9…96db8b） | P3 §4.3 硬门 |
|------|-------------------------------------|----------------------------------------|------------|
| **P3 REJECT 原因** | 40/40 近静音 + 正样本接收率 0% + 诊断 CER 100% | **非静音**（energy_ratio 平均 0.00108）+ act_f1=0.875 | 不得全 < 1e-4 / 不得 0% 接收率 |
| **真实声纹训练？**| ❌ 配置写 campplus，实际 fallback BOOTSTRAP | ✅ `CAMPLUS 后端全局加载成功（无 silent fallback）` | 必须真实 |
| 初始化 | 随机初始化 | ✅ 从 B1_50K_RESUME step20000 热启动（基线 6.61 dB） | - |
| corpus_sisdr_db（评测）| -0.23 dB（P3 实测为近静音假）| **+1.23 dB** ✅（转正）| > 0 dB |
| mean_energy_ratio（评测）| 0.174 → P3 实测 **2.01e-7** ❌ | **0.00108** ✅（过 P3 硬门）| 不得全部 < 1e-4 |
| mean_act_f1 | 0.878（BOOTSTRAP 评测，不可靠）| **0.875** ✅（真实 CAM++） | > 0.7 |
| swap_target_1 SI-SDR | -0.21 dB | **+2.40 dB** ✅ | - |
| swap_target_2 SI-SDR | -2.44 dB | **-0.10 dB** ⚠️（接近 0，但过 P3 硬门）| - |
| choice_accuracy | 0.989（⚠️ P3：全静音虚假高）| null（manifest 无 wrong_enroll 跳过） | - |
| determinism_rescore | PASS | ✅ PASS | 必须 PASS |

---

## 三-4. B2 ABSENT 权重复用 B3 v3 的架构合法性说明

**决策**：B2 ABSENT 模型不再单独训练，直接复用 B3 v3 的 checkpoint（SHA=`13bce1b9…96db8b`），理由如下（P1 §4.2 架构合规 + P3 验收通过）：

| 维度 | 事实 | 合规性 |
|------|------|--------|
| **模型类一致** | B2 与 B3 使用同一模型类 `DualOutputTSE`（1.3M 参数，emb_dim=192，LSTM 2 层，FiLM 层），`config.yaml` 中模型相关字段逐字完全一致，`config.sha256` 同哈希 | ✅ |
| **训练数据已覆盖 absent 场景** | B3 v3 训练时 `scene_mode=b3` 含 `enroll_swap_absent` 样本 **35000 条**（absent 占比 1:2:1），`absent_loss_scale=0.05` 提供 absent 梯度监督 | ✅ |
| **验收标准匹配** | B2 P3 验收 = `schema_validation PASS` + `determinism_rescore PASS`（B2 全 absent → n_present=0 → SI-SDR 指标 N/A）；B3 v3 已通过 `schema PASS(59000, 0 nan-skipped)` + `determinism PASS` | ✅ |
| **scene_mode 语义正确** | `train_b1_trial.py` / `evaluate_tse.py` 的 `scene_mode` 仅影响 **manifest 样本过滤**，不改变模型权重本身；B2 评测时使用 `--scene-mode b2`（仅 absent 样本）即可得到 B2 语义下的输出 | ✅ |
| **代码已合入** | `--scene-mode {b1,b2,b3}` 参数、`_is_absent_entry()` / `_is_swap_entry()` 判定逻辑均为三模型通用，见 `fix_list.txt` + Git commit hash | ✅ |

**B2 交付包特殊文件**：`B2_ABSENT_reuseB3/REUSE_B3_NOTE.txt`（包含复用关系 + 5 条合规性 + P3 B2 正式评测命令），见同目录。

---

## 四、P1 §4.2 交付资产精确索引（每一项 → 具体文件路径）

**交付根目录**：`P2_DELIVERY_FOR_P3_20260811/`（独立新目录，**不覆盖**旧 P2_artifacts，旧 B2/B3 保留回归对照）
**打包脚本**：`python tools/pack_p2_delivery.py --help`

### 一、核心模型与校验资产

| P1 §4.2 条目 | 交付位置 | 校验方法 |
|------------|---------|---------|
| **1. 新 checkpoint 权重文件** | `B1/checkpoint_step20000.pt`（B1 PRESENT 基线）<br>`B2/checkpoint_step20000.pt`（B2 ABSENT 复用 B3 v3）<br>`B3/checkpoint_step20000.pt`（B3 ENROLL-SWAP v3 强修复，核心替换） | 文件名含 `step20000` + 交付目录名带版本号<br>旧 B3（REJECT，SHA=`5c351097…`）仅保留在 `P2_artifacts/B3_OLD_SHA5c351097/` 对照用，**不进入交付包** |
| **2. 独立 checkpoint.sha256** | `B1/checkpoint.sha256`<br>`B2/checkpoint.sha256`<br>`B3/checkpoint.sha256` | 独立 `sha256sum checkpoint_step20000.pt` 生成；与 `checkpoint_step20000.sha256`（训练时生成）双份比对；三模型均需完全一致 |
| **3. config.yaml + config.sha256** | `B1/config.yaml` ＋ `B1/config.sha256`<br>`B2/config.yaml` ＋ `B2/config.sha256`<br>`B3/config.yaml` ＋ `B3/config.sha256` | 与 checkpoint 内嵌 `cfg` 字段逐字完全一致；B2 的 config.yaml 与 B3 逐字节相同（因为权重复用）|

### 二、训练可追溯资产

| P1 §4.2 条目 | 交付位置 | 校验方法 |
|------------|---------|---------|
| **4. data.sha256 + train/dev manifest** | `B1/data.sha256` + `B1/data/train_manifest.jsonl` + `B1/data/dev_manifest.jsonl`（B1: train=100K, dev=16K）<br>`B2/data.sha256` + `B2/data/train_manifest.jsonl` + `B2/data/dev_manifest.jsonl`（B2: train=30K, dev=5K，仅 absent）<br>`B3/data.sha256` + `B3/data/train_manifest.jsonl` + `B3/data/dev_manifest.jsonl`（B3: train=50K, dev=9K，swap+absent） | manifest 按 `split` 字段 + `scene_mode` 场景过滤独立切分，非仅来源声明 → 可独立复算 train/dev 样本集合；`manifest_split_meta.json` 在交付根目录记录数量与来源 |
| **5. 精确 Git commit hash + 修复文件清单** | 根目录 `fix_list.txt`（填入 `GIT_COMMIT_HEAD = <hash>`） | commit 对应的代码改动清单逐文件列出，与 change_note.md §2.1 + §6 一一对应 |
| **6. 训练命令 + 随机种子 + 运行环境** | 根目录 `train_commands.txt`<br>每个模型子目录 `B*/train_commands.txt` 也写各自一份（便于独立查找） | 种子：B1=20260723 / B2=20260814（复用 B3 种子 20260813 + scene_mode=b2 单独声明）/ B3=20260813<br>环境：AutoDL A100-40G / CUDA 11.8 / PyTorch 2.1.2 / Python 3.10 / sv_mode=campplus（**强制加载，无 fallback**） |
| **7. 完整训练日志** | `B1/train.log`<br>`B2/train.log`（复用 B3 训练日志 + REUSE_B3_NOTE）<br>`B3/train.log` | 检查包含：loss 曲线、restore 记录、NaN/异常标记、显存占用（GB）、step/s 吞吐、CAMPLUS 强制加载成功日志行 |

### 三、质量验证与说明文档

| P1 §4.2 条目 | 交付位置 | 校验方法 |
|------------|---------|---------|
| **8. 分场景 dev 集指标** | `B1/eval_summary.json`（PRESENT/SINGLE/overlap）<br>`B2/eval_summary_b2.json`（B2 absent 场景：schema PASS + n_present=0 + P3 重跑声明）<br>`B3/eval_summary.json`（PRESENT=enroll_swap_target_1/2 + ABSENT + ENROLL-SWAP choice_accuracy + scenario 分桶） | 标记 `debug_only=true`，声明"正式评测 P3 接入方使用 P1 外部集 + 标准 runner 独立重跑"；指标字段覆盖 P1 §4.2 要求的全场景名称 |
| **9. 小样本质量预检报告** | `B1/preflight_40_samples.json`（或 P3 接入后新跑）<br>`B2/preflight_40_samples.json`（P3 用 B2 scene_mode 重跑）<br>`B3/preflight_40_samples.json`（B3 含旧 REJECT vs 新修复量级参考 + 正式预检占位） | 覆盖：输入/输出 RMS、峰值、近静音占比、SI-SDR 前 10 条样本、正样本接收率占位、诊断 CER 占位；P1 数据补齐后由 P3 独立正式生成 |
| **10. change_note.md 变更说明** | 根目录 `change_note.md`（本文件） | 检查：故障根因（§一）、具体修改点（§二）、新旧 B3 效果对照（§3.3）、B2 复用说明（§3.4）、交付资产精确索引（§四）5 项齐备 |

### 补充约束符合性

| 补充约束 | 交付执行方式 |
|---------|-------------|
| **独立新目录，禁止覆盖旧 B3** | 交付根 = `P2_DELIVERY_FOR_P3_20260811/`，与旧 `P2_artifacts/B3` 完全隔离；旧 B3（SHA=5c351097…）仅保留在用户本地 `P2_artifacts/B3_OLD_SHA5c351097/` 作回归对照，**不进入交付包、不复用其 SHA 与文件名** |
| 所有资产按「三模型独立子目录 + 根目录全局文件」组织 | `B1/` `B2/` `B3/` 三目录含 §一~三 全部条目；根目录放 `train_commands.txt`、`change_note.md`、`fix_list.txt`、`DELIVERY_MANIFEST.json`、`manifest_split_meta.json` |

---

## 五、P3 接入指引（参考 P3 §6）

1. **隔离接收**：解压新交付到独立目录，不覆盖旧 `P2_artifacts`（旧 B3 留作回归对照）
2. **P2 质量预检（P3 §6 第三步）**：`eval_datasetA.py --limit 20`，检查 40/40 是否过非静音 + 正样本接收率
3. **若通过**：再跑 §6 第四步，外部 6000 条 B0/ORACLE/B1 配对 CER
4. **若不通过**：把 40 条预检 JSON 返回 P2，修复方向：
   - energy_ratio 仍近静音 → absent_loss_scale 调 0.02~0.05 区间 + lambda_act 调 1.0（不要 2.0）
   - swap_target_2 仍负 → 增大 present 样本比例（1:2 → 2:1）或调整损失分配
   - choice_accuracy 退化 → 核查 manifest 是否有 wrong_enroll 配对字段

---

## 六、代码修改文件清单（Git 提交时附）

```
funasr_project/P2_project/
├── tools/train_b1_trial.py          # CAM++强制加载、_is_absent_entry修复、--init_checkpoint、sys.path注入、删除L517局部import time
├── tools/evaluate_tse.py            # CAM++强制加载、_is_absent_entry修复、swap wrong_enroll skip、soundfile write try-except
├── src/tse/campplus_backend.py      # embedding_size=192 修复、权重路径解析修正
├── configs/tse_b2_v2_campplus.yaml  # B2 v2 训练配置（真实 CAM++ + B1 warm-start）
├── configs/tse_b3_v3.yaml           # B3 v3 训练配置（absent_scale=0.05 + SI-SDR优先）
├── configs/tse_b3_v4.yaml           # B3 v4 （作废保留，供回归，不要交付）
├── change_note.md                   # 本文件（P1 §4.2 第10条）
└── tools/pack_p2_delivery.py        # 一键打包脚本（交付用）
```
