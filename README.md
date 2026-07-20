# FunASR 目标发音人抗干扰语音指令识别

当前版本：`v0.3.3-competition-compliance`

本项目面向“目标发音人语音指令识别 + 非目标发音人拒识”任务。在远场噪声、多人重叠语音和非目标说话人干扰下，系统只输出目标说话人的识别文本；非目标说话人则输出空字符串。

核心代码位于 [funasr_project](./funasr_project)。完整操作手册见 [funasr_project/README.md](./funasr_project/README.md)，ASR 实验结果见 [ASR 公开数据实验记录](./funasr_project/docs/ASR_PUBLIC_DEV.md)。

## 任务指标

- CER：使用 `DataSetA/pos` 计算目标发音人的字错率，越低越好。
- RR：使用 `DataSetA/neg` 计算非目标语音的正确拒识率，越高越好。
- 推理效率：关注单条推理时间和内存占用。

`DataSetA` 用于阶段性开发、临时排行榜和结果提交；测试集 B 才是最终脚本核查与排位依据。两者的 `识别文本` 都是标签，不能作为模型输入、训练数据或短语库先验。B 集不会提供 `pos/neg` 标识或 `识别文本`。

## 系统流程

```text
唤醒音频 + 识别音频
        |
        v
CAM++ 目标说话人验证
        |
        +-- 非目标说话人 -> 输出 ""
        |
        v
FSMN-VAD + Paraformer-large ASR
        |
        +-- 可选：CAM++ 引导的目标语音净化
        |
        v
文本归一化 + 指令短语匹配 + 轻量门控
        |
        v
识别文本或空字符串
```

主要模型：Paraformer-large ASR、FSMN-VAD、CAM++ 声纹验证，以及 Logistic Regression 轻量拒识门控。

## 版本记录

### v0.3.2-datasetA-baseline

- 新增 DataSetA 全量公平端到端基线：1,364 条正样本、474 条负样本。
- 公平配置为硬声纹门控 `--sv-threshold 0.30`，禁用意图过滤和短语纠错。
- 基线结果：语料级 CER `52.87%`、RR `91.14%`（432/474）、端到端耗时 `408.4 s`。
- 明确区分 AISHELL 纯净开发 CER、公开重叠语音 smoke test 与 DataSetA 比赛端到端指标。

### v0.3.3-competition-compliance

- 根据比赛 FAQ 修正 DataSetA 与测试集 B 的职责边界：A 用于阶段验证，B 用于最终核查。
- CER 默认按原始字符编辑距离进行本地调试；负样本不得进入 CER，正样本误拒识按删除错误计入 CER。
- DataSetA 提交 JSON 统一为 `results / avg_cer / avg_rr / duration(ms)`，并保留输入的原始 `id`。
- 新增 `predict_jsonl.py`，只使用 `id`、唤醒音频、唤醒文本和识别音频，适配没有标签和 `pos/neg` 标识的 B 集输入。
- 新口径全量阶段复测：本地原始字符 CER `119.17%`、RR `91.14%`（432/474）、端到端耗时 `618.4 s`；以主办方 scorer 的 B 集结果为最终依据。

### v0.3.1-asr-dev

- 修复 AISHELL-1 内部说话人分卷的自动展开，并避免解压全部语料。
- 新增纯净、说话人隔离的 ASR 训练/开发清单构建脚本。
- 新增独立 ASR CER 评测入口和公开开发实验记录。
- 为目标语音净化加入 `--purify-sim-trigger`，可仅对低相似度语音启用。
- 公开开发基线：100 条纯净开发语音的 CER 为 `1.19%`，平均 ASR 推理为 `0.363 s/条`。
- 公开重叠语音 smoke test：目标净化将 CER 从 `35.43%` 降至 `25.56%`。

### v0.3.0-public-data

- 支持 AISHELL-1 国内镜像、ModelScope 备源、下载进度、断点续传和代理。
- 将公开语料转换为比赛所需的 `pos/neg/jsonl/phrase_bank.txt` 格式。
- 保持 DataSetA 与外部训练集隔离，并支持外部短语库和轻量门控训练。

## 快速开始

```powershell
cd funasr_project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

构建公开训练数据：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1
```

对 DataSetA 进行全量阶段基线评测：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct
```

当前全量结果：本地原始字符 CER `119.17%`（9,515 个参考字）、RR `91.14%`、正样本接收率 `69.35%`、耗时 `618.4 s`（约 `0.336 s/条`）。这是 DataSetA 阶段基线，不代表测试集 B 的最终成绩；历史 v0.3.2 的 `52.87%` 使用项目文本归一化，不能直接横向比较。

测试集 B 到达后，使用无标签推理入口：

```powershell
.\.venv\Scripts\python.exe predict_jsonl.py `
  --input-jsonl data\datasetB\input.jsonl `
  --audio-root data\datasetB `
  --output-jsonl outputs\datasetB_predictions.jsonl
```

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `funasr_project/prepare_public_dataset.py` | 下载 AISHELL-1 并转换比赛格式 |
| `funasr_project/prepare_asr_finetune_manifest.py` | 构建纯净、说话人隔离的 ASR 清单 |
| `funasr_project/eval_asr_manifest.py` | 在纯净开发集计算 ASR CER 和延迟 |
| `funasr_project/eval_datasetA.py` | 端到端 CER、RR 和耗时评测 |
| `funasr_project/predict_jsonl.py` | 测试集 B 无标签、无 pos/neg 标识的推理入口 |
| `funasr_project/train_lightweight_gate.py` | 训练轻量拒识门控 |
| `funasr_project/docs/ASR_PUBLIC_DEV.md` | ASR 公开数据实验与结论 |

## 文档

- [完整使用手册](./funasr_project/README.md)
- [DataSetA 公平调参说明](./funasr_project/docs/DATASET_A_FAIR_TUNING.md)
- [ASR 公开开发记录](./funasr_project/docs/ASR_PUBLIC_DEV.md)
- [轻量门控训练计划](./funasr_project/docs/LIGHTWEIGHT_TRAINING_PLAN.md)

## 团队协作

仓库负责人需先在 GitHub `Settings -> Collaborators` 邀请队员。队员接受邀请后，应从 `main` 创建自己的功能分支、提交并推送分支，再通过 Pull Request 合并；不要直接推送 `main`。详细本地协作规范见 [项目协作要求](./funasr_project/README.md#团队协作要求)。

数据集、模型缓存、虚拟环境和评测输出均保留在本地，不上传 GitHub。
