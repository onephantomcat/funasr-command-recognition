# 复杂交互场景抗干扰语音指令识别系统

版本：`v0.3.5-gpu-runtime`

本目录包含比赛项目的完整实现。系统以唤醒音频注册目标说话人，通过 CAM++ 声纹验证过滤非目标说话人，再对通过的语音执行 VAD、Paraformer ASR、文本归一化和指令匹配。

## 模型与模块

| 模块 | 实现 |
| --- | --- |
| ASR | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| 声纹验证 | `iic/speech_campplus_sv_zh-cn_16k-common` |
| 目标语音净化 | 基于 CAM++ 窗口相似度的训练无关软掩蔽 |
| 拒识门控 | Logistic Regression JSON 模型 |

指令识别默认关闭标点恢复模型，以减少内存占用；Paraformer 输出会移除格式空白，避免中文字符间空格形成无意义 CER 插入。检测到 CUDA 时 ASR 与 CAM++ 自动使用 GPU，否则回退 CPU。DataSetA 本地 CER 默认按原始字符计算；`--local-normalize` 仅用于旧结果调试，最终以主办方 scorer 为准。

## 环境安装

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如使用 NVIDIA GPU，请继续安装已验证的 CUDA 运行时版本：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu118.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

项目已在 RTX 4060 Laptop + 驱动 610.62 上验证 `torch 2.7.1+cu118`。无需安装完整 CUDA Toolkit 或 `nvcc`；`asr_demo.py` 和 `speaker_verify.py` 会在可用时自动选择 CUDA。详见 [docs/GPU_SETUP.md](./docs/GPU_SETUP.md)。

首次运行会下载 FunASR/CAM++ 模型；下载完成后可复用本地缓存。

## 构建公开训练数据

DataSetA 用于阶段性验证和临时排行榜，不得混入训练数据。项目使用公开 AISHELL-1 构建外部训练与开发数据；测试集 B 才用于最终脚本核查，且不会提供 `pos/neg` 标识或识别文本标签。

### 国内镜像下载

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py `
  --dataset aishell1 --out data\public_train\aishell1
```

下载器支持进度、断点续传和完整性校验。若本机代理端口为 `7890`：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py `
  --dataset aishell1 --source auto --proxy http://127.0.0.1:7890 `
  --out data\public_train\aishell1
```

### ModelScope 备源

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
.\.venv\Scripts\python.exe -m modelscope.cli.cli download `
  --dataset OmniData/AISHELL-1 `
  --local_dir data\public\aishell1\modelscope --max-workers 4

.\.venv\Scripts\python.exe prepare_public_dataset.py `
  --archive data\public\aishell1\modelscope\raw\33\data_aishell.tgz `
  --skip-download --out data\public_train\aishell1
```

脚本会自动展开构建当前训练集所需的说话人分卷，输出：

```text
data/public_train/aishell1/
  pos/
  neg/
  pos.jsonl
  neg.jsonl
  phrase_bank.txt
```

其中 `pos.jsonl` 的 `识别文本` 是目标说话人标签；`neg.jsonl` 的 `识别文本` 为空，用于拒识训练和验证。

## ASR 开发与评测

不要把重叠语音的目标标签直接用于通用 ASR 微调。`prepare_asr_finetune_manifest.py` 会从原始 AISHELL 音频生成纯净、说话人隔离的 `source/target` 清单：

```powershell
.\.venv\Scripts\python.exe prepare_asr_finetune_manifest.py
```

默认输出 `data/asr_finetune/aishell1_clean/`，当前公开数据构建结果为 4,558 条训练语音和 1,053 条开发语音。

评测未微调 ASR 基线：

```powershell
.\.venv\Scripts\python.exe eval_asr_manifest.py `
  --manifest data\asr_finetune\aishell1_clean\dev.jsonl --limit 100
```

当前 100 条开发语音结果：CER `1.19%`，平均 ASR 推理 `0.363 s/条`。

针对目标说话人重叠语音，可启用净化；`--purify-sim-trigger` 仅对低声纹相似度的已接收语音执行，避免为干净语音增加额外耗时：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\public_train\aishell1 --limit 16 --sv-threshold -1 `
  --no-intent-filter --no-phrase-correct `
  --purify --purify-sim-trigger 0.80
```

公开重叠语音 smoke test 中，净化将 CER 从 `35.43%` 降至 `25.56%`。这是开发阶段结论，DataSetA 不参与该调参。

## DataSetA 阶段基线

DataSetA 用于阶段性开发与临时排行榜。为了避免标签泄漏，本轮基线不构建 DataSetA 短语库，不启用意图过滤或短语纠错：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --out data\datasetA\eval_report_fair_v032_full.json
```

| 项目 | 全量公平基线 |
| --- | ---: |
| 正样本 / 负样本 | 1,364 / 474 |
| 本地原始字符 CER | 53.43%（9,515 个参考字） |
| 正样本接收率 | 69.35% |
| RR | 91.14%（432 / 474） |
| 端到端耗时 | 438.9 s，约 0.239 s/条 |
| 本机进程工作集 | 3,073.22 MB |

此结果不同于 AISHELL 纯净开发集的 1.19% CER，也不同于公开混叠 smoke test 的 25.56% CER：前两者分别衡量基础转写能力与净化增益；本表只衡量 DataSetA 的阶段端到端效果，包含声纹门控的误拒绝损失，不代表 B 集最终成绩。v0.3.3 的 119.17% 是未清理 Paraformer 格式空白时的原始字符基线；v0.3.4 在推理输出侧统一移除空白，不涉及标签、词表或短语纠错。

`eval_datasetA.py` 的后续报告会附带本机进程工作集内存与 CUDA 峰值已分配显存；这两个字段用于调试部署占用，最终效率评分仍以主办方统一硬件和计时方式为准。

对于重叠语音，可选用经过公开开发集验证的鲁棒模式：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --purify --purify-keep-ratio 0.45 --purify-floor-gain 0.03
```

该模式会增加约三倍前端耗时，默认快速路径不会启用。实验细节和后续外部训练计划见 [docs/ASR_OPTIMIZATION_V034.md](./docs/ASR_OPTIMIZATION_V034.md)。

## 训练轻量拒识门控

```powershell
.\.venv\Scripts\python.exe train_lightweight_gate.py `
  --root data\public_train\aishell1 --split-mode random `
  --train-ratio 0.80 --min-dev-rr 0.98 `
  --model-out models\lightweight_gate_public.json
```

## DataSetA 结果提交

DataSetA 目录必须包含：

```text
data/datasetA/
  pos/
  neg/
  pos.jsonl
  neg.jsonl
```

全量阶段评测：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct
```

生成 DataSetA 临时排行榜 JSON：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct `
  --submission-out outputs\submission.json
```

正式推理计时不应使用 `--asr-cache` 或 `--embedding-cache` 的缓存结果。

## 测试集 B 无标签推理

测试集 B 预计提供输入 JSONL，其中每条只包含 `id`、`唤醒音频`、`唤醒文本` 和 `识别音频`。不要根据 DataSetA 的 `pos/neg` 目录或标签设计推理分支。使用以下脚本输出只含 `id` 和 `content` 的 JSONL：

```powershell
.\.venv\Scripts\python.exe predict_jsonl.py `
  --input-jsonl data\datasetB\input.jsonl `
  --audio-root data\datasetB `
  --output-jsonl outputs\datasetB_predictions.jsonl
```

正式 B 集提交格式和脚本封装以主办方后续通知为准；不要自行填充未知的 `label`、CER、RR 或内存字段。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `asr_demo.py` | VAD + Paraformer 单文件识别 |
| `speaker_verify.py` | CAM++ 声纹注册、相似度和拒识 |
| `eval_datasetA.py` | 端到端 CER/RR/耗时评测 |
| `predict_jsonl.py` | B 集无标签输入的声纹门控 + ASR 推理 |
| `prepare_public_dataset.py` | 公共数据下载、校验和比赛格式转换 |
| `prepare_asr_finetune_manifest.py` | 纯净 ASR 训练/开发清单 |
| `eval_asr_manifest.py` | 纯净 ASR 开发集 CER 评测 |
| `target_purify.py` | 可选目标语音净化 |
| `docs/ASR_PUBLIC_DEV.md` | ASR 公开开发实验记录 |

## 团队协作要求

仓库负责人在 GitHub `Settings -> Collaborators` 邀请队员；队员必须先接受邀请，才能推送自己的分支。

首次获取项目：

```powershell
git clone https://github.com/onephantomcat/funasr-command-recognition.git
cd funasr-command-recognition
git checkout main
git pull origin main
```

每项工作都从最新 `main` 创建独立分支。分支名使用 `feature/<姓名或模块>-<工作内容>`，例如 `feature/zhangsan-asr-purify`：

```powershell
git checkout -b feature/<name>-<task>
# 修改并验证代码
git add <changed-files>
git commit -m "Describe the change"
git push -u origin feature/<name>-<task>
```

推送后在 GitHub 创建 Pull Request，写明改动、验证命令和结果，由至少一名队员检查后再合并。`main` 只接受已审核的 Pull Request，不直接推送。

- 合并前先同步最新 `main`，处理冲突后重新验证。
- 不提交 `data/`、模型缓存、`.venv/`、评测输出、音频数据或 Office 临时文件。
- 不得将 DataSetA 的 `识别文本` 用作训练、短语库、阈值调优或提交前的标签泄漏。
- 每次修改评测逻辑都要说明使用的配置；正式成绩必须关闭缓存并保留报告文件名。

## 注意事项

- DataSetA 与 B 集的 `识别文本` 都不得用于模型输入、训练或短语库构建；B 集不会提供该字段。
- 不要将 DataSetA 的 `pos/neg` 目录作为推理先验；B 集可能混合正负样本。
- 当前 RTX 4060 Laptop GPU 约有 4GB 显存，不建议全参数微调 Paraformer-large；清单适合后续 LoRA 或冻结前端实验。
- `data/`、`.venv/`、模型缓存和评测输出默认不上传 GitHub。
- 详细开发结论见 [docs/ASR_PUBLIC_DEV.md](./docs/ASR_PUBLIC_DEV.md) 和 [docs/DATASET_A_FAIR_TUNING.md](./docs/DATASET_A_FAIR_TUNING.md)。
