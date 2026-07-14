# 复杂交互场景抗干扰语音指令识别系统

版本：`v0.3.1-asr-dev`

本目录包含比赛项目的完整实现。系统以唤醒音频注册目标说话人，通过 CAM++ 声纹验证过滤非目标说话人，再对通过的语音执行 VAD、Paraformer ASR、文本归一化和指令匹配。

## 模型与模块

| 模块 | 实现 |
| --- | --- |
| ASR | `iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| VAD | `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| 声纹验证 | `iic/speech_campplus_sv_zh-cn_16k-common` |
| 目标语音净化 | 基于 CAM++ 窗口相似度的训练无关软掩蔽 |
| 拒识门控 | Logistic Regression JSON 模型 |

指令识别默认关闭标点恢复模型，以减少内存占用；CER 前会进行文本归一化。

## 环境安装

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

首次运行会下载 FunASR/CAM++ 模型；下载完成后可复用本地缓存。

## 构建公开训练数据

`DataSetA` 必须保留为测试集。项目使用公开 AISHELL-1 构建外部训练与开发数据。

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

## 训练轻量拒识门控

```powershell
.\.venv\Scripts\python.exe train_lightweight_gate.py `
  --root data\public_train\aishell1 --split-mode random `
  --train-ratio 0.80 --min-dev-rr 0.98 `
  --model-out models\lightweight_gate_public.json
```

## DataSetA 最终评测

DataSetA 目录必须包含：

```text
data/datasetA/
  pos/
  neg/
  pos.jsonl
  neg.jsonl
```

使用外部短语库评测：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --phrase-bank data\public_train\aishell1\phrase_bank.txt
```

生成提交格式 JSON：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --phrase-bank data\public_train\aishell1\phrase_bank.txt `
  --submission-out outputs\submission.json
```

正式推理计时不应使用 `--asr-cache` 或 `--embedding-cache` 的缓存结果。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `asr_demo.py` | VAD + Paraformer 单文件识别 |
| `speaker_verify.py` | CAM++ 声纹注册、相似度和拒识 |
| `eval_datasetA.py` | 端到端 CER/RR/耗时评测 |
| `prepare_public_dataset.py` | 公共数据下载、校验和比赛格式转换 |
| `prepare_asr_finetune_manifest.py` | 纯净 ASR 训练/开发清单 |
| `eval_asr_manifest.py` | 纯净 ASR 开发集 CER 评测 |
| `target_purify.py` | 可选目标语音净化 |
| `docs/ASR_PUBLIC_DEV.md` | ASR 公开开发实验记录 |

## 注意事项

- `DataSetA` 的 `识别文本` 不得用于训练、短语库构建或阈值调优。
- 当前 RTX 4060 Laptop GPU 约有 4GB 显存，不建议全参数微调 Paraformer-large；清单适合后续 LoRA 或冻结前端实验。
- `data/`、`.venv/`、模型缓存和评测输出默认不上传 GitHub。
- 详细开发结论见 [docs/ASR_PUBLIC_DEV.md](./docs/ASR_PUBLIC_DEV.md) 和 [docs/DATASET_A_FAIR_TUNING.md](./docs/DATASET_A_FAIR_TUNING.md)。
