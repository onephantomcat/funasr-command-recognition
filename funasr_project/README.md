# 复杂交互场景抗干扰语音指令识别系统

当前版本：`v0.3.0-public-data`

本项目面向“目标发音人语音指令识别 + 非目标发音人拒识”的比赛任务。系统输入一轮交互中的唤醒音频、唤醒文本和识别音频，输出目标发音人的识别文本；若识别音频来自非目标发音人，则输出空字符串。

当前代码重点覆盖三类指标：

- 目标发音人识别字错率 CER：用于 `DataSetA/pos`。
- 非目标语音拒识率 RR：用于 `DataSetA/neg`。
- 推理效率：缓存调参、轻量门控、ASR 前拒识减少无效识别。

## 核心方案

```text
唤醒音频 + 识别音频
        |
        v
CAM++ 声纹注册/验证
        |
        +-- 低相似度：拒识，输出 ""
        |
        v
FSMN-VAD + Paraformer ASR
        |
        v
文本归一化 / 指令短语匹配 / 融合门控
        |
        +-- 判定为目标：输出识别文本
        +-- 判定为非目标：输出 ""
```

使用的主要模型：

- ASR：FunASR Paraformer-large，`iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- VAD：FSMN-VAD，`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
- 声纹验证：CAM++，`iic/speech_campplus_sv_zh-cn_16k-common`
- 轻量门控：本项目训练的 Logistic Regression JSON 小模型

## 本轮主要修改

### v0.3.0-public-data

- 新增公开数据集构建脚本：[prepare_public_dataset.py](./prepare_public_dataset.py)
  - 支持下载 AISHELL-1 并转换成比赛需要的 `pos/neg/jsonl/phrase_bank.txt` 格式。
  - 默认使用国内 OpenSLR 镜像 `openslr.magicdatatech.com`。
  - 支持断点续传、下载进度条、速度和 ETA。
  - 支持 `--proxy http://127.0.0.1:7890` 走本地代理/VPN。
  - 下载连接提前中断时会保留 `.part` 文件，不会将不完整压缩包误标记为已下载。
- 合并 `Fix tar extraction EOFError` 对话中的处理经验：
  - 不完整的 `data_aishell.tgz` 会导致 `gzip EOFError`。
  - 当前下载器会校验已接收字节数，未完整下载时保留 `.part` 供续传。
  - 文档补充了 ModelScope 备用下载、代理设置和隐藏临时目录进度监控方式。
- 新增/完善外部训练集构建脚本：[build_external_trainset.py](./build_external_trainset.py)
  - 将公开语音数据转换为比赛格式。
  - `识别文本` 字段只作为训练/验证标签，不使用 DataSetA 标签泄漏。
- 修改 DataSetA 评测逻辑：[eval_datasetA.py](./eval_datasetA.py)
  - DataSetA 仅作为测试集。
  - 默认不再从 DataSetA 正样本标签构造短语库。
  - 可用 `--phrase-bank` 指定外部训练集短语库。
  - 保留 `--use-test-label-phrase-bank` 仅用于复现实验，不建议作为公平测试成绩。
- 新增说明文档：[docs/DATASET_A_FAIR_TUNING.md](./docs/DATASET_A_FAIR_TUNING.md)
  - 说明 DataSetA 公平使用方式、公开训练集构建和评测命令。
- GitHub 推送配置
  - 大型数据、模型缓存、虚拟环境、评测输出已通过 `.gitignore` 排除。
  - 仓库只保存代码、文档、小型 JSON 模型和必要配置。

## 目录说明

| 文件/目录 | 作用 |
|---|---|
| `asr_demo.py` | VAD + Paraformer ASR 单文件识别 |
| `speaker_verify.py` | CAM++ 声纹注册、相似度计算和拒识 |
| `eval_datasetA.py` | DataSetA 正负样本评测，输出 CER/RR/耗时 |
| `prepare_public_dataset.py` | 下载公开数据集并转换成比赛训练格式 |
| `build_external_trainset.py` | 从本地 wav/transcript 构建外部训练集 |
| `train_lightweight_gate.py` | 训练轻量拒识门控模型 |
| `lightweight_gate.py` | 轻量门控模型推理 |
| `command_match.py` | 指令短语模糊匹配 |
| `text_norm.py` | 中文文本归一化 |
| `cer.py` | 字错率 CER 计算 |
| `models/` | 小型门控模型 JSON |
| `docs/` | 调参和数据集说明 |
| `data/` | 本地数据目录，默认不上传 GitHub |

## 环境安装

建议在 Windows + Python 虚拟环境中运行：

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

模型会由 ModelScope/FunASR 下载到本机缓存。已下载过的模型后续可离线复用。

## 构建公开训练集

比赛要求 DataSetA 作为测试集，因此训练集需要使用外部公开数据。当前默认使用 AISHELL-1。

普通下载，默认国内镜像：

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1
```

如果下载仍然很慢，并且本地代理/VPN 端口是 `7890`：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1 --source auto --proxy http://127.0.0.1:7890
```

下载中断后重新运行同一命令即可断点续传。脚本完成后会生成：

```text
data/public_train/aishell1/
  pos/
  neg/
  pos.jsonl
  neg.jsonl
  phrase_bank.txt
```

### 使用 ModelScope 下载（推荐的备用方式）

如果 OpenSLR 下载不稳定，可改用 ModelScope 的 AISHELL-1 镜像。先在项目目录执行：

```powershell
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
.\.venv\Scripts\python.exe -m modelscope.cli.cli download --dataset OmniData/AISHELL-1 --local_dir data\public\aishell1\modelscope --max-workers 4
```

代理未使用时，删除前两行即可。ModelScope 下载完成后，复用下载到的归档文件构建训练集：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --archive data\public\aishell1\modelscope\raw\33\data_aishell.tgz --skip-download --out data\public_train\aishell1
```

下载过程中若要按文件大小监控进度，扫描时需包含 ModelScope 的隐藏临时目录：

```powershell
Get-ChildItem data\public\aishell1\modelscope -Recurse -File -Force |
  Measure-Object -Property Length -Sum
```

样本字段格式：

```json
{"id":"...","唤醒音频":"...","唤醒文本":"hi colmo","识别音频":"...","识别文本":"..."}
```

其中 `识别文本` 是标签。对 DataSetA 测试时，该字段不能作为输入。

## 使用已有本地数据快速验证

如果本机已有 AISHELL 子集，可跳过大文件下载：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --use-existing-local --out data\public_train\aishell1_local
```

也可以直接调用旧的构建器：

```powershell
.\.venv\Scripts\python.exe build_external_trainset.py
```

## DataSetA 公平测试

DataSetA 目录需要包含：

```text
datasetA/
  pos/
  neg/
  pos.jsonl
  neg.jsonl
```

快速 smoke test：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --limit 20 --no-intent-filter --no-phrase-correct
```

使用外部训练集短语库评测：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --phrase-bank data\public_train\aishell1\phrase_bank.txt
```

正式生成提交 JSON：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --phrase-bank data\public_train\aishell1\phrase_bank.txt --submission-out outputs\submission.json
```

调参时可加缓存减少重复计算：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --limit 100 --phrase-bank data\public_train\aishell1\phrase_bank.txt --embedding-cache outputs\emb_cache.pkl --asr-cache outputs\asr_cache.pkl
```

正式计时不要使用缓存结果作为最终耗时依据。

## 训练轻量拒识门控

可用外部训练集训练一个小型门控模型：

```powershell
.\.venv\Scripts\python.exe train_lightweight_gate.py --root data\public_train\aishell1 --split-mode random --train-ratio 0.80 --min-dev-rr 0.98 --model-out models\lightweight_gate_public.json
```

评测时加载该模型：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --phrase-bank data\public_train\aishell1\phrase_bank.txt --gate-model models\lightweight_gate_public.json
```

## 当前注意事项

- DataSetA 是测试集，只能用于最终评测或不泄漏标签的观察。
- 不要用 DataSetA 的 `识别文本` 训练模型、调阈值或构造短语库。
- `--use-test-label-phrase-bank` 是旧实验复现开关，不用于公平提交。
- `data/`、`.venv/`、模型缓存和输出文件不上传 GitHub，避免仓库过大。
- AISHELL-1 是普通朗读语音，不完全等同比赛的远场、重叠语音和拒识场景；后续可继续加入 AISHELL-4、MobvoiHotwords 或自录噪声/重叠数据增强。

## 参考文档

- [DataSetA 公平调参说明](./docs/DATASET_A_FAIR_TUNING.md)
- [轻量门控训练计划](./docs/LIGHTWEIGHT_TRAINING_PLAN.md)
- [ASR 优化报告](./ASR_Optimization_Report.md)
- [项目总结报告](./Project_Summary_Report.md)
