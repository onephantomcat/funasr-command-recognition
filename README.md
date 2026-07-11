# FunASR 目标发音人抗干扰语音指令识别

本仓库是“复杂交互场景抗干扰语音指令识别”项目代码。系统目标是在远场噪声、多人说话和非目标发音人干扰下，只识别目标发音人的语音指令；若输入来自非目标发音人，则输出空字符串完成拒识。

核心工程目录：

```text
funasr_project/
```

完整使用说明见：[funasr_project/README.md](./funasr_project/README.md)

## 任务指标

- 目标发音人识别字错率 CER：`DataSetA/pos` 用于测试识别文本准确率。
- 非目标语音拒识率 RR：`DataSetA/neg` 用于测试拒识能力。
- 推理效率：关注推理时间和内存占用。

注意：`DataSetA` 只作为测试集，不能用其中的 `识别文本` 字段训练模型、调阈值或构建短语库。

## 当前方案

```text
唤醒音频 / 识别音频
        |
        v
CAM++ 声纹验证
        |
        +-- 非目标：输出 ""
        |
        v
FSMN-VAD + Paraformer ASR
        |
        v
文本归一化 + 指令匹配 + 融合门控
        |
        v
输出识别文本或空字符串
```

主要模型：

- ASR：FunASR Paraformer-large
- VAD：FSMN-VAD
- 声纹验证：CAM++
- 拒识增强：轻量 Logistic Regression 门控

## 本仓库主要修改

- 增加公开训练集构建脚本：`funasr_project/prepare_public_dataset.py`
- 支持 AISHELL-1 下载并转换为比赛格式：`pos/neg/jsonl/phrase_bank.txt`
- 默认使用国内 OpenSLR 镜像，并支持下载进度条、断点续传和 `7890` 本地代理
- 修改 DataSetA 评测流程，默认不再使用测试集标签构造短语库
- 增加外部短语库参数 `--phrase-bank`
- 增加轻量拒识门控训练与加载流程
- 排除大文件、数据集、虚拟环境和模型缓存，避免 GitHub 仓库过大

## 快速开始

进入项目目录：

```powershell
cd C:\Users\13238\Desktop\挑战杯1号语音识别\新\funasr_project
```

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

构建公开训练集：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1
```

如果需要走本地代理/VPN：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1 --source auto --proxy http://127.0.0.1:7890
```

DataSetA 公平测试：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --phrase-bank data\public_train\aishell1\phrase_bank.txt
```

生成提交结果：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py --root data\datasetA --phrase-bank data\public_train\aishell1\phrase_bank.txt --submission-out outputs\submission.json
```

## 重要文件

| 路径 | 说明 |
|---|---|
| `funasr_project/README.md` | 完整项目说明和使用方法 |
| `funasr_project/eval_datasetA.py` | DataSetA CER/RR 评测入口 |
| `funasr_project/prepare_public_dataset.py` | 公开数据集下载与格式转换 |
| `funasr_project/build_external_trainset.py` | 本地语音数据转比赛格式 |
| `funasr_project/train_lightweight_gate.py` | 训练轻量拒识门控 |
| `funasr_project/asr_demo.py` | ASR 基础识别 |
| `funasr_project/speaker_verify.py` | 声纹验证与拒识 |
| `funasr_project/docs/DATASET_A_FAIR_TUNING.md` | DataSetA 公平调参说明 |

## 数据与大文件说明

以下内容默认不上传 GitHub：

- `datasetA/`
- `funasr_project/data/`
- `.venv/`
- 模型缓存
- 评测输出和中间缓存

公开数据集下载后请保留在本地 `funasr_project/data/` 下使用。
