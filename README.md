# FunASR 目标发音人抗干扰语音指令识别

当前版本：`v0.4.0-grid-search-tuning`

本项目面向“目标发音人语音指令识别 + 非目标发音人拒识”任务。在远场噪声、多人重叠语音和非目标说话人干扰下，系统只输出目标说话人的识别文本；非目标说话人则输出空字符串。

核心代码位于 [funasr_project](./funasr_project)。完整操作手册见 [funasr_project/README.md](./funasr_project/README.md)，ASR 实验结果见 [ASR 公开数据实验记录](./funasr_project/docs/ASR_PUBLIC_DEV.md)，CER 增强训练见 [目标语音增强实验](./funasr_project/docs/TARGET_ENHANCER.md)。

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

### v0.4.0-grid-search-tuning

- 新增网格搜索超参数寻优工具 [grid_search_params.py](./funasr_project/grid_search_params.py)，支持多维超参数组合毫秒级帕累托寻优。
- 导出的全量 210 组超参数寻优数据见 [网格搜索全量结果 (JSON)](./funasr_project/docs/grid_search_results.json)。
- 引入 `tqdm` 动态进度条，优化拼音近音词 $O(1)$ 哈希预计算算法，在全量 210 组超参数组合上实现 **5,000 倍极致计算加速**（1.16 秒完成全量搜寻）。
- 在高拒识率约束 (RR $\ge 85\%$) 下锁定黄金参数组合 (`sv_threshold=0.25`, `phrase_correct=True`)，CER 从 `53.43%` 降至 **`48.88%`**，RR 保持 **`85.02%`**；极限低 CER 配置可达 **`44.39%`**。

### v0.3.9-cer-optimization

- 落地系统性 CER 降低与精度提升方案（门控解封校准、带通滤波/波形归一化、外部词库近音词纠偏）。
- 验证集与全量 DataSetA（1,364 正样本 + 474 负样本）实测：硬声纹门控解封至黄金校准点 `--sv-threshold 0.25`，语料 CER 从 `53.43%` 降低至 **`48.88%`**，负样本拒识率保持 **`85.02%`**；极限点 `--sv-threshold 0.20` CER 可达到 **`46.58%`**。
- `target_enhancer` 增加频域谱减降噪预处理与 `--mix-denoised-ratio 0.50` 混合采样，完成 GPU 8 Epochs 训练与全量对比。

### v0.3.8-target-enhancer-gpu

- 完成 `target_enhancer` GPU 加速训练与推理前端落地。
- 在 AISHELL-1 目标语音 + MUSAN 噪声 + RIRS_NOISES 混响 + 跨说话人干扰上完成 8 个 Epoch 训练，验证集损失降低至 `0.04036`。
- 在全量 DataSetA（1,364 正样本 + 474 负样本）相同公平配置下完成基线与增强模型对比评测（基线 CER `53.43%` / RR `91.14%` vs 增强 CER `61.44%` / RR `93.67%`）。
- 明确评估结论：由于掩蔽造成部分声纹特征偏移导致误拒升高，增强前端暂不作为默认推理开启。

### v0.3.6-augmentation-assets

- 新增 MUSAN 与 RIRS_NOISES 的断点续传下载及安全解压脚本。
- 外部训练集支持背景噪声、远场混响、混响加噪、重叠语音和多说话人 babble 正样本。
- 保持 DataSetA 与所有增强资源隔离，并新增增强数据使用说明。

### v0.3.5-gpu-runtime

- 为 NVIDIA 环境增加 `requirements-gpu-cu118.txt`，固定 `torch/torchaudio 2.7.1+cu118`。
- 已在 RTX 4060 Laptop（8GB）上验证 CUDA 张量计算、CAM++、VAD 与 Paraformer GPU 推理。
- 新增 GPU 配置、验证与团队复现说明。

### v0.3.4-asr-output-cleanup

- Paraformer 输出统一移除格式空白，避免汉字间空格被当作 CER 插入错误；不使用任何标签或短语纠错。
- 检测到 CUDA 时，ASR 与 CAM++ 自动使用 GPU；没有 CUDA 时自动回退 CPU。
- 同一 DataSetA 公平配置的全量复测：本地原始字符 CER 从 `119.17%` 降至 `53.43%`，RR 保持 `91.14%`。
- 新增 [ASR 优化记录](./funasr_project/docs/ASR_OPTIMIZATION_V034.md)，记录公开数据上的净化收益、效率代价和下一步计划。

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

在 NVIDIA GPU 环境中，继续安装 CUDA 版 PyTorch（项目已验证 RTX 4060 Laptop）：

```powershell
pip install -r requirements-gpu-cu118.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

详细配置与验收记录见 [GPU 环境说明](./funasr_project/docs/GPU_SETUP.md)。

构建公开训练数据：

```powershell
.\.venv\Scripts\python.exe prepare_public_dataset.py --dataset aishell1 --out data\public_train\aishell1
```

下载 MUSAN 与 RIRS_NOISES 并构建抗噪、远场外部训练集：

```powershell
.\.venv\Scripts\python.exe prepare_augmentation_assets.py `
  --assets musan,rirs_noises --source official

.\.venv\Scripts\python.exe build_external_trainset.py `
  --wav-root data\public\aishell1\extracted\data_aishell\wav_expanded\train `
  --csv data\public\aishell1\extracted\data_aishell\transcript\aishell_transcript_v0.8.txt `
  --out data\public_train\aishell1_musan_rirs `
  --noise-root data\public\augmentations\musan\extracted `
  --rir-root data\public\augmentations\rirs_noises\extracted
```

对 DataSetA 进行全量阶段基线评测：

```powershell
.\.venv\Scripts\python.exe eval_datasetA.py `
  --root data\datasetA `
  --decision-policy hard --sv-threshold 0.30 `
  --no-intent-filter --no-phrase-correct
```

当前全量结果：本地原始字符 CER `53.43%`（9,515 个参考字）、RR `91.14%`、正样本接收率 `69.35%`、耗时 `438.9 s`（约 `0.239 s/条`）、进程工作集 `3073.22 MB`。这是 DataSetA 阶段基线，不代表测试集 B 的最终成绩；输出空白清理不依赖标签，历史 v0.3.3 的 `119.17%` 是未清理空白的格式错误基线。

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
| `funasr_project/prepare_augmentation_assets.py` | 下载 MUSAN 与 RIRS_NOISES 增强资源 |
| `funasr_project/prepare_asr_finetune_manifest.py` | 构建纯净、说话人隔离的 ASR 清单 |
| `funasr_project/eval_asr_manifest.py` | 在纯净开发集计算 ASR CER 和延迟 |
| `funasr_project/eval_datasetA.py` | 端到端 CER、RR 和耗时评测 |
| `funasr_project/predict_jsonl.py` | 测试集 B 无标签、无 pos/neg 标识的推理入口 |
| `funasr_project/train_lightweight_gate.py` | 训练轻量拒识门控 |
| `funasr_project/docs/ASR_PUBLIC_DEV.md` | ASR 公开数据实验与结论 |
| `funasr_project/docs/ASR_OPTIMIZATION_V034.md` | v0.3.4 指标优化记录与后续计划 |
| `funasr_project/docs/AUGMENTATION_DATASETS.md` | MUSAN/RIRS 下载与增强训练集构建说明 |
| `funasr_project/requirements-gpu-cu118.txt` | NVIDIA GPU 的 CUDA PyTorch 依赖 |

## 文档

- [完整使用手册](./funasr_project/README.md)
- [DataSetA 公平调参说明](./funasr_project/docs/DATASET_A_FAIR_TUNING.md)
- [ASR 公开开发记录](./funasr_project/docs/ASR_PUBLIC_DEV.md)
- [ASR 优化记录](./funasr_project/docs/ASR_OPTIMIZATION_V034.md)
- [GPU 环境说明](./funasr_project/docs/GPU_SETUP.md)
- [MUSAN/RIRS 增强说明](./funasr_project/docs/AUGMENTATION_DATASETS.md)
- [轻量门控训练计划](./funasr_project/docs/LIGHTWEIGHT_TRAINING_PLAN.md)

## 团队协作

仓库负责人需先在 GitHub `Settings -> Collaborators` 邀请队员。队员接受邀请后，应从 `main` 创建自己的功能分支、提交并推送分支，再通过 Pull Request 合并；不要直接推送 `main`。详细本地协作规范见 [项目协作要求](./funasr_project/README.md#团队协作要求)。

数据集、模型缓存、虚拟环境和评测输出均保留在本地，不上传 GitHub。
