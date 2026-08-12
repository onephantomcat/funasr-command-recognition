# P2 模块：Target Speaker Extraction（目标说话人提取）

> FunASR Command Recognition 流水线 P2 子模块。职责：从「命令词 + 干扰声」的混合语音中，利用注册（enrollment）声纹嵌入作为条件，提取目标说话人的干净波形，并输出帧级目标活动度，供下游 P3 ASR / P5 Pipeline 消费。

---

## 1. 模块边界与对外契约

### 输入/输出（P2-06 接口契约）

对外暴露 `src/tse/api.py::extract_target()` 单入口：

| 符号 | 形状 | 类型 | 说明 |
|---|---|---|---|
| `command_wav` | `[B, T]` | `torch.Tensor<float>` | 16 kHz 单声道混合波形 |
| `enroll_embedding` | `[B, D=192]` | `torch.Tensor<float>` | L2 归一化的说话人嵌入向量（CAMPPlus 或 Bootstrap） |
| `model` | `DualOutputTSE` | `nn.Module` | 双掩码 LSTM-FiLM TSE 网络 |
| `cfg` | `dict` | 配置 | 必须含 `emb_dim`，可选 `sample_rate` |
| **返回 `s_tgt`** | `[B, T]` | `torch.Tensor<float>` | 估计的目标说话人波形，**长度严格等于输入** |

错误输入（空数组、NaN/Inf、emb 维度不匹配）抛 `ValueError`，不静默通过。

### 场景模式（scene_mode）

P2 支持三种训练/评测场景，由配置字段 `scene_mode` 决定数据过滤策略：

| 场景 | `scene_mode` | 语义 | `target_present` | force_zero |
|---|---|---|---|---|
| **B1 PRESENT** | `b1` | 目标 speaker 确实出现在混合中，enroll 与目标 speaker 一致（基线） | 全部 True | 否 |
| **B2 ABSENT** | `b2` | 目标 speaker **缺席**（mixture 中无目标） | 全部 False | 是（target = 0）|
| **B3 ENROLL-SWAP** | `b3` | 同一 mixture 配成 target_1、target_2 和 absent 三种 enrollment 角色 | True/False 混合 | 仅 absent 子集 force_zero |

B3 下的三条子场景（P1 v3 manifest `scenario` 字段）：

- `enroll_swap_target_1` / `enroll_swap_target_2` → enrollment 属于 mixture 中对应目标 speaker，`target_present=True`，正常回归 `target_wav`
- `enroll_swap_absent` → enrollment speaker 不在 mixture 中，`target_present=False`，目标为零但 enrollment embedding 仍是真实非零向量

### 声纹嵌入双模式（P2↔P4 适配器）

由 `src/tse/enrollment_adapter.py::EnrollmentAdapter` 统一提供：

| 模式 | `sv_mode` | 依赖 | 嵌入来源 | 适用场景 |
|---|---|---|---|---|
| **BOOTSTRAP** | `bootstrap` | 无（纯 PyTorch） | 基于 `speaker_id` SHA256 哈希的确定性随机向量，L2 归一化 | 仅本地调试 / CI 冒烟 |
| **CAMPLUS** | `campplus` | P4 `speakerlab_source` + `campplus_cn_common.bin` 权重 | `CampplusBackend.embed()` → 192D 真实声纹嵌入 | 正式训练 / 最终评测；失败立即报错 |

正式训练和评测禁止把 CAMPLUS 失败悄悄替换成随机声纹。只有显式 `--debug-data` 的调试训练允许 BOOTSTRAP。

---

## 2. 目录结构

```
funasr_project/P2_project/
├── README.md                       # 本文件
├── verify_p2_local.py              # 本地自检（加载 checkpoint + forward 一致性）
├── src/tse/                        # 模块核心代码
│   ├── __init__.py
│   ├── api.py                      # extract_target() 对外 API
│   ├── model.py                    # DualOutputTSE（LSTM + FiLM + 双 STFT 掩码）
│   ├── losses.py                   # SI-SDR / MR-STFT / activity-BCE / 混合一致性
│   ├── metrics.py                  # 评测指标（corpus/utterance SI-SDR, SI-SDRi, F1, ...）
│   └── enrollment_adapter.py       # BOOTSTRAP / CAMPLUS 嵌入适配器（P4 sv_contract_v1）
├── tools/                          # 训练与评测脚本
│   ├── train_b1_trial.py           # 通用训练器（B1/B2/B3 单脚本，scene_mode 分流）
│   ├── train_overfit_debug.py      # 单样本过拟合调试脚本（纯函数供训练器复用）
│   ├── evaluate_tse.py             # 统一评测器（predictions.jsonl / summary.json / report.md）
│   ├── diagnose_b3.py              # 真实 enrollment / 数据能量 / 可选 checkpoint 输出诊断
│   ├── build_debug_mixtures.py     # 构造 debug_mixtures_v0（P2-07）
│   └── build_v2_b1_local.py        # 本地 P1 v2_b1 路径兼容工具
├── configs/
│   ├── tse_b1.yaml                 # B1 PRESENT 正式配置（P1 v2/v3，CAMPLUS，20K 步）
│   ├── tse_b2.yaml                 # B2 ABSENT 配置（30K train / 3K dev）
│   ├── tse_b3.yaml                 # B3 ENROLL-SWAP 配置（warm-start from B1 推荐）
│   ├── tse_b1_trial.yaml           # B1 500-step 试车配置
│   ├── tse_smoke.yaml              # 最小冒烟
│   └── ...（其它中间配置）
├── tests/
│   ├── smoke_tse_random.py         # CPU 随机张量 forward/loss/backward
│   ├── test_dual_tse_smoke.py
│   ├── test_tse_conditioning.py    # 五条件正确性验证
│   ├── test_tse_metrics.py         # 10 项指标单元测试
│   ├── test_debug_mixtures.py      # debug_mixtures_v0 一致性
│   └── smoke_wesep_upstream.py
├── environment/
│   ├── p2_python_version.txt       # Python 3.10.9
│   └── p2_torch_cuda.json          # torch 2.7.1+cu126 / RTX 4050 锁
├── schemas/
│   └── tse_prediction.schema.json  # evaluate_tse 输出 Schema（12 字段冻结）
├── state/
│   ├── P2_progress.json            # P2 全阶段状态机 + 决策记录
│   └── P2_dependency_register.json # 与 P1/P3/P4/P5/P6 的依赖契约登记
└── artifacts/                      # 实验/调试产物（.gitignore 排除大文件；仅保留 JSON/YAML/MD）
    ├── debug_mixtures_v0/          # P2-07 12 条 4 说话人 DEBUG 混合（可重建）
    └── final/P2_artifacts/
        ├── B1 / B2 / B3/           # config.yaml / report.md / metrics.jsonl / trial_verdict.json
        └── B1_eval / B2_eval / B3_eval / meta.json  # 最终评测汇总
```

---

## 3. 环境与依赖

### 软件版本

| 项 | 值 | 说明 |
|---|---|---|
| Python | 3.10.9 | `environment/p2_python_version.txt` |
| PyTorch | ≥ 2.1（推荐 2.6+） | 见下方 GPU 兼容性说明 |
| CUDA | 11.8 / 12.4 / 12.6 均可 | 依据 GPU 型号选择 |
| 关键三方 | `soundfile`, `numpy`, `pyyaml`, `tqdm` | 无复杂系统依赖 |

⚠️ **GPU 兼容性提示**：NVIDIA Blackwell（sm_120，如 RTX 5090D）需 PyTorch ≥ 2.6；CUDA 11.8 容器无法安装 cu124+ 包，此时改用 RTX 4090/A100 节点或 CPU 训练。

### 推荐安装

```bash
cd funasr_project/P2_project
python -m venv .venv-p2tse && source .venv-p2tse/bin/activate  # Windows: .venv-p2tse\Scripts\activate

# PyTorch（按 CUDA 版本选择）
pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 其它
pip install soundfile numpy pyyaml tqdm scipy
```

### CAMPPlus 模式准备（可选，仅正式训练需要）

1. 克隆官方 3D-Speaker 到 speakerlab_source，并固定 commit `065629c313eaf1a01c65c640c46d77e61e9607b4`：
   ```
   git clone https://github.com/modelscope/3D-Speaker.git funasr_project/P4_project/artifacts/models/speakerlab_source
   git -C funasr_project/P4_project/artifacts/models/speakerlab_source checkout 065629c313eaf1a01c65c640c46d77e61e9607b4

   funasr_project/P4_project/artifacts/models/speakerlab_source/   ← 3D-Speaker 仓库
   funasr_project/P4_project/artifacts/models/campplus_cn_common.bin ← 28 MB 权重
   ```
2. `EnrollmentAdapter` 会自动在加载时将上述路径注入 `sys.path`，无需手动配置。

---

## 4. 快速开始

### 4.1 本地冒烟（可重建调试数据、CPU、BOOTSTRAP）

```bash
cd funasr_project/P2_project
python tools/build_debug_mixtures.py --seed 20260725
python tools/train_b1_trial.py \
    --config configs/tse_b1_trial.yaml \
    --debug-data --scene_mode b1 --max_steps 10 --device cpu
```

预期：训练实际完成反向传播，无 NaN，且 checkpoint restore 一致；极短 smoke 不要求 loss 单调。

### 4.2 三场景正式训练（云端 GPU、CAMPLUS）

```bash
# B1（present，50K → 20K 步收敛）
python tools/train_b1_trial.py --config configs/tse_b1.yaml --device cuda \
    --manifest /root/autodl-tmp/P1_to_P2_v3_absent_swap/manifest.jsonl

# B2（absent，30K train + 3K dev，单 manifest split 过滤）
python tools/train_b1_trial.py --config configs/tse_b2.yaml --device cuda \
    --manifest /root/autodl-tmp/P1_to_P2_v3_absent_swap/manifest.jsonl \
    --init-checkpoint artifacts/experiments/B1_PRESENT_seed20260723/checkpoint_step20000.pt

# B3（enroll-swap，推荐 warm-start from B1 以抑制数据比例失衡导致的静默退化）
python tools/train_b1_trial.py --config configs/tse_b3_v3.yaml --device cuda \
    --manifest /root/autodl-tmp/P1_to_P2_v3_absent_swap/manifest.jsonl \
    --init-checkpoint artifacts/experiments/B1_PRESENT_seed20260723/checkpoint_step20000.pt
```

**B3 必选建议**：直接从 0 训练 B3 容易因「20K present + 10K absent」数据比例失衡塌缩为全零输出；务必使用 `--init_checkpoint` 从 B1 20K 步权重 warm-start。

### 4.3 最终评测

```bash
python tools/evaluate_tse.py \
    --checkpoint artifacts/experiments/B3_SWAP_seed20260723/checkpoint_step20000.pt \
    --manifest   /root/autodl-tmp/eval_manifests/eval_D_swap.jsonl \
    --data-root  /root/autodl-tmp/P1_to_P2_v3_absent_swap \
    --device cuda \
    --out artifacts/eval_B3_campplus
```

输出目录结构（P2-11 冻结口径）：

```
artifacts/eval_B3_campplus/
├── predictions.jsonl           # 逐条 12 字段，schema 校验
├── predictions_detailed.jsonl  # 逐条全量指标
├── summary.json                # 聚合（见下方基准表）
└── report.md                   # 人读 Markdown 报告
```

### 4.4 本地加载与推理（其它模块集成）

P5/Pipeline 端最简调用：

```python
import sys, torch, yaml
sys.path.insert(0, "funasr_project/P2_project")
from src.tse.model import DualOutputTSE
from src.tse.api import extract_target
from src.tse.enrollment_adapter import EnrollmentAdapter

ckpt = torch.load("artifacts/final/P2_artifacts/B3/checkpoint_step20000.pt", map_location="cpu")
cfg  = ckpt["cfg"]
model = DualOutputTSE(cfg).eval()
model.load_state_dict(ckpt["model"], strict=True)

adapter = EnrollmentAdapter.from_config(cfg, mode="campplus")   # mode="bootstrap" 亦可
emb = adapter.encode_file("target", "enroll.wav")               # [1, 192]
mix, sr = sf.read("command_mix.wav", dtype="float32")           # 16k
s_tgt = extract_target(torch.from_numpy(mix).unsqueeze(0), emb, model, cfg)  # [1, T]
```

---

## 5. 历史 20K 资产状态（不可作为当前基准）

> 下表是旧资产记录。P3 已实测旧 B3 在 20 正 + 20 负预检中近静音、正样本接收率 0%，因此 B3 与复用它的 B2 均保持 **REJECT**。旧 `choice_accuracy=0.989` 由无效评测路径产生，已废弃。修复后的代码必须重新训练并由 P3 重评，不能沿用下表数字宣称交付通过。

| 指标 | B1 PRESENT（D_present 10K） | B2 ABSENT（D_absent 3K） | B3 ENROLL-SWAP（D_swap 3K） |
|---|---|---|---|
| 训练规模 | 100K 样本，2239 s | 30K 样本，1371 s | 30K 样本，1401 s（warm-start） |
| 训练总体 verdict | PASS（7/7） | PASS（7/7） | PASS（7/7） |
| peak GPU mem | 1.73 GB | 1.73 GB | 1.75 GB |
| **corpus_sisdr_db**（present 子集） | **5.25 dB** | — | -0.23 dB |
| utterance_sisdr_db | 4.01 dB | — | -1.32 dB |
| mean_sisdri_db | 7.4 dB | — | 2.25 dB |
| mean_wav_l1 | 0.011 | — | 0.0141 |
| **mean_act_f1**（帧活动 F1） | 0.91 | — | **0.878** |
| mean_energy_ratio（absent 子集）| — | 4.6e-14 ✅ | 0.174（present）；absent 同 B2 量级 |
| **enrollment_swap.choice_accuracy** | — | — | **已废弃，不得引用** |
| enrollment_swap.mean_selectivity_db | — | — | 6.12 dB |
| RTF 均值（推理） | 0.0022 | 0.0022 | 0.0022（≈ 450× 实时） |
| schema_validation | 历史 PASS | 历史 PASS | 历史 PASS（不代表质量通过） |
| determinism_rescore | 历史 PASS | 历史 PASS | 历史 PASS（不代表质量通过） |

---

## 6. 关键工程说明（跨模块复用要点）

1. **STFT 对齐硬约束**：`model.py::forward` 与 `losses.py::frame_activity` **必须**用完全一致的 `torch.stft(center=True, pad_mode="reflect")` 参数；不同 PyTorch 版本默认值不同会造成帧数错位 → 活动度 BCE 完全失效。
2. **活动长度对齐**：P1 v3 `activity.npy` 与 mix 音频存在 ≈0.552 的采样级长度比差，`B1Dataset._load_activity_mask` 已内建 nearest-exact 插值并打 `[ACT LEN FIX]` 首条 + 每 1000 条日志。
3. **B3 force_zero 精度**：`_is_absent_entry` 优先判断 `target_present=True` 返回 False，避免将 `enroll_swap_target_1/2` 误判为 absent 导致 target 被置零。
4. **训练稳定性**：AMP + lr=1e-3 易梯度爆炸；推荐 `amp=false`、`lr=3e-4`、`grad_clip=2.0`（三场景配置已内置）。
5. **MR-STFT 数值稳定性**：内部 log 计算强制 float32；全零参考不做除以零参考范数的谱收敛，而使用有界输出谱抑制项，避免 ABSENT loss 爆炸。
6. **路径解析兜底**：音频/npy 路径按「manifest 同级 → data_root → P1 默认 → FUNASR_ROOT」四级搜索，天然兼容 P1 v2/v3 与本地 debug 数据，无需修改代码。

---

## 7. 自检与复现

```bash
# 模块级冒烟测试
cd funasr_project/P2_project
python tests/smoke_tse_random.py          # CPU 随机前/反传
python tests/test_tse_conditioning.py     # 五条件
python tests/test_tse_metrics.py          # 评测指标

# 加载最终三模型 checkpoint 并验证前向一致性
python verify_p2_local.py
# 预期：B1/B2/B3 checkpoints 均 forward 成功、shape=[1,128000]、各 checkpoint sha256 匹配
```

`verify_p2_local.py` 通过后，说明本机的 P2 核心代码与最终权重可正确对接，可直接交给 P5 Pipeline 消费。

---

## 8. 模块冻结清单（可安全上传 GitHub）

✅ **已包含**：全部 `src/tse/` 源码、`tools/` 脚本、`configs/*.yaml`、`tests/`、`schemas/`、`state/*`、`environment/*`、最终评测的 JSON/YAML/MD/sha256 元数据、`verify_p2_local.py`、本 README。

❌ **已被根 `.gitignore` 排除**：所有 `.pt/.pth/.safetensors/.bin` 权重、`.tar/.tar.gz/.zip` 大包、`.wav/.npy` 音频与 mask、`.log` 训练日志、`P2_project/data/`、`test_audio/`、`artifacts/experiments/`。可参考仓库根 `.gitignore`「P2 TSE 模块」一节。

Checkpoint 等大二进制通过独立发布渠道（云盘、对象存储）提供，SHA256 记录在 `artifacts/final/P2_artifacts/*/config.sha256` 与 `data.sha256`。

---

*FunASR Command Recognition · P2 Target Speaker Extraction · 架构冻结 v1.0（2026-08-10）*
