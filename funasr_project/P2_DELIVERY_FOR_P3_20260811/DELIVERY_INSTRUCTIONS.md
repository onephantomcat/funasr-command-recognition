# P2 交付说明文档 v2.2 — 发送给 P3

> **修订版本**: v2.2  
> **打包日期**: 2026-08-11  
> **tar.gz 文件名**: `P2_DELIVERY_FOR_P3_20260811_v2.2.tar.gz`  
> **tar.gz SHA256**: `2f7e8b7def8649bbcd9ad4b58f40d2ff8ee0a6e3eadcde0f711496d0ebc2862d`  
> **相比 v2.1 修复项**: 5 项（详见 §7 版本变更说明）

---

## 1. 解压步骤

### 1.1 环境要求

| 组件 | 版本要求 |
|------|----------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0（GPU 环境需 CUDA） |
| numpy | ≥ 1.20 |
| soundfile | ≥ 0.12 |
| yaml | 任意 |
| GPU（可选） | NVIDIA GPU + CUDA，CPU 也可跑但较慢 |

### 1.2 解压

**Linux/macOS:**
```bash
tar -xzf P2_DELIVERY_FOR_P3_20260811_v2.2.tar.gz
```

**Windows PowerShell:**
```powershell
tar -xzf P2_DELIVERY_FOR_P3_20260811_v2.2.tar.gz
```

### 1.3 完整性校验（可选，推荐）

```bash
# 校验 tar.gz 哈希
echo "2f7e8b7def8649bbcd9ad4b58f40d2ff8ee0a6e3eadcde0f711496d0ebc2862d  P2_DELIVERY_FOR_P3_20260811_v2.2.tar.gz" | sha256sum -

# 校验包内工具脚本哈希（解压后）
cd P2_DELIVERY_FOR_P3_20260811/tools
for f in evaluate_tse.py train_b1_trial.py diagnose_b3.py train_overfit_debug.py; do
    echo "$(cat ${f}.sha256)  $f" | sha256sum -
done
```

### 1.4 目录结构

```
P2_DELIVERY_FOR_P3_20260811/
├── B1/                          # B1 场景（单说话人 PRESENT）
│   ├── config.yaml              # B1 模型配置
│   ├── checkpoint_step20000.pt  # B1 模型权重
│   ├── data/train_manifest.jsonl
│   ├── data/dev_manifest.jsonl
│   └── ...（metrics / report / verdict 等）
├── B2/                          # B2 场景（全 ABSENT，复用 B3 权重）
│   ├── config.yaml
│   ├── checkpoint_step20000.pt
│   ├── data/train_manifest.jsonl
│   ├── data/dev_manifest.jsonl
│   ├── REUSE_B3_NOTE.txt
│   └── ...
├── B3/                          # B3 场景（ENROLL-SWAP 双说话人）
│   ├── config.yaml
│   ├── checkpoint_step20000.pt
│   ├── data/train_manifest.jsonl
│   ├── data/dev_manifest.jsonl
│   └── ...
├── src/tse/                     # TSE 核心模块（6 个 .py 文件）
│   ├── model.py
│   ├── metrics.py
│   ├── losses.py
│   ├── enrollment_adapter.py
│   ├── api.py
│   └── __init__.py
├── tools/                       # 工具脚本
│   ├── evaluate_tse.py          # 评测脚本
│   ├── train_b1_trial.py        # 训练脚本
│   ├── diagnose_b3.py           # B3 集成自检脚本
│   └── train_overfit_debug.py   # Debug 辅助（evaluate_tse.py 引用）
├── DELIVERY_MANIFEST.json       # 交付清单（含所有 SHA256）
├── THIRD_PARTY_PINNINGS.txt     # 第三方依赖锁定（speakerlab/CAMPPlus）
├── change_note.md               # 变更日志
└── fix_list.txt                 # 修复清单
```

---

## 2. 运行前环境准备

### 2.1 安装依赖

```bash
pip install numpy soundfile pyyaml torch
```

如需 GPU 加速请根据 CUDA 版本安装对应 PyTorch。

### 2.2 CAMPPlus 声纹模型下载

根据 [THIRD_PARTY_PINNINGS.txt](file:///D:/Users/xuanmo0205/Desktop/funasr-command-recognition/funasr_project/P2_DELIVERY_FOR_P3_20260811/THIRD_PARTY_PINNINGS.txt) 中的说明下载 CAMPPlus 权重：

```
模型 ID: iic/speech_campplus_sv_zh-cn_16k-common
SHA256: 3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8
大小: 26.74 MB
```

下载后放入 `funasr` 模型缓存目录（默认 `~/.cache/modelscope/hub/`），或在 config.yaml 中通过 `sv_model_id` 指定路径。

### 2.3 speakerlab 依赖

```
commit: 065629c313eaf1a01c65c640c46d77e61e9607b4
安装: pip install git+https://github.com/alibaba-damo-academy/speakerlab.git@065629c3
```

---

## 3. 运行验证命令

### 3.1 Step 1：B3 集成自检（推荐首先执行）

这是 v2.2 新增的自检脚本，会依次验证：CAMPPlus 后端加载 → 嵌入有效性 → 数据能量分布 → 声纹区分度。

```bash
cd P2_DELIVERY_FOR_P3_20260811

python tools/diagnose_b3.py \
    --config B3/config.yaml \
    --manifest B3/data/dev_manifest.jsonl \
    --device cuda \
    --max_samples 50
```

**预期输出**：4 项检查全部 PASS，末尾输出 `ALL CHECKS PASSED`。

若任何一项失败，会明确标注是哪个环节的问题（CAMPPlus 加载失败？能量分布异常？声纹不可区分？），便于快速定位。

### 3.2 Step 2：B3 评测（主验证）

```bash
cd P2_DELIVERY_FOR_P3_20260811

python tools/evaluate_tse.py \
    --checkpoint B3/checkpoint_step20000.pt \
    --manifest B3/data/dev_manifest.jsonl \
    --device cuda \
    --out B3/eval_summary_v22.json
```

**预期结果**（与 v2.1 修复后的指标一致）：

| 指标 | 预期值 | 说明 |
|------|--------|------|
| corpus_sisdr_db | ~1.23 dB | 平均 SI-SDRi |
| mean_act_f1 | ~0.875 | 活动检测 F1 |
| mean_energy_ratio | ~0.001 | 能量比（>0 表示非近静音） |
| utterance_sisdr_by_scenario | | |
| ├ enroll_swap_target_1 | ~-0.60 dB | 目标说话人 1 |
| └ enroll_swap_target_2 | ~-1.99 dB | 目标说话人 2 |
| choice_accuracy | ⚠️ 已废弃 | v2.2 禁止 fallback 后不再可信 |

### 3.3 Step 3：B1 评测

```bash
python tools/evaluate_tse.py \
    --checkpoint B1/checkpoint_step20000.pt \
    --manifest B1/data/dev_manifest.jsonl \
    --device cuda \
    --out B1/eval_summary_v22.json
```

**预期结果**：

| 指标 | 预期值 |
|------|--------|
| corpus_sisdr_db | ~5.25 dB |
| mean_act_f1 | ~0.975 |
| mean_energy_ratio | ~0.304 |
| utterance_sisdr_by_scenario.single | ~9.99 dB |

### 3.4 Step 4（可选）：B1 空跑训练验证

```bash
python tools/train_b1_trial.py \
    --config B1/config.yaml \
    --manifest B1/data/train_manifest.jsonl \
    --scene_mode b1 \
    --device cuda \
    --max_steps 10 \
    --out B1/trial_verdict_v22.json
```

`--max_steps 10` 仅跑 10 步做连通性验证，确认无 CAMPPlus 加载/编码错误即可停止。

### 3.5 CPU 模式验证

如无 GPU，所有命令加 `--device cpu` 即可：

```bash
python tools/diagnose_b3.py --config B3/config.yaml --manifest B3/data/dev_manifest.jsonl --device cpu
```

---

## 4. 关键行为变化（v2.1 → v2.2）

| 场景 | v2.1（旧） | v2.2（新） |
|------|-----------|-----------|
| CAMPPlus 加载失败 | ⚠️ **静默 fallback** 到 bootstrap，评测继续但输出为假静音 | ❌ **直接报错退出**，不产生虚假结果 |
| CAMPPlus 编码失败 | ⚠️ 同上 | ❌ 同上 |
| `--debug_data` 模式下 CAMPPlus 失败 | 报错退出 | ✅ 允许 fallback 到 bootstrap（仅调试用） |
| `cfg["__debug_data"]` 键 | 可能与用户 YAML 冲突 | 改用 `cfg["_p2_internal"]["debug_data"]` 命名空间 |
| 交付包 tools 路径 | 假设 `parents[1]` 即 `P2_project/`，换目录就 ImportError | 灵活探测含 `src/tse/` 的父目录，自动适配 |

---

## 5. 常见问题排查

### Q: `ImportError: cannot import name 'DualOutputTSE' from 'src.tse.model'`

**原因**: `src/tse/` 目录缺失或路径不对。  
**解决**: 确认交付包根目录下有 `src/tse/` 文件夹。若用 git clone 方式获取，请确保检出了 `src/tse/` 子目录。

### Q: `RuntimeError: CAMPLUS embedding 失败 (spk_xxx)，禁止 fallback BOOTSTRAP`

**原因**: CAMPPlus 权重未下载或 speakerlab 版本不对。  
**解决**:  
1. 确认 CAMPPlus 权重已下载到 modelscope 缓存目录  
2. 确认 speakerlab 版本与 `THIRD_PARTY_PINNINGS.txt` 中 commit 一致  
3. 先用 `diagnose_b3.py` 单独验证 CAMPPlus 加载

### Q: 评测结果 SI-SDR 很低 / 能量比接近 0

**原因**: CAMPPlus 加载失败但旧版本 fallback 到 bootstrap 产生假静音（v2.1 的已知 bug）。  
**解决**: v2.2 已修复此问题。若仍出现，确认使用的是本交付包 v2.2 的工具脚本（检查 `tools/*.py.sha256`）。

---

## 6. 交付包校验清单

请确认以下文件的 SHA256 与 `DELIVERY_MANIFEST.json` 中记录的一致：

```bash
cd P2_DELIVERY_FOR_P3_20260811

# 工具脚本
for f in tools/evaluate_tse.py tools/train_b1_trial.py tools/diagnose_b3.py; do
    sha256sum $f
done

# 配置文件
for f in B1/config.yaml B2/config.yaml B3/config.yaml; do
    sha256sum $f
done

# 检查点
for f in B1/checkpoint_step20000.pt B3/checkpoint_step20000.pt; do
    sha256sum $f
done
```

对照值见 [DELIVERY_MANIFEST.json](file:///D:/Users/xuanmo0205/Desktop/funasr-command-recognition/funasr_project/P2_DELIVERY_FOR_P3_20260811/DELIVERY_MANIFEST.json) 中的 `models.*.checkpoint_sha256` / `config_sha256` 字段。

---

## 7. v2.2 版本变更说明（相对 v2.1）

v2.2 修复了 P3 验收中提出的 5 项问题：

| # | 问题 | 修复 |
|---|------|------|
| 1 | DatasetA PRESENT 输出坍塌（CAMPPlus 失败时静默 fallback 到 bootstrap） | `evaluate_tse.py` + `train_b1_trial.py`：CAMPlus 模式失败直接 `raise RuntimeError`，禁止 silent fallback |
| 2 | train/dev SHA 标注与实际文件不一致（两文件用了同一哈希） | 重算并更新 `B{1,2,3}/data.sha256` 中 train/dev 各自的 SHA256 |
| 3 | speakerlab commit 和 CAMPPlus 权重 SHA 未固定 | `THIRD_PARTY_PINNINGS.txt` 锁定 commit `065629c3` + SHA `3388cf5f` |
| 4 | 文档说"禁止 fallback"但代码实际允许 | 删除 `evaluate_tse.py` 中两处 bootstrap fallback 分支，文档与代码对齐 |
| 5 | choice_accuracy 与分场景 SI-SDR 指标冲突 | `B3/eval_summary.json` 标记 `choice_accuracy` 为废弃，等 P3 用 v2.2 重跑 |

---

## 附录：关键文件路径速查

| 用途 | 路径 |
|------|------|
| 评测 B3 | `python tools/evaluate_tse.py --checkpoint B3/checkpoint_step20000.pt --manifest B3/data/dev_manifest.jsonl --device cuda` |
| 评测 B1 | `python tools/evaluate_tse.py --checkpoint B1/checkpoint_step20000.pt --manifest B1/data/dev_manifest.jsonl --device cuda` |
| B3 自检 | `python tools/diagnose_b3.py --config B3/config.yaml --manifest B3/data/dev_manifest.jsonl --device cuda` |
| B1 空跑训练 | `python tools/train_b1_trial.py --config B1/config.yaml --manifest B1/data/train_manifest.jsonl --scene_mode b1 --device cuda --max_steps 10` |
| 交付清单 | `DELIVERY_MANIFEST.json` |
| 依赖锁定 | `THIRD_PARTY_PINNINGS.txt` |
| 变更日志 | `change_note.md` |
