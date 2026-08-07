# P1 v2_b1 本机替代数据集（v2 分支）

## 元信息

- 生成脚本: `tools/build_v2_b1_local.py`
- 生成时间: 2026-08-04 21:03:10
- 源数据: `../data/aishell_test` (AISHELL-1 测试集，20 说话人)
- 说话人分配:
  - train:   target=10 spk, interferer=10 spk
  - dev:     target=5 spk, interferer=5 spk
  - eval:    target=5 spk, interferer=5 spk
- 配置矩阵: 9 格 (overlap × SIR) × 2 repeat + ABSENT
- 总样本: 400 条
- 片段时长: 8.0s @ 16000Hz
- manifest 路径相对 `funasr_project/` 根目录

## 使用

```bash
# B1 500 步正式训练（v2 分支）
python tools/train_b1_trial.py --device auto

# 覆盖 manifest
python tools/train_b1_trial.py --manifest data/p1_v2_b1/train.jsonl --device auto
```

## 注意

- 本数据为 P1 v2_b1 本机替代，正式数据交付后需替换
- dev/eval target 说话人与 train 部分重叠（使用不同 utt 段）
- embedding 使用 BOOTSTRAP 模式（P4 对接未集成到 v2 分支）
