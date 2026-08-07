# P2-12 B1 10-step 试车报告（B1_TRIAL）

- 日期: 2026-08-06 20:51:26
- 设备: cuda（torch 2.7.1+cu126）
- 配置: `tse_b1_trial.yaml`
- 数据: train=12 条（manifest=manifest.jsonl）
- AMP: True；batch=12；seg=8s；accum=1；effective_batch=12

## 吞吐量

- samples/sec: 1.45
- step time P50: 8216.9 ms / P95: 8809.6 ms
- 数据等待占比: 93.9%
- 训练总耗时: 94.7s（10 步）

## 显存

- 峰值显存: 0.977 GB
- 预算: 4.0 GB；余量: 3.023 GB

## 收敛

- total loss 首 100 步均值: -3.2351
- total loss 末 100 步均值: -3.2935
- loss 下降: 是
- NaN step 数: 0

## 恢复

- checkpoint: checkpoint_step10.pt
- 恢复一致性 max|Δ|: 0.000e+00
- 判定: PASS

## 完整训练预估

- 1 epoch 步数: 1
- 1 epoch 预估时长: 8.2s（0.1 分钟）
- 100 epoch 预估: 0.23 小时

## 判定

| 项 | 结果 |
|---|---|
| no_nan | PASS |
| grad_finite | PASS |
| peak_mem_under_budget | PASS |
| throughput_measured | PASS |
| loss_decreasing | PASS |
| checkpoint_restore_ok | PASS |
| full_train_time_estimated | PASS |

**总体判定: PASS**
