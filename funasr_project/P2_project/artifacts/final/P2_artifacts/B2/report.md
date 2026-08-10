# P2-12 B1 20000-step 试车报告（B2_ABSENT）

- 日期: 2026-08-09 20:21:13
- 设备: cuda（torch 2.1.2+cu118）
- 配置: `tse_b2.yaml`
- 数据: train=30000 条（manifest=manifest.jsonl）
- AMP: False；batch=16；seg=8s；accum=1；effective_batch=16

## 吞吐量

- samples/sec: 338.03
- step time P50: 46.9 ms / P95: 52.2 ms
- 数据等待占比: 0.4%
- 训练总耗时: 1370.9s（20000 步）

## 显存

- 峰值显存: 1.733 GB
- 预算: 4.0 GB；余量: 2.267 GB

## 收敛

- total loss 首 100 步均值: 56937069.2275
- total loss 末 100 步均值: 100.5342
- loss 下降: 是
- NaN step 数: 0

## 恢复

- checkpoint: checkpoint_step20000.pt
- 恢复一致性 max|Δ|: 0.000e+00
- 判定: PASS

## 完整训练预估

- 1 epoch 步数: 1875
- 1 epoch 预估时长: 87.9s（1.5 分钟）
- 100 epoch 预估: 2.44 小时

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
