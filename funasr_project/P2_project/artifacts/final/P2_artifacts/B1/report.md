# P2-12 B1 20000-step 试车报告（B1_PRESENT）

- 日期: 2026-08-09 19:44:21
- 设备: cuda（torch 2.1.2+cu118）
- 配置: `tse_b1.yaml`
- 数据: train=100000 条（manifest=manifest.jsonl）
- AMP: False；batch=16；seg=8s；accum=1；effective_batch=16

## 吞吐量

- samples/sec: 337.64
- step time P50: 47.3 ms / P95: 50.1 ms
- 数据等待占比: 0.4%
- 训练总耗时: 2239.0s（20000 步）

## 显存

- 峰值显存: 1.733 GB
- 预算: 4.0 GB；余量: 2.267 GB

## 收敛

- total loss 首 100 步均值: 0.9599
- total loss 末 100 步均值: -4.0606
- loss 下降: 是
- NaN step 数: 0

## 恢复

- checkpoint: checkpoint_step20000.pt
- 恢复一致性 max|Δ|: 0.000e+00
- 判定: PASS

## 完整训练预估

- 1 epoch 步数: 6250
- 1 epoch 预估时长: 295.3s（4.9 分钟）
- 100 epoch 预估: 8.20 小时

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
