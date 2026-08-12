# P2-12 B1 20000-step 试车报告（B3_ENROLL_SWAP_V3）

- 日期: 2026-08-10 20:24:35
- 设备: cuda（torch 2.1.2+cu118）
- 配置: `tse_b3_v3.yaml`
- 数据: train=30000 条（manifest=manifest.jsonl）
- AMP: False；batch=16；seg=8s；accum=1；effective_batch=16

## 吞吐量

- samples/sec: 307.74
- step time P50: 51.0 ms / P95: 61.9 ms
- 数据等待占比: 0.4%
- 训练总耗时: 1501.2s（20000 步）

## 显存

- 峰值显存: 2.178 GB
- 预算: 4.0 GB；余量: 1.822 GB

## 收敛

- total loss 首 100 步均值: 150477.9991
- total loss 末 100 步均值: 0.6299
- loss 下降: 是
- NaN step 数: 0

## 恢复

- checkpoint: checkpoint_step20000.pt
- 恢复一致性 max|Δ|: 0.000e+00
- 判定: PASS

## 完整训练预估

- 1 epoch 步数: 1875
- 1 epoch 预估时长: 95.6s（1.6 分钟）
- 100 epoch 预估: 2.65 小时

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
