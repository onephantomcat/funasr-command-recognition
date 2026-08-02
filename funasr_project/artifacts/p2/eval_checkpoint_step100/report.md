# P2-11 TSE 评测报告（DEBUG_ONLY / BOOTSTRAP_ENCODER_ONLY）

- checkpoint: `checkpoint_step100.pt`（sha256 `199b4ef6c892f4bd…`）
- manifest: `manifest.jsonl`（sha256 `421d55452b1439f3…`）
- 设备: cuda；样本: 12 条（PRESENT 12 / ABSENT 0 / NaN 0）

## PRESENT 聚合

- SI-SDR 语料级 1.64 dB / 逐句平均 5.99 dB（两口径并列，手册 P2-11）
- SI-SDRi 均值 1.41 dB；wav_l1 0.0074；MR-STFT 1.2017；act_f1 0.982；energy_ratio 0.4816

| 场景 | utterance SI-SDR (dB) |
|---|---|
| overlap_100_sir_0 | 1.05 |
| overlap_25_sir_5 | 12.78 |
| overlap_50_sir_0 | 4.15 |

## 注册交换

- 选择正确率 0.667（tie_eps=1e-06，平局计 0.5）
- 平均选择性（q_e1_y1−q_e1_y2）15.98 dB

## 效率

- latency 均值 13.9 ms；RTF 均值 0.0035；峰值显存 0.021 GB

## 门禁

- schema 校验: PASS(12 validated, 0 nan-skipped)
- 确定性复评（逐字节）: PASS
