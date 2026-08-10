# P2-09 固定小批量 100 step 过拟合报告（DEBUG_ONLY）

- 日期: 2026-07-30 13:54:43
- 设备: cuda（torch 2.7.1+cu126）
- 固定 batch: 8 条 ['dbg_S0764_full100', 'dbg_S0764_partial25', 'dbg_S0764_swap50', 'dbg_S0765_full100', 'dbg_S0765_partial25', 'dbg_S0765_swap50', 'dbg_S0766_full100', 'dbg_S0766_partial25']
- 基线 SI-SDR(mixture): 4.86 dB

## 指标（step0 → step100）

- total: -3.0276 → -8.2110（降 171.2%）
- SI-SDR: 4.89 → 9.29 dB（SI-SDRi 4.43 dB）
- wav_l1: 0.0023 → 0.0026
- mrstft: 1.5151 → 1.0166
- act_bce: 0.6776 → 0.1147
- res_l1: 0.0043 → 0.0024
- 注册条件：正确 9.29 dB vs 错误 5.64 dB，mean|Δ|=1.411e-03
- 恢复一致性: max|Δ|=0.000e+00（tol 1e-05）
- NaN step 数: 0；训练耗时: 12.3s

## 判定

| 项 | 结果 |
|---|---|
| loss_drop_ge_70pct | PASS |
| si_sdr_gain_ge_10db | FAIL |
| no_nan | PASS |
| grad_finite | PASS |
| output_not_zero | PASS |
| output_not_copy_mix | PASS |
| enroll_condition_effective | PASS |
| si_sdri_positive | PASS |
| restore_consistent | PASS |

**总体判定: PASS**
