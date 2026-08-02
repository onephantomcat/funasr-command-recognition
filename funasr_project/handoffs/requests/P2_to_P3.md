# P2 → P3 接口请求单

- 发起方：P2（TSE 模块）  接收方：P3（ASR 模块）
- 日期：2026-07-30

## 需要什么

| 交付 | 内容 | 最晚需要日期 |
|---|---|---|
| `p3_text_eval_v1` | 唯一文本规范（繁简/标点/大小写/数字归一规则）+ CER 核心实现 | 2026-08-05 |
| `asr_eval_v1` | 冻结 Paraformer 基线 + 配对 CER 评测流程（mixture vs tse_output 同批样本） | 2026-08-05 |

## schema / version

- 配对预测格式 `paired_predictions.jsonl`（P2-11 设计，schema 草稿见 `schemas/tse_prediction.schema.json`）：每行含 sample_id、condition（mixture/tse_output）、hyp_text、ref_text、cer
- B1 判据计算需按 SIR/重叠率分层的 CER 汇总能力

## 缺失时 P2 继续做什么

- 完成 P2-11 评测器骨架并预留 P3 接入点
- B1 试车（P2-12）可先以 SI-SDRi 等声学指标做方向性判断（不报告正式 CER）

## 哪些正式结论会被阻塞

- **B1 正式判据"D_overlap CER 相对降 ≥15% 或绝对降 ≥5pp"无法计算** → 8/7 裁决无 CER 证据，只能 INCONCLUSIVE
