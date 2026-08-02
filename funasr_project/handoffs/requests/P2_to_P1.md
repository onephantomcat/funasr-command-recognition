# P2 → P1 接口请求单

- 发起方：P2（TSE 模块）  接收方：P1（数据模块）
- 日期：2026-07-30

## 需要什么

按优先级三个交付版本：

| 版本 | 内容 | 最晚需要日期 |
|---|---|---|
| `v1_smoke` | 100 条完整管线联调数据（含 ≥10 条 overlap=100%、SIR=-5dB） | **2026-07-31（已到期，最高优先级）** |
| `v2_b1` | B1 正式 train/dev/confirm 三分 + 泄漏报告（说话人无重叠、hash 去重） | 2026-08-03 |
| `v3_absent_swap` | ABSENT 子集（target_wav 等长全零）+ 同混合三注册 triplet | 2026-08-07 |

## schema / version

- manifest 单条遵守 `p1_to_p2.v1`（P2 手册 §4.3）：相对路径、`requested/measured` 双写的 sir/snr/overlap、`common_scale`、`generator_version`、`seed`、source/output SHA256
- 每版本含 `README.md / SCHEMA.json / manifest.jsonl / SHA256SUMS.txt / VERSION / FROZEN` 六件套
- 硬性规则（§4.3 九条）：ABSENT 全零 target、`target_present=false`、无干扰时 `interferer_wav=null`、activity_mask 必须来自干净源 VAD（禁对混合猜 VAD）
- 建议：`v2_b1` README 写明样本时长档（4s 为主），便于 P2 按本机 6GB 显存做预算（决策 D-2026-07-30-01：本机路线）

## 缺失时 P2 继续做什么

- P2-11 评测器骨架先行（不等数据）
- WeSep 上游克隆/环境锁/冒烟（P2-03/04/05）
- P2-09 过拟合诊断细化、B2 零抑制损失单测

## 哪些正式结论会被阻塞

- `v1_smoke` 不到 → P2-10/11 联调阻塞，B1 窗口（7/30–8/7）空转
- `v2_b1` 或泄漏报告不到 → **8/7 B1 裁决整体顺延**（三种子正式实验无法启动）
- `v3_absent_swap` 不到 → B2（ABSENT 静音）/B3（注册选择）无法启动
