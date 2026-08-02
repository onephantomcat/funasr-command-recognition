# P2 → P4 接口请求单

- 发起方：P2（TSE 模块）  接收方：P4（声纹模块）
- 日期：2026-07-30

## 需要什么

| 交付 | 内容 | 最晚需要日期 |
|---|---|---|
| `sv_contract_v1` | 冻结的注册 embedding 提取契约：encoder 版本/commit、embedding 维度（当前 debug 假设 192）、输入音频规范（时长/采样率/归一化）、注册质量 q_enroll 定义与取值范围 | **2026-08-03**（P2-12 试车前置） |

## schema / version

- 契约文档 + 冻结 encoder 标识（版本号或 commit hash）
- q_enroll 须满足 05A §2.1：只描述"能否提供稳定条件"，不泄漏"是不是目标人"；质量 dropout 独立于 SINGLE/OVERLAP/ABSENT 标签

## 缺失时 P2 继续做什么

- BOOTSTRAP 确定性随机 embedding（由注册路径哈希派生）继续支撑 debug 车道，保留正确/错误注册 swap 语义（P2-09 已验证条件有效性：正确 9.29dB vs 错误 5.64dB）

## 哪些正式结论会被阻塞

- **B1 正式实验（P2-13）不能启动**——BOOTSTRAP embedding 下训练的模型不具备真实声纹条件，结论无效
- post-TSE identity audit（05A 后验审计）无 encoder 可用
