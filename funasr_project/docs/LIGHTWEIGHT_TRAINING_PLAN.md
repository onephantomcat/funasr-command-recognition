# 轻量训练方案

目标: 在不微调 Paraformer 主干的前提下, 先把拒识和强干扰样本的决策稳定下来。所有训练先走小样本闭环, 指标稳定后再逐步扩样。

## 总体路线

当前瓶颈已经比较清楚:

1. 前 300 条正样本中, ASR 本身可用, 主要问题是负样本拒识。
2. 400 条以后正样本中, 大量强干扰/混叠导致 ASR 被非目标说话人带跑。
3. 训练免费的短窗声纹掩蔽没有明显提升, 所以不能继续只靠手调提纯参数。

因此轻量训练分三步走:

| 阶段 | 训练对象 | 是否动大模型 | 目标 | 预期收益 |
|---|---|---:|---|---|
| A | 小型拒识门控 | 否 | 学习 `声纹相似度 + 文本偏离度 + 输出形态` 的联合判定 | 提升 RR, 控制正样本误拒 |
| B | 命令短语纠偏/重排 | 否 | 对已放行文本映射到合法指令短语 | 降低 CER |
| C | 轻量目标说话人提纯前端 | 否, 只训小前端 | 用唤醒声纹引导抑制非目标语音 | 改善 400 条以后强混叠 CER |

阶段 A 已落地为 `train_lightweight_gate.py` 和 `eval_datasetA.py --gate-model`。

## 阶段 A: 小型拒识门控

训练特征全部来自现有流水线缓存:

- `speaker_similarity`: 唤醒音频与识别音频 CAM++ 相似度
- `intent_distance`: ASR 文本到已知指令短语库的归一化编辑距离
- `fusion_score`: 当前手写融合分数 `sim - 0.70 * intent`
- `hyp_len_norm`: 识别文本长度
- `has_hyp`: 是否有识别文本
- `known_phrase`: 是否接近已知指令短语
- `very_short_hyp`: 是否异常短

训练模型是一个逻辑回归门控, 输出“是否应该放行”的概率。它的好处是可解释、训练快、保存成 JSON, 不依赖 GPU。

推荐命令:

```bash
./venv/bin/python eval_datasetA.py --limit 474 \
  --embedding-cache data/datasetA/emb_cache_tune.pkl \
  --asr-cache data/datasetA/asr_cache_tune.pkl

./venv/bin/python train_lightweight_gate.py \
  --train-limit 300 \
  --dev-offset 300 \
  --dev-limit 174 \
  --model-out models/lightweight_gate.json

./venv/bin/python train_lightweight_gate.py \
  --train-limit 300 \
  --dev-offset 300 \
  --dev-limit 174 \
  --min-dev-rr 0.98 \
  --model-out models/lightweight_gate_safe.json

./venv/bin/python train_lightweight_gate.py \
  --split-mode random \
  --train-ratio 0.80 \
  --min-dev-rr 0.98 \
  --model-out models/lightweight_gate_random_safe.json

./venv/bin/python eval_datasetA.py --limit 20 \
  --gate-model models/lightweight_gate.json \
  --embedding-cache data/datasetA/emb_cache_tune.pkl \
  --asr-cache data/datasetA/asr_cache_tune.pkl
```

扩样顺序:

| 轮次 | 样本 | 目的 |
|---|---:|---|
| smoke | 20+20 | 检查模型能加载、报告字段正常 |
| small | 100+100 | 看 RR 是否明显高于纯 ASR, CER 是否不劣于默认融合 |
| mid | 300+300 | 覆盖容易段, 确认没有过拟合小样本 |
| hard | offset=400, 20+20 | 专门看强干扰段是否变坏 |
| full-neg | 474+474 | 覆盖所有负样本 |
| full | 1364+474 | 只在前面都稳定后运行 |

准入标准:

- 20+20 不低于当前默认融合结果。
- 474+474 的 RR 尽量保持在 0.96 以上。
- 全量 CER 不能因为追求 RR 明显升高; 若冲突, 优先保留两个工作点:
  - 安全版: RR 高, CER 稍高。
  - 均衡版: CER 较低, RR 稍低。

## 已跑通结果

2026-07-01 已用现有缓存跑通阶段 A:

| 模型 | 训练/开发设置 | 开发集 CER | 开发集 RR | 20+20 CER | 20+20 RR | 结论 |
|---|---|---:|---:|---:|---:|---|
| `models/lightweight_gate.json` | train 300+300, dev 174+174 | 0.2693 | 0.9598 | 0.0185 | 0.7500 | 均衡阈值, 小样本拒识偏松 |
| `models/lightweight_gate_safe.json` | 同上, `--min-dev-rr 0.98` | 0.3084 | 0.9828 | 0.0185 | 0.8000 | 安全阈值, 仍未超过手写融合 |
| `models/lightweight_gate_random.json` | 随机分层 80/20 | 0.4273 | 0.9684 | 0.0185 | 0.7000 | 随机切分后更保守, 但小样本拒识偏松 |
| `models/lightweight_gate_random_safe.json` | 随机分层 80/20, `--min-dev-rr 0.98` | 0.4905 | 0.9895 | 0.0185 | 0.8500 | RR 更高, CER 代价较大 |

判断: 训练链路已经可用, 但当前小门控暂不替换默认提交策略。默认仍保留手写融合
`sim - 0.70 * intent >= 0.03`。后续若要继续训练门控, 应优先改进训练/验证划分和加入困难段样本,
而不是直接全量提交。

补充对照: `models/lightweight_gate_random_safe.json` 在 100+100 上为 CER 0.1589 / RR 0.9000;
当前默认手写融合在同一 100+100 上为 CER 0.0701 / RR 0.8800。训练版只小幅提升 RR,
但明显增加正样本误拒, 因此暂不继续扩到 300 或全量。

## 阶段 B: 命令短语纠偏

当前 `eval_datasetA.py` 已经有 `--phrase-correct`, 会把接近短语库的识别结果归一到最近指令。下一步可以把它改成可训练重排:

1. 候选集合: 所有 `pos.jsonl` 里的唯一 `识别文本`。
2. 特征: 编辑距离、拼音距离、长度差、关键词重合、ASR 原文是否包含候选短语。
3. 模型: 仍用逻辑回归或感知机排序。
4. 训练目标: 正样本选择标签短语, 负样本不输出短语。

这一步不需要重新跑 ASR, 只依赖已有 ASR 缓存, 适合在阶段 A 稳定后做。

## 阶段 C: 轻量目标说话人提纯前端

不建议继续只调 `target_purify.py` 的短窗保留比例, 因为已验证困难段没有收益。更合适的轻量训练方案是训一个小型谱掩蔽前端:

输入:

- 混合语音的 log-mel 或 STFT 幅度
- 唤醒音频提取的 CAM++ 声纹向量

模型:

- 2 层 BiLSTM 或小型 Conv-TasNet 风格 TCN
- 声纹向量通过 FiLM/门控注入每一帧特征
- 输出一个软掩蔽 mask, 得到增强后的目标语音

训练数据构造:

1. 用 datasetA 正样本作为目标语音候选。
2. 随机采样其他说话人识别音频作为干扰。
3. 按 -5/0/5 dB 混合, 目标原音作为训练标签。
4. 加入噪声和混响增强, 但每轮只做小批量。

损失:

- 主损失: 多尺度 STFT loss 或 L1 幅度谱损失
- 辅助损失: 增强后音频与唤醒声纹相似度更高
- 轻量验证: 增强后再跑 ASR, 用 CER 做最终判断

扩样顺序:

| 轮次 | 训练混合数 | 验证样本 | 目标 |
|---|---:|---:|---|
| prototype | 200 | offset=400 的 20+20 | 看是否比无提纯下降 CER |
| small | 1000 | 100+100 | 检查不会破坏容易样本 |
| mid | 5000 | 474+474 | 稳定 RR/CER |

阶段 C 只有在阶段 A/B 到瓶颈后再启动。否则容易把时间花在训练前端上, 但最终提交分数主要还是被拒识策略拖住。

## 当前推荐下一步

先跑阶段 A:

1. 用缓存训练 `models/lightweight_gate.json`。
2. 用 20+20、100+100、300+300 验证。
3. 如果没有明显退化, 再跑 474+474 和全量。

如果小门控没有超过手写融合, 就保留手写融合当提交默认, 但训练结果仍可作为报告里的“参数高效门控学习”实验。
