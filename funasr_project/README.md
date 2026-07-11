# 面向复杂交互场景的目标说话人抗干扰语音识别与拒识系统

大创项目（国家级申报，杭电）代码工程，基于阿里 FunASR 中文基座。

## 系统架构 (对应申报表总体架构)

```
注册音频 ──> CAM++ 声纹提取 ──> 目标声纹库
                                   │
测试音频 ──> 声纹核验门控 ──┬─ 相似度 < 阈值 ──> 拒识, 输出空字符串 ""
                            └─ 相似度 ≥ 阈值 ──> fsmn-vad 端点检测
                                                  └─> Paraformer 识别
                                                       └─> ct-punc 标点
                                                            └─> 拼音模糊指令匹配
                                                                 └─> JSON 输出
```

## 环境

- macOS (Apple Silicon, 8GB 内存), Python 3.9, CPU 推理
- 依赖已装在 `venv/`，运行前激活:

```bash
cd ~/Desktop/6.9的大创/funasr_project
source venv/bin/activate
```

## 文件说明

| 文件 | 对应申报表模块 | 作用 |
|---|---|---|
| `asr_demo.py` | ASR 基线 (Step 1) | VAD + Paraformer + 标点单文件识别 |
| `speaker_verify.py` | 声纹鉴权 + 拒识门控 (Step 2/3) | CAM++ 声纹注册/核验, 阈值判决 |
| `command_match.py` | 文本后处理 | 拼音编辑距离滑窗模糊匹配, 抗同音/错字 |
| `trusted_pipeline.py` | 全链路闭环 | 声纹门控 -> 拒识或识别 -> 指令 -> JSON |
| `pipeline.py` | 抗干扰实验 | 干净 vs 10/5/0dB 噪声识别对比 |
| `mic_realtime.py` | 实时演示 | 麦克风流式识别 + 指令触发 |
| `make_test_audio.py` | 数据构建 | TTS 合成干净语音 + 叠加噪声 |
| `make_speaker_audio.py` | 数据构建 | 目标说话人(Tingting) vs 陌生人(Meijia/Sinji) |
| `text_norm.py` | 文本规范化 | 全半角/数字转中文/去标点, 评测前统一 |
| `cer.py` | 评测指标 | 字错误率 CER (句级 + 语料级) |
| `make_trials.py` | 多维评测桶 | AISHELL-1 真实录音构建 clean/混叠/babble/非目标 试验集 |
| `eval_trials.py` | 分桶评测 | 各桶 CER + FRR/FAR 拒识指标, 输出 JSON 报告 |
| `eval_datasetA.py` | 比赛测试集A评测 | 唤醒音频声纹预门控 -> ASR -> 声纹/文本融合拒识, 输出报告/提交JSON |
| `sweep_datasetA_gate.py` | 阈值小样本搜索 | 只跑声纹相似度, 快速比较正样本放行率与负样本RR |
| `simulate_datasetA_policy.py` | 联合策略离线模拟 | 复用已缓存ASR/声纹报告, 秒级比较声纹阈值和文本阈值组合 |
| `target_purify.py` | 实验性目标提纯 | CAM++短窗相似度掩蔽, 可通过 `eval_datasetA.py --purify` 小样本试验 |
| `train_lightweight_gate.py` | 轻量训练 | 用缓存特征训练 JSON 小门控, 可由 `eval_datasetA.py --gate-model` 加载 |
| `docs/LIGHTWEIGHT_TRAINING_PLAN.md` | 训练方案 | 分阶段训练/扩样/准入标准 |

## 快速开始

```bash
python make_test_audio.py      # 生成抗噪测试音频
python make_speaker_audio.py   # 生成声纹测试音频
python pipeline.py             # 实验一: 抗噪声识别对比
python trusted_pipeline.py     # 实验二: 目标说话人放行 / 陌生人拒识 (核心 demo)
python mic_realtime.py         # 实时麦克风识别
```

比赛测试集A建议先小样本验证, 不要直接全量跑:

```bash
./venv/bin/python sweep_datasetA_gate.py --limit 100
./venv/bin/python simulate_datasetA_policy.py
./venv/bin/python eval_datasetA.py --limit 20
./venv/bin/python eval_datasetA.py --limit 20 --submission-out data/datasetA/submission_20.json
./venv/bin/python eval_datasetA.py --offset 400 --limit 20 --purify
./venv/bin/python train_lightweight_gate.py --train-limit 300 --dev-offset 300 --dev-limit 174
./venv/bin/python train_lightweight_gate.py --train-limit 300 --dev-offset 300 --dev-limit 174 --min-dev-rr 0.98 --model-out models/lightweight_gate_safe.json
./venv/bin/python train_lightweight_gate.py --split-mode random --train-ratio 0.80 --min-dev-rr 0.98 --model-out models/lightweight_gate_random_safe.json
./venv/bin/python eval_datasetA.py --limit 20 --gate-model models/lightweight_gate.json
```

`eval_datasetA.py` 默认使用融合联合判定:
以 `声纹阈值=0.0` 做轻量预门控, 通过后运行 ASR, 再计算
`声纹相似度 - 0.70 * 文本偏离度 >= 0.03` 决定是否输出识别文本;
未通过则输出空字符串。已放行文本会在 `文本偏离度 <= 0.50` 时归一到最近的
已知指令/查询短语。反复调参时可加
`--embedding-cache data/datasetA/emb_cache.pkl --asr-cache data/datasetA/asr_cache.pkl`
复用声纹与ASR结果;
正式计时报告建议不使用缓存。

轻量训练方案见 `docs/LIGHTWEIGHT_TRAINING_PLAN.md`。当前已支持先训练一个
JSON 小门控, 再通过 `--gate-model models/lightweight_gate.json` 接入评测。
已跑通均衡版与安全版两个门控, 但小样本拒识尚未超过默认手写融合, 暂不作为默认提交策略。
新增随机分层训练模式 `--split-mode random`, 便于避免前后样本难度分布造成的切分偏差。

注意: 不默认启用唤醒词硬保护。测试集A的正样本后半段包含多种唤醒词,
只允许 `hi colmo/你好科慕` 会误拒正样本。

模型缓存在 `~/.cache/modelscope/`，下载一次后离线可用。

## 已验证结果 (2026-06-12, 本机实测)

实验一 · 抗噪识别(Paraformer-large): 合成噪声 10/5/0dB 下转写**零错字**, 指令全部命中;
粤语口音说话人转写出错但指令匹配正确拒绝(距离0.36>阈值)。单句推理 0.7~1.2s。

实验二 · 全链路(声纹门控+ASR+指令): 5/5 全对——
目标说话人干净/5dB噪声均放行并正确执行"打开空调"; 三个陌生人用例全部拒识输出空串。
拒识路径仅 0.02s(门控在 ASR 之前, 陌生人不消耗识别算力); 目标全链路 ≤0.9s。
声纹阈值 0.6(本地小验证集搜索, 目标最低0.654 / 陌生人最高0.556)。

注意: 模型一律走本地路径加载(`asr_demo.py` 顶部的 `*_DIR`), 不联网检查;
模型文件齐全在 `~/.cache/modelscope/hub/models/iic/`(共约1.3GB, 含SHA256校验过的主模型)。

## 真实数据基线 (AISHELL-1 测试集, 60条试验, 2026-06-12)

4目标+4干扰真人说话人, 阈值0.6, CPU 28s 跑完:

| 桶 | 放行率 | 总CER(误拒=1) | 放行样本CER |
|---|---|---|---|
| clean | 1.00 | 0.051 | 0.051 |
| babble5 (4人嘈杂) | 0.92 | 0.118 | 0.038 |
| overlap5 (混叠5dB) | 0.92 | 0.115 | 0.034 |
| overlap0 (混叠0dB) | 0.83 | 0.290 | 0.148 |
| nontarget | 0.00(全拒✓) | - | - |

FRR=8.3% / FAR=0%(阈值0.6); 阈值0.35时 FRR=2.1% / FAR=0%。
核心发现: ① 0dB混叠使CER劣化约3~5倍, 且声纹相似度跌入非目标区间(0.29<0.33),
单一声纹证据在重叠场景不可分 —— 这正是 Step 2 目标提纯 + Step 3 双证据联合判决的研究动机;
② babble/5dB混叠下放行样本CER仅约4%, Paraformer 对真实人声噪声相当鲁棒。

## 测试集A小样本结果 (2026-07-01)

题目要求: 正样本计算目标发音人识别 CER, 拒识测试集计算非目标发音人拒识率 RR;
拒识样本提交内容应为空字符串。

| 设置 | 样本 | 正样本CER | 负样本RR | 说明 |
|---|---:|---:|---:|---|
| 纯ASR基线 | 100+100 | 0.0788 | 0.0100 | 几乎没有拒识能力 |
| 仅声纹门控, 阈值0.30 | 20+20 | 0.0741 | 0.7500 | 保留识别率, RR仍偏低 |
| 仅声纹门控, 阈值0.38 | 20+20 | 0.1728 | 0.8500 | 更安全, 但误拒正样本 |
| 声纹0.30 + 文本0.45 | 20+20 | 0.0741 | 0.9000 | 硬阈值联合判定 |
| 融合判定 `sim-0.7*intent>=0.14` | 20+20 | 0.0123 | 0.9000 | 明显降低误拒 |
| 融合判定 `sim-0.7*intent>=0.06` + 短语归一 | 474+474 | 0.1436 | 0.9684 | 负样本全覆盖小样本 |
| 融合判定 `sim-0.7*intent>=0.03` + 短语归一 | 1364+474 | 0.3408 | 0.9578 | 当前默认, 全测试集A |

100条声纹分数 + 已缓存ASR结果的离线模拟显示, 融合判定整体优于单一声纹阈值
和硬阈值联合判定。
逐步扩样发现: 前300条正样本较容易, 400条以后存在大量强干扰/混叠样本,
纯ASR在800条正样本上 CER 已达0.3059。因此后续主要瓶颈已从拒识阈值转向
目标说话人提纯与抗干扰转写。

实验性目标提纯: 已实现 `target_purify.py` 的训练免费短窗声纹掩蔽方案, 但在
offset=400 的20+20困难段上, CER/RR 与未提纯一致(0.3615/0.9500), 未作为默认。
FunASR 自带 `spk_model` 说话人分段在强干扰样本上也只得到单一说话人段。
这说明当前本地模型不足以做真正的重叠语音目标提纯, 后续需要引入或训练
SpeakerBeam/VoiceFilter/目标说话人提取类模型。

## 下一步 (按申报表技术路线)

- [ ] 拒识阈值搜索: 在验证集上扫 0.3~0.7, 画 误拒率-误受率 曲线选工作点
- [ ] 双证据联合判决: ASR 解码置信度 + 声纹相似度融合 (申报表 Step 3)
- [ ] 接入降噪前端 FRCRN (`speech_frcrn_ans_cirm_16k`), 对比降噪前后指标
- [ ] 真实数据: 自录注册/指令音频 + MUSAN 真实噪声, 统计 CER 与拒识率
- [ ] 多人混叠场景: 两路 TTS 按不同信干比混合, 测目标提纯能力
- [ ] Paraformer 热词 (hotword) 定制提升指令词识别
- [ ] ONNX 导出 + 量化, 测 RTF (端侧部署, Step 4)
