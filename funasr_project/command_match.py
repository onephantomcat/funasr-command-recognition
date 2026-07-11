# -*- coding: utf-8 -*-
"""
指令匹配模块: 把 ASR 输出的自由文本映射到预定义指令集
策略: 拼音层面的模糊匹配(编辑距离), 抗同音字/识别错字干扰
例: ASR 输出"打开恐铁" -> 命中指令"打开空调"
"""
from pypinyin import lazy_pinyin

# 指令集: 可按项目场景(车载/家居/工业)自行扩充
COMMANDS = [
    "打开空调", "关闭空调", "调高温度", "调低温度",
    "打开车窗", "关闭车窗", "打开音乐", "关闭音乐",
    "增大音量", "减小音量", "导航回家", "取消导航",
    "打开灯光", "关闭灯光", "拨打电话", "挂断电话",
]


def to_pinyin(text):
    return " ".join(lazy_pinyin(text))


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def match_command(asr_text, threshold=0.35):
    """返回 (最佳指令, 归一化距离); 距离超过阈值视为未命中返回 None"""
    for ch in "，。、？！?!,. ":
        asr_text = asr_text.replace(ch, "")
    if not asr_text:
        return None, 1.0
    best_cmd, best_score = None, 1.0
    for cmd in COMMANDS:
        cmd_py = to_pinyin(cmd)
        n = len(cmd)
        # 滑窗: 在长文本里截取与指令长度相近的片段逐一比对, 取最小距离
        for win in (n - 1, n, n + 1):
            if win < 1:
                continue
            for i in range(max(1, len(asr_text) - win + 1)):
                seg_py = to_pinyin(asr_text[i:i + win])
                dist = edit_distance(seg_py, cmd_py)
                score = dist / max(len(seg_py), len(cmd_py))
                if score < best_score:
                    best_cmd, best_score = cmd, score
    if best_score <= threshold:
        return best_cmd, best_score
    return None, best_score


if __name__ == "__main__":
    tests = ["打开恐铁", "请帮我大开空调", "导航回家吧", "今天天气不错"]
    for t in tests:
        cmd, score = match_command(t)
        print(f"输入: {t!r:18s} -> 指令: {cmd}  (距离 {score:.2f})")
