# -*- coding: utf-8 -*-
"""
文本规范化 (申报表"多维语料规范化"模块的评测版):
计算 CER 前对参考文本和识别文本做统一: 去标点/空格、全角转半角、
阿拉伯数字转中文读法(简版), 保证评测只比"读出来的内容"
"""
import re

_PUNC = r"[，。、？！?!,.；;：:""\"''‘’（）()\[\]【】<>《》~·…—\-_+=*/\\|@#$%^&{}\s]"
_TONE = r"[啊啦吧呢哦哟哈嘛的了]"

_DIGITS = "零一二三四五六七八九"


def num_to_zh(num_str):
    """阿拉伯数字串 -> 中文读法(逐位或十百千万, 简版覆盖常见指令场景)"""
    n = int(num_str)
    if n < 0 or n > 99999:
        return "".join(_DIGITS[int(c)] for c in num_str)
    if n < 10:
        return _DIGITS[n]
    units = ["", "十", "百", "千", "万"]
    s, digits = "", str(n)
    L = len(digits)
    for i, c in enumerate(digits):
        d = int(c)
        if d:
            s += ("" if (d == 1 and L - i == 2 and i == 0) else _DIGITS[d]) + units[L - i - 1]
        elif not s.endswith("零"):
            s += "零"
    return s.rstrip("零") or "零"


def full_to_half(text):
    out = []
    for ch in text:
        code = ord(ch)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        out.append(chr(code))
    return "".join(out)


def normalize(text):
    """评测用规范化: 返回只含汉字/字母的字符串"""
    text = full_to_half(text)
    text = re.sub(r"\d+", lambda m: num_to_zh(m.group()), text)
    text = re.sub(_TONE, "", text)
    text = re.sub(_PUNC, "", text)
    return text.lower()


if __name__ == "__main__":
    cases = ["温度调到２６度。", "打开空调，调到 26 度！", "音量调到105"]
    for c in cases:
        print(f"{c!r} -> {normalize(c)!r}")
