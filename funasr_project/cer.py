# -*- coding: utf-8 -*-
"""
字错误率 CER 统计 (申报表评测指标): CER = (替换+删除+插入) / 参考长度
"""
from command_match import edit_distance
from text_norm import normalize


def cer(ref, hyp, do_norm=True):
    """返回 (cer, 参考长度)。ref/hyp 为字符串"""
    if do_norm:
        ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return (0.0 if not hyp else 1.0), 0
    return edit_distance(ref, hyp) / len(ref), len(ref)


def corpus_cer(pairs, do_norm=True):
    """语料级 CER: pairs = [(ref, hyp), ...], 按总编辑距离/总参考长度"""
    total_err, total_len = 0, 0
    for ref, hyp in pairs:
        if do_norm:
            ref, hyp = normalize(ref), normalize(hyp)
        if not ref:
            continue
        total_err += edit_distance(ref, hyp)
        total_len += len(ref)
    return (total_err / total_len if total_len else 0.0), total_len


if __name__ == "__main__":
    tests = [
        ("打开空调温度调到二十六度", "打开空调温度调到二十六度"),
        ("打开空调温度调到二十六度", "打开恐铁温度调到二十六度"),
        ("温度调到26度", "温度调到二十六度"),
    ]
    for ref, hyp in tests:
        c, L = cer(ref, hyp)
        print(f"ref={ref!r} hyp={hyp!r} -> CER={c:.3f} (len={L})")
