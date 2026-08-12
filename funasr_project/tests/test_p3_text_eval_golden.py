import unittest

from cer import cer_stats


class P3TextEvalGoldenTests(unittest.TestCase):
    def test_eighteen_frozen_utf8_cases(self):
        long_text = "空调" * 50
        cases = [
            ("exact_chinese", "打开空调", "打开空调", False, (0, 0, 0, 4)),
            ("substitution", "甲乙", "甲丙", False, (1, 0, 0, 2)),
            ("deletion", "打开空调", "打开", False, (0, 2, 0, 4)),
            ("insertion", "打开", "请打开", False, (0, 0, 1, 2)),
            ("rejected_positive", "打开空调", "", False, (0, 4, 0, 4)),
            ("raw_punctuation", "打开。", "打开", False, (0, 1, 0, 3)),
            ("raw_space", "开 空", "开空", False, (0, 1, 0, 3)),
            ("normalized_digits", "调到26度", "调到二十六度", True, (0, 0, 0, 6)),
            ("fullwidth_digits", "温度２６度", "温度26度", True, (0, 0, 0, 6)),
            ("latin_case", "ECO模式", "eco模式", True, (0, 0, 0, 5)),
            ("normalized_punctuation", "打开，空调！", "打开空调", True, (0, 0, 0, 4)),
            ("tone_particle", "打开空调啊", "打开空调", True, (0, 0, 0, 4)),
            ("traditional_raw", "開空調", "开空调", False, (2, 0, 0, 3)),
            ("emoji_deletion", "开😊空调", "开空调", False, (0, 1, 0, 4)),
            ("repeated_character", "开开空调", "开空调", False, (0, 1, 0, 4)),
            ("latin_insertion", "ECO模式", "ECO模式X", False, (0, 0, 1, 5)),
            ("mixed_substitution", "打开AC", "打开DC", False, (1, 0, 0, 4)),
            ("long_exact", long_text, long_text, False, (0, 0, 0, 100)),
        ]
        self.assertEqual(len(cases), 18)
        for name, ref, hyp, do_norm, expected in cases:
            with self.subTest(name=name):
                stats = cer_stats(ref, hyp, do_norm=do_norm)
                actual = (
                    stats.substitutions,
                    stats.deletions,
                    stats.insertions,
                    stats.reference_chars,
                )
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
