import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cer import cer, cer_stats, corpus_cer, corpus_cer_stats


class CerTests(unittest.TestCase):
    def test_substitution_and_deletion_example(self):
        value, reference_length = cer("我爱中国", "我很国")
        self.assertEqual(reference_length, 4)
        self.assertEqual(value, 0.5)

    def test_rejected_positive_is_all_deletions(self):
        value, reference_length = cer("打开空调", "")
        self.assertEqual(reference_length, 4)
        self.assertEqual(value, 1.0)

    def test_negative_reference_is_not_a_cer_sample(self):
        with self.assertRaises(ValueError):
            cer("", "任意文本")
        with self.assertRaises(ValueError):
            corpus_cer([("", "")])

    def test_missing_or_non_string_text_fails_loudly(self):
        with self.assertRaises(TypeError):
            cer("打开空调", None)
        with self.assertRaises(TypeError):
            cer("打开空调", ("打开空调", 0.12))
        with self.assertRaises(TypeError):
            cer(123, "123")

    def test_corpus_cer_is_not_sentence_average(self):
        pairs = [("甲", ""), ("甲乙丙丁戊己庚辛壬", "甲乙丙丁戊己庚辛壬")]
        value, chars = corpus_cer(pairs)
        self.assertEqual(chars, 10)
        self.assertAlmostEqual(value, 0.1)
        self.assertNotEqual(value, (1.0 + 0.0) / 2)

    def test_utf8_spaces_punctuation_and_digits_are_counted_raw(self):
        stats = cer_stats("开 空调，调到26度", "开空调调到二十六度")
        self.assertEqual(stats.reference_chars, len("开 空调，调到26度"))
        self.assertEqual(stats.errors, stats.substitutions + stats.deletions + stats.insertions)
        self.assertGreater(stats.errors, 0)

    def test_sdi_counts_and_corpus_aggregation(self):
        deleted = cer_stats("打开空调", "打开")
        inserted = cer_stats("打开", "请打开")
        substituted = cer_stats("打开", "关开")
        self.assertEqual((deleted.substitutions, deleted.deletions, deleted.insertions), (0, 2, 0))
        self.assertEqual((inserted.substitutions, inserted.deletions, inserted.insertions), (0, 0, 1))
        self.assertEqual((substituted.substitutions, substituted.deletions, substituted.insertions), (1, 0, 0))

        total = corpus_cer_stats([("打开空调", "打开"), ("打开", "请打开")])
        self.assertEqual(total.reference_chars, 6)
        self.assertEqual(total.errors, 3)

    def test_empty_corpus_is_undefined(self):
        with self.assertRaises(ValueError):
            corpus_cer([])


if __name__ == "__main__":
    unittest.main()
