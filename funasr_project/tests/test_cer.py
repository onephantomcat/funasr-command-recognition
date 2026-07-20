import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cer import cer, corpus_cer


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


if __name__ == "__main__":
    unittest.main()
