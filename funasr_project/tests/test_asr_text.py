import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr_demo import compact_asr_text


class AsrTextTests(unittest.TestCase):
    def test_compacts_formatting_whitespace_without_changing_characters(self):
        self.assertEqual(compact_asr_text("开 屏 幕\tECO 模 式\n"), "开屏幕ECO模式")

    def test_handles_empty_text(self):
        self.assertEqual(compact_asr_text(None), "")


if __name__ == "__main__":
    unittest.main()
