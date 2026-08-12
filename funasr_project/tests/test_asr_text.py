import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asr_demo import compact_asr_text, recognize, recognize_result


class AsrTextTests(unittest.TestCase):
    def test_compacts_formatting_whitespace_without_changing_characters(self):
        self.assertEqual(compact_asr_text("开 屏 幕\tECO 模 式\n"), "开屏幕ECO模式")

    def test_handles_empty_text(self):
        self.assertEqual(compact_asr_text(None), "")

    def test_structured_result_preserves_raw_normalized_and_final(self):
        class FakeModel:
            def generate(self, **_kwargs):
                return [{"text": "开 空 调\t26 度"}]

        result = recognize_result(FakeModel(), "unused.wav")
        self.assertEqual(result.raw_text, "开 空 调\t26 度")
        self.assertEqual(result.normalized_text, "开空调26度")
        self.assertEqual(result.final_text, "开空调26度")
        self.assertEqual(result.status, "OK")
        self.assertGreaterEqual(result.elapsed_sec, 0.0)

        legacy_text, legacy_elapsed = recognize(FakeModel(), "unused.wav")
        self.assertEqual(legacy_text, result.final_text)
        self.assertGreaterEqual(legacy_elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
