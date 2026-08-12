import json
import tempfile
import unittest
from pathlib import Path

from eval_datasetA import save_report, summarize_p2_output_quality


class DatasetAEvaluatorIoTests(unittest.TestCase):
    def test_save_report_creates_nested_parent_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            save_report(str(output), {"result_valid": True})
            self.assertTrue(output.is_file())
            self.assertFalse(Path(str(output) + ".tmp").exists())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"result_valid": True},
            )

    def test_p2_quality_summary_is_machine_readable(self):
        details = [
            {
                "p2_tse_info": {
                    "output_to_input_rms_ratio": 2e-7,
                    "output_near_silent": True,
                }
            },
            {
                "p2_tse_info": {
                    "output_to_input_rms_ratio": 4e-7,
                    "output_near_silent": True,
                }
            },
        ]
        summary = summarize_p2_output_quality(details)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["near_silent_samples"], 2)
        self.assertEqual(summary["near_silent_rate"], 1.0)
        self.assertAlmostEqual(summary["rms_ratio_median"], 3e-7)


if __name__ == "__main__":
    unittest.main()
