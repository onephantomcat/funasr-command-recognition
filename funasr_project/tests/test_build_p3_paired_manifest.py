import json
import tempfile
import unittest
from pathlib import Path

from build_p3_paired_manifest import build_manifest, compact_reference_text


class BuildP3PairedManifestTests(unittest.TestCase):
    def test_compact_reference_text_removes_token_spaces(self):
        self.assertEqual(compact_reference_text("打 开 空 调"), "打开空调")

    def test_build_maps_p1_fields_and_reports_missing_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.txt"
            transcript.write_text("UTT001 打 开 空 调\n", encoding="utf-8")
            manifest = root / "D_single.jsonl"
            manifest.write_text(
                json.dumps({
                    "sample_id": "sample-1",
                    "target_utt": "UTT001",
                    "mixture_wav": "audio/mixture/sample-1.wav",
                    "target_wav": "audio/target/sample-1.wav",
                    "enroll_wav": "audio/enroll/sample-1.wav",
                    "scenario": "single",
                    "split": "D_single",
                    "measured_overlap": 0.0,
                    "measured_snr_db": 3.0,
                    "seed": 7,
                }) + "\n",
                encoding="utf-8",
            )

            rows, report = build_manifest([manifest], transcript, root)
            self.assertEqual(rows[0]["ref_text"], "打开空调")
            self.assertEqual(rows[0]["scene"], "SINGLE")
            self.assertEqual(report["rows"], 1)
            self.assertFalse(report["audio_readiness"]["ready_for_paired_eval"])
            self.assertEqual(report["audio_readiness"]["missing_total"], 3)

    def test_duplicate_sample_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.txt"
            transcript.write_text("UTT001 文 本\n", encoding="utf-8")
            payload = {
                "sample_id": "duplicate",
                "target_utt": "UTT001",
                "mixture_wav": "m.wav",
                "target_wav": "t.wav",
                "enroll_wav": "e.wav",
            }
            manifests = []
            for index in range(2):
                path = root / f"m{index}.jsonl"
                path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                manifests.append(path)
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                build_manifest(manifests, transcript, root)


if __name__ == "__main__":
    unittest.main()
