import unittest
from types import SimpleNamespace

from p3_eval_contracts import (
    RecognitionContractError,
    RecognitionOutcome,
    negative_is_rejected,
    recognize_safely,
    recognize_result_safely,
    require_unique_sample_ids,
    unpack_recognition,
)


class P3EvalContractTests(unittest.TestCase):
    def test_asr_tuple_is_unpacked(self):
        text, elapsed = unpack_recognition(("打开空调", 0.125))
        self.assertEqual(text, "打开空调")
        self.assertEqual(elapsed, 0.125)

    def test_malformed_asr_result_fails_loudly(self):
        for result in ("打开空调", ("打开空调",), (("打开空调", 0.1), 0.2), (None, 0.1)):
            with self.subTest(result=result):
                with self.assertRaises(RecognitionContractError):
                    unpack_recognition(result)

    def test_runtime_error_is_preserved(self):
        def failing_recognizer(_model, _path):
            raise RuntimeError("decoder failed")

        outcome = recognize_safely(failing_recognizer, object(), "sample.wav")
        self.assertEqual(outcome.status, "ERROR")
        self.assertIn("decoder failed", outcome.error)

    def test_structured_stages_are_preserved(self):
        def recognizer(_model, _path):
            return SimpleNamespace(
                raw_text="开 空 调",
                normalized_text="开空调",
                final_text="开空调",
                elapsed_sec=0.2,
                status="OK",
            )

        outcome = recognize_result_safely(recognizer, object(), "sample.wav")
        self.assertEqual(outcome.raw_text, "开 空 调")
        self.assertEqual(outcome.normalized_text, "开空调")
        self.assertEqual(outcome.final_text, "开空调")

    def test_emit_then_empty_asr_counts_as_rejected(self):
        outcome = RecognitionOutcome(text="", elapsed_sec=0.1, status="OK")
        self.assertTrue(negative_is_rejected(emit_allowed=True, outcome=outcome))

    def test_asr_error_does_not_masquerade_as_rejection(self):
        outcome = RecognitionOutcome(
            text="", elapsed_sec=0.0, status="ERROR", error="RuntimeError: failed"
        )
        self.assertFalse(negative_is_rejected(emit_allowed=True, outcome=outcome))

    def test_duplicate_sample_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            require_unique_sample_ids([
                {"sample_id": "x"},
                {"sample_id": "x"},
            ])

    def test_missing_sample_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            require_unique_sample_ids([{"sample_id": ""}])


if __name__ == "__main__":
    unittest.main()
