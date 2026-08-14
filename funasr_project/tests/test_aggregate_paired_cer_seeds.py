import unittest

from aggregate_paired_cer_seeds import aggregate_summaries
from eval_paired_cer import CONTRACT_VERSION


def summary(training_seed, *, passed=True, manifest_sha="m" * 64):
    return {
        "contract": CONTRACT_VERSION,
        "training_seed": training_seed,
        "result_valid": True,
        "asr_errors": 0,
        "samples": 6000,
        "manifest_sha256": manifest_sha,
        "asr": {
            "model_id": "asr",
            "vad_model_id": "vad",
            "with_punctuation": False,
            "device": "cuda",
        },
        "p2": {"checkpoint_sha256": f"{training_seed:064x}"},
        "p2_output_quality": {"samples": 6000, "near_silent_samples": 0},
        "acceptance": {
            "verdict": (
                "PASS_SINGLE_RUN_THRESHOLDS"
                if passed
                else "FAIL_SINGLE_RUN_THRESHOLDS"
            )
        },
        "b1_vs_b0": {
            "OVERALL": {"absolute_reduction_pp": 3.0},
            "OVERLAP_100": {"absolute_reduction_pp": 6.0},
            "SINGLE": {"absolute_change_pp": -1.0},
            "OVERLAP_100_SIR_-5DB": {"absolute_reduction_pp": 5.0},
        },
    }


class AggregatePairedCerSeedsTests(unittest.TestCase):
    def test_accepts_two_of_three_when_frozen_candidate_passes(self):
        result = aggregate_summaries(
            [summary(1), summary(2), summary(3, passed=False)],
            frozen_training_seed=1,
        )
        self.assertEqual(result["verdict"], "ACCEPT_B1_CANDIDATE")
        self.assertEqual(result["passed_training_seed_count"], 2)

    def test_rejects_when_only_one_seed_passes(self):
        result = aggregate_summaries(
            [summary(1), summary(2, passed=False), summary(3, passed=False)],
            frozen_training_seed=1,
        )
        self.assertEqual(result["verdict"], "REJECT_CURRENT_TSE")

    def test_rejects_when_frozen_candidate_fails(self):
        result = aggregate_summaries(
            [summary(1, passed=False), summary(2), summary(3)],
            frozen_training_seed=1,
        )
        self.assertEqual(result["verdict"], "REJECT_CURRENT_TSE")

    def test_rejects_duplicate_training_seeds(self):
        with self.assertRaisesRegex(ValueError, "training_seed values must be distinct"):
            aggregate_summaries(
                [summary(1), summary(1), summary(3)],
                frozen_training_seed=1,
            )

    def test_rejects_mismatched_manifest(self):
        with self.assertRaisesRegex(ValueError, "same manifest_sha256"):
            aggregate_summaries(
                [summary(1), summary(2), summary(3, manifest_sha="x" * 64)],
                frozen_training_seed=1,
            )


if __name__ == "__main__":
    unittest.main()
