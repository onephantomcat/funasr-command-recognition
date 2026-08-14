import unittest

from eval_paired_cer import (
    acceptance_summary,
    analysis_bucket,
    summarize_p2_quality,
    summarize_predictions,
    validate_manifest,
)


def prediction(condition, errors, reference_chars, buckets):
    return {
        "condition": condition,
        "substitutions": errors,
        "deletions": 0,
        "insertions": 0,
        "reference_chars": reference_chars,
        "asr_status": "OK",
        "buckets": buckets,
    }


class PairedCerTests(unittest.TestCase):
    def test_extreme_overlap_bucket(self):
        buckets = analysis_bucket({"scene": "overlap", "overlap_ratio": 1.0, "sir_db": -5})
        self.assertIn("OVERLAP", buckets)
        self.assertIn("OVERLAP_100_SIR_-5DB", buckets)

    def test_manifest_mixture_seed_is_not_an_analysis_bucket(self):
        buckets = analysis_bucket({
            "scene": "overlap",
            "overlap_ratio": 1.0,
            "sir_db": -5,
            "seed": 12345,
        })
        self.assertFalse(any(bucket.startswith("seed:") for bucket in buckets))

    def test_manifest_requires_unique_ids_and_fields(self):
        with self.assertRaises(ValueError):
            validate_manifest([
                {"sample_id": "x", "ref_text": "甲", "mixture": "m", "target": "t", "tse_target": "p"},
                {"sample_id": "x", "ref_text": "乙", "mixture": "m2", "target": "t2", "tse_target": "p2"},
            ], require_precomputed_tse=True)

    def test_summary_keeps_conditions_paired(self):
        records = []
        for condition, errors in (("B0_MIXTURE", 4), ("ORACLE_TARGET", 1), ("B1_P2_TARGET", 2)):
            records.append(prediction(condition, errors, 10, ["OVERLAP"]))
        overall, buckets, comparisons = summarize_predictions(records)
        self.assertEqual(overall["B0_MIXTURE"]["cer"], 0.4)
        self.assertEqual(overall["B1_P2_TARGET"]["cer"], 0.2)
        self.assertEqual(buckets["OVERLAP"]["ORACLE_TARGET"]["cer"], 0.1)
        self.assertAlmostEqual(comparisons["OVERALL"]["relative_reduction"], 0.5)

    def test_summary_ignores_legacy_manifest_seed_buckets(self):
        records = []
        for condition, errors in (("B0_MIXTURE", 4), ("ORACLE_TARGET", 1), ("B1_P2_TARGET", 2)):
            records.append(prediction(condition, errors, 10, ["OVERLAP", "seed:123:OVERLAP"]))
        _, buckets, comparisons = summarize_predictions(records)
        self.assertNotIn("seed:123:OVERLAP", buckets)
        self.assertNotIn("seed:123:OVERLAP", comparisons)

    def test_single_run_acceptance_passes_without_manifest_seed_buckets(self):
        result = acceptance_summary({
            "OVERLAP_100": {
                "b0_cer": 0.4,
                "b1_cer": 0.3,
                "absolute_change_pp": -10.0,
                "absolute_reduction_pp": 10.0,
                "relative_reduction": 0.25,
            },
            "SINGLE": {
                "b0_cer": 0.1,
                "b1_cer": 0.11,
                "absolute_change_pp": 1.0,
                "absolute_reduction_pp": -1.0,
                "relative_reduction": -0.1,
            },
            "OVERLAP_100_SIR_-5DB": {
                "b0_cer": 0.8,
                "b1_cer": 0.7,
                "absolute_change_pp": -10.0,
                "absolute_reduction_pp": 10.0,
                "relative_reduction": 0.125,
            },
        }, result_valid=True, asr_errors=0, p2_output_quality={
            "samples": 6000,
            "near_silent_samples": 0,
        }, expected_samples=6000)
        self.assertEqual(result["verdict"], "PASS_SINGLE_RUN_THRESHOLDS")

    def test_single_run_acceptance_fails_near_silent_output(self):
        comparisons = {
            "OVERLAP_100": {
                "b0_cer": 0.4,
                "b1_cer": 0.3,
                "absolute_change_pp": -10.0,
                "absolute_reduction_pp": 10.0,
                "relative_reduction": 0.25,
            },
            "SINGLE": {
                "b0_cer": 0.1,
                "b1_cer": 0.1,
                "absolute_change_pp": 0.0,
                "absolute_reduction_pp": 0.0,
                "relative_reduction": 0.0,
            },
            "OVERLAP_100_SIR_-5DB": {
                "b0_cer": 0.8,
                "b1_cer": 0.7,
                "absolute_change_pp": -10.0,
                "absolute_reduction_pp": 10.0,
                "relative_reduction": 0.125,
            },
        }
        result = acceptance_summary(
            comparisons,
            result_valid=True,
            asr_errors=0,
            p2_output_quality={"samples": 20, "near_silent_samples": 1},
            expected_samples=20,
        )
        self.assertEqual(result["verdict"], "FAIL_SINGLE_RUN_THRESHOLDS")

    def test_p2_quality_summary_flags_near_silent_outputs(self):
        records = [
            {
                "condition": "B1_P2_TARGET",
                "p2_info": {
                    "output_to_input_rms_ratio": 1e-7,
                    "output_near_silent": True,
                },
            },
            {
                "condition": "B1_P2_TARGET",
                "p2_info": {
                    "output_to_input_rms_ratio": 0.5,
                    "output_near_silent": False,
                },
            },
            {"condition": "B0_MIXTURE", "p2_info": None},
        ]
        summary = summarize_p2_quality(records)
        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["near_silent_samples"], 1)
        self.assertEqual(summary["near_silent_rate"], 0.5)
        self.assertAlmostEqual(summary["rms_ratio_median"], 0.25000005)


if __name__ == "__main__":
    unittest.main()
