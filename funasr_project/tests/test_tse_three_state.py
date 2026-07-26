"""Unit tests for TSE Dual-Output and Three-State Audit modules."""
import unittest
import torch

from tse_dual_output import TSEDualOutputNet
from three_state_audit import ThreeStateAudit, AuditReasonCode


class TestTSEDualOutput(unittest.TestCase):
    def setUp(self):
        self.net = TSEDualOutputNet()

    def test_forward_shapes_and_values(self):
        mix = torch.randn(2, 32000)
        enroll = torch.randn(2, 16000)
        target, residual = self.net.forward_audio(mix, enroll)
        self.assertEqual(target.shape, mix.shape)
        self.assertEqual(residual.shape, mix.shape)
        self.assertFalse(torch.isnan(target).any())
        self.assertFalse(torch.isnan(residual).any())


class TestThreeStateAudit(unittest.TestCase):
    def setUp(self):
        self.auditor = ThreeStateAudit(present_threshold=0.35, empty_threshold=0.18)

    def test_present_state(self):
        res = self.auditor.audit(target_sim=0.45)
        self.assertEqual(res["state"], "PRESENT")
        self.assertEqual(res["reason_code"], AuditReasonCode.TARGET_PRESENT_HIGH_CONF)
        self.assertTrue(res["emit_allowed"])

    def test_empty_low_sim(self):
        res = self.auditor.audit(target_sim=0.12)
        self.assertEqual(res["state"], "EMPTY")
        self.assertEqual(res["reason_code"], AuditReasonCode.TARGET_ABSENT_LOW_SIM)
        self.assertFalse(res["emit_allowed"])

    def test_empty_leaked_residual(self):
        res = self.auditor.audit(target_sim=0.22, residual_sim=0.42)
        self.assertEqual(res["state"], "EMPTY")
        self.assertEqual(res["reason_code"], AuditReasonCode.TARGET_LEAKED_TO_RESIDUAL)
        self.assertFalse(res["emit_allowed"])

    def test_gray_state(self):
        res = self.auditor.audit(target_sim=0.28)
        self.assertEqual(res["state"], "GRAY")
        self.assertEqual(res["reason_code"], AuditReasonCode.GRAY_LOW_SNR_AMBIGUOUS)


if __name__ == "__main__":
    unittest.main()
