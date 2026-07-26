"""Unit tests for TSE Dual-Output model and training logic."""
import unittest
import torch
import numpy as np

from tse_dual_output import TSEDualOutputNet, SR
from train_tse_dual_output import dual_loss_fn


class TestTSEDualOutputMVP(unittest.TestCase):
    def setUp(self):
        self.net = TSEDualOutputNet()

    def test_forward_audio_shapes(self):
        mix = torch.randn(2, 32000)
        enroll = torch.randn(2, 16000)
        target, residual = self.net.forward_audio(mix, enroll)
        self.assertEqual(target.shape, mix.shape)
        self.assertEqual(residual.shape, mix.shape)
        self.assertFalse(torch.isnan(target).any())
        self.assertFalse(torch.isnan(residual).any())

    def test_dual_loss_computation(self):
        target_est = torch.randn(2, 32000)
        residual_est = torch.randn(2, 32000)
        target_gt = torch.randn(2, 32000)
        residual_gt = torch.randn(2, 32000)
        loss = dual_loss_fn(target_est, residual_est, target_gt, residual_gt)
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.isnan(loss).any())


if __name__ == "__main__":
    unittest.main()
