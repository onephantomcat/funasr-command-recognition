import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


P2_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = P2_ROOT / "tools"
sys.path.insert(0, str(P2_ROOT))
sys.path.insert(0, str(TOOLS_ROOT))

from train_b1_trial import optimizer_step_if_finite


def disabled_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=False)


class OptimizerStepIfFiniteTest(unittest.TestCase):
    def test_nonfinite_gradient_does_not_mutate_parameter_or_optimizer_state(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        scaler = disabled_grad_scaler()
        parameter = next(model.parameters())
        before = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, float("inf"))

        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = optimizer_step_if_finite(model, optimizer, scaler, grad_clip=1.0)

        step.assert_not_called()
        self.assertTrue(torch.equal(parameter, before))
        self.assertEqual(optimizer.state, {})
        self.assertIsNone(parameter.grad)
        self.assertFalse(result["grad_finite"])
        self.assertFalse(result["optimizer_step_applied"])
        self.assertTrue(result["optimizer_step_skipped"])
        self.assertEqual(result["skip_reason"], "nonfinite_gradient")
        self.assertEqual(result["nonfinite_gradient_tensors"], 1)

    def test_finite_gradient_applies_clipped_optimizer_update(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = disabled_grad_scaler()
        parameter = next(model.parameters())
        before = parameter.detach().clone()
        parameter.grad = torch.full_like(parameter, 2.0)

        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = optimizer_step_if_finite(model, optimizer, scaler, grad_clip=1.0)

        step.assert_called_once_with()
        self.assertFalse(torch.equal(parameter, before))
        self.assertIsNone(parameter.grad)
        self.assertTrue(result["grad_finite"])
        self.assertTrue(result["optimizer_step_applied"])
        self.assertFalse(result["optimizer_step_skipped"])
        self.assertIsNone(result["skip_reason"])
        self.assertTrue(result["clipped"])
        self.assertAlmostEqual(result["grad_norm"], 2.0, places=5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP scale backoff")
    def test_cuda_nonfinite_gradient_skips_update_and_backs_off_scale(self):
        model = torch.nn.Linear(1, 1, bias=False).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
        except AttributeError:
            scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=1024.0)

        parameter = next(model.parameters())
        before = parameter.detach().clone()
        loss = model(torch.ones(1, 1, device="cuda")).sum()
        scaler.scale(loss).backward()
        parameter.grad.fill_(float("inf"))

        with mock.patch.object(optimizer, "step", wraps=optimizer.step) as step:
            result = optimizer_step_if_finite(model, optimizer, scaler, grad_clip=1.0)

        step.assert_not_called()
        self.assertTrue(torch.equal(parameter, before))
        self.assertEqual(optimizer.state, {})
        self.assertIsNone(parameter.grad)
        self.assertFalse(result["grad_finite"])
        self.assertTrue(result["optimizer_step_skipped"])
        self.assertEqual(result["amp_scale_before"], 1024.0)
        self.assertEqual(result["amp_scale_after"], 512.0)


if __name__ == "__main__":
    unittest.main()
