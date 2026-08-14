import sys
import unittest
from pathlib import Path

import torch


P2_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = P2_ROOT / "tools"
for path in (P2_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.tse.losses import activity_bce_loss, si_sdr
from src.tse.model import DualOutputTSE
from train_overfit_debug import compute_losses


class AmpLossStabilityTests(unittest.TestCase):
    def test_fp16_boundary_probabilities_have_finite_bce(self):
        probability = torch.tensor(
            [[1.0, 0.0]], dtype=torch.float16, requires_grad=True
        )
        label = torch.tensor([[1.0, 0.0]], dtype=torch.float16)

        loss = activity_bce_loss(probability, label)

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_fp16_saturated_logits_have_finite_bce_and_gradient(self):
        logits = torch.tensor(
            [[100.0, -100.0]], dtype=torch.float16, requires_grad=True
        )
        label = torch.tensor([[1.0, 0.0]], dtype=torch.float16)

        loss = activity_bce_loss(logits, label, from_logits=True)
        loss.backward()

        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_eight_second_fp16_si_sdr_reduction_is_finite(self):
        samples = 8 * 16000
        estimate = torch.ones((1, samples), dtype=torch.float16)
        reference = torch.ones((1, samples), dtype=torch.float16)

        value = si_sdr(estimate, reference)

        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(value)))

    def test_default_activity_interface_is_unchanged(self):
        cfg = {
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "emb_dim": 8,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        }
        model = DualOutputTSE(cfg).eval()
        mixture = torch.randn(1, 512)
        embedding = torch.randn(1, 8)

        with torch.no_grad():
            public_output = model(mixture, embedding)
            training_output = model(
                mixture, embedding, return_activity_logits=True
            )

        self.assertEqual(len(public_output), 3)
        self.assertEqual(len(training_output), 4)
        self.assertTrue(torch.equal(public_output[0], training_output[0]))
        self.assertTrue(torch.equal(public_output[1], training_output[1]))
        self.assertTrue(torch.equal(public_output[2], training_output[2]))
        self.assertTrue(
            torch.allclose(
                training_output[2], torch.sigmoid(training_output[3]), atol=0, rtol=0
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for AMP regression")
    def test_cuda_amp_forward_backward_keeps_all_loss_terms_and_gradients_finite(self):
        cfg = {
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "emb_dim": 8,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
            "zero_ref_kappa": 1.0e-3,
            "mrstft_resolutions": [[64, 16, 64]],
            "lambda_sisdr": 1.0,
            "lambda_wav": 1.0,
            "lambda_stft": 1.0,
            "lambda_act": 0.5,
            "lambda_residual": 0.5,
            "lambda_mix": 0.5,
        }
        device = torch.device("cuda")
        model = DualOutputTSE(cfg).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        try:
            scaler = torch.amp.GradScaler(
                "cuda", enabled=True, init_scale=128.0
            )
        except AttributeError:
            # PyTorch 2.1 exposes GradScaler only from torch.cuda.amp.
            scaler = torch.cuda.amp.GradScaler(enabled=True, init_scale=128.0)
        generator = torch.Generator(device=device).manual_seed(20260814)
        mixture = torch.randn(2, 4096, device=device, generator=generator) * 0.05
        target = torch.randn(2, 4096, device=device, generator=generator) * 0.02
        embedding = torch.randn(2, 8, device=device, generator=generator)
        frame_count = mixture.shape[-1] // cfg["hop_length"] + 1
        batch = {
            "mix": mixture,
            "target": target,
            "interferer": mixture - target,
            "emb": embedding,
            "frame_act": torch.ones(2, frame_count, device=device),
            "is_absent": torch.zeros(2, dtype=torch.bool, device=device),
        }

        with torch.amp.autocast("cuda", enabled=True):
            output = model(
                batch["mix"], batch["emb"], return_activity_logits=True
            )
        total, terms = compute_losses(cfg, output, batch)
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)

        self.assertEqual(total.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(total)))
        self.assertTrue(all(bool(torch.isfinite(v)) for v in terms.values()))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertIsNotNone(model.act_head.weight.grad)
        self.assertGreater(model.act_head.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
