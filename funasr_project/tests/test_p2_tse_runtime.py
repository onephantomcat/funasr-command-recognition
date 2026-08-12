import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from P2_project.src.tse import DualOutputTSE, extract_target
from p2_tse_runtime import P2TSERuntime, sha256_file


class P2TSERuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cfg = {
            "sample_rate": 16000,
            "n_fft": 64,
            "hop_length": 16,
            "win_length": 64,
            "emb_dim": 4,
            "lstm_hidden": 8,
            "lstm_layers": 1,
            "dropout": 0.0,
        }
        model = DualOutputTSE(self.cfg)
        self.checkpoint = self.root / "p2.pt"
        torch.save({"model": model.state_dict(), "cfg": self.cfg, "step": 7}, self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def test_load_hash_and_extract_file(self):
        expected_hash = sha256_file(self.checkpoint)
        runtime = P2TSERuntime(
            self.checkpoint,
            device="cpu",
            expected_sha256=expected_hash,
        )
        command_path = self.root / "command.wav"
        output_path = self.root / "out" / "target.wav"
        sf.write(command_path, np.linspace(-0.1, 0.1, 800, dtype=np.float32), 8000)

        info = runtime.extract_file(
            command_path,
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            output_path,
        )
        output, sample_rate = sf.read(output_path, dtype="float32")
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(len(output), 1600)
        self.assertTrue(np.isfinite(output).all())
        self.assertEqual(info["checkpoint_sha256"], expected_hash)
        self.assertEqual(info["step"], 7)
        self.assertGreater(info["input_rms"], 0)
        self.assertGreaterEqual(info["output_rms"], 0)
        self.assertGreaterEqual(info["output_to_input_rms_ratio"], 0)
        self.assertIsInstance(info["output_near_silent"], bool)
        self.assertFalse(info["cached"])

    def test_wrong_checkpoint_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            P2TSERuntime(self.checkpoint, device="cpu", expected_sha256="0" * 64)

    def test_embedding_dimension_is_strict(self):
        runtime = P2TSERuntime(self.checkpoint, device="cpu")
        command_path = self.root / "command.wav"
        sf.write(command_path, np.zeros(800, dtype=np.float32), 16000)
        with self.assertRaisesRegex(ValueError, "embedding"):
            runtime.extract_file(command_path, np.ones(3, dtype=np.float32), self.root / "out.wav")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_frozen_api_cuda_smoke(self):
        cfg = {
            "sample_rate": 16000,
            "n_fft": 512,
            "hop_length": 128,
            "win_length": 512,
            "emb_dim": 192,
            "lstm_hidden": 256,
            "lstm_layers": 2,
            "dropout": 0.0,
        }
        model = DualOutputTSE(cfg).cuda().eval()
        command = torch.randn(1, 16000, device="cuda")
        embedding = torch.nn.functional.normalize(
            torch.randn(1, 192, device="cuda"), dim=-1
        )
        torch.cuda.reset_peak_memory_stats()
        target = extract_target(command, embedding, model, cfg)
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        self.assertEqual(target.shape, command.shape)
        self.assertTrue(torch.isfinite(target).all())
        self.assertLess(peak_gb, 6.0)


if __name__ == "__main__":
    unittest.main()
