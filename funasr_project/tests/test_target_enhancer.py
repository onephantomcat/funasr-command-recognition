import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from target_enhancer import (
    SR,
    TargetEnhancer,
    enhance_audio,
    enhance_file,
    enhancement_loss,
    load_target_enhancer,
    write_audio,
)
from train_target_enhancer import ExternalMixtureDataset


class TargetEnhancerTests(unittest.TestCase):
    def test_forward_and_loss_are_finite(self):
        model = TargetEnhancer(channels=8, blocks=2)
        mixture = torch.randn(2, 4096) * 0.02
        wake = torch.randn(2, 4096) * 0.02
        clean = torch.randn(2, 4096) * 0.01
        enhanced, mask = model(mixture, wake)
        loss, parts = enhancement_loss(enhanced, clean)
        self.assertEqual(tuple(enhanced.shape), tuple(mixture.shape))
        self.assertEqual(mask.shape[:2], (2, 257))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(parts["logmag_l1"]))

    def test_checkpoint_and_wav_output(self):
        model = TargetEnhancer(channels=8, blocks=2).eval()
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            checkpoint = temp / "enhancer.pt"
            torch.save({"model_config": model.config(), "model_state": model.state_dict()}, checkpoint)
            loaded = load_target_enhancer(checkpoint, device="cpu")
            wake = np.sin(np.linspace(0, 20, SR // 2, dtype=np.float32)) * 0.1
            mixture = np.concatenate([wake, wake]).astype(np.float32)
            wake_path = temp / "wake.wav"
            mixture_path = temp / "mixture.wav"
            out_path = temp / "enhanced.wav"
            write_audio(wake_path, wake)
            write_audio(mixture_path, mixture)
            info = enhance_file(loaded, wake_path, mixture_path, out_path, chunk_sec=0.3, overlap_sec=0.05)
            self.assertEqual(info["samples"], len(mixture))
            written, sr = sf.read(out_path)
            self.assertEqual(sr, SR)
            self.assertEqual(len(written), len(mixture))
            self.assertTrue(np.isfinite(written).all())
            direct = enhance_audio(loaded, wake, mixture, chunk_sec=0.3, overlap_sec=0.05)
            self.assertEqual(len(direct), len(mixture))

    def test_external_mixture_dataset_returns_supervised_triplet(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            speakers = {}
            for speaker_index, speaker in enumerate(("S0001", "S0002")):
                paths = []
                for item in range(2):
                    path = temp / f"BAC009{speaker}W{item:04d}.wav"
                    write_audio(path, np.full(SR // 4, 0.02 * (speaker_index + item + 1), dtype=np.float32))
                    paths.append(path)
                speakers[speaker] = paths
            noise = temp / "musan.wav"
            rir = temp / "rir.wav"
            write_audio(noise, np.random.default_rng(1).normal(0, 0.01, SR // 4).astype(np.float32))
            write_audio(rir, np.array([1.0, 0.3, 0.1], dtype=np.float32))
            dataset = ExternalMixtureDataset(
                speakers, [noise], [rir], items=2, segment_sec=0.25, seed=4,
            )
            mixture, wake, clean = dataset[0]
            self.assertEqual(tuple(mixture.shape), (SR // 4,))
            self.assertEqual(tuple(wake.shape), (SR // 4,))
            self.assertEqual(tuple(clean.shape), (SR // 4,))
            self.assertTrue(torch.isfinite(mixture).all())


if __name__ == "__main__":
    unittest.main()
