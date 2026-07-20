import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_external_trainset import build
from prepare_augmentation_assets import candidate_urls


class ExternalAugmentationTests(unittest.TestCase):
    def write_wav(self, path, frequency):
        samples = np.arange(4000, dtype=np.float32) / 16000
        sf.write(path, 0.15 * np.sin(2 * np.pi * frequency * samples), 16000)

    def test_builds_competition_format_with_noise_and_rir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_root = root / "source"
            wav_root.mkdir()
            transcript = root / "transcript.csv"
            with open(transcript, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for speaker in range(1, 5):
                    for utterance in range(7):
                        name = f"S{speaker:04d}_{utterance:02d}.wav"
                        self.write_wav(wav_root / name, 180 + speaker * 30 + utterance)
                        writer.writerow([name, f"测试语句{speaker}{utterance}"])

            noise_root = root / "musan"
            rir_root = root / "rirs"
            noise_root.mkdir()
            rir_root.mkdir()
            self.write_wav(noise_root / "noise.wav", 70)
            sf.write(rir_root / "rir.wav", np.array([1.0, 0.3, 0.1], dtype=np.float32), 16000)

            out = root / "out"
            build(SimpleNamespace(
                wav_root=str(wav_root),
                csv=str(transcript),
                out=str(out),
                target_speakers=2,
                interferer_speakers=2,
                enroll_count=1,
                trials_per_speaker=1,
                wake_text="hi colmo",
                seed=7,
                noise_root=str(noise_root),
                rir_root=str(rir_root),
                noise_snr_db=5.0,
                reverb_noise_snr_db=5.0,
            ))

            pos = [json.loads(line) for line in (out / "pos.jsonl").read_text(encoding="utf-8").splitlines()]
            neg = [json.loads(line) for line in (out / "neg.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pos), 12)
            self.assertEqual(len(neg), 4)
            self.assertTrue(all(row["识别文本"] for row in pos))
            self.assertTrue(all(row["识别文本"] is None for row in neg))
            self.assertTrue(any("noise5" in row["识别音频"] for row in pos))
            self.assertTrue(any("reverb_noise5" in row["识别音频"] for row in pos))
            self.assertEqual(sf.info(out / pos[0]["识别音频"]).samplerate, 16000)

    def test_auto_source_prefers_mirror_then_official(self):
        spec = {"mirror_url": "https://mirror", "official_url": "https://official"}
        self.assertEqual(candidate_urls(spec, "auto"), ["https://mirror", "https://official"])


if __name__ == "__main__":
    unittest.main()
