# -*- coding: utf-8 -*-
"""B3 near-silence and enrollment-semantics regression tests."""

import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

P2_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = P2_ROOT / "tools"
for path in (P2_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.tse.enrollment_adapter import EnrollmentAdapter
from src.tse.losses import (
    activity_bce_loss,
    mrstft_loss,
    scale_sensitive_l1,
    si_sdr,
)
from evaluate_tse import resolve_wrong_enrollment
from train_b1_trial import B1Dataset
from train_overfit_debug import compute_losses


def _loss_cfg(absent_scale=0.05):
    return {
        "zero_ref_kappa": 1.0e-3,
        "absent_loss_scale": absent_scale,
        "mrstft_resolutions": [[16, 4, 16]],
        "lambda_sisdr": 2.0,
        "lambda_wav": 0.5,
        "lambda_stft": 0.5,
        "lambda_act": 1.0,
        "lambda_residual": 0.5,
        "lambda_mix": 0.5,
    }


def _write_dataset_fixture(tmp_path, target_present=False):
    sr = 16000
    n = 160
    t = np.arange(n, dtype=np.float32) / sr
    mix = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    target = mix.copy() if target_present else np.zeros_like(mix)
    residual = mix - target
    activity = np.ones(n, dtype=np.float32) if target_present else np.zeros(n, dtype=np.float32)
    paths = {}
    for name, wav in (("mix", mix), ("target", target), ("residual", residual), ("enroll", mix)):
        paths[name] = tmp_path / f"{name}.wav"
        sf.write(str(paths[name]), wav, sr, subtype="FLOAT")
    paths["activity"] = tmp_path / "activity.npy"
    np.save(str(paths["activity"]), activity)
    row = {
        "sample_id": "fixture",
        "mixture_wav": str(paths["mix"]),
        "target_wav": str(paths["target"]),
        "interferer_wav": str(paths["residual"]),
        "activity_mask": str(paths["activity"]),
        "enroll_wav": str(paths["enroll"]),
        "target_speaker": "S_ENROLLED",
        "target_present": target_present,
        "scenario": "enroll_swap_target_1" if target_present else "absent",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    cfg = {
        "sample_rate": sr,
        "segment_length": n / sr,
        "win_length": 16,
        "hop_length": 4,
        "act_frame_ratio": 0.5,
        "emb_dim": 8,
        "scene_mode": "b2" if not target_present else "b3",
        "sv_mode": "bootstrap",
    }
    return manifest, cfg


class B3TrainingRegressionTests(unittest.TestCase):
    def test_waveform_l1_penalizes_present_gain_collapse(self):
        ref = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
        near_silent = ref * 0.01
        loss = scale_sensitive_l1(near_silent, ref, 1.0e-3)
        self.assertAlmostEqual(loss.item(), (near_silent - ref).abs().mean().item(), places=7)
        self.assertGreater(loss.item(), 0.7)

    def test_absent_scale_and_residual_reference(self):
        g = torch.Generator().manual_seed(20260812)
        target = torch.stack((torch.randn(64, generator=g) * 0.2, torch.zeros(64)))
        mix = torch.randn(2, 64, generator=g) * 0.3
        s_tgt = torch.randn(2, 64, generator=g) * 0.1
        s_res = mix - target
        p_tgt = torch.sigmoid(torch.randn(2, 17, generator=g))
        frame_act = torch.stack((torch.ones(17), torch.zeros(17)))
        batch = {
            "target": target,
            "interferer": torch.full_like(mix, 99.0),
            "mix": mix,
            "frame_act": frame_act,
            "is_absent": torch.tensor([0, 1], dtype=torch.uint8),
        }
        cfg = _loss_cfg()

        _total, terms = compute_losses(cfg, (s_tgt, s_res, p_tgt), batch)
        weights = torch.tensor([1.0, cfg["absent_loss_scale"]])
        wav_ps = scale_sensitive_l1(s_tgt, target, cfg["zero_ref_kappa"], reduction="none")
        stft_ps = mrstft_loss(s_tgt, target, cfg["mrstft_resolutions"], reduction="none")
        act_ps = activity_bce_loss(p_tgt, frame_act, reduction="none")
        sisdr_ps = si_sdr(s_tgt, target, reduction="none")

        self.assertAlmostEqual(terms["wav_l1"].item(), float((wav_ps * weights).sum() / weights.sum()), places=6)
        self.assertAlmostEqual(terms["mrstft"].item(), float((stft_ps * weights).sum() / weights.sum()), places=6)
        self.assertAlmostEqual(terms["act_bce"].item(), float((act_ps * weights).sum() / weights.sum()), places=6)
        self.assertAlmostEqual(terms["si_sdr_db"].item(), float(sisdr_ps[0]), places=6)
        self.assertAlmostEqual(terms["res_l1"].item(), 0.0, places=8)
        self.assertLess(float(stft_ps[1]), 10.0)
        self.assertLess(abs(float(_total)), 100.0)

    def test_absent_sample_keeps_enrollment_condition(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, cfg = _write_dataset_fixture(Path(td), target_present=False)
            sample = B1Dataset(manifest, cfg, seed=1)[0]
        self.assertGreater(torch.count_nonzero(sample["emb"]).item(), 0)
        self.assertAlmostEqual(sample["emb"].norm().item(), 1.0, places=6)
        self.assertEqual(sample["is_absent"].item(), 1)
        self.assertEqual(torch.count_nonzero(sample["target"]).item(), 0)

    def test_target_present_true_overrides_empty_activity(self):
        self.assertFalse(B1Dataset._is_absent_entry(
            {"target_present": True, "scenario": "enroll_swap_target_1", "is_absent": True},
            np.zeros(8, dtype=np.float32),
        ))

    def test_worker_rng_uses_torch_worker_seed(self):
        dataset = object.__new__(B1Dataset)
        dataset.rng = np.random.default_rng(1)
        dataset._rng_worker_id = None
        with mock.patch("train_b1_trial.data.get_worker_info", return_value=None), \
                mock.patch("train_b1_trial.torch.initial_seed", return_value=12345):
            first = dataset._worker_rng().integers(0, 2**31, size=4).tolist()
        expected = np.random.default_rng(12345).integers(0, 2**31, size=4).tolist()
        self.assertEqual(first, expected)

    def test_formal_campplus_encode_failure_is_not_silently_replaced(self):
        class BrokenCampplus:
            mode = "campplus"
            emb_dim = 8
            _backend = object()

            def encode_file(self, speaker_id, wav_path):
                raise RuntimeError("synthetic encoder failure")

        with tempfile.TemporaryDirectory() as td:
            manifest, cfg = _write_dataset_fixture(Path(td), target_present=True)
            dataset = B1Dataset(manifest, cfg, seed=1, adapter=BrokenCampplus())
            with self.assertRaisesRegex(RuntimeError, "CAMPLUS encode 失败"):
                dataset[0]

    def test_bootstrap_embedding_is_stable_across_python_processes(self):
        script = (
            "import json,sys;"
            f"sys.path.insert(0,{str(P2_ROOT)!r});"
            "from src.tse.enrollment_adapter import EnrollmentAdapter;"
            "a=EnrollmentAdapter(emb_dim=8,seed=123);"
            "print(json.dumps(a.get_embedding('S004').tolist()))"
        )
        first = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        second = subprocess.check_output([sys.executable, "-c", script], text=True).strip()
        self.assertEqual(first, second)

    def test_wrong_enrollment_requires_explicit_or_triplet_pair(self):
        first = {"triplet_id": "T1", "target_present": True, "target_speaker": "A", "enroll_wav": "a.wav"}
        second = {"triplet_id": "T1", "target_present": True, "target_speaker": "B", "enroll_wav": "b.wav"}
        self.assertEqual(resolve_wrong_enrollment(first, {"T1": [first, second]}), ("B", "b.wav"))
        self.assertEqual(resolve_wrong_enrollment(first, {}), (None, None))


if __name__ == "__main__":
    unittest.main()
