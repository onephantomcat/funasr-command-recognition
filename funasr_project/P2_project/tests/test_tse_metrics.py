# -*- coding: utf-8 -*-
"""P2-11 评测指标单测（手册 P2-11 必须测试 10 项）。v2 分支版。"""

import sys
from pathlib import Path

import pytest
import torch

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
sys.path.insert(0, str(P2_ROOT))

from src.tse.metrics import (
    activity_prf,
    activity_ratio,
    clipped_ratio,
    corpus_sisdr,
    energy_ratio,
    mrstft_eval,
    si_sdr_components,
    si_sdr_eval,
    si_sdri,
    utterance_sisdr,
    waveform_l1,
)

torch.manual_seed(0)
N = 16000


def _sig(freq=440.0):
    t = torch.arange(N).float() / 16000
    return torch.sin(2 * torch.pi * freq * t)


def test_perfect_estimate():
    y = _sig()
    est = y.clone()
    assert si_sdr_eval(est, y) > 60.0
    assert waveform_l1(est, y) < 1e-7
    mix = y + 0.1 * _sig(880.0)
    assert energy_ratio(est, mix) == pytest.approx(1.0 / 1.01, rel=1e-3)
    assert si_sdri(est, y, mix) > 20.0


def test_estimate_equals_mixture():
    y = _sig()
    mix = y + 0.2 * _sig(880.0)
    assert si_sdri(mix.clone(), y, mix) == pytest.approx(0.0, abs=1e-5)


def test_zero_estimate_present():
    y = _sig()
    est = torch.zeros(N)
    assert si_sdr_eval(est, y) < -60.0
    mix = y + 0.1 * _sig(880.0)
    assert energy_ratio(est, mix) == pytest.approx(0.0)


def test_absent_zero_output():
    mix = _sig(550.0)
    est = torch.zeros(N)
    assert energy_ratio(est, mix) == pytest.approx(0.0)
    p = torch.zeros(125)
    assert activity_ratio(p) == pytest.approx(0.0)


def test_absent_nonzero_output():
    mix = _sig(550.0)
    est = 0.1 * _sig(330.0)
    assert energy_ratio(est, mix) == pytest.approx(0.01, rel=1e-2)
    p = torch.zeros(125)
    p[:10] = 0.9
    assert activity_ratio(p) == pytest.approx(10 / 125, rel=1e-5)


def test_length_mismatch_fails():
    with pytest.raises(ValueError):
        si_sdr_eval(torch.zeros(N), torch.zeros(N + 1))
    with pytest.raises(ValueError):
        waveform_l1(torch.zeros(100), torch.zeros(200))


def test_valid_mask():
    y = _sig()
    est = y + 0.01 * torch.randn(N)
    valid = torch.zeros(N, dtype=torch.bool)
    valid[:8000] = True
    masked = si_sdr_eval(est, y, valid)
    manual = si_sdr_eval(est[:8000], y[:8000])
    assert masked == pytest.approx(manual, abs=1e-6)
    with pytest.raises(ValueError):
        si_sdr_eval(est, y, torch.zeros(10, dtype=torch.bool))


def test_nan_inf_fails():
    bad = torch.zeros(N)
    bad[0] = float("nan")
    with pytest.raises(ValueError):
        si_sdr_eval(bad, torch.zeros(N))
    bad[0] = float("inf")
    with pytest.raises(ValueError):
        energy_ratio(bad, torch.ones(N))
    with pytest.raises(ValueError):
        activity_prf(torch.full((10,), float("nan")), torch.zeros(10))


def test_corpus_vs_utterance_aggregation():
    g = torch.Generator().manual_seed(1)
    ya = _sig()
    est_a = ya + 0.001 * torch.randn(N, generator=g)
    nb = 4000
    yb = _sig()[:nb]
    est_b = yb + 0.3 * torch.randn(nb, generator=g)
    comps = [si_sdr_components(est_a, ya), si_sdr_components(est_b, yb)]
    c = corpus_sisdr(comps)
    u = utterance_sisdr(comps)
    assert c != pytest.approx(u, abs=1e-3)
    assert c < u
    import math
    sig = sum(x[0] for x in comps)
    err = sum(x[1] for x in comps)
    assert c == pytest.approx(10.0 * math.log10(sig / (err + 1e-8) + 1e-8), rel=1e-9)
    assert u == pytest.approx((comps[0][2] + comps[1][2]) / 2, rel=1e-9)


def test_shuffle_pairing_invariant():
    y1, y2 = _sig(440.0), _sig(660.0)
    est1 = y1 + 0.01 * torch.randn(N)
    est2 = y2 + 0.05 * torch.randn(N)

    def evaluate(pairs):
        return {sid: si_sdr_eval(est, ref) for sid, est, ref in pairs}

    forward = evaluate([("s1", est1, y1), ("s2", est2, y2)])
    shuffled = evaluate([("s2", est2, y2), ("s1", est1, y1)])
    assert forward == shuffled


def test_activity_prf_basic():
    act = torch.zeros(100)
    act[:50] = 1.0
    p = torch.zeros(100)
    p[:40] = 0.9
    p[50:60] = 0.9
    r = activity_prf(p, act)
    assert r["precision"] == pytest.approx(40 / 50, rel=1e-4)
    assert r["recall"] == pytest.approx(40 / 50, rel=1e-4)


def test_mrstft_and_clip():
    y = _sig()
    assert mrstft_eval(y.clone(), y) < 1e-3
    clipped = torch.ones(1000) * 0.999
    assert clipped_ratio(clipped) == pytest.approx(1.0)
    assert clipped_ratio(0.5 * _sig()) == pytest.approx(0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))