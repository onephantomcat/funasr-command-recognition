# -*- coding: utf-8 -*-
"""P2-07 DEBUG_ONLY 小语音集验收测试。v2 分支版。

运行：python -m pytest tests/test_debug_mixtures.py -v
前置：python tools/build_debug_mixtures.py --seed 20260725
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from src.tse import DualOutputTSE

_NEW_OUT = P2_ROOT / "artifacts" / "debug_mixtures_v0"
_OLD_OUT = FUNASR_ROOT / "artifacts" / "p2" / "debug_mixtures_v0"
OUT = _NEW_OUT if _NEW_OUT.exists() else _OLD_OUT
MANIFEST = OUT / "manifest.jsonl"
BASE = FUNASR_ROOT
N = 64000


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@pytest.fixture(scope="module")
def entries():
    assert MANIFEST.exists(), f"manifest 不存在，先运行 build_debug_mixtures.py"
    with open(MANIFEST, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_manifest_count_and_flags(entries):
    assert len(entries) == 12
    assert all(e["debug_only"] for e in entries)
    assert sum(e["target_present"] for e in entries) == 8
    assert sum(not e["target_present"] for e in entries) == 4
    assert {e["scenario"] for e in entries} == {
        "enroll_swap_target_1", "enroll_swap_target_2", "enroll_swap_absent"
    }


def test_required_groups(entries):
    full100 = [e for e in entries if e["overlap_ratio"] == 1.0]
    swap = [e for e in entries if e["scenario"] == "enroll_swap_absent"]
    assert len(full100) >= 4, f"100% 重叠不足 4 条: {len(full100)}"
    assert len(swap) >= 4, f"enrollment-swap 不足 4 组: {len(swap)}"
    for e in entries:
        assert e["interferer_speaker"] != e["target_speaker"]
        assert Path(BASE / e["enrollment"]).exists()


def test_files_exist_and_sha256(entries):
    for e in entries:
        for key in ("mixture", "target", "interferer", "activity"):
            assert (BASE / e[key]).exists(), f"{e['id']} 缺 {key}"
        assert _sha256(BASE / e["mixture"]) == e["sha256_mixture"]
    assert (OUT / "SHA256SUMS.txt").exists()
    assert (OUT / "_meta.json").exists()


def test_audio_shapes_and_closure(entries):
    for e in entries:
        mix, sr1 = sf.read(str(BASE / e["mixture"]), dtype="float32")
        tgt, sr2 = sf.read(str(BASE / e["target"]), dtype="float32")
        itf, sr3 = sf.read(str(BASE / e["interferer"]), dtype="float32")
        assert sr1 == sr2 == sr3 == 16000
        assert len(mix) == len(tgt) == len(itf) == N
        err = np.max(np.abs(mix - (tgt + itf)))
        assert err < 1e-6, f"{e['id']} 闭合误差 {err}"
        assert np.max(np.abs(mix)) <= 0.99
        if e["target_present"]:
            assert np.sqrt(np.mean(tgt ** 2)) > 1e-3
        else:
            assert np.max(np.abs(tgt)) == 0.0


def test_activity_mask(entries):
    for e in entries:
        act = np.load(str(BASE / e["activity"]))
        assert act.shape == (N,)
        assert set(np.unique(act)).issubset({0.0, 1.0})
        if e["target_present"]:
            assert act.sum() > N * 0.3
        else:
            assert act.sum() == 0.0
        tgt, _ = sf.read(str(BASE / e["target"]), dtype="float32")
        outside = np.abs(tgt[act < 0.5])
        assert outside.max() < 1e-3 if len(outside) else True


def test_mixture_feeds_model(entries):
    """P2-07 数据与 P2-06 模型端到端：真实 mixture 可前向。"""
    cfg = yaml.safe_load(open(P2_ROOT / "configs" / "tse_smoke.yaml", encoding="utf-8"))
    torch.manual_seed(20260725)
    model = DualOutputTSE(cfg).eval()
    e = entries[0]
    mix, _ = sf.read(str(BASE / e["mixture"]), dtype="float32")
    x = torch.from_numpy(mix).unsqueeze(0)
    emb = torch.randn(1, int(cfg["emb_dim"]), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        s, r, _ = model(x, emb)
    assert s.shape == x.shape and r.shape == x.shape
    assert torch.isfinite(s).all() and torch.isfinite(r).all()
    assert (s + r - x).abs().max().item() < 1e-5
