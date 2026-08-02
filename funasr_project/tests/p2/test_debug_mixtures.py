# -*- coding: utf-8 -*-
"""P2-07 DEBUG_ONLY 小语音集验收测试。

运行：python -m pytest tests/p2/test_debug_mixtures.py -v
前置：python p2_tools/build_debug_mixtures.py --seed 20260725
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tse import DualOutputTSE  # noqa: E402

OUT = ROOT / "artifacts" / "p2" / "debug_mixtures_v0"
MANIFEST = OUT / "manifest.jsonl"
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


def test_required_groups(entries):
    full100 = [e for e in entries if e["overlap_ratio"] == 1.0]
    swap = [e for e in entries if e["enrollment_speaker"] != e["target_speaker"]]
    assert len(full100) >= 4, f"100% 重叠不足 4 条: {len(full100)}"
    assert len(swap) >= 4, f"enrollment-swap 不足 4 组: {len(swap)}"
    for e in entries:
        assert e["interferer_speaker"] != e["target_speaker"]
        assert Path(ROOT / e["enrollment"]).exists()


def test_files_exist_and_sha256(entries):
    for e in entries:
        for key in ("mixture", "target", "interferer", "activity"):
            assert (ROOT / e[key]).exists(), f"{e['id']} 缺 {key}"
        assert _sha256(ROOT / e["mixture"]) == e["sha256_mixture"]
    assert (OUT / "SHA256SUMS.txt").exists()
    assert (OUT / "_meta.json").exists()


def test_audio_shapes_and_closure(entries):
    for e in entries:
        mix, sr1 = sf.read(str(ROOT / e["mixture"]), dtype="float32")
        tgt, sr2 = sf.read(str(ROOT / e["target"]), dtype="float32")
        itf, sr3 = sf.read(str(ROOT / e["interferer"]), dtype="float32")
        assert sr1 == sr2 == sr3 == 16000
        assert len(mix) == len(tgt) == len(itf) == N
        # 混合闭合：mixture 必须等于 target + interferer（float wav，容差 1e-6）
        err = np.max(np.abs(mix - (tgt + itf)))
        assert err < 1e-6, f"{e['id']} 闭合误差 {err}"
        assert np.max(np.abs(mix)) <= 0.99  # 防削波检查
        assert np.sqrt(np.mean(tgt ** 2)) > 1e-3  # target 非静音


def test_activity_mask(entries):
    for e in entries:
        act = np.load(str(ROOT / e["activity"]))
        assert act.shape == (N,)
        assert set(np.unique(act)).issubset({0.0, 1.0})
        assert act.sum() > N * 0.3  # target 应大部分活动（4s 语音窗）
        tgt, _ = sf.read(str(ROOT / e["target"]), dtype="float32")
        # 掩码与 target 能量一致：活动区外的 target 必须接近零
        outside = np.abs(tgt[act < 0.5])
        assert outside.max() < 1e-3 if len(outside) else True


def test_mixture_feeds_model(entries):
    """P2-07 数据与 P2-06 模型端到端：真实 mixture 可前向。"""
    cfg = yaml.safe_load(open(ROOT / "configs" / "p2" / "tse_smoke.yaml", encoding="utf-8"))
    torch.manual_seed(20260725)
    model = DualOutputTSE(cfg).eval()
    e = entries[0]
    mix, _ = sf.read(str(ROOT / e["mixture"]), dtype="float32")
    x = torch.from_numpy(mix).unsqueeze(0)
    emb = torch.randn(1, int(cfg["emb_dim"]), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        s, r, _ = model(x, emb)
    assert s.shape == x.shape and r.shape == x.shape
    assert torch.isfinite(s).all() and torch.isfinite(r).all()
    assert (s + r - x).abs().max().item() < 1e-5
