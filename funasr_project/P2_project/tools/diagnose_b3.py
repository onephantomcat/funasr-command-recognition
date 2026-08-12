# -*- coding: utf-8 -*-
"""P3 B3 诊断脚本：验证 CAMPPlus 声纹嵌入、能量统计和域差异。

用途：在 P3 集成环境下运行，快速定位 B3 失败原因：
  1. CAMPPlus 后端是否正确加载（权重、speakerlab 依赖）
  2. 声纹嵌入是否有效（非零、区分度足够）
  3. P3 数据能量分布是否正常（非近静音、无异常截断）
  4. 与 P2 外部集的域差异量化（可选）

运行：
  python tools/diagnose_b3.py \
      --config P2_DELIVERY_FOR_P3_20260811/B3/config.yaml \
      --manifest /path/to/datasetA_manifest.jsonl \
      --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# v2.3：灵活路径探测——自动在「P2_project/tools/*.py」「交付包 tools/*.py」两种目录结构下定位 src/
_HERE = Path(__file__).resolve()
_P2_ROOT = None
for _parent in [_HERE.parents[1], _HERE.parents[0], *_HERE.parents[2:]]:
    if (_parent / "src" / "tse" / "enrollment_adapter.py").exists():
        _P2_ROOT = _parent
        break
if _P2_ROOT is None:
    _P2_ROOT = _HERE.parents[1]  # 退回到旧逻辑
P2_ROOT = _P2_ROOT
FUNASR_ROOT = P2_ROOT.parent
sys.path.insert(0, str(P2_ROOT))

from src.tse.enrollment_adapter import EnrollmentAdapter

LOG = logging.getLogger("b3_diagnose")


def _energy_ratio(wav: torch.Tensor, mix: torch.Tensor) -> float:
    """计算能量比（RMS）。"""
    rms_wav = torch.sqrt(torch.mean(wav ** 2) + 1e-12)
    rms_mix = torch.sqrt(torch.mean(mix ** 2) + 1e-12)
    return (rms_wav / (rms_mix + 1e-12)).item()


def _load_wav(path: str, base=None) -> torch.Tensor:
    wav, sr = sf.read(str((base or FUNASR_ROOT) / path), dtype="float32")
    return torch.from_numpy(wav), sr


def diagnose_campplus(cfg: dict, device: torch.device) -> dict:
    """验证 CAMPPlus 声纹嵌入后端。"""
    LOG.info("=== 诊断 1：CAMPPlus 后端 ===")
    result = {"campplus_load": False, "embedding_valid": False, "error": None}

    if cfg.get("sv_mode") != "campplus":
        LOG.warning("sv_mode != campplus，跳过声纹诊断")
        result["note"] = "sv_mode 非 campplus"
        return result

    adapter = EnrollmentAdapter.from_config(cfg)
    LOG.info("adapter mode=%s, model_id=%s", adapter.mode, adapter._model_id)

    # 强制加载后端
    try:
        adapter.load_backend()
        result["campplus_load"] = True
        LOG.info("CAMPPlus 后端加载成功")
    except Exception as e:
        result["error"] = f"load_backend 失败: {e}"
        LOG.error(result["error"])
        return result

    # 用随机噪声测试嵌入是否非零
    try:
        test_wav = torch.randn(1, 16000).to(device)
        emb = adapter._backend.embed(test_wav)
        if emb.shape[-1] == cfg["emb_dim"] and not torch.allclose(emb, torch.zeros_like(emb)):
            result["embedding_valid"] = True
            LOG.info("CAMPPlus 嵌入有效 (shape=%s, mean=%.6f)", emb.shape, emb.mean().item())
        else:
            result["error"] = f"嵌入无效: shape={emb.shape}, allclose_zero={torch.allclose(emb, torch.zeros_like(emb))}"
            LOG.error(result["error"])
    except Exception as e:
        result["error"] = f"嵌入测试失败: {e}"
        LOG.error(result["error"])

    return result


def diagnose_data_energy(manifest_path: str, base: Path, max_samples: int = 50) -> dict:
    """分析数据能量分布，判断是否存在近静音或异常。"""
    LOG.info("=== 诊断 2：数据能量分布 ===")
    result = {"n_samples": 0, "near_silent_ratio": 0.0, "mean_energy_ratio": 0.0, "error": None}

    manifest = Path(manifest_path)
    if not manifest.exists():
        result["error"] = f"manifest 不存在: {manifest}"
        LOG.error(result["error"])
        return result

    entries = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    entries = entries[:max_samples]
    energies = []
    near_silent_count = 0

    for e in entries:
        try:
            mix, _ = _load_wav(e["mixture"], base)
            target_present = bool(e.get("target_present", True))
            if target_present and "target" in e:
                target, _ = _load_wav(e["target"], base)
                ratio = _energy_ratio(target, mix)
                energies.append(ratio)
                if ratio < 0.01:  # 能量比 < 0.01 判定为近静音
                    near_silent_count += 1
            else:
                # absent 样本，只看 mix 能量
                rms_mix = torch.sqrt(torch.mean(mix ** 2) + 1e-12).item()
                energies.append(rms_mix)
                if rms_mix < 0.001:
                    near_silent_count += 1
        except Exception as ex:
            LOG.warning("样本 %s 加载失败: %s", e.get("id", "?"), ex)

    if energies:
        result["n_samples"] = len(energies)
        result["near_silent_ratio"] = near_silent_count / len(energies)
        result["mean_energy_ratio"] = float(np.mean(energies))
        result["min_energy_ratio"] = float(np.min(energies))
        result["max_energy_ratio"] = float(np.max(energies))
        result["std_energy_ratio"] = float(np.std(energies))
        LOG.info("样本数=%d, 近静音比=%.4f, 能量均值=%.4f, std=%.4f",
                 result["n_samples"], result["near_silent_ratio"],
                 result["mean_energy_ratio"], result["std_energy_ratio"])
    else:
        result["error"] = "无有效样本"
        LOG.error(result["error"])

    return result


def diagnose_embedding_distinction(cfg: dict, manifest_path: str, base: Path, max_speakers: int = 5) -> dict:
    """检查不同说话人嵌入的区分度（Cosine 相似度）。"""
    LOG.info("=== 诊断 3：声纹区分度 ===")
    result = {"distinction_ok": False, "error": None, "mean_cosine_same": 0.0, "mean_cosine_diff": 0.0}

    if cfg.get("sv_mode") != "campplus":
        result["note"] = "sv_mode 非 campplus"
        LOG.warning(result["note"])
        return result

    adapter = EnrollmentAdapter.from_config(cfg)
    try:
        adapter.load_backend()
    except Exception as e:
        result["error"] = f"后端加载失败: {e}"
        LOG.error(result["error"])
        return result

    manifest = Path(manifest_path)
    if not manifest.exists():
        result["error"] = f"manifest 不存在: {manifest}"
        LOG.error(result["error"])
        return result

    # 收集不同说话人的 enrollment 路径
    speaker_enrolls = {}
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            spk = e.get("target_speaker", e.get("enrollment", ""))
            if spk and "enrollment" in e and spk not in speaker_enrolls:
                speaker_enrolls[spk] = str(base / e["enrollment"])
            if len(speaker_enrolls) >= max_speakers * 3:  # 多采样一些
                break

    # 提取嵌入
    embs = {}
    for spk, path in list(speaker_enrolls.items())[:max_speakers]:
        try:
            emb = adapter.encode_file(spk, path)
            embs[spk] = emb.squeeze(0)
            LOG.info("说话人 %s 嵌入已提取 (norm=%.4f)", spk, emb.norm().item())
        except Exception as ex:
            LOG.warning("说话人 %s 嵌入失败: %s", spk, ex)

    if len(embs) < 2:
        result["error"] = f"有效说话人不足: {len(embs)}"
        LOG.error(result["error"])
        return result

    spk_list = list(embs.keys())
    # 计算两两 cosine 相似度
    cos_matrix = torch.zeros(len(spk_list), len(spk_list))
    for i, s1 in enumerate(spk_list):
        for j, s2 in enumerate(spk_list):
            cos_matrix[i, j] = torch.cosine_similarity(embs[s1].unsqueeze(0), embs[s2].unsqueeze(0))

    # 同说话人（对角线）和 不同说话人（非对角线）
    diag_mean = cos_matrix.diag().mean().item()
    off_diag = cos_matrix.fill_diagonal_(0).sum() / (len(spk_list) * (len(spk_list) - 1) + 1e-12)
    off_diag_mean = off_diag.item()

    result["mean_cosine_same"] = diag_mean
    result["mean_cosine_diff"] = off_diag_mean
    result["distinction_ok"] = (diag_mean > 0.99) and (off_diag_mean < 0.8)

    LOG.info("同说话人 cosine 均值=%.4f, 异说话人 cosine 均值=%.4f, 区分度OK=%s",
             diag_mean, off_diag_mean, result["distinction_ok"])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="B3 config.yaml 路径")
    ap.add_argument("--manifest", required=True, help="DatasetA manifest 路径")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_samples", type=int, default=50, help="诊断样本上限")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        LOG.error("配置文件不存在: %s", cfg_path)
        sys.exit(1)

    import yaml
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cuda" if args.device == "cuda" else "cpu"))
    LOG.info("device=%s", device)

    base = FUNASR_ROOT

    summary = {
        "config": str(cfg_path),
        "manifest": args.manifest,
        "device": str(device),
        "campplus": diagnose_campplus(cfg, device),
        "data_energy": diagnose_data_energy(args.manifest, base, args.max_samples),
        "embedding_distinction": diagnose_embedding_distinction(cfg, args.manifest, base),
    }

    # 综合判定
    pass_checks = []
    fail_checks = []

    if summary["campplus"]["campplus_load"] and summary["campplus"]["embedding_valid"]:
        pass_checks.append("CAMPPlus 加载与嵌入 ✓")
    else:
        fail_checks.append(f"CAMPPlus 失败: {summary['campplus'].get('error', '未知原因')}")

    energy = summary["data_energy"]
    if energy.get("error") is None and energy.get("near_silent_ratio", 1.0) < 0.3:
        ratio_disp = f"{energy.get('near_silent_ratio', '?'):.2f}"
        pass_checks.append(f"数据能量分布正常 (近静音比={ratio_disp}) ✓")
    else:
        err = energy.get("error")
        if err is None:
            ratio_disp = f"{energy.get('near_silent_ratio', '?'):.2f}"
            err = f"近静音比={ratio_disp}"
        fail_checks.append(f"数据能量异常: {err}")

    emb = summary["embedding_distinction"]
    if emb.get("distinction_ok", False):
        pass_checks.append("声纹区分度 ✓")
    elif emb.get("note"):
        pass_checks.append(f"声纹区分度: {emb['note']}（跳过）")
    else:
        fail_checks.append(f"声纹区分度失败: {emb.get('error', '未通过')}")

    summary["verdict"] = {
        "pass": pass_checks,
        "fail": fail_checks,
        "overall_pass": len(fail_checks) == 0,
    }

    LOG.info("========== 诊断结论 ==========")
    for p in pass_checks:
        LOG.info("  ✓ %s", p)
    for f in fail_checks:
        LOG.error("  ✗ %s", f)

    out_path = Path(args.manifest).parent / "b3_diagnosis.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOG.info("诊断报告: %s", out_path)

    return 0 if summary["verdict"]["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
