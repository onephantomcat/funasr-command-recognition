# -*- coding: utf-8 -*-
"""P2-11 TSE 评测器：同一批样本 + 同一套逐条输出，供模型间公平比较。

输入：manifest.jsonl（P2-07 格式）+ checkpoint（P2-09 格式，内嵌 cfg）。
输出（--out 目录）：
  predictions.jsonl          逐条预测记录（schemas/tse_prediction.schema.json 12 字段）
  predictions_detailed.jsonl 逐条全量指标（wav_l1/mrstft/activity PRF/RTF/显存/注册交换）
  summary.json               语料级 + 逐句级聚合、分场景、注册交换选择正确率、RTF/显存
  report.md                  人读报告
  audio/<id>__est_target.wav 逐条估计音频（供确定性复评）

冻结口径：
- 指标算法全部来自 src/tse/metrics.py（SI-SDR 去均值+投影+eps=1e-8；tau_debug=0.5）；
- 注册交换选择：q_e1_y1 vs q_e2_y1，|Δ|<=TIE_EPS(1e-6) 判平计 0.5；
- 确定性复评：从磁盘 est 音频重算指标字段（latency_ms 取自原记录，属测量值非评分），
  全量 JSONL 逐字节一致才判 PASS（手册 P2-11 PASS 条件）。

ABSENT：manifest 条目带 "target_present": false 时禁止 SI-SDR（置 null），
记录 rho_abs(=energy_ratio) 与 A_abs(=activity_ratio)。DEBUG 集暂无 ABSENT 样本。
NaN/Inf 输出：记录 nan=true，派生指标置 null 并跳过该条 schema 校验（不伪造假数）。

标记：DEBUG_ONLY / BOOTSTRAP_ENCODER_ONLY（embedding 来源与训练一致的哈希派生随机向量）

运行：
  .\\.venv-p2tse\\Scripts\\python.exe p2_tools/evaluate_tse.py \
      --checkpoint artifacts/experiments/DEBUG_OVERFIT_seed20260723/checkpoint_step100.pt \
      --device auto
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tse.model import DualOutputTSE  # noqa: E402
from src.tse.metrics import (  # noqa: E402
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
from train_overfit_debug import bootstrap_embedding, frame_activity, sha256_file  # noqa: E402

LOG = logging.getLogger("p2_eval")

TIE_EPS = 1e-6  # 注册交换选择平局阈值（冻结）
SCHEMA_PATH = ROOT / "schemas" / "tse_prediction.schema.json"


def scenario_of(entry):
    """手册风格场景串：overlap_100_sir_minus5 / overlap_25_sir_5。"""
    ov = int(round(float(entry["overlap_ratio"]) * 100))
    sir = float(entry["sir_db"])
    sir_s = f"minus{abs(sir):g}" if sir < 0 else f"{sir:g}"
    return f"overlap_{ov}_sir_{sir_s}"


def _load_wav(rel_path):
    wav, sr = sf.read(str(ROOT / rel_path), dtype="float32")
    return torch.from_numpy(wav), sr


@torch.no_grad()
def _forward_timed(model, mix, emb, device):
    """单条前向 + 毫秒计时（CUDA 同步保证口径）。"""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    s_tgt, s_res, p_tgt = model(mix.unsqueeze(0).to(device), emb.unsqueeze(0).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return s_tgt[0].cpu(), s_res[0].cpu(), p_tgt[0].cpu(), latency_ms


def score_present(est, ref, mix, p_tgt, frame_act):
    """PRESENT 评分纯函数（determinism 复评与首评共用同一入口）。"""
    prf = activity_prf(p_tgt, frame_act)
    return {
        "sisdr": si_sdr_eval(est, ref),
        "sisdri": si_sdri(est, ref, mix),
        "wav_l1": waveform_l1(est, ref),
        "mrstft": mrstft_eval(est, ref),
        "act_precision": prf["precision"],
        "act_recall": prf["recall"],
        "act_f1": prf["f1"],
        "energy_ratio": energy_ratio(est, mix),
        "activity_mean": p_tgt.mean().item(),
        "clipped": clipped_ratio(est) > 0.0,
        "nan": False,
    }


def score_absent(est, mix, p_tgt):
    """ABSENT 评分：禁止 SI-SDR；rho_abs + A_abs + 活动均值。"""
    return {
        "sisdr": None,
        "sisdri": None,
        "energy_ratio": energy_ratio(est, mix),  # 即 rho_abs
        "a_abs": activity_ratio(p_tgt),
        "activity_mean": p_tgt.mean().item(),
        "clipped": clipped_ratio(est) > 0.0,
        "nan": False,
    }


def nan_record_present():
    return {"sisdr": None, "sisdri": None, "wav_l1": None, "mrstft": None,
            "act_precision": None, "act_recall": None, "act_f1": None,
            "energy_ratio": None, "activity_mean": None, "clipped": False, "nan": True}


def schema_record(sample_id, scenario, target_present, ckpt_sha, data_sha, scored, latency_ms):
    """裁剪为 schema 12 字段（nan=true 时 energy_ratio/activity_mean 允许 None 并跳过校验）。"""
    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "target_present": target_present,
        "checkpoint_sha256": ckpt_sha,
        "data_sha256": data_sha,
        "sisdr": scored["sisdr"],
        "sisdri": scored["sisdri"],
        "energy_ratio": scored["energy_ratio"],
        "activity_mean": scored["activity_mean"],
        "nan": scored["nan"],
        "clipped": scored["clipped"],
        "latency_ms": latency_ms,
    }


def validate_schema(records):
    """jsonschema 可用则逐条校验；nan=true 记录跳过（见模块 docstring）。"""
    try:
        import jsonschema
    except ImportError:
        LOG.warning("jsonschema 未安装，跳过 schema 校验（pip install jsonschema 后自动启用）")
        return "SKIPPED_NO_LIB"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    skipped = 0
    for r in records:
        if r["nan"]:
            skipped += 1
            continue
        jsonschema.validate(r, schema)
    return f"PASS({len(records) - skipped} validated, {skipped} nan-skipped)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "artifacts" / "p2" / "debug_mixtures_v0" / "manifest.jsonl"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    ckpt_sha = sha256_file(ckpt_path)
    data_sha = sha256_file(args.manifest)
    LOG.info("checkpoint=%s sha256=%s", ckpt_path.name, ckpt_sha[:16])

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cuda" if args.device == "cuda" else "cpu"))
    model = DualOutputTSE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    LOG.info("device=%s 模型已加载（step=%s）", device, ckpt.get("step"))

    out_dir = Path(args.out) if args.out else ROOT / "artifacts" / "p2" / f"eval_{ckpt_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)

    entries = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    entries = sorted(entries, key=lambda e: e["id"])  # 按 id 排序，与训练取数口径一致

    records, detailed, comps = [], [], []
    swap_rows = []
    latencies, rtf_list = [], []
    peak_gpu_gb = 0.0

    for i, e in enumerate(entries):
        sid = e["id"]
        target_present = bool(e.get("target_present", True))
        mix, sr = _load_wav(e["mixture"])
        assert sr == cfg["sample_rate"]
        emb = bootstrap_embedding(e["enrollment"], int(cfg["emb_dim"]))

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        s_tgt, _s_res, p_tgt, latency_ms = _forward_timed(model, mix, emb, device)
        if device.type == "cuda":
            peak_gpu_gb = max(peak_gpu_gb, torch.cuda.max_memory_allocated() / 1024 ** 3)
        if i == 0:  # 首条含 warmup，重测一次
            s_tgt, _s_res, p_tgt, latency_ms = _forward_timed(model, mix, emb, device)

        # subtype=FLOAT：float32 WAV 往返位精确（默认 PCM_16 量化会破坏确定性复评）
        sf.write(str(out_dir / "audio" / f"{sid}__est_target.wav"),
                 s_tgt.numpy().astype("float32"), sr, subtype="FLOAT")
        np.save(str(out_dir / "audio" / f"{sid}__p_tgt.npy"),
                p_tgt.numpy().astype("float32"))  # 复评从磁盘取，不再跑模型

        duration = mix.numel() / sr
        latencies.append(latency_ms)
        rtf_list.append(latency_ms / 1000.0 / duration)
        scenario = scenario_of(e)

        nan_out = not bool(torch.isfinite(s_tgt).all())
        if nan_out:
            LOG.warning("%s 输出含 NaN/Inf，派生指标置 null", sid)
            scored = nan_record_present()
            det = {"sample_id": sid, "scenario": scenario, "target_present": target_present,
                   **scored, "metrics_skipped": True}
        elif target_present:
            ref, _ = _load_wav(e["target"])
            act = np.load(str(ROOT / e["activity"]))
            fa = frame_activity(act, cfg["win_length"], cfg["hop_length"], float(cfg["act_frame_ratio"]))
            scored = score_present(s_tgt, ref, mix, p_tgt, fa)
            comps.append(si_sdr_components(s_tgt, ref))

            # ---- 注册交换（手册：e1/e2 对 y1/y2 四象限 + 选择正确率）----
            itr, _ = _load_wav(e["interferer"])
            wrong_enroll = f"data/trials/enroll_{e['interferer_speaker']}.wav"
            emb_w = bootstrap_embedding(wrong_enroll, int(cfg["emb_dim"]))
            s_w, _, _, _ = _forward_timed(model, mix, emb_w, device)
            q_e1_y1 = si_sdr_eval(s_tgt, ref)
            q_e1_y2 = si_sdr_eval(s_tgt, itr)
            q_e2_y2 = si_sdr_eval(s_w, itr)
            q_e2_y1 = si_sdr_eval(s_w, ref)
            delta = q_e1_y1 - q_e2_y1
            choice = 0.5 if abs(delta) <= TIE_EPS else (1.0 if delta > 0 else 0.0)
            swap_rows.append({"sample_id": sid, "q_e1_y1": q_e1_y1, "q_e1_y2": q_e1_y2,
                              "q_e2_y2": q_e2_y2, "q_e2_y1": q_e2_y1,
                              "selectivity_db": q_e1_y1 - q_e1_y2, "choice_score": choice})
            det = {"sample_id": sid, "scenario": scenario, "target_present": True,
                   "config": e["config"], **scored,
                   "activity_ratio": activity_ratio(p_tgt),
                   "rtf": latency_ms / 1000.0 / duration,
                   "swap": swap_rows[-1]}
        else:  # ABSENT
            scored = score_absent(s_tgt, mix, p_tgt)
            det = {"sample_id": sid, "scenario": scenario, "target_present": False,
                   "config": e["config"], **scored,
                   "rtf": latency_ms / 1000.0 / duration}

        records.append(schema_record(sid, scenario, target_present, ckpt_sha, data_sha,
                                     scored, latency_ms))
        detailed.append(det)
        LOG.info("%s [%s] sisdr=%s energy_ratio=%s lat=%.1fms",
                 sid, scenario,
                 f"{scored['sisdr']:.2f}" if scored["sisdr"] is not None else "null",
                 f"{scored['energy_ratio']:.4f}" if scored["energy_ratio"] is not None else "null",
                 latency_ms)

    # ---- 写逐条记录 ----
    def _dump_jsonl(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pred_path = out_dir / "predictions.jsonl"
    _dump_jsonl(pred_path, records)
    _dump_jsonl(out_dir / "predictions_detailed.jsonl", detailed)

    schema_status = validate_schema(records)
    LOG.info("schema 校验: %s", schema_status)

    # ---- 确定性复评：从磁盘 est 音频重算指标字段，逐字节比对 ----
    verify = []
    for e, rec in zip(entries, records):
        est, _ = sf.read(str(out_dir / "audio" / f"{e['id']}__est_target.wav"), dtype="float32")
        est = torch.from_numpy(est)
        p_tgt = torch.from_numpy(np.load(str(out_dir / "audio" / f"{e['id']}__p_tgt.npy")))
        mix, _ = _load_wav(e["mixture"])
        target_present = bool(e.get("target_present", True))
        if rec["nan"]:
            scored = nan_record_present()
        elif target_present:
            ref, _ = _load_wav(e["target"])
            act = np.load(str(ROOT / e["activity"]))
            fa = frame_activity(act, cfg["win_length"], cfg["hop_length"], float(cfg["act_frame_ratio"]))
            scored = score_present(est, ref, mix, p_tgt, fa)
        else:
            scored = score_absent(est, mix, p_tgt)
        verify.append(schema_record(e["id"], rec["scenario"], target_present,
                                    ckpt_sha, data_sha, scored, rec["latency_ms"]))
    det_pass = all(json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
                   for a, b in zip(records, verify))
    LOG.info("确定性复评: %s", "PASS（逐字节一致）" if det_pass else "FAIL")
    verify_path = out_dir / "predictions.verify.jsonl"
    if not det_pass:
        _dump_jsonl(verify_path, verify)
    elif verify_path.exists():
        verify_path.unlink()  # 清除上一轮 FAIL 的残留，避免误读

    # ---- 聚合 ----
    present_dets = [d for d in detailed if d["target_present"] and not d.get("metrics_skipped")]
    summary = {
        "checkpoint": str(ckpt_path), "checkpoint_sha256": ckpt_sha,
        "manifest": args.manifest, "data_sha256": data_sha,
        "device": str(device), "n_samples": len(records),
        "n_present": len(present_dets),
        "n_absent": sum(1 for d in detailed if not d["target_present"]),
        "n_nan": sum(1 for r in records if r["nan"]),
        "present": {
            "corpus_sisdr_db": corpus_sisdr(comps) if comps else None,
            "utterance_sisdr_db": utterance_sisdr(comps) if comps else None,
            "mean_sisdri_db": float(np.mean([d["sisdri"] for d in present_dets])) if present_dets else None,
            "mean_wav_l1": float(np.mean([d["wav_l1"] for d in present_dets])) if present_dets else None,
            "mean_mrstft": float(np.mean([d["mrstft"] for d in present_dets])) if present_dets else None,
            "mean_act_f1": float(np.mean([d["act_f1"] for d in present_dets])) if present_dets else None,
            "mean_energy_ratio": float(np.mean([d["energy_ratio"] for d in present_dets])) if present_dets else None,
        },
        "enrollment_swap": {
            "tie_eps": TIE_EPS,
            "choice_accuracy": float(np.mean([r["choice_score"] for r in swap_rows])) if swap_rows else None,
            "mean_selectivity_db": float(np.mean([r["selectivity_db"] for r in swap_rows])) if swap_rows else None,
        },
        "efficiency": {
            "latency_ms_mean": float(np.mean(latencies)),
            "rtf_mean": float(np.mean(rtf_list)),
            "peak_gpu_mem_gb": peak_gpu_gb,
        },
        "schema_validation": schema_status,
        "determinism_rescore": "PASS" if det_pass else "FAIL",
        "debug_only": True,
    }
    # 分场景 utterance SI-SDR
    by_scen = {}
    for d in present_dets:
        by_scen.setdefault(d["scenario"], []).append(d["sisdr"])
    summary["present"]["utterance_sisdr_by_scenario"] = {
        k: float(np.mean(v)) for k, v in sorted(by_scen.items())}

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # ---- report.md ----
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("# P2-11 TSE 评测报告（DEBUG_ONLY / BOOTSTRAP_ENCODER_ONLY）\n\n")
        f.write(f"- checkpoint: `{ckpt_path.name}`（sha256 `{ckpt_sha[:16]}…`）\n")
        f.write(f"- manifest: `{Path(args.manifest).name}`（sha256 `{data_sha[:16]}…`）\n")
        f.write(f"- 设备: {device}；样本: {len(records)} 条"
                f"（PRESENT {summary['n_present']} / ABSENT {summary['n_absent']} / NaN {summary['n_nan']}）\n\n")
        if present_dets:
            p = summary["present"]
            f.write("## PRESENT 聚合\n\n")
            f.write(f"- SI-SDR 语料级 {p['corpus_sisdr_db']:.2f} dB / 逐句平均 {p['utterance_sisdr_db']:.2f} dB"
                    "（两口径并列，手册 P2-11）\n")
            f.write(f"- SI-SDRi 均值 {p['mean_sisdri_db']:.2f} dB；wav_l1 {p['mean_wav_l1']:.4f}；"
                    f"MR-STFT {p['mean_mrstft']:.4f}；act_f1 {p['mean_act_f1']:.3f}；"
                    f"energy_ratio {p['mean_energy_ratio']:.4f}\n\n")
            f.write("| 场景 | utterance SI-SDR (dB) |\n|---|---|\n")
            for k, v in summary["present"]["utterance_sisdr_by_scenario"].items():
                f.write(f"| {k} | {v:.2f} |\n")
            f.write("\n")
        if swap_rows:
            sw = summary["enrollment_swap"]
            f.write("## 注册交换\n\n")
            f.write(f"- 选择正确率 {sw['choice_accuracy']:.3f}（tie_eps={TIE_EPS}，平局计 0.5）\n")
            f.write(f"- 平均选择性（q_e1_y1−q_e1_y2）{sw['mean_selectivity_db']:.2f} dB\n\n")
        eff = summary["efficiency"]
        f.write("## 效率\n\n")
        f.write(f"- latency 均值 {eff['latency_ms_mean']:.1f} ms；RTF 均值 {eff['rtf_mean']:.4f}；"
                f"峰值显存 {eff['peak_gpu_mem_gb']:.3f} GB\n\n")
        f.write("## 门禁\n\n")
        f.write(f"- schema 校验: {schema_status}\n")
        f.write(f"- 确定性复评（逐字节）: {summary['determinism_rescore']}\n")

    LOG.info("产物目录: %s", out_dir)
    schema_ok = schema_status.startswith("PASS") or schema_status == "SKIPPED_NO_LIB"
    return 0 if (det_pass and schema_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
