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

正式 checkpoint 使用真实 CAMPPlus；BOOTSTRAP 只适用于其配置明确声明的调试 checkpoint。

运行：
  python tools/evaluate_tse.py \
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

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
sys.path.insert(0, str(P2_ROOT))

from src.tse.model import DualOutputTSE
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
from src.tse.enrollment_adapter import EnrollmentAdapter
from train_overfit_debug import frame_activity, sha256_file

LOG = logging.getLogger("p2_eval")

TIE_EPS = 1e-6
SCHEMA_PATH = P2_ROOT / "schemas" / "tse_prediction.schema.json"


def scenario_of(entry):
    # P1 v3 优先使用 scenario 字段（absent / enroll_swap_target_1 / 等）
    sc = entry.get("scenario", "")
    if sc:
        return str(sc)
    # P1 v2 使用 overlap_ratio + sir_db 构造场景标签
    ov_raw = entry.get("overlap_ratio",
                        entry.get("requested_overlap",
                                  entry.get("measured_overlap", 0.0)))
    sir_raw = entry.get("sir_db",
                         entry.get("requested_sir_db",
                                   entry.get("measured_sir_db", 0.0)))
    ov = int(round(float(ov_raw) * 100))
    sir = float(sir_raw)
    sir_s = f"minus{abs(sir):g}" if sir < 0 else f"{sir:g}"
    return f"overlap_{ov}_sir_{sir_s}"


def resolve_wrong_enrollment(entry, triplet_rows):
    """返回同一 mixture 中另一说话人的 (speaker_id, enrollment)。

    优先使用显式 swap 字段；P1 v3 没有这些字段时，从同一 triplet 的
    target_1/target_2 兄弟行配对。无法证明配对关系时返回 ``(None, None)``，
    调用方必须跳过 choice 指标，不能猜默认路径。
    """
    explicit_path = (entry.get("swap_enrollment") or entry.get("swap_enroll_wav")
                     or entry.get("wrong_enrollment") or entry.get("wrong_enroll_wav"))
    explicit_speaker = (entry.get("swap_speaker") or entry.get("swap_target_speaker")
                        or entry.get("wrong_speaker"))
    if explicit_path and explicit_speaker:
        return str(explicit_speaker), str(explicit_path)

    triplet_id = entry.get("triplet_id")
    if not triplet_id:
        return None, None
    current_speaker = str(entry.get("target_speaker", ""))
    for sibling in triplet_rows.get(str(triplet_id), []):
        sibling_speaker = str(sibling.get("target_speaker", ""))
        sibling_enroll = sibling.get("enrollment", sibling.get("enroll_wav", ""))
        if (sibling.get("target_present") is True and sibling_speaker
                and sibling_speaker != current_speaker and sibling_enroll):
            return sibling_speaker, str(sibling_enroll)
    return None, None


def _load_wav(rel_path, base=None):
    """加载音频：绝对路径直接读；相对路径先试 base 再试 FUNASR_ROOT。"""
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        wav, sr = sf.read(str(p), dtype="float32")
        return torch.from_numpy(wav), sr
    candidates = []
    if base:
        candidates.append(Path(base) / rel_path)
    candidates.append(FUNASR_ROOT / rel_path)
    for c in candidates:
        if c.exists():
            wav, sr = sf.read(str(c), dtype="float32")
            return torch.from_numpy(wav), sr
    raise FileNotFoundError(f"找不到音频: {rel_path} (搜索: {[str(c) for c in candidates]})")


def _load_npy(rel_path, base=None):
    """加载 npy：绝对路径直接读；相对路径先试 base 再试 FUNASR_ROOT。"""
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return np.load(str(p))
    candidates = []
    if base:
        candidates.append(Path(base) / rel_path)
    candidates.append(FUNASR_ROOT / rel_path)
    for c in candidates:
        if c.exists():
            return np.load(str(c))
    raise FileNotFoundError(f"找不到 npy: {rel_path} (搜索: {[str(c) for c in candidates]})")


@torch.no_grad()
def _forward_timed(model, mix, emb, device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    s_tgt, s_res, p_tgt = model(mix.unsqueeze(0).to(device), emb.unsqueeze(0).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return s_tgt[0].cpu(), s_res[0].cpu(), p_tgt[0].cpu(), latency_ms


def score_present(est, ref, mix, p_tgt, frame_act):
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
    return {
        "sisdr": None,
        "sisdri": None,
        "energy_ratio": energy_ratio(est, mix),
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
    try:
        import jsonschema
    except ImportError:
        LOG.warning("jsonschema 未安装，跳过 schema 校验")
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
    ap.add_argument("--manifest", default=str(P2_ROOT / "artifacts" / "debug_mixtures_v0" / "manifest.jsonl"))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--data_root", "--data-root", dest="data_root", default=None,
                    help="数据根目录（音频/npy 相对路径的基目录；不指定则用 FUNASR_ROOT）")
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

    out_dir = Path(args.out) if args.out else P2_ROOT / "artifacts" / f"eval_{ckpt_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)

    entries = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    # P1 v2/v3 字段兼容：v2 用 mixture/target/enrollment/activity，v3 用 mixture_wav/target_wav/enroll_wav/activity_mask
    for e in entries:
        e.setdefault("id", e.get("sample_id", e.get("triplet_id", "")))
        e.setdefault("mixture", e.get("mixture_wav", ""))
        e.setdefault("target", e.get("target_wav", ""))
        e.setdefault("interferer", e.get("interferer_wav", ""))
        e.setdefault("enrollment", e.get("enroll_wav", ""))
        e.setdefault("activity", e.get("activity_mask", ""))
        e.setdefault("config", e.get("generator_version", "unknown"))
    entries = sorted(entries, key=lambda e: str(e.get("id", "")))
    triplet_rows = {}
    for entry in entries:
        if entry.get("triplet_id"):
            triplet_rows.setdefault(str(entry["triplet_id"]), []).append(entry)
    data_root = Path(args.data_root) if args.data_root else FUNASR_ROOT
    base = data_root
    LOG.info("data_root=%s (FUNASR_ROOT=%s)", data_root, FUNASR_ROOT)

    adapter = EnrollmentAdapter.from_config(cfg)
    LOG.info("EnrollmentAdapter mode=%s", adapter.mode)
    if adapter.mode == "campplus":
        try:
            adapter.load_backend()
            LOG.info("CAMPLUS 后端加载成功")
        except Exception as ex:
            raise RuntimeError(
                "正式评估需要真实 CAMPLUS embedding，禁止退化为 BOOTSTRAP"
            ) from ex

    records, detailed, comps = [], [], []
    swap_rows = []
    swap_skips = []
    latencies, rtf_list = [], []
    peak_gpu_gb = 0.0

    for i, e in enumerate(entries):
        sid = str(e.get("id", e.get("sample_id", str(i))))
        target_present = bool(e.get("target_present", True))
        mix, sr = _load_wav(e.get("mixture", e.get("mixture_wav", "")), base)
        assert sr == cfg["sample_rate"]
        spk_id = e.get("target_speaker", e.get("enrollment", sid))
        if adapter.mode == "campplus":
            enroll_path = str(base / e.get("enrollment", e.get("enroll_wav", "")))
            try:
                emb = adapter.encode_file(spk_id, enroll_path).squeeze(0)
            except Exception as ex:
                raise RuntimeError(
                    f"样本 {sid} 的 CAMPLUS enrollment 编码失败: {enroll_path}"
                ) from ex
        else:
            emb = adapter.get_embedding(spk_id).squeeze(0)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        s_tgt, _s_res, p_tgt, latency_ms = _forward_timed(model, mix, emb, device)
        if device.type == "cuda":
            peak_gpu_gb = max(peak_gpu_gb, torch.cuda.max_memory_allocated() / 1024 ** 3)
        if i == 0:
            s_tgt, _s_res, p_tgt, latency_ms = _forward_timed(model, mix, emb, device)

        sf.write(str(out_dir / "audio" / f"{sid}__est_target.wav"),
                 s_tgt.numpy().astype("float32"), sr, subtype="FLOAT")
        np.save(str(out_dir / "audio" / f"{sid}__p_tgt.npy"),
                p_tgt.numpy().astype("float32"))

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
            ref, _ = _load_wav(e.get("target", e.get("target_wav", "")), base)
            # P1 v3 activity_mask 采样数可能 ≠ mix（10ms 帧级），需先 nearest 对齐到 mix 长度
            act = _load_npy(e.get("activity", e.get("activity_mask", "")), base)
            if act.size != mix.numel():
                _src = torch.from_numpy(act.astype("float32")).reshape(1, 1, -1)
                act = torch.nn.functional.interpolate(_src, size=mix.numel(), mode="nearest-exact").squeeze().numpy().astype("float32")
            fa = frame_activity(act, cfg["win_length"], cfg["hop_length"], float(cfg["act_frame_ratio"]))
            scored = score_present(s_tgt, ref, mix, p_tgt, fa)
            comps.append(si_sdr_components(s_tgt, ref))

            # 修复：interferer 可能为 None（D_single 单说话人场景），跳过 swap 评测
            itr_rel = e.get("interferer") or e.get("interferer_wav") or ""
            if itr_rel:
                itr, _ = _load_wav(itr_rel, base)
                wrong_spk, wrong_enroll = resolve_wrong_enrollment(e, triplet_rows)
                if wrong_spk and wrong_enroll:
                    if adapter.mode == "campplus":
                        wrong_path = str(base / wrong_enroll)
                        try:
                            emb_w = adapter.encode_file(wrong_spk, wrong_path).squeeze(0)
                        except Exception as ex:
                            raise RuntimeError(
                                f"样本 {sid} 的 wrong-enrollment 编码失败: {wrong_path}"
                            ) from ex
                    else:
                        emb_w = adapter.get_embedding(wrong_spk).squeeze(0)
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
                else:
                    swap_skips.append({"sample_id": sid, "reason": "no_paired_wrong_enrollment"})
                det = {"sample_id": sid, "scenario": scenario, "target_present": True,
                       "config": e.get("config", "unknown"), **scored,
                       "activity_ratio": activity_ratio(p_tgt),
                       "rtf": latency_ms / 1000.0 / duration}
                if wrong_spk and wrong_enroll:
                    det["swap"] = swap_rows[-1]
                else:
                    det["swap_skipped_reason"] = "no_paired_wrong_enrollment"
            else:
                det = {"sample_id": sid, "scenario": scenario, "target_present": True,
                       "config": e.get("config", "unknown"), **scored,
                       "activity_ratio": activity_ratio(p_tgt),
                       "rtf": latency_ms / 1000.0 / duration}
        else:
            scored = score_absent(s_tgt, mix, p_tgt)
            det = {"sample_id": sid, "scenario": scenario, "target_present": False,
                   "config": e.get("config", "unknown"), **scored,
                   "rtf": latency_ms / 1000.0 / duration}

        records.append(schema_record(sid, scenario, target_present, ckpt_sha, data_sha,
                                     scored, latency_ms))
        detailed.append(det)
        LOG.info("%s [%s] sisdr=%s energy_ratio=%s lat=%.1fms",
                 sid, scenario,
                 f"{scored['sisdr']:.2f}" if scored["sisdr"] is not None else "null",
                 f"{scored['energy_ratio']:.4f}" if scored["energy_ratio"] is not None else "null",
                 latency_ms)

    def _dump_jsonl(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pred_path = out_dir / "predictions.jsonl"
    _dump_jsonl(pred_path, records)
    _dump_jsonl(out_dir / "predictions_detailed.jsonl", detailed)

    schema_status = validate_schema(records)
    LOG.info("schema 校验: %s", schema_status)

    verify = []
    for e, rec in zip(entries, records):
        sid_v = str(e.get("id", ""))
        est, _ = sf.read(str(out_dir / "audio" / f"{sid_v}__est_target.wav"), dtype="float32")
        est = torch.from_numpy(est)
        p_tgt = torch.from_numpy(np.load(str(out_dir / "audio" / f"{sid_v}__p_tgt.npy")))
        mix, _ = _load_wav(e.get("mixture", ""), base)
        target_present = bool(e.get("target_present", True))
        if rec["nan"]:
            scored = nan_record_present()
        elif target_present:
            ref, _ = _load_wav(e.get("target", ""), base)
            act = _load_npy(e.get("activity", ""), base)
            if act.size != mix.numel():
                _src = torch.from_numpy(act.astype("float32")).reshape(1, 1, -1)
                act = torch.nn.functional.interpolate(_src, size=mix.numel(), mode="nearest-exact").squeeze().numpy().astype("float32")
            fa = frame_activity(act, cfg["win_length"], cfg["hop_length"], float(cfg["act_frame_ratio"]))
            scored = score_present(est, ref, mix, p_tgt, fa)
        else:
            scored = score_absent(est, mix, p_tgt)
        verify.append(schema_record(sid_v, rec["scenario"], target_present,
                                    ckpt_sha, data_sha, scored, rec["latency_ms"]))
    det_pass = all(json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
                   for a, b in zip(records, verify))
    LOG.info("确定性复评: %s", "PASS（逐字节一致）" if det_pass else "FAIL")
    verify_path = out_dir / "predictions.verify.jsonl"
    if not det_pass:
        _dump_jsonl(verify_path, verify)
    elif verify_path.exists():
        verify_path.unlink()

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
            "n_scored": len(swap_rows),
            "n_skipped": len(swap_skips),
            "skip_reasons": sorted({r["reason"] for r in swap_skips}),
        },
        "efficiency": {
            "latency_ms_mean": float(np.mean(latencies)),
            "rtf_mean": float(np.mean(rtf_list)),
            "peak_gpu_mem_gb": peak_gpu_gb,
        },
        "schema_validation": schema_status,
        "determinism_rescore": "PASS" if det_pass else "FAIL",
        "debug_only": all(bool(e.get("debug_only", False)) for e in entries),
    }
    by_scen = {}
    for d in present_dets:
        by_scen.setdefault(d["scenario"], []).append(d["sisdr"])
    summary["present"]["utterance_sisdr_by_scenario"] = {
        k: float(np.mean(v)) for k, v in sorted(by_scen.items())}

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        scope = ("DEBUG_ONLY" if summary["debug_only"] else "FORMAL_INPUT")
        f.write(f"# P2-11 TSE 评测报告（{scope} / {adapter.mode.upper()}）\n\n")
        f.write(f"- checkpoint: `{ckpt_path.name}`（sha256 `{ckpt_sha[:16]}…`）\n")
        f.write(f"- manifest: `{Path(args.manifest).name}`（sha256 `{data_sha[:16]}…`）\n")
        f.write(f"- 设备: {device}；样本: {len(records)} 条"
                f"（PRESENT {summary['n_present']} / ABSENT {summary['n_absent']} / NaN {summary['n_nan']}）\n\n")
        if present_dets:
            p = summary["present"]
            f.write("## PRESENT 聚合\n\n")
            f.write(f"- SI-SDR 语料级 {p['corpus_sisdr_db']:.2f} dB / 逐句平均 {p['utterance_sisdr_db']:.2f} dB\n")
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
        if swap_skips:
            f.write(f"- 注册交换未计分 {len(swap_skips)} 条：缺少可证明配对的 wrong-enrollment\n\n")
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
