# -*- coding: utf-8 -*-
"""Inspect B3 data, real enrollment embeddings, and optional model outputs.

This diagnostic never substitutes BOOTSTRAP embeddings for a CAMPPlus failure.
It uses real manifest enrollment audio and reports measurements; it does not
rewrite the manifest, checkpoint, or evaluation output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import yaml

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
sys.path.insert(0, str(P2_ROOT))

from src.tse.enrollment_adapter import EnrollmentAdapter
from src.tse.model import DualOutputTSE

LOG = logging.getLogger("p2_b3_diagnose")
MODEL_KEYS = (
    "sample_rate", "n_fft", "hop_length", "win_length", "emb_dim",
    "lstm_hidden", "lstm_layers", "dropout", "film_scale",
)


def _normalize_entry(entry: dict) -> dict:
    row = dict(entry)
    row.setdefault("id", row.get("sample_id", row.get("triplet_id", "")))
    row.setdefault("mixture", row.get("mixture_wav", ""))
    row.setdefault("target", row.get("target_wav", ""))
    row.setdefault("enrollment", row.get("enroll_wav", ""))
    row.setdefault("target_present", True)
    row.setdefault("scenario", "unknown")
    return row


def _read_manifest(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"manifest row is not an object at {path}:{line_number}")
            rows.append(_normalize_entry(value))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return sorted(rows, key=lambda row: str(row["id"]))


def _resolve_path(value: str, manifest_dir: Path, data_root: Path) -> Path:
    if not value:
        raise FileNotFoundError("empty asset path")
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [
        manifest_dir / raw,
        data_root / raw,
        FUNASR_ROOT / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"asset not found: {value} (searched {[str(path) for path in candidates]})"
    )


def _load_wav(path: Path, expected_sr: int) -> torch.Tensor:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if sample_rate != expected_sr:
        raise ValueError(f"sample rate mismatch for {path}: {sample_rate} != {expected_sr}")
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    if not len(mono) or not np.isfinite(mono).all():
        raise ValueError(f"invalid audio: {path}")
    return torch.from_numpy(mono)


def _rms(audio: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(audio.float() ** 2)))


def _sample_across_scenarios(rows: Iterable[dict], limit: int) -> List[dict]:
    """Round-robin scenarios so a prefix-ordered manifest cannot hide a class."""
    groups: Dict[str, List[dict]] = {}
    for row in rows:
        key = f"{row.get('scenario', 'unknown')}|present={bool(row.get('target_present', True))}"
        groups.setdefault(key, []).append(row)
    selected = []
    keys = sorted(groups)
    offset = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if offset < len(groups[key]) and len(selected) < limit:
                selected.append(groups[key][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def inspect_inputs(rows: Iterable[dict], manifest_dir: Path, data_root: Path,
                   sample_rate: int, max_samples: int) -> dict:
    selected = _sample_across_scenarios(rows, max_samples)
    errors = []
    present_ratios = []
    absent_target_rms = []
    mixture_rms = []
    scenarios = Counter()
    for row in selected:
        sample_id = str(row["id"])
        scenarios[str(row.get("scenario", "unknown"))] += 1
        try:
            mix_path = _resolve_path(row["mixture"], manifest_dir, data_root)
            enroll_path = _resolve_path(row["enrollment"], manifest_dir, data_root)
            mix = _load_wav(mix_path, sample_rate)
            _load_wav(enroll_path, sample_rate)
            mix_rms = _rms(mix)
            mixture_rms.append(mix_rms)
            if bool(row["target_present"]):
                target = _load_wav(
                    _resolve_path(row["target"], manifest_dir, data_root), sample_rate
                )
                present_ratios.append(_rms(target) / max(mix_rms, 1.0e-12))
            elif row.get("target"):
                target = _load_wav(
                    _resolve_path(row["target"], manifest_dir, data_root), sample_rate
                )
                absent_target_rms.append(_rms(target))
        except Exception as exc:
            errors.append({"sample_id": sample_id, "error": str(exc)})

    def stats(values: List[float]) -> Optional[dict]:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "min": float(array.min()),
            "mean": float(array.mean()),
            "max": float(array.max()),
        }

    return {
        "sampled": len(selected),
        "scenario_counts": dict(sorted(scenarios.items())),
        "mixture_rms": stats(mixture_rms),
        "present_target_to_mix_rms_ratio": stats(present_ratios),
        "absent_target_rms": stats(absent_target_rms),
        "errors": errors,
    }


def load_real_adapter(cfg: dict, device: torch.device) -> EnrollmentAdapter:
    run_cfg = dict(cfg)
    run_cfg["device"] = str(device)
    adapter = EnrollmentAdapter.from_config(run_cfg)
    if adapter.mode != "campplus":
        raise ValueError(f"formal B3 diagnosis requires sv_mode=campplus, got {adapter.mode}")
    adapter.load_backend()
    return adapter


def inspect_embeddings(adapter: EnrollmentAdapter, rows: Iterable[dict],
                       manifest_dir: Path, data_root: Path,
                       max_speakers: int) -> Tuple[dict, Dict[str, torch.Tensor]]:
    embeddings = {}
    errors = []
    for row in rows:
        speaker = str(row.get("target_speaker", ""))
        if not speaker or speaker in embeddings:
            continue
        try:
            enroll = _resolve_path(row["enrollment"], manifest_dir, data_root)
            embedding = adapter.encode_file(speaker, str(enroll)).squeeze(0).detach().cpu()
            if embedding.numel() != adapter.emb_dim or not torch.isfinite(embedding).all():
                raise ValueError(f"invalid embedding shape/values: {tuple(embedding.shape)}")
            embeddings[speaker] = embedding
        except Exception as exc:
            errors.append({"speaker": speaker, "sample_id": str(row["id"]), "error": str(exc)})
        if len(embeddings) >= max_speakers:
            break

    norms = [float(value.norm()) for value in embeddings.values()]
    cosine_values = []
    values = list(embeddings.values())
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            cosine_values.append(float(torch.nn.functional.cosine_similarity(
                values[left].unsqueeze(0), values[right].unsqueeze(0)
            )))
    if len(embeddings) < 2:
        errors.append({
            "speaker": "*", "sample_id": "*",
            "error": f"fewer than two speakers encoded: {len(embeddings)}",
        })
    return {
        "speakers_encoded": len(embeddings),
        "embedding_norm": None if not norms else {
            "min": min(norms), "mean": float(np.mean(norms)), "max": max(norms),
        },
        "different_speaker_cosine": None if not cosine_values else {
            "min": min(cosine_values),
            "mean": float(np.mean(cosine_values)),
            "max": max(cosine_values),
        },
        "errors": errors,
    }, embeddings


def _checkpoint_state(checkpoint: dict) -> dict:
    for key in ("model", "model_state_dict", "model_state", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError("checkpoint has no model state")


@torch.no_grad()
def inspect_checkpoint(checkpoint_path: Path, config: dict, adapter: EnrollmentAdapter,
                       rows: Iterable[dict], manifest_dir: Path, data_root: Path,
                       device: torch.device, max_samples: int) -> dict:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    checkpoint_cfg = checkpoint.get("cfg", config)
    mismatches = {
        key: {"config": config.get(key), "checkpoint": checkpoint_cfg.get(key)}
        for key in MODEL_KEYS if config.get(key) != checkpoint_cfg.get(key)
    }
    if mismatches:
        raise ValueError(f"config/checkpoint model fields differ: {mismatches}")
    model = DualOutputTSE(checkpoint_cfg).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    model.eval()

    present = []
    absent = []
    errors = []
    for row in _sample_across_scenarios(rows, max_samples):
        try:
            mix = _load_wav(
                _resolve_path(row["mixture"], manifest_dir, data_root),
                int(checkpoint_cfg["sample_rate"]),
            )
            enroll = _resolve_path(row["enrollment"], manifest_dir, data_root)
            speaker = str(row.get("target_speaker", row["id"]))
            embedding = adapter.encode_file(speaker, str(enroll)).to(device)
            estimate, _residual, _activity = model(mix.unsqueeze(0).to(device), embedding)
            ratio = _rms(estimate[0].cpu()) / max(_rms(mix), 1.0e-12)
            (present if bool(row["target_present"]) else absent).append(ratio)
        except Exception as exc:
            errors.append({"sample_id": str(row["id"]), "error": str(exc)})

    def output_stats(values: List[float]) -> Optional[dict]:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": int(array.size),
            "min": float(array.min()),
            "mean": float(array.mean()),
            "max": float(array.max()),
            "near_silent_below_1e-4": int(np.count_nonzero(array < 1.0e-4)),
        }

    return {
        "checkpoint": str(checkpoint_path),
        "step": checkpoint.get("step"),
        "strict_state_load": True,
        "config_mismatches": mismatches,
        "present_output_to_mix_rms_ratio": output_stats(present),
        "absent_output_to_mix_rms_ratio": output_stats(absent),
        "errors": errors,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", "--data_root", dest="data_root", type=Path, default=FUNASR_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="optional: measure actual B3 output energy without writing audio")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=20)
    parser.add_argument("--max-speakers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = _read_manifest(args.manifest)
    manifest_dir = args.manifest.resolve().parent
    data_root = args.data_root.resolve()

    report = {
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "input": inspect_inputs(
            rows, manifest_dir, data_root, int(config["sample_rate"]), args.max_samples
        ),
        "embedding": None,
        "checkpoint": None,
        "errors": [],
    }
    adapter = None
    try:
        adapter = load_real_adapter(config, device)
        report["embedding"], _embeddings = inspect_embeddings(
            adapter, rows, manifest_dir, data_root, args.max_speakers
        )
    except Exception as exc:
        report["errors"].append(f"CAMPPlus: {exc}")

    if args.checkpoint is not None:
        if adapter is None:
            report["errors"].append("checkpoint output skipped because CAMPPlus is unavailable")
        else:
            try:
                report["checkpoint"] = inspect_checkpoint(
                    args.checkpoint.resolve(), config, adapter, rows, manifest_dir,
                    data_root, device, args.max_samples,
                )
            except Exception as exc:
                report["errors"].append(f"checkpoint: {exc}")

    report["errors"].extend(
        f"input/{item['sample_id']}: {item['error']}" for item in report["input"]["errors"]
    )
    if report["embedding"] is not None:
        report["errors"].extend(
            f"embedding/{item['speaker']}: {item['error']}"
            for item in report["embedding"]["errors"]
        )
    if report["checkpoint"] is not None:
        report["errors"].extend(
            f"checkpoint/{item['sample_id']}: {item['error']}"
            for item in report["checkpoint"]["errors"]
        )
        present_output = report["checkpoint"].get("present_output_to_mix_rms_ratio")
        if (present_output and present_output["count"] > 0
                and present_output["near_silent_below_1e-4"] == present_output["count"]):
            report["errors"].append("checkpoint: all sampled PRESENT outputs are near-silent")
    report["status"] = "PASS" if not report["errors"] else "FAIL"

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        LOG.info("report=%s", args.output.resolve())
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
