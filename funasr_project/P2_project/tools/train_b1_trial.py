# -*- coding: utf-8 -*-
"""P2-12 B1 500-step 试车脚本（通用训练器，支持 AMP/梯度累积/warmup+cosine/checkpoint 恢复）。

相对 train_overfit_debug.py 的核心差异：
- Dataset + DataLoader（支持 shuffle / num_workers / pin_memory）
- 按 segment_length 随机截取等长片段（mixture/target/interferer/activity 同起点）
- AMP 混合精度（torch.cuda.amp.autocast + GradScaler，CPU 自动禁用）
- 学习率 warmup(线性) → cosine 衰减
- 梯度累积（effective_batch = batch_size × gradient_accumulation）
- 每 save_every 步存 checkpoint（含 optimizer/scaler/step），支持 --resume 续训
- 记录吞吐量（samples/sec）、数据等待占比、step time P50/P95
- 试车判定 trial_verdict.json + 人读报告 report.md

复用 train_overfit_debug 的纯函数：bootstrap_embedding / frame_activity /
sha256_file / sha256_text / compute_losses（不复制，直接 import）。

标记：B1_TRIAL / BOOTSTRAP_ENCODER_ONLY（P4 契约未交付前 embedding 用哈希随机）

运行示例：
  # P1 v2_b1 未交付时用 DEBUG 数据空跑 10 步验证脚本
  python tools/train_b1_trial.py --debug_data --max_steps 10

  # 正式 500 步试车（需 P1 v2_b1 交付）
  python tools/train_b1_trial.py --device auto

  # 从 checkpoint 续训
  python tools/train_b1_trial.py --resume artifacts/experiments/B1_TRIAL_seed20260723/checkpoint_step100.pt
"""

import argparse
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.utils.data as data
import yaml

P2_ROOT = Path(__file__).resolve().parents[1]
FUNASR_ROOT = P2_ROOT.parent
P1_DATA_ROOT = Path("/root/autodl-tmp/P1_to_P2_v2_b1")
sys.path.insert(0, str(P2_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.tse.model import DualOutputTSE
from src.tse.losses import si_sdr
from src.tse.enrollment_adapter import EnrollmentAdapter
from train_overfit_debug import (
    compute_losses,
    frame_activity,
    sha256_file,
)

LOG = logging.getLogger("p2_b1_trial")

DEBUG_MANIFEST = P2_ROOT / "artifacts" / "debug_mixtures_v0" / "manifest.jsonl"


class B1Dataset(data.Dataset):
    """B1 / B2 / B3 多场景训练数据集。

    场景模式 scene_mode:
      - b1 (PRESENT):    目标 speaker 出现在 mixture 中，enroll 与目标一致 (基线场景)
      - b2 (ABSENT):     目标 speaker 不在 mixture 中，target=0，emb=零向量（P1 v3 数据）
      - b3 (ENROLL-SWAP):enroll 被替换为干扰 speaker 的音频，target=0（P1 v3 数据）

    按 segment_length 随机截取等长片段；mixture/target/interferer/activity 必须同起点
    （否则破坏对齐）。不足段长则右侧补零。embedding 由 CAMPPlus / BOOTSTRAP 派生。
    """

    SUPPORTED_MODES = ("b1", "b2", "b3")

    def _resolve_path(self, p):
        """Resolve audio path.
        搜索顺序（从高到低）：
          1) manifest_dir：manifest.jsonl 所在目录（P1 v3：assets/ 与 manifest 同目录）
          2) cfg.datasets.data_root（可选，配置覆盖）
          3) P1_DATA_ROOT 硬编码默认（P1 v2 兼容）
          4) FUNASR_ROOT 项目根（P2 debug data 路径 P2_project/artifacts/...）
        对省略后缀（.npy/.wav）的 activity_mask 路径自动补全后缀（兼容 P1 v3 hash 路径）。"""
        p = str(p)
        if not p or p == "None":
            return None
        roots = []
        # 1) manifest dir（最高优先级：manifest 同级 assets）
        if hasattr(self, "_manifest_dir") and self._manifest_dir is not None:
            roots.append(self._manifest_dir)
        # 2) cfg 可选 data_root
        if hasattr(self, "cfg") and self.cfg is not None:
            cfg_dr = self.cfg.get("datasets", {}).get("data_root") or self.cfg.get("data_root")
            if cfg_dr:
                roots.append(Path(cfg_dr))
        # 3) 硬编码 P1 v2 默认
        roots.append(P1_DATA_ROOT)
        # 4) 项目根兜底（debug data）
        roots.append(FUNASR_ROOT)
        suffixes = ("", ".npy", ".wav")  # 先试原路径 → .npy → .wav
        for root in roots:
            candidate_base = Path(root) / p
            for suf in suffixes:
                cand = Path(str(candidate_base) + suf)
                if cand.exists():
                    return str(cand)
        return str(FUNASR_ROOT / p)

    @staticmethod
    def _load_activity_mask(path):
        """加载 activity_mask：.npy（优先）或 .wav（单通道 mask）。"""
        p = Path(path)
        if p.suffix == ".npy" or p.suffix == "":
            try:
                arr = np.load(str(p))
                return arr.astype("float32")
            except Exception:
                pass
        arr, _sr = sf.read(str(p), dtype="float32")
        if arr.ndim > 1:
            arr = arr[:, 0]
        return arr.astype("float32")

    def __init__(self, manifest_path, cfg, seed=None, adapter=None, scene_mode=None, split_name=None):
        self.entries = []
        self.cfg = cfg  # 提前赋值，供 _resolve_path 内取 cfg.datasets.data_root
        # ---- manifest 所在目录（P1 v3：assets/ 相对 manifest 位置）----
        self._manifest_dir = Path(manifest_path).resolve().parent
        # ---- scene_mode / split_name：CLI > cfg > 默认 ----
        self.scene_mode = (scene_mode or cfg.get("scene_mode") or "b1").lower()
        assert self.scene_mode in self.SUPPORTED_MODES, (
            f"scene_mode={self.scene_mode} 不受支持，允许: {self.SUPPORTED_MODES}")
        self.split_name = split_name or cfg.get("split_name")  # None=不过滤，否则按 e['split'] 匹配（如 train/dev）
        # ---- scenario 过滤：按 scene_mode 自动限定（P1 v3 支持）----
        if self.scene_mode == "b2":
            self._scenario_filter = {"absent", "enroll_swap_absent"}  # target_present=False
        elif self.scene_mode == "b3":
            self._scenario_filter = {"enroll_swap_target_1", "enroll_swap_target_2",  # target_present=True（同 speaker swap 注册句）
                                     "enroll_swap_absent"}                           # target_present=False（注册句 swap + absent）
        else:  # b1：全 scenario（含 v2 P1 数据）
            self._scenario_filter = None
        # ---- 读 manifest + 过滤 ----
        skipped_split = 0
        skipped_scenario = 0
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                # split 过滤（P1 v3：e['split'] = train/dev/D_absent/D_swap）
                if self.split_name and e.get("split", "") != self.split_name:
                    skipped_split += 1
                    continue
                # scenario 过滤（P1 v3 有 scenario 字段时严格过滤；v1/v2 P2 debug 无 scenario 字段 → 全通过，靠 is_absent/is_swap/target_present 旧字段判定）
                sc = e.get("scenario")
                if self._scenario_filter and sc is not None and sc not in self._scenario_filter:
                    skipped_scenario += 1
                    continue
                self.entries.append(e)
        LOG.info("加载 manifest=%s scene_mode=%s split_name=%s total=%d (skipped_split=%d scenario=%d)",
                 Path(manifest_path).name, self.scene_mode, self.split_name or "ALL",
                 len(self.entries), skipped_split, skipped_scenario)
        self.seg_samples = int(cfg["segment_length"] * cfg["sample_rate"])
        self.win_length = int(cfg["win_length"])
        self.hop_length = int(cfg["hop_length"])
        self.act_frame_ratio = float(cfg["act_frame_ratio"])
        self.emb_dim = int(cfg["emb_dim"])
        self.rng = np.random.default_rng(seed)
        self.adapter = adapter or EnrollmentAdapter.from_config(cfg)
        self._camplus_fallback = False
        if self.adapter.mode == "campplus":
            try:
                self.adapter.load_backend()
                LOG.info("CAMPLUS 后端加载成功 (scene_mode=%s)", self.scene_mode)
            except Exception as e:
                LOG.warning(f"CAMPLUS 后端加载失败，fallback BOOTSTRAP: {e}")
                self._camplus_fallback = True
                self.adapter = EnrollmentAdapter.from_config(cfg, mode="bootstrap")
        # 场景统计（每 1000 条 __getitem__ 打一条 summary log）
        self._stats = {"present": 0, "absent": 0, "swap": 0}
        LOG.info("Dataset 场景模式: %s split=%s (总 %d 条 manifest)",
                 self.scene_mode, self.split_name or "ALL", len(self.entries))

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _is_absent_entry(e, act_array):
        """判定是否 absent 场景（应输出 0）。
        优先级：显式 target_present=False（P1 v3）> scenario=*absent*（P1 v3）> 旧字段 is_absent > activity 全零。"""
        if e.get("target_present") is False:
            return True
        scenario = str(e.get("scenario", ""))
        if scenario in ("absent", "enroll_swap_absent"):
            return True
        if e.get("is_absent") in (True, 1, "true", "True", "1"):
            return True
        if act_array is not None and act_array.size > 0 and float(act_array.sum()) == 0.0:
            return True
        return False

    @staticmethod
    def _is_swap_entry(e):
        """判定是否 enroll-swap 场景（P1 v3 scenario 含 enroll_swap_* 或 P2 旧字段 is_swap）。
        注意：is_swap 只是场景统计用，不等于应该输出 0。B3 target_1/2（target_present=True）不应输出 0。"""
        scenario = str(e.get("scenario", ""))
        if "enroll_swap" in scenario:
            return True
        if e.get("is_swap") in (True, 1, "true", "True", "1"):
            return True
        swap_field = e.get("swap_enrollment") or e.get("swap_enroll_wav") or e.get("swap_enroll")
        return bool(swap_field)

    def __getitem__(self, idx):
        e = self.entries[idx]

        # P1 field mapping: mixture_wav→mixture, target_wav→target, etc.
        mix_path = e.get("mixture", e.get("mixture_wav", ""))
        tgt_path = e.get("target", e.get("target_wav", ""))
        itr_path = e.get("interferer", e.get("interferer_wav", ""))
        act_path = e.get("activity", e.get("activity_mask", ""))
        enroll_path = e.get("enrollment", e.get("enroll_wav", ""))
        swap_enroll_path = (e.get("swap_enrollment") or e.get("swap_enroll_wav")
                            or e.get("swap_enroll") or "")
        sample_id = e.get("id", e.get("sample_id", str(idx)))

        mix, sr = sf.read(self._resolve_path(mix_path), dtype="float32")
        tgt, _ = sf.read(self._resolve_path(tgt_path), dtype="float32")

        # Handle null interferer (single speaker case)
        itr_resolved = self._resolve_path(itr_path)
        if itr_resolved and Path(itr_resolved).exists():
            itr, _ = sf.read(itr_resolved, dtype="float32")
        else:
            itr = np.zeros_like(mix, dtype="float32")

        # Activity mask：.npy（P1/P2 原生）或 .wav（P1 v3 也可存 mask），自动补全后缀
        act_resolved = self._resolve_path(act_path)
        act = self._load_activity_mask(act_resolved)
        assert sr == self.cfg["sample_rate"], f"采样率不一致: {sr} vs {self.cfg['sample_rate']}"

        # ============================================================
        # P1 v3 兼容修复：activity.npy 采样数 ≠ mix 采样数（常见 ratio≈0.552）
        #   → 先 nearest-exact 将 act（0/1 标签，保语义无插值污染）对齐到 len(mix)
        #   → 再走统一 crop/pad，保证后续 frame_activity() 帧数与 model STFT 严格一致
        # ============================================================
        _T_mix = len(mix)
        _orig_act_len = int(act.size)
        if _orig_act_len != _T_mix:
            try:
                if _orig_act_len == 0:
                    act = np.zeros(_T_mix, dtype="float32")
                else:
                    _src = torch.from_numpy(act.astype("float32")).reshape(1, 1, -1)
                    _aligned = torch.nn.functional.interpolate(
                        _src, size=_T_mix, mode="nearest-exact"
                    ).squeeze()
                    act = _aligned.numpy().astype(act.dtype if hasattr(act, 'dtype') else 'float32')
                if not getattr(self, "_actlen_fix_count", 0):
                    LOG.info(
                        "[ACT LEN FIX] sample-level activity %d → %d (len(mix)), ratio=%.3f  [后续将每 1000 次汇总 1 条]",
                        _orig_act_len, _T_mix, _T_mix / max(1, _orig_act_len),
                    )
                self._actlen_fix_count = getattr(self, "_actlen_fix_count", 0) + 1
                if self._actlen_fix_count % 1000 == 0:
                    LOG.info("[ACT LEN FIX] 累计 %d 次 (最近 ratio=%.3f)",
                             self._actlen_fix_count, _T_mix / max(1, _orig_act_len))
            except Exception as _e:
                LOG.warning("[ACT LEN FIX] align fail (%s), fallback pad/trim", _e)
                if _orig_act_len < _T_mix:
                    act = np.pad(act, (0, _T_mix - _orig_act_len))
                else:
                    act = act[:_T_mix]

        # ========== 场景判定 ==========
        is_absent = self._is_absent_entry(e, act)
        is_swap = self._is_swap_entry(e)

        # force_zero_target：只有目标 speaker 不在 mixture 中（target_present=False / absent）才强制输出 0
        #   B3 enroll_swap_target_1/2（target_present=True，注册句是同 speaker 另一句）→ 不输出 0，正常回归 target_wav
        #   兼容 P2 debug_data：没有 target_present 字段时，回退旧逻辑（is_absent or is_swap → zero）
        target_present = e.get("target_present")
        if target_present is not None:
            force_zero_target = (target_present is False)
        else:
            force_zero_target = is_absent or is_swap

        T = len(mix)
        if T >= self.seg_samples:
            start = int(self.rng.integers(0, T - self.seg_samples + 1))
            mix = mix[start:start + self.seg_samples]
            tgt = tgt[start:start + self.seg_samples]
            itr = itr[start:start + self.seg_samples]
            act = act[start:start + self.seg_samples]
        else:
            pad = self.seg_samples - T
            mix = np.pad(mix, (0, pad))
            tgt = np.pad(tgt, (0, pad))
            itr = np.pad(itr, (0, pad))
            act = np.pad(act, (0, pad))

        # ---- 监督信号置零（B2 ABSENT + B3 swap_absent）----
        if force_zero_target:
            tgt = np.zeros_like(mix, dtype="float32")
            act = np.zeros_like(act, dtype=act.dtype)
            if is_absent:
                self._stats["absent"] += 1
            else:  # P2 debug 的旧 is_swap=true（老数据兼容）
                self._stats["swap"] += 1
        else:
            # 非零 target：B1 present / B3 enroll_swap_target_1/2（同 speaker 注册句 swap，target 仍存在）
            if is_swap:
                self._stats["swap"] += 1  # B3：注册句有 swap
            else:
                self._stats["present"] += 1  # B1：普通场景

        # ---- Embedding 选择 ----
        # B2 (absent)          : 零向量（P2 设计要求：注册 speaker 不存在则给零 emb）
        # B1 (present)          : 正常 enroll_path
        # B3 enroll_swap        : P1 v3 直接把 swap 后的 enroll 填在 enroll_wav 字段 → 正常 enroll_path 计算 emb（无需 swap_enroll_path）
        # 旧 P2 debug 兼容      : 若显式传 swap_enroll_path（P1 v2 之前设计）→ 走 swap 分支（独立缓存 key）
        if is_absent:
            emb = torch.zeros(self.emb_dim, dtype=torch.float32)
        elif is_swap and swap_enroll_path:
            # 旧 P2 debug data：swap embedding 走缓存（key 是 swap enroll 的 MD5，避免冲突）
            emb = self._get_embedding(e, swap_enroll_path, speaker_hint="swap")
        else:
            # B1 普通场景 + B3 v3 swap（enroll_wav 已被 P1 swap 成正确值）→ 统一走 enroll_path
            emb = self._get_embedding(e, enroll_path, speaker_hint="target")

        # ---- 场景统计：每 1000 条打一条 ----
        total = sum(self._stats.values())
        if total % 1000 == 0 and total > 0:
            LOG.info("Dataset 样本统计 [total=%d] present=%d absent=%d swap=%d",
                     total, self._stats["present"], self._stats["absent"], self._stats["swap"])

        return {
            "id": sample_id,
            "mix": torch.from_numpy(mix),
            "target": torch.from_numpy(tgt),
            "interferer": torch.from_numpy(itr),
            "frame_act": frame_activity(act, self.win_length, self.hop_length, self.act_frame_ratio),
            "emb": emb,
            # 给 loss 层 / debugging 用的场景标签
            "is_absent": torch.tensor(1 if is_absent else 0, dtype=torch.uint8),
            "is_swap": torch.tensor(1 if is_swap else 0, dtype=torch.uint8),
        }

    def _get_embedding(self, e, enroll_path=None, speaker_hint="target"):
        """获取 speaker embedding。

        speaker_hint: 'target' / 'swap'  —— 仅用于 absent 分支的缓存 key 不冲突；
                       absent 场景下返回零向量，不走此函数。
        """
        spk_id = e.get("target_speaker", e.get("enrollment", e.get("sample_id", "unknown")))
        if speaker_hint == "swap":
            # swap 场景：speaker id 换成 swap 字段（如果有的话），否则用 swap_enroll 派生
            spk_id = (e.get("swap_speaker") or e.get("swap_target_speaker")
                      or f"swap__{spk_id}")
        if self.adapter.mode == "campplus" and not self._camplus_fallback:
            enroll = enroll_path or e.get("enrollment", e.get("enroll_wav", ""))
            enroll_resolved = self._resolve_path(enroll) if enroll else None
            if enroll_resolved and Path(enroll_resolved).exists():
                cache_key = hashlib.md5(str(enroll_resolved).encode()).hexdigest() + ".pt"
                cache_dir = P1_DATA_ROOT / "emb_cache_campplus"
                cache_file = cache_dir / cache_key
                if cache_file.exists():
                    emb = torch.load(cache_file, weights_only=False, map_location="cpu")
                    return emb.squeeze(0)
                try:
                    emb = self.adapter.encode_file(spk_id, enroll_resolved)
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(emb, cache_file)
                    return emb.squeeze(0)
                except Exception as ex:
                    if not getattr(self, "_camplus_fail_count", 0):
                        LOG.warning(f"CAMPLUS encode 失败 ({speaker_hint}/{spk_id}), fallback BOOTSTRAP: {ex}  [后续同类错误将每 1000 次汇总 1 条]")
                    self._camplus_fail_count = getattr(self, "_camplus_fail_count", 0) + 1
                    if self._camplus_fail_count % 1000 == 0:
                        LOG.warning(f"CAMPLUS fallback BOOTSTRAP 累计 {self._camplus_fail_count} 次 (最近 spk_id={spk_id})")
            # 兜底：enroll 路径无效或 encode 异常 → BOOTSTRAP 确定性 emb（同 enroll→同 emb）
            #   避免：① adapter.get_embedding(spk_id) → KeyError 未注册；
            #        ② 调用模块级 bootstrap_embedding(...) → 云端脚本 import 被补丁打乱时 NameError。
            #   直接内联 bootstrap 实现（与 train_overfit_debug.bootstrap_embedding 等价，0 外部 import 依赖）
            _bo_text = enroll or enroll_path or str(spk_id)
            try:
                _bo_seed = int(sha256_text(_bo_text)[:8], 16)
            except Exception:
                _bo_seed = int(hashlib.sha256(str(_bo_text).encode()).hexdigest()[:8], 16)
            _bo_gen = torch.Generator().manual_seed(_bo_seed)
            return torch.randn(self.emb_dim, generator=_bo_gen)
        return self.adapter.get_embedding(spk_id).squeeze(0)


def collate_fn(batch):
    """所有样本等长（由 Dataset 保证），直接 stack。is_absent/is_swap 是场景标签。"""
    return {
        "ids": [b["id"] for b in batch],
        "mix": torch.stack([b["mix"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "interferer": torch.stack([b["interferer"] for b in batch]),
        "frame_act": torch.stack([b["frame_act"] for b in batch]),
        "emb": torch.stack([b["emb"] for b in batch]),
        "is_absent": torch.stack([b["is_absent"] for b in batch]),
        "is_swap": torch.stack([b["is_swap"] for b in batch]),
    }


def get_lr(step, peak_lr, warmup_steps, total_steps, schedule="cosine"):
    """warmup(线性) → cosine/linear 衰减到 0；constant 不衰减。"""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if schedule == "cosine":
        return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    if schedule == "linear":
        return peak_lr * (1.0 - progress)
    return peak_lr


@torch.no_grad()
def evaluate_dev(model, loader, cfg, device, use_amp):
    """dev 集平均损失 + SI-SDR（不反向，不记录吞吐量）。"""
    model.eval()
    total_loss, total_sisdr, n = 0.0, 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch["mix"], batch["emb"])
            loss, terms = compute_losses(cfg, out, batch)
        total_loss += float(loss.detach().cpu())
        total_sisdr += float(terms["si_sdr_db"].detach().cpu())
        n += 1
    model.train()
    return {"dev_loss": total_loss / max(1, n), "dev_sisdr": total_sisdr / max(1, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(P2_ROOT / "configs" / "tse_b1_trial.yaml"))
    ap.add_argument("--manifest", default=None,
                    help="覆盖 cfg.datasets：P1 v2 传 train_manifest；P1 v3 传单 manifest.jsonl（配合 train_split/dev_split 过滤）")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="覆盖 cfg.steps（空跑测试用）")
    ap.add_argument("--debug_data", action="store_true", help="用 P2-07 DEBUG 集空跑，非正式 B1/B2/B3")
    ap.add_argument("--resume", default=None, help="从 checkpoint 恢复续训")
    ap.add_argument("--init_checkpoint", default=None,
                    help="从已有 checkpoint 热启动（仅加载 model 权重，不加载 optimizer/scaler/step，用于 B3 从 B1 预训练权重起步）")
    ap.add_argument("--scene_mode", default=None, choices=["b1", "b2", "b3"],
                    help="场景模式覆盖配置：b1=PRESENT, b2=ABSENT, b3=ENROLL-SWAP")
    ap.add_argument("--train_split", default=None,
                    help="P1 v3 过滤 e['split']=此值作为 train（例：train / D_absent / D_swap）")
    ap.add_argument("--dev_split", default=None,
                    help="P1 v3 过滤 e['split']=此值作为 dev（例：dev）")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

    tag = cfg.get("tag", "B1_TRIAL")
    out_dir = Path(args.out) if args.out else P2_ROOT / "artifacts" / "experiments" / tag
    if args.resume:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    LOG.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    LOG.addHandler(fh)
    LOG.addHandler(ch)

    LOG.info("config=%s device=%s seed=%d amp=%s sv_mode=%s", args.config, device, seed, use_amp, cfg.get("sv_mode", "bootstrap"))
    LOG.info("torch=%s cuda=%s", torch.__version__, torch.cuda.is_available())

    adapter = EnrollmentAdapter.from_config(cfg)
    LOG.info("EnrollmentAdapter mode=%s emb_dim=%d", adapter.mode, adapter.emb_dim)

    shutil.copy(args.config, out_dir / "config.yaml")
    (out_dir / "config.sha256").write_text(sha256_file(out_dir / "config.yaml") + "\n", encoding="utf-8")

    if args.debug_data:
        train_manifest = str(DEBUG_MANIFEST)
        dev_manifest = str(DEBUG_MANIFEST)
        train_split = None
        dev_split = None
        LOG.warning("--debug_data 模式：用 P2-07 DEBUG 集空跑，非正式 B1 训练")
    else:
        d = cfg.get("datasets", {})
        # ---- 单 manifest + split 过滤（P1 v3：manifest.jsonl 内含 59K，split 字段分 train/dev）----
        single_mf = args.manifest or d.get("manifest")
        if single_mf:
            if not Path(single_mf).is_absolute():
                single_mf = str(P2_ROOT / single_mf)
            train_manifest = single_mf
            dev_manifest = single_mf
            # CLI > cfg > 默认（train/dev）
            train_split = args.train_split or d.get("train_split") or "train"
            dev_split = args.dev_split or d.get("dev_split") or "dev"
            LOG.info("P1 v3 单 manifest 模式：manifest=%s train_split=%s dev_split=%s",
                     Path(single_mf).name, train_split, dev_split)
        else:
            # ---- 双 manifest（P1 v2 / 旧模式：train.jsonl + dev.jsonl 分开）----
            train_manifest = args.manifest or d["train_manifest"]
            dev_manifest = d["dev_manifest"]
            if not Path(train_manifest).is_absolute():
                train_manifest = str(P2_ROOT / train_manifest)
            if not Path(dev_manifest).is_absolute():
                dev_manifest = str(P2_ROOT / dev_manifest)
            train_split = args.train_split or d.get("train_split")
            dev_split = args.dev_split or d.get("dev_split")

    # ---- 先构建 dataset（内部会按 split+scenario 过滤，所以 len 才是真实样本数）----
    train_ds = B1Dataset(train_manifest, cfg, seed=seed, adapter=adapter,
                         scene_mode=args.scene_mode, split_name=train_split)
    dev_ds = B1Dataset(dev_manifest, cfg, seed=seed + 1, adapter=adapter,
                       scene_mode=args.scene_mode, split_name=dev_split)
    batch_size = int(cfg["batch_size"])
    n_train_samples = len(train_ds)
    if n_train_samples == 0:
        LOG.error("train dataset 为空 0 条！请检查：scene_mode=%s 是否过滤了全部样本，或 manifest=%s 是否包含对应 split/scenario。",
                  train_ds.scene_mode, train_manifest)
        raise RuntimeError(f"train dataset 为空 (scene_mode={train_ds.scene_mode})")
    if len(dev_ds) == 0:
        LOG.warning("dev dataset 为空 0 条（跳过 dev 评测）。scene_mode=%s split=%s",
                    dev_ds.scene_mode, dev_split or "-")
    if batch_size > n_train_samples:
        LOG.warning("batch_size=%d > 样本数=%d，降为 %d", batch_size, n_train_samples, n_train_samples)
        batch_size = n_train_samples
    drop_last = len(train_ds) >= batch_size
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=bool(cfg["shuffle"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]) and device.type == "cuda",
        prefetch_factor=int(cfg["prefetch_factor"]) if int(cfg["num_workers"]) > 0 else None,
        collate_fn=collate_fn, drop_last=drop_last)
    dev_loader = data.DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    LOG.info("scene_mode=%s split=[train=%s,dev=%s] train=%d 条 dev=%d 条 batch=%d seg=%ds drop_last=%s",
             train_ds.scene_mode, train_split or "-", dev_split or "-",
             len(train_ds), len(dev_ds), batch_size, int(cfg["segment_length"]), drop_last)

    with open(out_dir / "data.sha256", "w", encoding="utf-8") as f:
        f.write(f"{sha256_file(train_manifest)}  train_manifest.jsonl  split={train_split or '-'}\n")
        f.write(f"{sha256_file(dev_manifest)}    dev_manifest.jsonl  split={dev_split or '-'}\n")

    model = DualOutputTSE(cfg).to(device)
    LOG.info("参数量 %d", sum(p.numel() for p in model.parameters()))

    # ---- Warm-start: 从已有 checkpoint 加载 model 权重（仅权重，不含 optimizer/scaler/step）----
    if args.init_checkpoint:
        init_ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        init_state = init_ckpt["model"] if "model" in init_ckpt else init_ckpt
        missing, unexpected = model.load_state_dict(init_state, strict=False)
        if missing:
            LOG.warning("init_checkpoint 缺失参数: %s", missing[:5])
        if unexpected:
            LOG.warning("init_checkpoint 多余参数: %s", unexpected[:5])
        LOG.info("Warm-start 从 %s 加载权重（step=%s）", args.init_checkpoint, init_ckpt.get("step", "?"))

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except AttributeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = int(args.max_steps if args.max_steps else cfg["steps"])
    warmup = int(cfg["lr_warmup_steps"])
    grad_accum = int(cfg["gradient_accumulation"])
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    save_every = int(cfg["save_every"])
    if save_every > total_steps:
        save_every = max(1, total_steps)
        LOG.warning("save_every > total_steps，降为 %d（空跑模式）", save_every)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if use_amp and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"]) + 1
        LOG.info("从 step %d 恢复（ckpt=%s）", start_step, args.resume)

    metrics_f = open(out_dir / "metrics.jsonl", "a" if args.resume else "w", encoding="utf-8")
    dev_metrics_f = open(out_dir / "dev_metrics.jsonl", "a" if args.resume else "w", encoding="utf-8")
    nan_steps = 0
    si_sdr_mix_baseline = None
    t_train0 = time.time()

    model.train()
    loader_iter = iter(train_loader)

    for step in range(start_step, total_steps + 1):
        t_step0 = time.time()
        t_data0 = time.time()
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        data_time = time.time() - t_data0

        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}

        if si_sdr_mix_baseline is None:
            si_sdr_mix_baseline = float(si_sdr(batch["mix"], batch["target"]).item())
            LOG.info("基线 SI-SDR(mixture)=%.2f dB", si_sdr_mix_baseline)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(batch["mix"], batch["emb"])
            # step 0 诊断：打印 batch 与 model 输出各张量形状
            #   (用于定位云端 PyTorch 版本差异导致的 STFT 帧数不一致问题)
            if step == 0:
                try:
                    s_tgt_0, s_res_0, p_tgt_0 = out
                    LOG.info(
                        "[SHAPE step0] PyTorch=%s device=%s | "
                        "mix=%s target=%s frame_act(label)=%s | "
                        "s_tgt=%s s_res=%s p_tgt(model)=%s | "
                        "seg_samples(cfg)=%d n_fft=%d hop=%d win=%d sr=%d → "
                        "理论 STFT 帧数(center=True)=%d  frame_activity 期望帧数=%d",
                        torch.__version__, str(batch["mix"].device),
                        tuple(batch["mix"].shape), tuple(batch["target"].shape),
                        tuple(batch["frame_act"].shape),
                        tuple(s_tgt_0.shape), tuple(s_res_0.shape), tuple(p_tgt_0.shape),
                        int(cfg["segment_length"] * cfg["sample_rate"]),
                        int(cfg["n_fft"]), int(cfg["hop_length"]), int(cfg["win_length"]),
                        int(cfg["sample_rate"]),
                        int(cfg["segment_length"] * cfg["sample_rate"]) // int(cfg["hop_length"]) + 1,
                        int(cfg["segment_length"] * cfg["sample_rate"]) // int(cfg["hop_length"]) + 1,
                    )
                except Exception as _ex:
                    LOG.warning("[SHAPE step0] 打印失败: %s", _ex)
            total, terms = compute_losses(cfg, out, batch)

        grad_norm, clipped = 0.0, False
        if step > 0:
            if not torch.isfinite(total):
                nan_steps += 1
                LOG.warning("step=%d 非有限损失，跳过本步", step)
            else:
                scaler.scale(total / grad_accum).backward()
                if step % grad_accum == 0:
                    scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), grad_clip).item()
                    clipped = grad_norm > grad_clip
                    cur_lr = get_lr(step, float(cfg["lr"]), warmup,
                                    total_steps, cfg["lr_schedule"])
                    for pg in opt.param_groups:
                        pg["lr"] = cur_lr
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

        step_time = time.time() - t_step0
        cur_lr = get_lr(step, float(cfg["lr"]), warmup, total_steps, cfg["lr_schedule"])

        rec = {
            "step": step,
            "total": float(total.detach().cpu()),
            "si_sdr_db": float(terms["si_sdr_db"].detach().cpu()),
            "wav_l1": float(terms["wav_l1"].detach().cpu()),
            "mrstft": float(terms["mrstft"].detach().cpu()),
            "act_bce": float(terms["act_bce"].detach().cpu()),
            "res_l1": float(terms["res_l1"].detach().cpu()),
            "mix": float(terms["mix"].detach().cpu()),
            "lr": cur_lr,
            "grad_norm": grad_norm,
            "clipped": clipped,
            "nan": not bool(torch.isfinite(total)),
            "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3)
                          if device.type == "cuda" else 0.0,
            "step_time_ms": step_time * 1000,
            "data_time_ms": data_time * 1000,
            "data_wait_pct": (data_time / step_time * 100) if step_time > 0 else 0,
            "samples_per_sec": (batch_size * grad_accum / step_time) if step_time > 0 else 0,
            "effective_batch": batch_size * grad_accum,
            "si_sdri_db": float(terms["si_sdr_db"].detach().cpu()) - (si_sdr_mix_baseline or 0),
        }
        metrics_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        metrics_f.flush()

        if step % log_every == 0 or step == total_steps:
            LOG.info("step=%d/%d total=%.4f si_sdr=%.2f si_sdri=%.2f lr=%.2e |g|=%.3f "
                     "mem=%.2fGB %.0fms/step data=%.0f%%",
                     step, total_steps, rec["total"], rec["si_sdr_db"], rec["si_sdri_db"],
                     cur_lr, grad_norm, rec["gpu_mem_gb"], rec["step_time_ms"],
                     rec["data_wait_pct"])

        if step > 0 and step % save_every == 0:
            ckpt_path = out_dir / f"checkpoint_step{step}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "step": step,
                "cfg": cfg,
                "seed": seed,
            }, ckpt_path)
            (out_dir / f"checkpoint_step{step}.sha256").write_text(
                sha256_file(ckpt_path) + "\n", encoding="utf-8")
            LOG.info("存 checkpoint: %s", ckpt_path.name)

            dev_rec = {"step": step, **evaluate_dev(model, dev_loader, cfg, device, use_amp)}
            dev_metrics_f.write(json.dumps(dev_rec, ensure_ascii=False) + "\n")
            dev_metrics_f.flush()
            LOG.info("  dev_loss=%.4f dev_sisdr=%.2f dB",
                     dev_rec["dev_loss"], dev_rec["dev_sisdr"])

    metrics_f.close()
    dev_metrics_f.close()
    train_time = time.time() - t_train0

    last_ckpt = out_dir / f"checkpoint_step{total_steps}.pt"
    restore_ok = False
    restore_diff = float("inf")
    if last_ckpt.exists():
        model2 = DualOutputTSE(cfg).to(device)
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model2.load_state_dict(ckpt["model"])
        model2.eval()
        with torch.no_grad():
            out1 = model(batch["mix"], batch["emb"])
            out2 = model2(batch["mix"], batch["emb"])
        restore_diff = max((out1[i] - out2[i]).abs().max().item() for i in range(3))
        restore_ok = restore_diff < float(cfg.get("restore_tol", 1e-5))
        LOG.info("恢复一致性 max|Δ|=%.3e → %s", restore_diff,
                 "PASS" if restore_ok else "FAIL")

    all_metrics = [json.loads(l) for l in open(out_dir / "metrics.jsonl", encoding="utf-8")]
    train_metrics = [m for m in all_metrics if m["step"] > 0]

    if total_steps >= 200:
        first_window = [m for m in train_metrics if m["step"] <= 100]
        last_window = [m for m in train_metrics if m["step"] > total_steps - 100]
    else:
        mid = max(1, total_steps // 2)
        first_window = [m for m in train_metrics if m["step"] <= mid]
        last_window = [m for m in train_metrics if m["step"] > mid]
    loss_first = float(np.mean([m["total"] for m in first_window])) if first_window else 0.0
    loss_last = float(np.mean([m["total"] for m in last_window])) if last_window else 0.0
    loss_decreasing = loss_last < loss_first

    peak_mem = max((m["gpu_mem_gb"] for m in all_metrics), default=0.0)
    mean_sps = float(np.mean([m["samples_per_sec"] for m in train_metrics])) if train_metrics else 0.0
    p50_step = float(np.percentile([m["step_time_ms"] for m in train_metrics], 50)) if train_metrics else 0.0
    p95_step = float(np.percentile([m["step_time_ms"] for m in train_metrics], 95)) if train_metrics else 0.0
    mean_data_wait = float(np.mean([m["data_wait_pct"] for m in train_metrics])) if train_metrics else 0.0

    epoch_steps = max(1, len(train_ds) // batch_size)
    est_epoch_time = epoch_steps * p50_step / 1000.0

    verdicts = {
        "no_nan": nan_steps == 0,
        "grad_finite": all(m["grad_norm"] >= 0 and np.isfinite(m["grad_norm"])
                           for m in train_metrics) if train_metrics else False,
        "peak_mem_under_budget": peak_mem < 4.0,
        "throughput_measured": mean_sps > 0,
        "loss_decreasing": bool(loss_decreasing),
        "checkpoint_restore_ok": restore_ok,
        "full_train_time_estimated": train_time > 0,
    }
    must_pass = ["no_nan", "grad_finite", "peak_mem_under_budget",
                 "throughput_measured", "checkpoint_restore_ok", "full_train_time_estimated"]
    overall = "PASS" if all(verdicts[k] for k in must_pass) else "FAIL"

    summary = {
        "config": args.config,
        "device": str(device),
        "torch_version": torch.__version__,
        "amp": use_amp,
        "batch_size": batch_size,
        "segment_length_sec": int(cfg["segment_length"]),
        "total_steps": total_steps,
        "train_time_sec": float(train_time),
        "n_train_samples": len(train_ds),
        "n_nan_steps": nan_steps,
        "peak_gpu_mem_gb": float(peak_mem),
        "mem_budget_gb": 4.0,
        "mem_margin_gb": float(4.0 - peak_mem),
        "samples_per_sec_mean": mean_sps,
        "step_time_ms_p50": p50_step,
        "step_time_ms_p95": p95_step,
        "data_wait_pct_mean": mean_data_wait,
        "loss_first_100_mean": loss_first,
        "loss_last_100_mean": loss_last,
        "loss_decreasing": bool(loss_decreasing),
        "restore_diff": float(restore_diff),
        "restore_ok": restore_ok,
        "est_epoch_time_sec": float(est_epoch_time),
        "est_epoch_steps": int(epoch_steps),
        "verdicts": verdicts,
        "verdict_overall": overall,
    }
    (out_dir / "trial_verdict.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(f"# P2-12 B1 {total_steps}-step 试车报告"
                f"（{cfg.get('asset_class', 'B1_TRIAL')}）\n\n")
        f.write(f"- 日期: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 设备: {device}（torch {torch.__version__}）\n")
        f.write(f"- 配置: `{Path(args.config).name}`\n")
        f.write(f"- 数据: train={len(train_ds)} 条（manifest={Path(train_manifest).name}）\n")
        f.write(f"- AMP: {use_amp}；batch={batch_size}；seg={cfg['segment_length']}s；"
                f"accum={grad_accum}；effective_batch={batch_size * grad_accum}\n\n")
        f.write("## 吞吐量\n\n")
        f.write(f"- samples/sec: {mean_sps:.2f}\n")
        f.write(f"- step time P50: {p50_step:.1f} ms / P95: {p95_step:.1f} ms\n")
        f.write(f"- 数据等待占比: {mean_data_wait:.1f}%\n")
        f.write(f"- 训练总耗时: {train_time:.1f}s（{total_steps} 步）\n\n")
        f.write("## 显存\n\n")
        f.write(f"- 峰值显存: {peak_mem:.3f} GB\n")
        f.write(f"- 预算: 4.0 GB；余量: {4.0 - peak_mem:.3f} GB\n\n")
        f.write("## 收敛\n\n")
        f.write(f"- total loss 首 100 步均值: {loss_first:.4f}\n")
        f.write(f"- total loss 末 100 步均值: {loss_last:.4f}\n")
        f.write(f"- loss 下降: {'是' if loss_decreasing else '否'}\n")
        f.write(f"- NaN step 数: {nan_steps}\n\n")
        f.write("## 恢复\n\n")
        f.write(f"- checkpoint: {last_ckpt.name if last_ckpt.exists() else 'N/A'}\n")
        f.write(f"- 恢复一致性 max|Δ|: {restore_diff:.3e}\n")
        f.write(f"- 判定: {'PASS' if restore_ok else 'FAIL'}\n\n")
        f.write("## 完整训练预估\n\n")
        f.write(f"- 1 epoch 步数: {epoch_steps}\n")
        f.write(f"- 1 epoch 预估时长: {est_epoch_time:.1f}s"
                f"（{est_epoch_time / 60:.1f} 分钟）\n")
        f.write(f"- 100 epoch 预估: {est_epoch_time * 100 / 3600:.2f} 小时\n\n")
        f.write("## 判定\n\n| 项 | 结果 |\n|---|---|\n")
        for k, v in verdicts.items():
            f.write(f"| {k} | {'PASS' if v else 'FAIL'} |\n")
        f.write(f"\n**总体判定: {overall}**\n")

    LOG.info("产物目录: %s", out_dir)
    LOG.info("总体判定: %s", overall)
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())