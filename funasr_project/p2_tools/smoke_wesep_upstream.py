# -*- coding: utf-8 -*-
"""P2-05 WeSep 上游冒烟：证明冻结 commit 在本机环境可 import、可加载、可执行最小前向。

证据：reports/upstream/wesep_smoke.json
- import 版本（torch/torchaudio/wesep 路径）
- 小配置 ConvTasNet 实例化（参数量）
- 两种前向约定逐一尝试：A) 外部 spk embedding 向量；B) 注册波形联合编码
- CPU + CUDA 双端（CUDA 可用时）
失败不硬修：完整 traceback 存入 JSON，按手册建兼容异常单。

运行：
  .\\.venv-p2tse\\Scripts\\python.exe p2_tools/smoke_wesep_upstream.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "wesep"))
sys.path.insert(0, str(ROOT / "third_party" / "wespeaker"))

OUT = ROOT / "reports" / "upstream" / "wesep_smoke.json"

# 上游硬编码约束（已读源码确认）：
# - ResNet4SpExplus 内部 Conv1D(3*256,256,1) 硬编码 → joint_training 路径必须 N=256
# - 说话人融合层按 spk_emb_dim=256 构建 → 外部 embedding 必须 256 维
# 规模仅验证可加载/可前向，不代表正式 B1 模型
SMALL_CFG = dict(N=256, L=16, B=128, H=512, P=3, X=8, R=3, spk_emb_dim=256)
T_SAMPLES = 8000  # 0.5s @16k


def try_forward(model, mode, device):
    import torch
    x = torch.randn(1, T_SAMPLES, device=device)
    if mode == "A_external_emb":
        model.eval()
        with torch.no_grad():
            emb = torch.randn(1, SMALL_CFG["spk_emb_dim"], device=device)
            y = model(x, emb)
    else:  # B_enroll_wav（joint_training 内部编码注册波形）
        model.eval()
        with torch.no_grad():
            aux = torch.randn(1, T_SAMPLES, device=device)
            y = model(x, aux)
    return y


def main():
    report = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "attempts": []}
    try:
        import torch
        import torchaudio
        import wesep
        from wesep.models import get_model
        report["import"] = {
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "wesep_file": wesep.__file__,
            "status": "OK",
        }
    except Exception:
        report["import"] = {"status": "FAIL", "traceback": traceback.format_exc()}
        OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("IMPORT_FAIL →", OUT)
        return 1

    print(f"import OK: torch={report['import']['torch']} torchaudio={report['import']['torchaudio']}")

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    overall_pass = False
    for joint_training, mode in [(False, "A_external_emb"), (True, "B_enroll_wav")]:
        for device in devices:
            rec = {"mode": mode, "joint_training": joint_training, "device": device}
            try:
                cls = get_model("ConvTasNet")
                model = cls(**SMALL_CFG, joint_training=joint_training).to(device)
                rec["n_params"] = sum(p.numel() for p in model.parameters())
                y = try_forward(model, mode, device)
                if isinstance(y, (list, tuple)):
                    rec["out_shapes"] = [list(t.shape) for t in y if hasattr(t, "shape")]
                    t0 = y[0]
                else:
                    rec["out_shapes"] = [list(y.shape)]
                    t0 = y
                rec["finite"] = bool(torch.isfinite(t0).all().item())
                rec["gpu_mem_gb"] = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else 0.0
                rec["status"] = "PASS" if rec["finite"] else "FAIL_NONFINITE"
                overall_pass = overall_pass or (rec["status"] == "PASS")
            except Exception:
                rec["status"] = "FAIL"
                rec["traceback"] = traceback.format_exc().splitlines()[-6:]
            report["attempts"].append(rec)
            print(f"{mode} @ {device}: {rec['status']}"
                  + (f" out={rec.get('out_shapes')} params={rec.get('n_params')}" if rec["status"] == "PASS" else ""))
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

    report["overall"] = "PASS" if overall_pass else "FAIL"
    report["note"] = ("上游冒烟只验证 import/加载/最小前向；"
                      "A/B 任一约定任一设备 PASS 即 PASS（另一约定失败入 traceback 备查）")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("overall:", report["overall"], "→", OUT)
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
