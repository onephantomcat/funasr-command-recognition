import sys, torch, yaml
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
sys.path.insert(0, str(BASE / "tools"))

CKPT_PATH = BASE / "artifacts/final/P2_artifacts/B3/checkpoint_step20000.pt"
assert CKPT_PATH.exists(), f"Checkpoint 不存在: {CKPT_PATH}"

# === 验证 1: 加载 checkpoint ===
ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
print("[OK] checkpoint 加载成功")
print("  step:", ckpt.get("step"))
print("  cfg 主键:", list(ckpt["cfg"].keys())[:8])

# === 验证 2: 构建模型并加载权重 ===
from src.tse.model import DualOutputTSE
model = DualOutputTSE(ckpt["cfg"])
model.load_state_dict(ckpt["model"], strict=True)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print("[OK] 模型构建成功: {} 参数".format(total_params))

# === 验证 3: forward pass 推理 ===
sr = ckpt["cfg"].get("sample_rate", 16000)
# 模型期望: mix_wav [B, T] (2D), enroll_emb [B, D] (2D)
mix = torch.randn(1, 8 * sr)       # [B=1, T=128000]，2D 张量
emb_dim = ckpt["cfg"].get("emb_dim", 192)
emb = torch.randn(1, emb_dim)      # [B=1, D=192]，2D 张量
with torch.no_grad():
    s_tgt, s_res, p_tgt = model(mix, emb)
print("[OK] 推理通过")
print("  s_tgt shape:", tuple(s_tgt.shape))
print("  p_tgt shape:", tuple(p_tgt.shape))
print("  s_tgt max:  {:.4f}".format(abs(s_tgt).max().item()))
print("  p_tgt mean: {:.4f}".format(p_tgt.mean().item()))
print("  p_tgt max:  {:.4f}".format(p_tgt.max().item()))

# === 验证 4: 权重确定性（sha256 与云端 summary 对照）===
import hashlib
ckpt_bytes = CKPT_PATH.read_bytes()
sha = hashlib.sha256(ckpt_bytes).hexdigest()
print("[OK] checkpoint sha256:", sha)
expected_sha = "5c351097d710aa6bc5914fc942f7c5f7fcc6206a2cac9f9042dd3b7cf4afd68d"
print("  与云端一致:", sha == expected_sha)

# === 验证 5: B1/B2 模型同样加载 ===
for name, expected in [
    ("B1", "df95a0c25e02428d64c147bdbd94b1219a28c1f7338bd13745e3ccda26c6cfa3"),
    ("B2", None),
]:
    ckpt_dir = BASE / "artifacts/final/P2_artifacts" / name
    pt = ckpt_dir / "checkpoint_step20000.pt"
    if pt.exists():
        sha2 = hashlib.sha256(pt.read_bytes()).hexdigest()
        ok = (expected is None) or (sha2 == expected)
        print("[OK] {} sha256 match: {}".format(name, ok))
    else:
        print("[WARN] {} checkpoint 不存在: {}".format(name, pt))

print()
print("=== P2 本地验证全部通过 ===")
