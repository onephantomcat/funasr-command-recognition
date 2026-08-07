"""CAMPPlus 预训练权重下载脚本。

下载 iic/speech_campplus_sv_zh-cn_16k-common 权重到 P4_project/artifacts/models/。

使用方式:
    python P4_project/tools/download_campplus_weights.py
    python P4_project/tools/download_campplus_weights.py --force  # 强制覆盖
    python P4_project/tools/download_campplus_weights.py --model-id <ID>  # 自定义模型
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
MODELS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "models"
TARGET_PT = "campplus_sv_zh-cn_16k-common.pt"
MODELSCOPE_BASE = "https://www.modelscope.cn/api/v1/models"


def download_with_modelscope(model_id: str, dest: Path) -> bool:
    """优先使用 modelscope 库下载（断点续传 + 缓存）。"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        return False

    print(f"[modelscope] 下载 {model_id} ...")
    try:
        cache_dir = snapshot_download(
            model_id=model_id,
            cache_dir=str(dest.parent / ".modelscope_cache"),
        )
        # 在缓存中查找 .pt 文件
        cache_path = Path(cache_dir)
        pt_files = list(cache_path.rglob("*.pt"))
        if pt_files:
            src = pt_files[0]
            shutil.copy2(src, dest)
            print(f"[modelscope] 已保存: {dest}")
            return True

        # 如果是 .tar.gz 压缩包
        tar_files = list(cache_path.rglob("*.tar.gz"))
        if tar_files:
            extract_tar(tar_files[0], dest.parent)
            return True

        print(f"[modelscope] 下载成功但未找到 .pt 文件，缓存内容:")
        for f in cache_path.rglob("*"):
            print(f"  {f}")
        return False
    except Exception as e:
        print(f"[modelscope] 下载失败: {e}")
        return False


def download_with_http(model_id: str, dest: Path) -> bool:
    """备选：直接 HTTP 下载（从 ModelScope 开放接口）。"""
    # ModelScope 模型文件下载 URL 格式
    file_url = f"{MODELSCOPE_BASE}/{model_id}/resolve/master/{TARGET_PT}"
    print(f"[http] 尝试下载: {file_url}")

    try:
        req = urllib.request.Request(file_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                if total:
                    downloaded = 0
                    block = 8192
                    while True:
                        chunk = resp.read(block)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = downloaded * 100 // total
                        bar = "#" * (pct // 2) + "-" * (50 - pct // 2)
                        print(f"\r  [{bar}] {pct}%", end="", flush=True)
                    print()
                else:
                    shutil.copyfileobj(resp, f)
        print(f"[http] 已保存: {dest}")
        return True
    except Exception as e:
        print(f"[http] 下载失败: {e}")
        return False


def download_with_msdl(model_id: str, dest: Path) -> bool:
    """备选：使用 modelscope-sdk CLI（msdl）。"""
    try:
        import subprocess
        result = subprocess.run(
            ["msdl", "download", model_id, "--local_dir", str(dest.parent)],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"[msdl] 下载成功")
            return True
        print(f"[msdl] 失败: {result.stderr}")
        return False
    except Exception as e:
        print(f"[msdl] 不可用: {e}")
        return False


def extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """从 tar.gz 中提取 .pt 文件。"""
    print(f"[extract] 解压 {tar_path} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)
        for f in Path(tmpdir).rglob("*.pt"):
            shutil.copy2(f, dest_dir / TARGET_PT)
            print(f"[extract] 已保存: {dest_dir / TARGET_PT}")
            return
    print("[extract] 未在压缩包中找到 .pt 文件")


def try_all_methods(model_id: str, dest: Path, force: bool = False) -> bool:
    """依次尝试所有下载方式。"""
    if dest.exists() and not force:
        print(f"[skip] 权重已存在: {dest}")
        print(f"       使用 --force 覆盖")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    for method_name, method in [
        ("modelscope", download_with_modelscope),
        ("msdl", download_with_msdl),
        ("http", download_with_http),
    ]:
        print(f"\n── 尝试 {method_name} ──")
        if method(model_id, dest):
            return True

    print("\n所有下载方式均失败。请手动下载：")
    print(f"  1. 访问 https://www.modelscope.cn/models/{model_id}")
    print(f"  2. 下载 {TARGET_PT}")
    print(f"  3. 放入 {dest}")
    return False


def verify_weights(dest: Path) -> bool:
    """验证权重文件可被加载。"""
    try:
        import torch
        state = torch.load(dest, map_location="cpu", weights_only=True)
        if isinstance(state, dict):
            keys = list(state.keys())[:5]
            print(f"[verify] 权重加载成功，共 {len(state)} 个键")
            print(f"         前 5 个键: {keys}")
            return True
        else:
            print(f"[verify] 权重格式异常: {type(state)}")
            return False
    except Exception as e:
        print(f"[verify] 权重验证失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 CAMPPlus 预训练权重")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="ModelScope 模型 ID")
    parser.add_argument("--force", action="store_true", help="强制覆盖已存在的权重")
    parser.add_argument("--verify-only", action="store_true", help="仅验证已有权重，不下载")
    args = parser.parse_args()

    dest = MODELS_DIR / TARGET_PT

    if args.verify_only:
        ok = verify_weights(dest)
        sys.exit(0 if ok else 1)

    print(f"模型 ID: {args.model_id}")
    print(f"目标路径: {dest}")

    ok = try_all_methods(args.model_id, dest, force=args.force)
    if ok and dest.exists():
        verify_weights(dest)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()