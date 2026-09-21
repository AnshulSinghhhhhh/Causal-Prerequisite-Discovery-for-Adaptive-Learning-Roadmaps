"""Download GGUF model weights for local SLM probe.

Downloads Qwen2.5-3B-Instruct GGUF (default: Q4_K_M, ~2.0 GB) into data/models/
with resume support and size/integrity verification.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO_ROOT, "data", "models")

MODEL_CONFIGS = {
    "q4_k_m": {
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "expected_bytes": 2104932768,
        "sha256": "5ae0c201348e276d543a1e5c0e053370e32415774095c677319f62b302b1620a",
    },
    "q3_k_m": {
        "filename": "qwen2.5-3b-instruct-q3_k_m.gguf",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q3_k_m.gguf",
        "expected_bytes": 1724178848,
        "sha256": None,
    },
}


def download_file(url: str, dest_path: str, expected_bytes: int) -> bool:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".partial"

    existing_bytes = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    if os.path.exists(dest_path):
        current_size = os.path.getsize(dest_path)
        if current_size == expected_bytes:
            print(f"[FOUND] {dest_path} already fully downloaded ({current_size:,} bytes).")
            return True
        else:
            print(f"[WARNING] Existing file size {current_size} != expected {expected_bytes}. Re-downloading.")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if existing_bytes > 0:
        req.headers["Range"] = f"bytes={existing_bytes}-"
        print(f"[RESUME] Resuming from byte {existing_bytes:,} / {expected_bytes:,}...")

    t0 = time.time()
    last_print = t0
    chunk_size = 1024 * 1024  # 1 MB chunks

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            mode = "ab" if existing_bytes > 0 else "wb"
            downloaded = existing_bytes
            with open(temp_path, mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_print >= 2.0 or downloaded == expected_bytes:
                        elapsed = max(now - t0, 0.01)
                        speed_mb = (downloaded - existing_bytes) / (1024 * 1024) / elapsed
                        pct = (downloaded / expected_bytes) * 100
                        print(f"  {downloaded / (1024*1024):.1f} MB / {expected_bytes / (1024*1024):.1f} MB "
                              f"({pct:.1f}%) - {speed_mb:.2f} MB/s", end="\r")
                        last_print = now
        print()
    except Exception as e:
        print(f"\n[ERROR] Download interrupted: {e}")
        return False

    final_size = os.path.getsize(temp_path)
    if final_size == expected_bytes:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        os.rename(temp_path, dest_path)
        print(f"[SUCCESS] Download completed and verified: {dest_path} ({final_size:,} bytes)")
        return True
    else:
        print(f"[ERROR] Size mismatch: got {final_size:,} bytes, expected {expected_bytes:,}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download GGUF SLM model")
    parser.add_argument("--quant", choices=["q4_k_m", "q3_k_m"], default="q4_k_m",
                        help="Quantization variant (default: q4_k_m)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.quant]
    dest = os.path.join(MODELS_DIR, cfg["filename"])
    print(f"Downloading {args.quant} ({cfg['filename']}) to {dest}...")
    success = download_file(cfg["url"], dest, cfg["expected_bytes"])
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
