"""Push, poll, and pull Kaggle kernels from your local machine.

Usage:
    python scripts/kaggle_submit.py push
    python scripts/kaggle_submit.py status
    python scripts/kaggle_submit.py pull
"""

import sys
import time
import subprocess
from pathlib import Path

KERNEL_SLUG = "anshulsingh45/lightgap-cdp-probe-qwen"
KERNEL_DIR = Path(__file__).resolve().parents[1] / "kaggle"
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def push():
    """Push the kernel to Kaggle."""
    print(f"Pushing kernel from {KERNEL_DIR}...")
    subprocess.run(
        ["kaggle", "kernels", "push", "-p", str(KERNEL_DIR)],
        check=True,
    )
    print("Kernel pushed. Use 'status' to monitor.")


def status():
    """Check kernel execution status."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", "-k", KERNEL_SLUG],
        capture_output=True, text=True,
    )
    print(result.stdout)
    return "complete" in result.stdout.lower()


def pull():
    """Download kernel output artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Pulling output to {ARTIFACT_DIR}...")
    subprocess.run(
        ["kaggle", "kernels", "output", "-k", KERNEL_SLUG, "-p", str(ARTIFACT_DIR)],
        check=True,
    )
    print("Output downloaded.")


def poll(interval: int = 60, max_wait: int = 7200):
    """Poll until kernel completes, then pull output."""
    elapsed = 0
    while elapsed < max_wait:
        if status():
            pull()
            return
        print(f"Still running... (waited {elapsed}s)")
        time.sleep(interval)
        elapsed += interval
    print("Timeout waiting for kernel completion.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"push": push, "status": status, "pull": pull, "poll": poll}[cmd]()
