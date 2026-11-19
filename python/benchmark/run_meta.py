"""Метаданные прогона: окружение, железо, версии, хэши (воспроизводимость, CLAUDE.md)."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

KEY_PACKAGES = (
    "torch", "torchvision", "numpy", "opencv-python-headless", "pillow", "pycocotools",
    "hydra-core", "iopath", "timm", "onnx", "onnxruntime-gpu",
)  # fmt: skip


def file_md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _cmd(args: list[str]) -> str:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30, check=False)
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def collect(root: Path) -> dict:
    versions = {}
    for name in KEY_PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return {
        "utc_start": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": _cmd(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "packages": versions,
        "all_packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()),
    }
