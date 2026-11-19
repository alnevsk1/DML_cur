"""Сборка PVS-моделей семейства SAM 2 в едином режиме для бенчмарка.

Единые условия для всех моделей:
  * eager-режим: `compile_image_encoder=false` (у EfficientTAM в конфиге по умолчанию true,
    у SAM 2.1 и EdgeTAM — false; компиляция сделала бы скорость несопоставимой);
  * `fill_hole_area=0`: CUDA-расширение `_C` не собираем, C++/TensorRT-версия этот шаг
    тоже не делает;
  * остальная постобработка — как в официальных build_*_video_predictor (apply_postprocessing).
EdgeTAM: `TimmBackbone` жёстко создаёт RepViT с `pretrained=True` (качает веса ImageNet
с HF Hub); эти веса всё равно перезаписываются чекпойнтом EdgeTAM, поэтому create_model
подменяется на вызов с pretrained=False. Загрузка чекпойнта строгая
(missing/unexpected keys -> исключение в _load_checkpoint).

Каждая модель — в отдельном процессе с PYTHONPATH=third_party/<repo>
(EdgeTAM и SAM 2 оба — пакет `sam2`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "checkpoints"


@dataclass(frozen=True)
class PVSModel:
    repo: str  # каталог в python/third_party/
    module: str
    builder: str
    config: str
    checkpoint: Path


_SAM2 = ("sam2.build_sam", "build_sam2_video_predictor")
_ETAM = ("efficient_track_anything.build_efficienttam", "build_efficienttam_video_predictor")

MODELS: dict[str, PVSModel] = {
    "sam2.1_tiny": PVSModel(
        "sam2", *_SAM2, "configs/sam2.1/sam2.1_hiera_t.yaml", CKPT / "sam2" / "sam2.1_hiera_tiny.pt"
    ),
    "sam2.1_small": PVSModel(
        "sam2", *_SAM2, "configs/sam2.1/sam2.1_hiera_s.yaml",
        CKPT / "sam2" / "sam2.1_hiera_small.pt",
    ),
    "efficienttam_ti": PVSModel(
        "EfficientTAM", *_ETAM, "configs/efficienttam/efficienttam_ti.yaml",
        CKPT / "efficienttam" / "efficienttam_ti.pt",
    ),
    "efficienttam_s": PVSModel(
        "EfficientTAM", *_ETAM, "configs/efficienttam/efficienttam_s.yaml",
        CKPT / "efficienttam" / "efficienttam_s.pt",
    ),
    "edgetam": PVSModel(
        "EdgeTAM", *_SAM2, "configs/edgetam.yaml", CKPT / "edgetam" / "edgetam.pt"
    ),
}  # fmt: skip


def _disable_timm_pretrained_download() -> None:
    timm_backbone = importlib.import_module("sam2.modeling.backbones.timm")
    original = timm_backbone.create_model

    def create_model_no_download(*args, **kwargs):  # noqa: ANN002, ANN003 — прокси к timm.create_model
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    timm_backbone.create_model = create_model_no_download


def build_predictor(name: str, device: str = "cuda") -> torch.nn.Module:
    spec = MODELS[name]
    package = importlib.import_module(spec.module.split(".")[0])
    origin = Path(package.__file__).resolve().as_posix()
    if f"third_party/{spec.repo}/" not in origin:
        raise RuntimeError(f"{name}: wrong package on PYTHONPATH: {origin}")
    if spec.repo == "EdgeTAM":
        _disable_timm_pretrained_download()
    build = getattr(importlib.import_module(spec.module), spec.builder)
    predictor = build(
        spec.config,
        str(spec.checkpoint),
        device=device,
        hydra_overrides_extra=["++model.compile_image_encoder=false"],
    )
    predictor.fill_hole_area = 0
    return predictor
