"""Маски COCO RLE (формат AeroVIS) через pycocotools.

Собственный декодер, использовавшийся до установки pycocotools, сверен с ним 2026-10-22:
площади совпали на всех 1 405 752 кадрах треков AeroVIS, маски целиком — на 3 200 из 3 200
(200 кадров инициализации выборки v1 + 3 000 случайных). Поле `areas` из JSON AeroVIS
не используем: в SeaDronesSee-части оно не совпадает с масками (docs/datasets.md).
"""

from __future__ import annotations

import numpy as np
import pycocotools.mask as mask_utils


def _to_coco(rle: dict) -> dict:
    """pycocotools ждёт сжатую строку counts как bytes."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    return {"size": list(rle["size"]), "counts": counts}


def rle_area(rle: dict) -> int:
    """Площадь маски в пикселях без декодирования."""
    return int(mask_utils.area(_to_coco(rle)))


def decode_rle(rle: dict) -> np.ndarray:
    """RLE {'size': [h, w], 'counts': str | bytes} -> маска uint8 (H, W), C-порядок."""
    return np.ascontiguousarray(mask_utils.decode(_to_coco(rle)))


def encode_mask(mask: np.ndarray) -> dict:
    """Бинарная маска (H, W) -> RLE с counts в виде str (сериализуется в JSON)."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}
