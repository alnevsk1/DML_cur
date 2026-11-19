"""Маски первого кадра VisDrone-SOT val по эталонной рамке (для VOS-моделей и PVS-варианта mask).

VisDrone-SOT размечен только рамками, а XMem / Cutie / DeAOT инициализируются маской.
Маска строится один раз SAM 2.1-small (режим изображения, подсказка — эталонная рамка,
multimask_output=False — так рекомендуют авторы для однозначной подсказки) и одна и та же
подаётся всем моделям. Проверка: IoU охватывающей рамки маски с эталонной рамкой.

Выход: data/annotations/visdrone_sot_val_init_masks.json (RLE),
       reports/figures/visdrone_sot_val_init_masks.png (вырезки с маской и рамкой).
Запуск (WSL, из python/):
  PYTHONPATH=third_party/sam2 uv run --no-sync python -m tools.visdrone_init_masks
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark.pvs_models import build_predictor  # noqa: E402
from tools.rle import encode_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOT_ROOT = ROOT / "data" / "external" / "VisDrone-SOT" / "val"
OUT = ROOT / "data" / "annotations" / "visdrone_sot_val_init_masks.json"
FIG = ROOT / "reports" / "figures" / "visdrone_sot_val_init_masks.png"
MODEL = "sam2.1_small"


def first_frame_and_box(seq: str) -> tuple[np.ndarray, list[float]]:
    frame = sorted((SOT_ROOT / "sequences" / seq).glob("*.jpg"))[0]
    line = (SOT_ROOT / "annotations" / f"{seq}.txt").read_text(encoding="utf-8").splitlines()[0]
    box = [float(v) for v in line.split(",")]  # x, y, w, h
    return np.asarray(Image.open(frame).convert("RGB")), box


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU рамок x, y, w, h."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def mask_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def main() -> None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor(build_predictor(MODEL))
    seqs = sorted(p.name for p in (SOT_ROOT / "sequences").iterdir() if p.is_dir())
    out: dict[str, dict] = {}
    crops = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for seq in seqs:
            img, (x, y, w, h) = first_frame_and_box(seq)
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(
                box=np.array([x, y, x + w, y + h], dtype=np.float32), multimask_output=False
            )
            mask = masks[0] > 0
            mb = mask_box(mask) if mask.any() else [0.0, 0.0, 0.0, 0.0]
            out[seq] = {
                "mask": encode_mask(mask),
                "gt_box_xywh": [x, y, w, h],
                "mask_box_xywh": mb,
                "mask_box_iou": box_iou(mb, [x, y, w, h]) if mask.any() else 0.0,
                "mask_area_px": int(mask.sum()),
                "score": float(scores[0]),
            }
            r = out[seq]
            print(
                f"{seq}: box {w:.0f}x{h:.0f}, mask {r['mask_area_px']} px, "
                f"bbox IoU {r['mask_box_iou']:.3f}, score {r['score']:.3f}",
                flush=True,
            )
            pad = max(w, h)
            x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
            x1, y1 = int(min(img.shape[1], x + w + pad)), int(min(img.shape[0], y + h + pad))
            crops.append((seq, img[y0:y1, x0:x1], mask[y0:y1, x0:x1], (x - x0, y - y0, w, h)))
    meta = {"model": MODEL, "prompt": "gt box, frame 1", "multimask_output": False}
    OUT.write_text(json.dumps({"meta": meta, "sequences": out}), encoding="utf-8")

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (seq, crop, m, (bx, by, bw, bh)) in zip(axes.flat, crops, strict=False):
        ax.imshow(crop)
        ax.imshow(np.ma.masked_where(~m, m), alpha=0.5, cmap="autumn")
        ax.add_patch(plt.Rectangle((bx, by), bw, bh, fill=False, color="lime", lw=1.5))
        ax.set_title(f"{seq}\nIoU рамок {out[seq]['mask_box_iou']:.2f}", fontsize=8)
    fig.suptitle("VisDrone-SOT val: маска SAM 2.1-small по эталонной рамке (зелёная), кадр 1")
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120, bbox_inches="tight")
    print(f"saved {OUT} and {FIG}")


if __name__ == "__main__":
    main()
