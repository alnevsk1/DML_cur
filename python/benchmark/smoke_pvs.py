"""Smoke-тест PVS-моделей семейства SAM 2 (SAM 2.1, EfficientTAM, EdgeTAM) на коротких треках.

Проверяет: сборку модели из конфига, строгую загрузку весов (missing/unexpected keys ->
исключение внутри _load_checkpoint), отключение fill_hole_area, инициализацию кликом,
IoU с эталоном AeroVIS, время и пик VRAM. Не бенчмарк: 3 коротких трека, без замеров по частям.

Каждая модель — отдельный процесс (EdgeTAM и SAM 2 оба называют пакет `sam2`):
  PYTHONPATH=third_party/sam2 uv run --no-sync python -m benchmark.smoke_pvs --model sam2.1_tiny
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark.pvs_models import MODELS, build_predictor
from tools.rle import decode_rle

ROOT = Path(__file__).resolve().parents[2]
SEQ_ROOT = ROOT / "data" / "external" / "AeroVIS" / "sequences"
SAMPLE = ROOT / "data" / "annotations" / "aerovis_sample_v1.csv"
SAMPLE_GT = ROOT / "data" / "annotations" / "aerovis_sample_v1_gt.json"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def pick_tracks(n_per_bin: int = 1, max_span: int = 60) -> pd.DataFrame:
    s = pd.read_csv(SAMPLE)
    s = s[(s["span"] <= max_span) & (s["group"] == "regular")]
    return (
        s[s["size_bin"].isin(["tiny", "medium", "large"])]
        .sort_values("ann_id")
        .groupby("size_bin")
        .head(n_per_bin)
    )


def run_track(predictor, track: pd.Series, gt: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="smoke_") as tmp:
        # Кадры трека -> 00000.jpg, 00001.jpg, ... (кадр 0 = кадр инициализации).
        for i, rel in enumerate(gt["file_names"]):
            shutil.copy(SEQ_ROOT / rel, Path(tmp) / f"{i:05d}.jpg")
        t0 = time.perf_counter()
        state = predictor.init_state(video_path=tmp, offload_video_to_cpu=True)
        t_load = time.perf_counter() - t0
        _, _, logits0 = predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1,
            points=np.array([[track.click_x, track.click_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )  # fmt: skip
        init_iou = iou((logits0[0, 0] > 0).cpu().numpy(), decode_rle(gt["segmentations"][0]) > 0)
        ious = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 0
        for f, _, logits in predictor.propagate_in_video(state):
            n += 1
            seg = gt["segmentations"][f]
            if seg is None or f == 0:
                continue
            ious.append(iou((logits[0, 0] > 0).cpu().numpy(), decode_rle(seg) > 0))
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        predictor.reset_state(state)
    return {
        "ann_id": int(track.ann_id), "size_bin": track.size_bin, "category": track.category,
        "frames": n, "init_iou": round(init_iou, 3), "mean_iou": round(float(np.mean(ious)), 3),
        "fps_propagate": round(n / dt, 1), "load_s": round(t_load, 1),
    }  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    args = ap.parse_args()
    set_seed(SEED)
    predictor = build_predictor(args.model)
    n_params = sum(p.numel() for p in predictor.parameters()) / 1e6
    compiled = type(predictor.image_encoder).__name__ != "ImageEncoder"
    print(
        f"[{args.model}] repo {MODELS[args.model].repo}, params {n_params:.1f} M, "
        f"fill_hole_area={predictor.fill_hole_area}, image_size={predictor.image_size}, "
        f"compiled_encoder={compiled}"
    )

    gts = json.loads(SAMPLE_GT.read_text(encoding="utf-8"))
    torch.cuda.reset_peak_memory_stats()
    rows = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for track in pick_tracks().itertuples():
            rows.append(run_track(predictor, track, gts[str(track.ann_id)]))
            print(f"[{args.model}] {rows[-1]}")
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[{args.model}] peak VRAM allocated: {peak:.0f} MiB")


if __name__ == "__main__":
    main()
