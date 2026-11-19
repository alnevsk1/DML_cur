"""Фиксированная выборка треков AeroVIS для бенчмарка (этап 3, схема утверждена 2026-10-22).

Схема:
  * треки iscrowd=0, длиной (span) >= 30 кадров;
  * 4 класса по размеру sqrt(медианной площади маски): tiny <16, small 16-32,
    medium 32-96, large >=96 px (границы COCO 32/96 + tiny);
  * в каждом классе: 40 треков без исчезновений + 10 треков с исчезновением внутри трека;
  * не больше 2 треков из одного видео в одном классе; seed 42.
Для каждого трека сохраняется кадр инициализации (первый кадр с маской) и точка клика —
максимум distance transform маски этого кадра (точка, наиболее удалённая от границы).

Вход:  data/annotations/aerovis_tracks.parquet (tools/dataset_stats.py), aero_vis.json.
Выход: data/annotations/aerovis_sample_v1.csv

Запуск (WSL, из python/):  uv run --no-sync python -m tools.sample_tracks
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tools.rle import decode_rle, rle_area

SEED = 42
MIN_SPAN = 30
PER_VIDEO_CAP = 2
N_REGULAR = 40
N_DISAPPEAR = 10
BINS = [0.0, 16.0, 32.0, 96.0, np.inf]
LABELS = ["tiny", "small", "medium", "large"]

ROOT = Path(__file__).resolve().parents[2]
TRACKS = ROOT / "data" / "annotations" / "aerovis_tracks.parquet"
VIDEOS = ROOT / "data" / "annotations" / "aerovis_videos.parquet"
AEROVIS_JSON = ROOT / "data" / "external" / "AeroVIS" / "aero_vis.json"
OUT = ROOT / "data" / "annotations" / "aerovis_sample_v1.csv"


def capped_sample(
    pool: pd.DataFrame, n: int, rng: np.random.Generator, used: dict[int, int]
) -> pd.DataFrame:
    """Случайно берёт до n строк, не больше PER_VIDEO_CAP на видео (учитывая уже взятые)."""
    picked = []
    for i in rng.permutation(len(pool)):
        row = pool.iloc[i]
        vid = int(row["video_id"])
        if used.get(vid, 0) >= PER_VIDEO_CAP:
            continue
        picked.append(row)
        used[vid] = used.get(vid, 0) + 1
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


def click_point(mask: np.ndarray) -> tuple[int, int, float]:
    """(x, y, расстояние до границы в px) — максимум distance transform; при равенстве — первый."""
    padded = np.pad(mask, 1)  # объект у края кадра: граница кадра считается границей маски
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return int(x), int(y), float(dist[y, x])


def main() -> None:
    rng = np.random.default_rng(SEED)
    tracks = pd.read_parquet(TRACKS)
    videos = pd.read_parquet(VIDEOS).set_index("video_id")

    t = tracks[(tracks["iscrowd"] == 0) & (tracks["span"] >= MIN_SPAN)].copy()
    t["size_bin"] = pd.cut(t["size_median_px"], BINS, labels=LABELS, right=False).astype(str)
    t = t.sort_values("ann_id").reset_index(drop=True)  # детерминированный порядок до перемешивания

    parts = []
    for label in LABELS:
        in_bin = t[t["size_bin"] == label]
        used: dict[int, int] = {}
        dis = capped_sample(in_bin[in_bin["n_gaps"] > 0], N_DISAPPEAR, rng, used)
        reg = capped_sample(in_bin[in_bin["n_gaps"] == 0], N_REGULAR, rng, used)
        dis["group"] = "disappear"
        reg["group"] = "regular"
        for name, got, want in (("disappear", dis, N_DISAPPEAR), ("regular", reg, N_REGULAR)):
            if len(got) < want:
                print(f"WARNING: {label}/{name}: only {len(got)} of {want} tracks (per-video cap)")
        parts += [dis, reg]
    sample = pd.concat(parts, ignore_index=True)

    # Кадр инициализации и клик: декодируем маску первого кадра каждого трека.
    data = json.loads(AEROVIS_JSON.read_text(encoding="utf-8"))
    wanted = set(sample["ann_id"])
    anns = {a["id"]: a for a in data["annotations"] if a["id"] in wanted}
    file_names = {v["id"]: v["file_names"] for v in data["videos"]}  # ['vd_001/0000001.jpg', ...]
    clicks = []
    for _, row in sample.iterrows():
        ann = anns[int(row["ann_id"])]
        f = int(row["first_frame"])
        rle = ann["segmentations"][f]
        mask = decode_rle(rle)
        # Поле `areas` не используем: в SeaDronesSee оно местами не совпадает с маской.
        if int(mask.sum()) != rle_area(rle) or mask.sum() == 0:
            raise RuntimeError(
                f"bad init mask for ann {row['ann_id']}: decoded {mask.sum()}, rle {rle_area(rle)}"
            )
        x, y, d = click_point(mask)
        clicks.append(
            {
                "init_frame_file": file_names[int(row["video_id"])][f],
                "click_x": x,
                "click_y": y,
                "click_dist_px": round(d, 2),
                "init_area_px": int(mask.sum()),
            }
        )
    sample = pd.concat([sample, pd.DataFrame(clicks)], axis=1)
    sample["width"] = sample["video_id"].map(videos["width"])
    sample["height"] = sample["video_id"].map(videos["height"])

    cols = [
        "ann_id", "video_id", "video", "source", "category", "size_bin", "group",
        "first_frame", "last_frame", "span", "n_present", "n_gaps", "absent_frac",
        "size_first_px", "size_median_px", "width", "height",
        "init_frame_file", "click_x", "click_y", "click_dist_px", "init_area_px",
    ]  # fmt: skip
    sample = sample[cols].sort_values(["size_bin", "group", "ann_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)

    print(f"Saved {OUT} ({len(sample)} tracks); init masks checked (non-empty, area == RLE)")
    print(pd.crosstab(sample["size_bin"], sample["group"], margins=True).to_string())
    print(pd.crosstab(sample["size_bin"], sample["source"]).to_string())
    print(f"videos covered: {sample['video_id'].nunique()}")
    print(f"frames to process (sum of spans): {int(sample['span'].sum())}")
    dist = sample["click_dist_px"].describe().round(2).to_dict()
    print(f"click distance to mask boundary, px: {dist}")


if __name__ == "__main__":
    main()
