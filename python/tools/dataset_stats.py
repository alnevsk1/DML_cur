"""Статистика датасетов этапа 3: AeroVIS (маски треков) и VisDrone-SOT (рамки).

Пишет таблицы в data/annotations/ (вход для выборки треков) и печатает сводку
для docs/datasets.md: разрешения, длины видео, размеры объектов, исчезновения.

Площадь считается по самим RLE-маскам, а не по полю `areas`: в SeaDronesSee-части AeroVIS
поле `areas` у части треков не совпадает с маской (проверено 2026-10-22, см. docs/datasets.md).

Запуск (WSL, из python/):  uv run --no-sync python -m tools.dataset_stats
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tools.rle import rle_area

ROOT = Path(__file__).resolve().parents[2]
AEROVIS_JSON = ROOT / "data" / "external" / "AeroVIS" / "aero_vis.json"
VISDRONE_SOT = ROOT / "data" / "external" / "VisDrone-SOT" / "val"
OUT_DIR = ROOT / "data" / "annotations"


def aerovis_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = json.loads(AEROVIS_JSON.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data["categories"]}
    videos = pd.DataFrame(
        [
            {
                "video_id": v["id"],
                "video": v["name"],
                "source": v["source"],
                "width": v["width"],
                "height": v["height"],
                "length": v["length"],
            }
            for v in data["videos"]
        ]
    )
    vid_info = videos.set_index("video_id")

    rows = []
    for ann in data["annotations"]:
        frame_area = np.array(
            [rle_area(s) if s is not None else 0 for s in ann["segmentations"]], dtype=np.float64
        )
        present = frame_area > 0  # пустая RLE-маска = объекта в кадре нет
        n_empty_rle = sum(s is not None for s in ann["segmentations"]) - int(present.sum())
        if not present.any():
            continue
        idx = np.flatnonzero(present)
        first, last = int(idx[0]), int(idx[-1])
        span = present[first : last + 1]
        # Исчезновения: переходы «есть -> нет» внутри жизни трека.
        n_gaps = int(np.sum(span[:-1] & ~span[1:]))
        areas = frame_area[idx]
        n_area_field_mismatch = sum(frame_area[i] != ann["areas"][i] for i in (first, last))
        v = vid_info.loc[ann["video_id"]]
        rows.append(
            {
                "ann_id": ann["id"],
                "video_id": ann["video_id"],
                "video": v["video"],
                "source": v["source"],
                "category": cats[ann["category_id"]],
                "iscrowd": int(ann["iscrowd"]),
                "first_frame": first,
                "last_frame": last,
                "n_present": int(present.sum()),
                "span": last - first + 1,
                "n_gaps": n_gaps,
                "absent_frac": float(1.0 - span.mean()),
                "area_first": float(areas[0]),
                "area_median": float(np.median(areas)),
                # Поле `areas` != маска на кадрах first/last; кадры с пустой RLE-маской.
                "area_field_mismatch": int(n_area_field_mismatch),
                "n_empty_rle": int(n_empty_rle),
                # «Линейный размер» объекта, px: sqrt(площади маски).
                "size_first_px": float(np.sqrt(areas[0])),
                "size_median_px": float(np.sqrt(np.median(areas))),
                "rel_area_median_pct": float(100.0 * np.median(areas) / (v["width"] * v["height"])),
            }
        )
    return videos, pd.DataFrame(rows)


def visdrone_sot_table() -> pd.DataFrame:
    rows = []
    for ann_file in sorted((VISDRONE_SOT / "annotations").glob("*.txt")):
        seq = ann_file.stem
        boxes = np.loadtxt(ann_file, delimiter=",", ndmin=2)  # (T, 4): x, y, w, h
        frames = sorted((VISDRONE_SOT / "sequences" / seq).glob("*.jpg"))
        with Image.open(frames[0]) as im:
            width, height = im.size
        wh = boxes[:, 2] * boxes[:, 3]
        valid = wh > 0
        rows.append(
            {
                "sequence": seq,
                "width": width,
                "height": height,
                "n_frames": len(frames),
                "n_boxes": len(boxes),
                "size_first_px": float(np.sqrt(wh[0])),
                "size_median_px": float(np.sqrt(np.median(wh[valid]))) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def describe_sizes(s: pd.Series) -> str:
    q = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    return " | ".join(f"p{int(k * 100)}={v:.1f}" for k, v in q.items())


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    videos, tracks = aerovis_tables()
    videos.to_parquet(OUT_DIR / "aerovis_videos.parquet", index=False)
    tracks.to_parquet(OUT_DIR / "aerovis_tracks.parquet", index=False)

    print("=== AeroVIS: videos by source")
    res = videos.groupby(["source", "width", "height"]).size().rename("n_videos")
    print(res.to_string())
    lengths = videos.groupby("source")["length"].describe()
    print(lengths[["count", "min", "50%", "max", "mean"]].round(1))
    print(f"total frames: {videos['length'].sum()}")

    t = tracks[tracks["iscrowd"] == 0]
    print(f"\n=== AeroVIS: tracks (non-crowd {len(t)}, crowd {int(tracks['iscrowd'].sum())})")
    print(t.groupby(["source", "category"]).size().unstack(fill_value=0).to_string())
    print("\nsize_median_px (sqrt area) by source:")
    for src, g in t.groupby("source"):
        print(f"  {src:13s} n={len(g):5d} {describe_sizes(g['size_median_px'])}")
    print(f"  {'ALL':13s} n={len(t):5d} {describe_sizes(t['size_median_px'])}")
    print("\nspan (frames) by source:")
    for src, g in t.groupby("source"):
        print(f"  {src:13s} {describe_sizes(g['span'])}")
    print(
        f"\ntracks with >=1 disappearance inside lifetime: {(t['n_gaps'] > 0).sum()} "
        f"({100 * (t['n_gaps'] > 0).mean():.1f}%)"
    )
    print(f"tracks with span >= 30 frames: {(t['span'] >= 30).sum()}")
    print(f"frames with empty (zero-area) RLE inside tracks: {int(t['n_empty_rle'].sum())}")
    bad = t[t["area_field_mismatch"] > 0]
    print("\n'areas' field != RLE mask area (first/last frame), tracks by source:")
    print(bad.groupby("source").size().reindex(t["source"].unique(), fill_value=0).to_string())

    bins = [0.0, 16.0, 32.0, 96.0, np.inf]
    labels = ["tiny<16", "small16-32", "medium32-96", "large>=96"]
    long = t[t["span"] >= 30].copy()
    long["bin"] = pd.cut(long["size_median_px"], bins, labels=labels, right=False)
    print("\nspan>=30 tracks by size bin and source (+ with disappearance):")
    table = pd.crosstab(long["bin"], long["source"], margins=True)
    table["with_gaps"] = long[long["n_gaps"] > 0].groupby("bin", observed=False).size()
    print(table.to_string())

    if VISDRONE_SOT.exists():
        sot = visdrone_sot_table()
        sot.to_parquet(OUT_DIR / "visdrone_sot_val.parquet", index=False)
        print(f"\n=== VisDrone-SOT val: {len(sot)} sequences, {sot['n_frames'].sum()} frames")
        print(sot.groupby(["width", "height"]).size().rename("n_seq").to_string())
        print(f"n_frames: {describe_sizes(sot['n_frames'])}")
        print(f"size_median_px (sqrt w*h): {describe_sizes(sot['size_median_px'])}")
        mismatch = sot[sot["n_frames"] != sot["n_boxes"]]
        print(f"sequences with n_frames != n_boxes: {len(mismatch)}")


if __name__ == "__main__":
    main()
