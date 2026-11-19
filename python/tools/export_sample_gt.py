"""Выгружает эталонные маски треков выборки в компактный JSON (без 332 МБ aero_vis.json).

Выход: data/annotations/aerovis_sample_v1_gt.json
  {"<ann_id>": {"video": ..., "first_frame": f0, "file_names": [...], "segmentations": [...]}}
file_names и segmentations — срез [first_frame, last_frame] трека (кадры с исчезновением — null).

Запуск (WSL, из python/):  uv run --no-sync python -m tools.export_sample_gt
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "annotations" / "aerovis_sample_v1.csv"
AEROVIS_JSON = ROOT / "data" / "external" / "AeroVIS" / "aero_vis.json"
OUT = ROOT / "data" / "annotations" / "aerovis_sample_v1_gt.json"


def main() -> None:
    sample = pd.read_csv(SAMPLE)
    data = json.loads(AEROVIS_JSON.read_text(encoding="utf-8"))
    files = {v["id"]: v["file_names"] for v in data["videos"]}
    anns = {a["id"]: a for a in data["annotations"]}
    out = {}
    for r in sample.itertuples():
        a = anns[int(r.ann_id)]
        f0, f1 = int(r.first_frame), int(r.last_frame)
        out[str(r.ann_id)] = {
            "video": r.video,
            "first_frame": f0,
            "file_names": files[int(r.video_id)][f0 : f1 + 1],
            "segmentations": a["segmentations"][f0 : f1 + 1],
        }
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(out)} tracks)")


if __name__ == "__main__":
    main()
