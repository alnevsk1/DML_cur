"""Оценка прогонов бенчмарка: 𝒥&ℱ по протоколу DAVIS + скорость и VRAM.

Протокол: кадр инициализации (0) исключается; все остальные кадры трека входят в оценку,
включая кадры, где объект исчез (эталон пустой: пустое предсказание -> 𝒥 = ℱ = 1, как в DAVIS).
𝒥&ℱ трека = (среднее 𝒥 + среднее ℱ) / 2; по прогону — среднее по трекам (как DAVIS по объектам).

Треки с "oom": true (не поместились в лимит VRAM, run_vos.py) в 𝒥&ℱ и скорость не входят;
их число — колонка oom_tracks, «tracks» — число оценённых треков.
Выход: runs/eval/<run>_tracks.parquet (по трекам) и runs/eval/summary.csv (по прогонам).
Запуск (WSL, из python/):  uv run --no-sync python -m benchmark.evaluate
    [--runs ../runs/pvs ../runs/vos] [--eval-dir ../runs/eval]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.metrics import db_eval_iou, f_measure
from tools.rle import decode_rle

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "annotations" / "aerovis_sample_v1.csv"
SAMPLE_GT = ROOT / "data" / "annotations" / "aerovis_sample_v1_gt.json"
EVAL_DIR = ROOT / "runs" / "eval"
CLASS_GROUP = {
    "person": "person",
    "car": "vehicle", "vehicle": "vehicle", "truck": "vehicle", "bus": "vehicle",
    "bicycle": "two_wheeler", "motorcycle": "two_wheeler", "tricycle": "two_wheeler",
    "boat": "boat",
}  # fmt: skip

_GT: dict = {}


def _init_worker() -> None:
    _GT.update(json.loads(SAMPLE_GT.read_text(encoding="utf-8")))


def eval_track(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("oom"):  # трек не поместился в лимит VRAM (run_vos.py): без метрик, только факт
        nan = float("nan")
        return {
            "ann_id": r["ann_id"],
            "oom": True,
            "oom_frame": r["oom_frame"],
            "n_eval_frames": 0,
            "J": nan, "F": nan, "JF": nan, "J_absent_frames": nan,
            "t_total_ms_med": nan, "t_enc_ms_med": nan, "t_rest_ms_med": nan,
            "t_total_ms_all": [],
            "mem_growth_mib_per_100fr": nan,
            "peak_mib": r["peak_mib_over_baseline"] + r["baseline_mib"],
        }  # fmt: skip
    gt = _GT[str(r["ann_id"])]
    h, w = gt["segmentations"][0]["size"]
    js, fs, js_absent = [], [], []
    for i in range(1, r["n_frames"]):
        seg = gt["segmentations"][i]
        g = decode_rle(seg) if seg is not None else np.zeros((h, w), np.uint8)
        p = decode_rle(r["masks"][i]) if r["masks"][i] is not None else np.zeros((h, w), np.uint8)
        j = db_eval_iou(g, p)
        js.append(j)
        fs.append(f_measure(p, g))
        if seg is None:
            js_absent.append(j)
    tt = np.array(r["t_total_ms"], float)
    te = np.array(r["t_enc_ms"], float)
    mem = np.array(r["mem_state_mib"], float)
    ok = ~np.isnan(mem)
    frames = np.arange(len(mem))
    slope = float(np.polyfit(frames[ok], mem[ok], 1)[0] * 100) if ok.sum() > 2 else float("nan")
    return {
        "ann_id": r["ann_id"],
        "oom": False,
        "oom_frame": -1,
        "n_eval_frames": len(js),
        "J": float(np.mean(js)),
        "F": float(np.mean(fs)),
        "JF": float((np.mean(js) + np.mean(fs)) / 2),
        "J_absent_frames": float(np.mean(js_absent)) if js_absent else float("nan"),
        "t_total_ms_med": float(np.nanmedian(tt)),
        "t_enc_ms_med": float(np.nanmedian(te)),
        "t_rest_ms_med": float(np.nanmedian(tt - te)),
        "t_total_ms_all": tt[1:].tolist(),
        "mem_growth_mib_per_100fr": slope,
        "peak_mib": r["peak_mib_over_baseline"] + r["baseline_mib"],
    }


def evaluate_run(run_dir: Path, sample: pd.DataFrame, pool: ProcessPoolExecutor) -> pd.DataFrame:
    paths = sorted((run_dir / "tracks").glob("*.json"))
    rows = list(pool.map(eval_track, paths, chunksize=4))
    df = pd.DataFrame(rows).merge(sample, on="ann_id", how="left")
    df["class_group"] = df["category"].map(CLASS_GROUP)
    df.insert(0, "run", run_dir.name)
    return df


def summarize(df_all: pd.DataFrame, n_sample: int) -> dict:
    """Метрики — по трекам без OOM; число OOM-треков и пик VRAM — по всем."""
    model, init = df_all["run"].iloc[0].split("__")
    df = df_all[~df_all["oom"]]
    t_all = np.concatenate([np.asarray(x, float) for x in df["t_total_ms_all"]])
    out = {
        "model": model,
        "init": init,
        "tracks": f"{len(df)}/{n_sample}",
        "oom_tracks": int(df_all["oom"].sum()),
        "JF": df["JF"].mean(),
        "J": df["J"].mean(),
        "F": df["F"].mean(),
    }
    for col, values in (("size_bin", ["tiny", "small", "medium", "large"]),
                        ("class_group", ["person", "vehicle", "two_wheeler"])):  # fmt: skip
        for v in values:
            out[f"JF_{v}"] = df.loc[df[col] == v, "JF"].mean()
    out["JF_disappear"] = df.loc[df["group"] == "disappear", "JF"].mean()
    out["J_on_absent_frames"] = df["J_absent_frames"].mean()
    out["ms_per_frame_median"] = float(np.nanmedian(t_all))
    out["fps_from_median"] = 1000.0 / out["ms_per_frame_median"]
    out["enc_ms_median"] = df["t_enc_ms_med"].median()
    out["rest_ms_median"] = df["t_rest_ms_med"].median()
    out["peak_vram_mib_max"] = df_all["peak_mib"].max()
    out["state_growth_mib_per_100fr_median"] = df["mem_growth_mib_per_100fr"].median()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[str(ROOT / "runs" / r) for r in ("pvs", "vos")])
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(
        p for root in map(Path, args.runs) if root.is_dir()
        for p in root.iterdir() if (p / "tracks").is_dir()
    )  # fmt: skip
    sample = pd.read_csv(SAMPLE)
    summaries = []
    with ProcessPoolExecutor(max_workers=4, initializer=_init_worker) as pool:
        for run_dir in run_dirs:
            df = evaluate_run(run_dir, sample, pool)
            if df.empty:
                continue
            out = eval_dir / f"{run_dir.name}_tracks.parquet"
            df.drop(columns=["t_total_ms_all"]).to_parquet(out)
            summaries.append(summarize(df, len(sample)))
            print(f"evaluated {run_dir.name}: {len(df)} tracks", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(eval_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 3):
        print(summary.T.to_string(header=False))


if __name__ == "__main__":
    main()
