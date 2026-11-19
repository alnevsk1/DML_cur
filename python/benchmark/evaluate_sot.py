"""Оценка контрольного протокола VisDrone-SOT (OPE, как в OTB / VisDrone-SOT toolkit).

  * success — AUC кривой «доля кадров с IoU > порога», пороги 0, 0.05, ..., 1 (21 точка);
  * precision@20 — доля кадров с ошибкой центра ≤ 20 px (кривая — пороги 0..50 px);
  * success@0.5 — доля кадров с IoU > 0.5;
  * norm_precision — AUC нормализованной точности (как в LaSOT): ошибка центра в долях
    ширины/высоты эталонной рамки, пороги 0..0.5 шаг 0.01. Не зависит от того, насколько
    плотно размечена рамка: рамка по маске заметно плотнее свободных рамок VisDrone
    (у масок инициализации IoU рамок 0.46–0.88), поэтому success систематически занижен
    у всех моделей одинаково, а precision / norm_precision — нет.
Кадр инициализации исключён; пустая маска (рамки нет) -> IoU 0, ошибка центра ∞.
Кривые считаются по каждой последовательности и усредняются по последовательностям (OTB);
последовательности с OOM (run_sot.py) в метрики не входят, их число — колонка oom_seqs.

Выход: runs/eval_sot/summary.csv (по прогонам), runs/eval_sot/per_seq.csv (по последовательностям).
Запуск (WSL, из python/):  uv run --no-sync python -m benchmark.evaluate_sot [--runs ../runs/sot]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.run_sot import OUT_ROOT, load_sequence

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "runs" / "eval_sot"
IOU_THRESHOLDS = np.linspace(0, 1, 21)
CENTER_THRESHOLDS = np.arange(0, 51)
NORM_THRESHOLDS = np.linspace(0, 0.5, 51)


def iou_and_center_error(pred: list, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """pred: список рамок x, y, w, h или None; gt: (N, 4).

    Returns:
        IoU, ошибка центра в px, нормализованная ошибка центра (в долях w, h эталона).
    """
    n = len(gt)
    iou = np.zeros(n)
    err = np.full(n, np.inf)
    err_norm = np.full(n, np.inf)
    for i, b in enumerate(pred):
        if b is None:
            continue
        g = gt[i]
        ix = max(0.0, min(b[0] + b[2], g[0] + g[2]) - max(b[0], g[0]))
        iy = max(0.0, min(b[1] + b[3], g[1] + g[3]) - max(b[1], g[1]))
        inter = ix * iy
        iou[i] = inter / (b[2] * b[3] + g[2] * g[3] - inter)
        dx, dy = b[0] + b[2] / 2 - g[0] - g[2] / 2, b[1] + b[3] / 2 - g[1] - g[3] / 2
        err[i] = np.hypot(dx, dy)
        err_norm[i] = np.hypot(dx / g[2], dy / g[3])
    return iou, err, err_norm


def eval_seq(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    if r.get("oom"):
        peak = r["peak_mib_over_baseline"] + r["baseline_mib"]
        return {
            "seq": r["seq"],
            "n_frames": r["n_frames"],
            "oom": True,
            "oom_frame": r["oom_frame"],
            "peak_mib": peak,
        }
    _, gt = load_sequence(r["seq"])
    iou, err, err_norm = iou_and_center_error(r["boxes"][1:], gt[1:])  # без кадра инициализации
    success = np.array([(iou > t).mean() for t in IOU_THRESHOLDS])
    precision = np.array([(err <= t).mean() for t in CENTER_THRESHOLDS])
    norm_precision = np.array([(err_norm <= t).mean() for t in NORM_THRESHOLDS])
    tt = np.array(r["t_total_ms"], float)
    te = np.array(r["t_enc_ms"], float)
    return {
        "seq": r["seq"],
        "n_frames": r["n_frames"],
        "oom": False,
        "success_auc": float(success.mean()),
        "precision_20": float(precision[20]),
        "success_50": float((iou > 0.5).mean()),
        "norm_precision": float(norm_precision.mean()),
        "mean_iou": float(iou.mean()),
        "empty_frac": float(np.mean([b is None for b in r["boxes"][1:]])),
        "success_curve": success.tolist(),
        "precision_curve": precision.tolist(),
        "norm_precision_curve": norm_precision.tolist(),
        "t_total_ms_all": tt[1:].tolist(),
        "t_enc_ms_med": float(np.nanmedian(te)),
        "peak_mib": r["peak_mib_over_baseline"] + r["baseline_mib"],
    }


def summarize(run: str, df_all: pd.DataFrame) -> dict:
    model, init = run.split("__")
    df = df_all[~df_all["oom"]]
    success = np.mean(np.stack(df["success_curve"].to_list()), axis=0)
    precision = np.mean(np.stack(df["precision_curve"].to_list()), axis=0)
    norm_precision = np.mean(np.stack(df["norm_precision_curve"].to_list()), axis=0)
    t_all = np.concatenate([np.asarray(x, float) for x in df["t_total_ms_all"]])
    ms = float(np.nanmedian(t_all))
    return {
        "model": model,
        "init": init,
        "seqs": f"{len(df)}/{len(df_all)}",
        "oom_seqs": int(df_all["oom"].sum()),
        "success_auc": float(success.mean()),
        "precision_20": float(precision[20]),
        "success_50": float(df["success_50"].mean()),
        "norm_precision": float(norm_precision.mean()),
        "empty_frac": float(df["empty_frac"].mean()),
        "ms_per_frame_median": ms,
        "fps_from_median": 1000.0 / ms,
        "enc_ms_median": float(df["t_enc_ms_med"].median()),
        "peak_vram_mib_max": float(df_all["peak_mib"].max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(OUT_ROOT))
    ap.add_argument("--eval-dir", default=str(EVAL_DIR))
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    summaries, per_seq = [], []
    for run_dir in sorted(p for p in Path(args.runs).iterdir() if (p / "seqs").is_dir()):
        rows = [eval_seq(p) for p in sorted((run_dir / "seqs").glob("*.json"))]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        summaries.append(summarize(run_dir.name, df))
        keep = ["seq", "n_frames", "oom", "success_auc", "precision_20", "success_50",
                "norm_precision", "mean_iou", "empty_frac", "peak_mib"]  # fmt: skip
        per_seq.append(df.reindex(columns=keep).assign(run=run_dir.name))
        print(f"evaluated {run_dir.name}: {len(df)} seqs", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(eval_dir / "summary.csv", index=False)
    pd.concat(per_seq).to_csv(eval_dir / "per_seq.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.precision", 3):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
