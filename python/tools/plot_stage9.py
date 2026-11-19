"""Построить графики этапа 9 из сохранённых CSV без повторного инференса."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, GREEN = "#17618e", "#bc4b20", "#3b765a"
CLIPS = ("4072", "2548")
RUNTIMES = ("torch", "fp32", "fp16", "mixed", "cpu")
RUNTIME_LABELS = (
    "PyTorch\nBF16",
    "TRT\nFP32",
    "TRT\nFP16",
    "TRT mixed\nCUDA",
    "TRT mixed\nCPU",
)


def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"ann_id": str})


def finish(figure: plt.Figure, output: Path, name: str) -> None:
    figure.savefig(output / f"{name}.png", dpi=260, facecolor="white")
    plt.close(figure)


def paired(
    axis: plt.Axes,
    labels: tuple[str, ...],
    left: pd.Series,
    right: pd.Series,
    ylabel: str,
    *,
    names: tuple[str, str] = CLIPS,
) -> None:
    index = np.arange(len(labels))
    axis.bar(index - 0.19, left, 0.38, color=BLUE, label=names[0])
    axis.bar(index + 0.19, right, 0.38, color=ORANGE, label=names[1], hatch="//")
    axis.set_xticks(index, labels)
    axis.set_ylabel(ylabel)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.22)


def runtimes(summary: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    values = summary.set_index(["ann_id", "mode"])
    for axis, (field, name) in zip(
        axes.flat,
        (
            ("JF", "J&F"),
            ("p95_ms", "p95, мс"),
            ("effective_fps", "Кадров/с по среднему времени"),
            ("gpu_peak_system_mib", "Системный пик VRAM, МиБ"),
        ),
        strict=True,
    ):
        paired(
            axis,
            RUNTIME_LABELS,
            *(values.loc[clip, field].loc[list(RUNTIMES)] for clip in CLIPS),
            name,
        )
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend(title="Трек AeroVIS", ncol=2, frameon=False, loc="upper center")
    figure.suptitle("SAM 2.1-tiny на одинаковых кадрах · RTX 4060", fontweight="bold")
    finish(figure, output, "stage9_runtime")


def playback(summary: pd.DataFrame, common: pd.DataFrame, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.8), layout="constrained")
    modes = ["sync", "realtime"]
    labels = ("Покадрово", "Реальное время")
    for row, clip in zip(axes, CLIPS, strict=True):
        data = summary[summary["ann_id"] == clip].set_index("mode").loc[modes]
        quality = common[common["ann_id"] == clip].set_index("mode").loc[modes]
        paired(
            row[0],
            labels,
            data["display_fps"],
            data["mask_fps"],
            f"{clip}\nКадров или масок в секунду",
            names=("Показ видео", "Публикация масок"),
        )
        paired(
            row[1],
            labels,
            quality["JF"],
            data["coverage"],
            "Доля / J&F",
            names=("J&F на общих кадрах", "Покрытие"),
        )
        row[1].set_ylim(0, 1.05)
        row[2].bar(labels, data["gpu_peak_system_mib"], color=[BLUE, ORANGE], width=0.48)
        row[2].set_ylabel("Системный пик VRAM, МиБ")
        row[2].set_axisbelow(True)
        row[2].grid(axis="y", alpha=0.22)
    for axis in axes[0, :2]:
        axis.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=8)
    figure.suptitle("Режимы Win32 GUI · вход 24 кадра/с · RTX 4060", fontweight="bold")
    finish(figure, output, "stage9_playback")


def roi(previous: Path, refresh: Path, output: Path) -> None:
    figure, (timeline, quality, latency) = plt.subplots(
        1, 3, figsize=(13, 4.4), layout="constrained", width_ratios=(1.65, 1, 1)
    )
    frames = load(previous / "ab" / "quality_by_frame.csv")
    frames = frames[frames["ann_id"] == "4072"]
    diagnostic = load(refresh / "diagnostic_frames.csv")
    timeline.axvspan(23.5, 47.5, color="#e5e7eb", label="Цель отсутствует")
    for data, color, label, style in (
        (frames[frames["pass"] == "A1"], BLUE, "Полный кадр", "-"),
        (frames[frames["pass"] == "B1"], ORANGE, "ROI, банк заморожен", "--"),
        (diagnostic, GREEN, "ROI, банк обновляется", ":"),
    ):
        visible = data["J"].where(data["present"] == 1)
        timeline.plot(
            data["frame"], visible, color=color, label=label, linestyle=style, linewidth=2
        )
    timeline.axhline(0.5, color="#64748b", linewidth=0.8)
    timeline.set(xlabel="Кадр трека 4072", ylabel="IoU видимой цели", ylim=(0, 1))
    timeline.legend(loc="lower left", fontsize=8, frameon=False)

    held = load(refresh / "holdout_summary.csv")
    labels = ("3310", "7376", "2158")
    fixed = held.set_index(["pass", "ann_id"])
    paired(
        quality,
        labels,
        fixed.loc["A1", "JF"].loc[list(labels)],
        fixed.loc["B1", "JF"].loc[list(labels)],
        "J&F видимых кадров",
        names=("Полный кадр", "ROI, обновление каждые 8 кадров"),
    )
    quality.set_ylim(0, 1)
    quality.legend(loc="upper left", fontsize=8, frameon=False)
    medians = held.assign(mode=held["pass"].str[0]).groupby(["mode", "ann_id"])["p95_ms"].median()
    paired(
        latency,
        labels,
        medians.loc["A"].loc[list(labels)],
        medians.loc["B"].loc[list(labels)],
        "p95, мс\nМедиана двух прогонов",
        names=("Полный кадр", "ROI"),
    )
    figure.suptitle("ROI и восстановление после исчезновения · RTX 4060", fontweight="bold")
    finish(figure, output, "stage9_roi")


def models(table: Path, output: Path) -> None:
    data = pd.read_csv(table)
    groups = ("click", "mask")
    labels = {
        "sam2.1_tiny": "SAM 2.1-tiny",
        "sam2.1_small": "SAM 2.1-small",
        "efficienttam_ti": "EfficientTAM-Ti",
        "efficienttam_s": "EfficientTAM-S",
        "edgetam": "EdgeTAM",
        "cutie_base_s480_fp32": "Cutie-base",
        "xmem_s480_fp32": "XMem",
        "deaot_s_s480_fp32": "DeAOT-S",
        "deaot_t_s480_fp32": "DeAOT-T",
        "r50_deaot_l_s480_fp32": "R50-DeAOT-L",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, layout="constrained")
    for axis, init, title in zip(
        axes, groups, ("Подсказка кликом", "Подсказка маской"), strict=True
    ):
        rows = data[(data["init"] == init) & (data["oom_tracks"] == 0)].sort_values("JF")
        axis.barh(
            rows["model"].map(lambda name: labels.get(name, name)),
            rows["JF"],
            color=[ORANGE if name == "sam2.1_tiny" else BLUE for name in rows["model"]],
        )
        axis.set(xlabel="J&F", xlim=(0, 1), title=title)
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.22)
    figure.suptitle("Модели этапа 4 · 200 треков AeroVIS v1 · RTX 4060", fontweight="bold")
    finish(figure, output, "stage9_models")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="каталог замеров этапа 9")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output = root / "reports" / "figures"
    validation = root / "runs" / "validation"
    summary = load(args.run / "summary.csv")
    common = load(args.run / "gui_common.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    runtimes(summary, output)
    playback(summary, common, output)
    roi(
        validation / "stage8_roi_20261111_002456_681",
        validation / "stage8_roi_refresh_20261111_003657_987",
        output,
    )
    models(root / "runs" / "eval" / "summary.csv", output)


if __name__ == "__main__":
    main()
