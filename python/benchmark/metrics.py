"""Метрики 𝒥 (IoU) и ℱ (boundary F-measure) — как в официальной оценке DAVIS 2017.

Источник: davisvideochallenge/davis2017-evaluation, davis2017/metrics.py
(коммит ac7c43fca936f9722837b7fbd337d284ba37004b, BSD-3-Clause): db_eval_iou, f_measure, _seg2bmap.
Отличия от оригинала (на результат не влияют, проверяется `--selftest`):
  * `skimage.morphology.disk` заменён эквивалентной формулой (x^2 + y^2 <= r^2) — skimage не ставим;
  * ℱ считается на вырезке вокруг масок с запасом bound_pix + 2 px, а не на всём кадре:
    за пределами вырезки обе маски нулевые, поэтому границы, дилатация и совпадения те же,
    а кадр 2720×1530 не дилатируется целиком диском радиуса ~25 px.
Соглашения DAVIS: обе маски пустые -> 𝒥 = 1, ℱ = 1; void-пикселей в AeroVIS нет.

Самопроверка (WSL, из python/):  uv run --no-sync python -m benchmark.metrics --selftest
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

BOUND_TH = 0.008  # как в DAVIS: доля диагонали кадра


def db_eval_iou(annotation: np.ndarray, segmentation: np.ndarray) -> float:
    a = annotation.astype(bool)
    s = segmentation.astype(bool)
    union = np.sum(s | a)
    if np.isclose(union, 0):
        return 1.0
    return float(np.sum(s & a) / union)


def _disk(radius: float) -> np.ndarray:
    """= skimage.morphology.disk(radius) (strict_radius=True): x^2 + y^2 <= r^2."""
    r = int(radius)
    grid = np.arange(-r, r + 1)
    x, y = np.meshgrid(grid, grid)
    return (x**2 + y**2 <= radius**2).astype(np.uint8)


def _seg2bmap(seg: np.ndarray) -> np.ndarray:
    """Граница маски толщиной 1 px (DAVIS _seg2bmap при width/height = размеру маски)."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def _f_measure_full(foreground_mask: np.ndarray, gt_mask: np.ndarray, bound_pix: float) -> float:
    """Дословный перенос DAVIS f_measure (без void-пикселей) на переданном массиве."""
    fg_boundary = _seg2bmap(foreground_mask)
    gt_boundary = _seg2bmap(gt_mask)
    kernel = _disk(bound_pix)
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), kernel)
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), kernel)
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)
    if n_fg == 0 and n_gt > 0:
        precision, recall = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        precision, recall = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        precision, recall = 1.0, 1.0
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def f_measure(
    foreground_mask: np.ndarray, gt_mask: np.ndarray, bound_th: float = BOUND_TH
) -> float:
    """DAVIS ℱ; bound_pix — от размера полного кадра, вычисление — на вырезке."""
    h, w = gt_mask.shape
    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm((h, w)))
    union = foreground_mask.astype(bool) | gt_mask.astype(bool)
    if not union.any():
        return 1.0
    ys, xs = np.nonzero(union)
    m = int(bound_pix) + 2
    y0, y1 = max(0, ys.min() - m), min(h, ys.max() + m + 1)
    x0, x1 = max(0, xs.min() - m), min(w, xs.max() + m + 1)
    return _f_measure_full(foreground_mask[y0:y1, x0:x1], gt_mask[y0:y1, x0:x1], bound_pix)


def selftest(n: int = 300, seed: int = 42) -> None:
    """Вырезка == полный кадр на случайных масках, в т.ч. у краёв кадра и пустых."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in range(n):
        h, w = rng.choice([(540, 960), (756, 1344), (1530, 2720)], p=[0.4, 0.4, 0.2])
        gt = np.zeros((h, w), np.uint8)
        pr = np.zeros((h, w), np.uint8)
        for mask in (gt, pr):
            if rng.random() < 0.1:
                continue  # пустая маска
            # центр у левого/верхнего края, в середине или у правого/нижнего края
            cx = int(
                rng.choice([rng.integers(0, 40), rng.integers(0, w), w - 1 - rng.integers(0, 40)])
            )
            cy = int(
                rng.choice([rng.integers(0, 40), rng.integers(0, h), h - 1 - rng.integers(0, 40)])
            )
            axes = (int(rng.integers(2, 120)), int(rng.integers(2, 120)))
            cv2.ellipse(mask, (cx, cy), axes, float(rng.integers(0, 180)), 0, 360, 1, -1)
        bound_pix = np.ceil(BOUND_TH * np.linalg.norm((h, w)))
        full = _f_measure_full(pr, gt, bound_pix)
        crop = f_measure(pr, gt)
        worst = max(worst, abs(full - crop))
        if not np.isclose(full, crop, atol=1e-12):
            raise AssertionError(f"case {i}: full={full} crop={crop} shape={(h, w)}")
    assert db_eval_iou(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0
    assert f_measure(np.zeros((4, 4)), np.zeros((4, 4))) == 1.0
    print(f"selftest OK: {n} random cases, max |F_crop - F_full| = {worst:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
