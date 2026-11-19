"""Бенчмарк PVS-моделей (SAM 2.1, EfficientTAM, EdgeTAM) на выборке AeroVIS v1 (этап 4).

Один запуск = одна модель × один вариант инициализации (click | box | mask) по всем трекам
выборки. Результат по каждому треку пишется сразу (атомарно), повторный запуск пропускает
готовые треки — прогон можно прерывать и продолжать.

Условия (одинаковые для всех моделей): bf16 autocast (как в официальных примерах SAM 2),
eager, fill_hole_area=0 (benchmark/pvs_models.py), батч 1, один объект, кадры трека подаются
по порядку начиная с кадра инициализации (онлайн-режим, без доступа к будущим кадрам).

Замеры на кадр (кроме кадра инициализации):
  * t_total — от запроса следующего кадра у propagate_in_video до готовой маски на GPU
    в разрешении видео (cuda.synchronize), без ожидания чтения JPEG;
  * t_enc — энкодер изображения (_get_image_feature), без ожидания чтения JPEG; включает
    копирование кадра CPU->GPU в float64 и приведение к float32 (так в коде авторов,
    ~7 мс на RTX 4060 / PCIe 3.0 x8 — одинаково для всех трёх моделей);
  * mem — torch.cuda.memory_allocated после кадра (рост состояния со временем).
Чтение JPEG и ресайз идут в фоновом потоке с упреждением; ожидание вычитается из замеров.

Запуск (WSL, из python/; PYTHONPATH — репозиторий модели):
  PYTHONPATH=third_party/sam2 uv run --no-sync python -m benchmark.run_pvs \
      --model sam2.1_tiny --init click [--limit 3]
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # tqdm читает переменную при импорте (до sam2/...)
# WSL: при нехватке видеопамяти драйвер не выдаёт OOM, а отдаёт общую системную память (PCIe),
# и кадры замедляются в 10-20 раз. Состояние предиктора растёт каждый кадр -> кэш аллокатора
# фрагментируется и разрастается. Поэтому expandable_segments + лимит доли VRAM в main()
# (подробнее — run_vos.py, docs/progress.md 2026-10-23). До импорта torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import Future, ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from benchmark import run_meta  # noqa: E402
from benchmark.pvs_models import MODELS, build_predictor  # noqa: E402
from tools.rle import decode_rle, encode_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEQ_ROOT = ROOT / "data" / "external" / "AeroVIS" / "sequences"
SAMPLE = ROOT / "data" / "annotations" / "aerovis_sample_v1.csv"
SAMPLE_GT = ROOT / "data" / "annotations" / "aerovis_sample_v1_gt.json"
OUT_ROOT = ROOT / "runs" / "pvs"
SEED = 42
INITS = ("click", "box", "mask")
PREFETCH = 6  # кадров вперёд
VRAM_FRACTION = 0.85  # из 8 ГиБ RTX 4060; остальное — Windows (DWM и т. п.)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PrefetchFrameLoader:
    """Ленивая замена списка кадров в inference_state["images"].

    Предобработка — та же функция репозитория (`_load_img_as_tensor`: PIL resize до
    image_size x image_size, /255) и та же нормализация, что в AsyncVideoFrameLoader;
    в памяти держатся только кадры около текущего (а не все ~12.6 МБ x длина трека).
    """

    def __init__(self, paths: list[Path], image_size: int, load_fn, mean, std) -> None:  # noqa: ANN001 — функции/тензоры репозитория
        self.paths = paths
        self.image_size = image_size
        self.load_fn = load_fn
        self.mean = mean
        self.std = std
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.futures: dict[int, Future] = {}
        self.cache: dict[int, torch.Tensor] = {}
        self.wait_s = 0.0
        img, self.video_height, self.video_width = load_fn(str(paths[0]), image_size)
        self.cache[0] = self._normalize(img)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        img -= self.mean
        img /= self.std
        return img

    def _load(self, i: int) -> torch.Tensor:
        img, _, _ = self.load_fn(str(self.paths[i]), self.image_size)
        return self._normalize(img)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        for j in range(i, min(i + PREFETCH, len(self.paths))):
            if j not in self.cache and j not in self.futures:
                self.futures[j] = self.pool.submit(self._load, j)
        if i not in self.cache:
            t0 = time.perf_counter()
            self.cache[i] = self.futures.pop(i).result()
            self.wait_s += time.perf_counter() - t0
        for k in [k for k in self.cache if k < i - 1]:  # держим текущий и предыдущий кадр
            del self.cache[k]
        return self.cache[i]

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)


def install_loader_and_timers(predictor: torch.nn.Module) -> dict:
    """Подменяет load_video_frames в модуле предиктора и оборачивает _get_image_feature."""
    pred_module = sys.modules[type(predictor).__module__]
    misc = importlib.import_module(f"{type(predictor).__module__.split('.')[0]}.utils.misc")
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    ctx: dict = {"loader": None, "enc": {}}

    def load_video_frames(video_path, image_size, **_kwargs):  # noqa: ANN001, ANN003, ANN202
        loader = PrefetchFrameLoader(video_path, image_size, misc._load_img_as_tensor, mean, std)
        ctx["loader"] = loader
        return loader, loader.video_height, loader.video_width

    pred_module.load_video_frames = load_video_frames
    original = predictor._get_image_feature

    def timed_get_image_feature(inference_state, frame_idx, batch_size):  # noqa: ANN001, ANN202
        torch.cuda.synchronize()
        w0 = ctx["loader"].wait_s
        t0 = time.perf_counter()
        out = original(inference_state, frame_idx, batch_size)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0 - (ctx["loader"].wait_s - w0)
        ctx["enc"][frame_idx] = ctx["enc"].get(frame_idx, 0.0) + dt
        return out

    predictor._get_image_feature = timed_get_image_feature
    return ctx


def prompt(predictor, state: dict, init: str, track: pd.Series, gt0: np.ndarray) -> np.ndarray:  # noqa: ANN001
    if init == "click":
        pts = np.array([[track.click_x, track.click_y]], dtype=np.float32)
        _, _, logits = predictor.add_new_points_or_box(
            state, frame_idx=0, obj_id=1, points=pts, labels=np.array([1], dtype=np.int32)
        )
    elif init == "box":
        ys, xs = np.nonzero(gt0)
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)
        _, _, logits = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
    else:
        _, _, logits = predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=gt0)
    return (logits[0, 0] > 0).cpu().numpy()


def run_track(predictor, ctx: dict, init: str, track: pd.Series, gt: dict) -> dict:  # noqa: ANN001
    paths = [SEQ_ROOT / rel for rel in gt["file_names"]]
    gt0 = decode_rle(gt["segmentations"][0]).astype(bool)
    ctx["enc"] = {}
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    state = predictor.init_state(video_path=paths, offload_video_to_cpu=True)
    loader = ctx["loader"]
    try:
        init_mask = prompt(predictor, state, init, track, gt0)
        n = len(paths)
        masks: list[dict | None] = [None] * n
        masks[0] = encode_mask(init_mask)
        t_total = [float("nan")] * n
        mem = [float("nan")] * n
        gen = predictor.propagate_in_video(state)
        while True:
            torch.cuda.synchronize()
            w0 = loader.wait_s
            t0 = time.perf_counter()
            try:
                f, _, logits = next(gen)
            except StopIteration:
                break
            pred = logits[0, 0] > 0
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0 - (loader.wait_s - w0)
            mem[f] = (torch.cuda.memory_allocated() - mem_before) / 2**20
            if f == 0:
                continue  # кадр инициализации: выход уже посчитан при подсказке
            t_total[f] = dt * 1000
            masks[f] = encode_mask(pred.cpu().numpy())
        peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
        predictor.reset_state(state)
    finally:
        loader.close()
    t_enc = [ctx["enc"].get(i, float("nan")) * 1000 if i > 0 else float("nan") for i in range(n)]
    return {
        "ann_id": int(track.ann_id),
        "n_frames": n,
        "masks": masks,
        "t_total_ms": t_total,
        "t_enc_ms": t_enc,
        "mem_state_mib": mem,
        "peak_mib_over_baseline": peak,
        "baseline_mib": mem_before / 2**20,
        "jpeg_wait_s": loader.wait_s,
    }


def warmup(predictor, image_size: int, iters: int = 5) -> None:  # noqa: ANN001
    x = torch.randn(1, 3, image_size, image_size, device="cuda")
    for _ in range(iters):
        predictor.forward_image(x)
    torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--init", required=True, choices=INITS)
    ap.add_argument("--limit", type=int, default=None, help="только первые N треков (проверка)")
    args = ap.parse_args()

    set_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    out_dir = OUT_ROOT / f"{args.model}__{args.init}"
    (out_dir / "tracks").mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args.model)
    ctx = install_loader_and_timers(predictor)
    spec = MODELS[args.model]
    meta = run_meta.collect(ROOT) | {
        "model": args.model,
        "init": args.init,
        "repo": spec.repo,
        "config": spec.config,
        "checkpoint": str(spec.checkpoint.relative_to(ROOT)),
        "checkpoint_md5": run_meta.file_md5(spec.checkpoint),
        "sample_md5": run_meta.file_md5(SAMPLE),
        "precision": "bf16 autocast",
        "fill_hole_area": predictor.fill_hole_area,
        "image_size": predictor.image_size,
        "params_m": sum(p.numel() for p in predictor.parameters()) / 1e6,
        "seed": SEED,
        "cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "vram_fraction": VRAM_FRACTION,
    }
    (out_dir / f"meta_{meta['utc_start'].replace(':', '')}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    sample = pd.read_csv(SAMPLE).sort_values("ann_id")
    if args.limit:
        sample = sample.head(args.limit)
    gts = json.loads(SAMPLE_GT.read_text(encoding="utf-8"))

    t_start = time.perf_counter()
    done = skipped = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        warmup(predictor, predictor.image_size)
        for track in sample.itertuples():
            out = out_dir / "tracks" / f"{track.ann_id}.json"
            if out.exists():
                skipped += 1
                continue
            res = run_track(predictor, ctx, args.init, track, gts[str(track.ann_id)])
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res), encoding="utf-8")
            tmp.replace(out)  # атомарно: недописанный трек не считается готовым
            done += 1
            med = np.nanmedian(res["t_total_ms"])
            peak = res["peak_mib_over_baseline"]
            print(
                f"[{args.model}/{args.init}] {done + skipped}/{len(sample)} ann {track.ann_id} "
                f"({track.size_bin}, {res['n_frames']} fr): {med:.1f} ms/fr median, "
                f"peak +{peak:.0f} MiB, jpeg wait {res['jpeg_wait_s']:.1f} s",
                flush=True,
            )
    elapsed = time.perf_counter() - t_start
    print(f"done {done}, skipped (already present) {skipped}, {elapsed:.0f} s")


if __name__ == "__main__":
    main()
