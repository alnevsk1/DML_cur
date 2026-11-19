"""Бенчмарк VOS-моделей (XMem, Cutie, DeAOT) на выборке AeroVIS v1 (этап 4).

Пара к run_pvs.py: тот же формат результата (runs/vos/<run>/tracks/<ann_id>.json), поэтому
работает benchmark.evaluate. Инициализация — только маской первого кадра трека
(эталон AeroVIS), то есть сопоставимо с вариантом `mask` у PVS-моделей.

Условия: bf16 autocast (как у PVS-прогона; --precision fp32 — для проверки), eager, батч 1,
один объект, онлайн-режим (кадры подаются по порядку, без доступа к будущим).
Вход: кадр приводится к короткой стороне --short-side (PIL, билинейная интерполяция
со сглаживанием — как Resize в загрузчиках XMem/Cutie), маска первого кадра — так же
со сглаживанием и порогом 0.5 (у мелких объектов nearest может потерять маску целиком;
если маска всё же пустая — берётся пиксель с максимальным покрытием).
Выход: маска в исходном разрешении кадра (vos_models._to_out / evaluator DeAOT).

Трек, не поместившийся в лимит VRAM (VRAM_FRACTION), не роняет прогон: пишется JSON с
"oom": true, кадром и пиком памяти; evaluate исключает такие треки из 𝒥&ℱ и считает их число.

Замеры на кадр (кроме кадра инициализации):
  * t_total — копирование подготовленного кадра CPU->GPU + шаг модели + маска в разрешении
    видео (cuda.synchronize), без ожидания чтения JPEG;
  * t_enc — энкодер изображения (методы VOSAdapter.encoder_attrs);
  * mem — torch.cuda.memory_allocated после кадра (рост состояния со временем).

Запуск (WSL, из python/; PYTHONPATH — репозиторий модели):
  PYTHONPATH=third_party/XMem uv run --no-sync python -m benchmark.run_vos \
      --model xmem [--short-side 480] [--limit 3]
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
# WSL: при нехватке видеопамяти драйвер не выдаёт OOM, а молча отдаёт общую системную память
# (через PCIe) -> кадры в 10-20 раз медленнее. У моделей с растущей памятью (XMem, R50-DeAOT-L)
# каждый кадр нужны блоки нового размера, кэш аллокатора фрагментируется и разрастается
# (XMem: пик тензоров +0.7 ГиБ, а процесс занял 7.4 ГиБ + 4 ГиБ общей памяти). Поэтому:
# expandable_segments (меньше фрагментации) + лимит доли VRAM в main() (при его достижении
# аллокатор освобождает кэш, а не уходит в системную память). Задаётся до импорта torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from benchmark import run_meta  # noqa: E402
from benchmark.run_pvs import (  # noqa: E402
    SAMPLE,
    SAMPLE_GT,
    SEED,
    SEQ_ROOT,
    PrefetchFrameLoader,
    set_seed,
)
from benchmark.vos_models import MODELS, VOSAdapter, build_adapter  # noqa: E402
from tools.rle import decode_rle, encode_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "runs" / "vos"
VRAM_FRACTION = 0.85  # из 8 ГиБ RTX 4060; остальное — Windows (DWM и т. п.)
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def load_frame(path: str, size_hw: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
    """JPEG -> (3, h, w) float32 в [0, 1] во внутреннем разрешении + исходные H, W."""
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    h, w = size_hw
    if (h, w) != (h0, w0):
        img = img.resize((w, h), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div_(255)  # (3, h, w)
    return x, h0, w0


def resize_init_mask(mask: torch.Tensor, size_hw: tuple[int, int]) -> tuple[torch.Tensor, bool]:
    """(H, W) bool -> (h, w) bool; второе значение — сработал ли запасной вариант (1 пиксель)."""
    if tuple(mask.shape) == size_hw:
        return mask, False
    cover = F.interpolate(mask[None, None].float(), size_hw, mode="bilinear", antialias=True)[0, 0]
    out = cover >= 0.5
    if out.any():
        return out, False
    return cover == cover.max(), True


def install_encoder_timer(adapter: VOSAdapter) -> dict:
    ctx: dict = {"frame": 0, "enc": {}}
    for attr in adapter.encoder_attrs:
        original = getattr(adapter.network, attr)

        def timed(*args, _original=original, **kwargs):  # noqa: ANN002, ANN003, ANN202 — прокси к методу сети
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = _original(*args, **kwargs)
            torch.cuda.synchronize()
            f = ctx["frame"]
            ctx["enc"][f] = ctx["enc"].get(f, 0.0) + time.perf_counter() - t0
            return out

        setattr(adapter.network, attr, timed)
    return ctx


def run_track(adapter: VOSAdapter, ctx: dict, short_side: int, gt: dict) -> dict:
    paths = [SEQ_ROOT / rel for rel in gt["file_names"]]
    gt0 = decode_rle(gt["segmentations"][0]).astype(bool)
    h0, w0 = gt0.shape
    size_hw = adapter.internal_size(h0, w0, short_side)
    n = len(paths)
    ctx["enc"] = {}
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    mean = torch.tensor(adapter.mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(adapter.std, dtype=torch.float32)[:, None, None]
    loader = PrefetchFrameLoader(paths, size_hw, load_frame, mean, std)
    oom_frame = None
    try:
        init_mask, fallback = resize_init_mask(torch.from_numpy(gt0).cuda(), size_hw)
        ctx["frame"] = 0
        adapter.start(loader[0].cuda(), init_mask, n)
        masks: list[dict | None] = [None] * n
        masks[0] = encode_mask(gt0)
        t_total = [float("nan")] * n
        mem = [float("nan")] * n
        mem[0] = (torch.cuda.memory_allocated() - mem_before) / 2**20
        for f in range(1, n):
            torch.cuda.synchronize()
            w_start = loader.wait_s
            t0 = time.perf_counter()
            ctx["frame"] = f
            pred = adapter.step(loader[f].cuda(), (h0, w0), last=f == n - 1)
            torch.cuda.synchronize()
            t_total[f] = (time.perf_counter() - t0 - (loader.wait_s - w_start)) * 1000
            mem[f] = (torch.cuda.memory_allocated() - mem_before) / 2**20
            masks[f] = encode_mask(pred.cpu().numpy())
        peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
        adapter.reset()
    except torch.OutOfMemoryError:
        # трек не помещается в лимит VRAM (R50-DeAOT-L: долговременная память без ограничения)
        oom_frame = ctx["frame"]
        peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
    finally:
        loader.close()
    if oom_frame is not None:
        # освобождаем вне except: трассировка исключения держит ссылки на тензоры
        adapter.reset()
        torch.cuda.empty_cache()
        return {
            "ann_id": int(gt["ann_id"]),
            "n_frames": n,
            "oom": True,
            "oom_frame": oom_frame,
            "peak_mib_over_baseline": peak,
            "baseline_mib": mem_before / 2**20,
            "vram_limit_mib": torch.cuda.get_device_properties(0).total_memory
            * VRAM_FRACTION
            / 2**20,
            "internal_hw": list(size_hw),
        }
    t_enc = [ctx["enc"].get(i, float("nan")) * 1000 if i > 0 else float("nan") for i in range(n)]
    return {
        "ann_id": int(gt["ann_id"]),
        "n_frames": n,
        "masks": masks,
        "t_total_ms": t_total,
        "t_enc_ms": t_enc,
        "mem_state_mib": mem,
        "peak_mib_over_baseline": peak,
        "baseline_mib": mem_before / 2**20,
        "jpeg_wait_s": loader.wait_s,
        "internal_hw": list(size_hw),
        "init_mask_px_internal": int(init_mask.sum()),
        "init_mask_fallback": fallback,
    }


def warmup(adapter: VOSAdapter, size_hw: tuple[int, int], iters: int = 5) -> None:
    h, w = size_hw
    x = torch.rand(3, h, w, device="cuda")
    mask = torch.zeros(h, w, dtype=torch.bool, device="cuda")
    mask[h // 3 : h // 2, w // 3 : w // 2] = True
    adapter.start(x, mask, iters + 1)
    for i in range(iters):
        adapter.step(x, (h, w), last=i == iters - 1)
    adapter.reset()
    torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument(
        "--short-side", type=int, default=480, help="короткая сторона входа; -1 — исходная"
    )
    ap.add_argument("--precision", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--limit", type=int, default=None, help="только первые N треков (проверка)")
    args = ap.parse_args()

    set_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)
    size_tag = "orig" if args.short_side <= 0 else str(args.short_side)
    prec_tag = "" if args.precision == "bf16" else f"_{args.precision}"
    run_name = f"{args.model}_s{size_tag}{prec_tag}__mask"
    out_dir = Path(args.out_root) / run_name
    (out_dir / "tracks").mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(args.model)
    ctx = install_encoder_timer(adapter)
    spec = MODELS[args.model]
    meta = run_meta.collect(ROOT) | {
        "model": args.model,
        "init": "mask",
        "repo": spec.repo,
        "checkpoint": str(spec.checkpoint.relative_to(ROOT)),
        "checkpoint_md5": run_meta.file_md5(spec.checkpoint),
        "sample_md5": run_meta.file_md5(SAMPLE),
        "precision": f"{args.precision} autocast" if DTYPES[args.precision] else "fp32",
        "short_side": args.short_side,
        "model_config": adapter.config(),
        "params_m": sum(p.numel() for p in adapter.network.parameters()) / 1e6,
        "seed": SEED,
        "cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "vram_fraction": VRAM_FRACTION,
    }
    (out_dir / f"meta_{meta['utc_start'].replace(':', '')}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    sample = pd.read_csv(SAMPLE).sort_values("ann_id")
    if args.limit:
        sample = sample.head(args.limit)
    gts = json.loads(SAMPLE_GT.read_text(encoding="utf-8"))

    dtype = DTYPES[args.precision]
    autocast = torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()
    t_start = time.perf_counter()
    done = skipped = 0
    with torch.inference_mode(), autocast:
        warmup(adapter, adapter.internal_size(1080, 1920, args.short_side))
        for track in sample.itertuples():
            out = out_dir / "tracks" / f"{track.ann_id}.json"
            if out.exists():
                skipped += 1
                continue
            gt = gts[str(track.ann_id)] | {"ann_id": track.ann_id}
            res = run_track(adapter, ctx, args.short_side, gt)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res), encoding="utf-8")
            tmp.replace(out)  # атомарно: недописанный трек не считается готовым
            done += 1
            if res.get("oom"):
                print(
                    f"[{run_name}] {done + skipped}/{len(sample)} ann {track.ann_id} "
                    f"({track.size_bin}, {res['n_frames']} fr): OOM on frame {res['oom_frame']}, "
                    f"peak +{res['peak_mib_over_baseline']:.0f} MiB "
                    f"(limit {res['vram_limit_mib']:.0f} MiB) -> skipped",
                    flush=True,
                )
                continue
            med = np.nanmedian(res["t_total_ms"])
            print(
                f"[{run_name}] {done + skipped}/{len(sample)} ann {track.ann_id} "
                f"({track.size_bin}, {res['n_frames']} fr, in {res['internal_hw']}, "
                f"init {res['init_mask_px_internal']} px): {med:.1f} ms/fr median, "
                f"peak +{res['peak_mib_over_baseline']:.0f} MiB, "
                f"jpeg wait {res['jpeg_wait_s']:.1f} s",
                flush=True,
            )
    elapsed = time.perf_counter() - t_start
    print(f"done {done}, skipped (already present) {skipped}, {elapsed:.0f} s")


if __name__ == "__main__":
    main()
