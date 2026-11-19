"""Контрольный протокол VisDrone-SOT val (этап 4): рамки, размеченные людьми, OPE.

Зачем: эталон AeroVIS получен SAM 3 (возможен перекос в пользу семейства SAM); здесь эталон —
рамки людей. 11 последовательностей, 7 046 кадров, объект размечен на каждом кадре.

Инициализация (один проход с первого кадра, OPE):
  * PVS (SAM 2.1, EfficientTAM, EdgeTAM): `box` — эталонная рамка кадра 1; `mask` — маска
    SAM 2.1-small по этой рамке (tools/visdrone_init_masks.py), та же, что у VOS;
  * VOS (XMem, Cutie, DeAOT): только `mask` (та же маска SAM 2.1-small).
Выход на кадр — охватывающая рамка предсказанной маски (x, y, w, h); пустая маска -> None
(в оценке — промах). Условия как в основном бенчмарке: PVS — bf16, вход 1024x1024;
VOS — fp32, короткая сторона 480; замеры t_total / t_enc / VRAM те же (run_pvs.py, run_vos.py).

Результат: runs/sot/<run>/seqs/<seq>.json, возобновляемо; оценка — benchmark.evaluate_sot.
Запуск (WSL, из python/; PYTHONPATH — репозиторий модели):
  PYTHONPATH=third_party/sam2 uv run --no-sync python -m benchmark.run_sot \
      --model sam2.1_tiny --init box [--seqs uav0000053_00264_s]
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
# до импорта torch; почему — run_vos.py (WSL уходит в системную память вместо OOM)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from benchmark import pvs_models, run_meta, run_pvs, run_vos, vos_models  # noqa: E402
from tools.rle import decode_rle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOT_ROOT = ROOT / "data" / "external" / "VisDrone-SOT" / "val"
INIT_MASKS = ROOT / "data" / "annotations" / "visdrone_sot_val_init_masks.json"
OUT_ROOT = ROOT / "runs" / "sot"


def load_sequence(seq: str) -> tuple[list[Path], np.ndarray]:
    paths = sorted((SOT_ROOT / "sequences" / seq).glob("*.jpg"))
    lines = (SOT_ROOT / "annotations" / f"{seq}.txt").read_text(encoding="utf-8").splitlines()
    boxes = np.array([[float(v) for v in ln.split(",")] for ln in lines if ln.strip()])  # (N, 4)
    if len(boxes) != len(paths):
        raise ValueError(f"{seq}: {len(paths)} frames, {len(boxes)} boxes")
    return paths, boxes


def mask_to_box(mask: torch.Tensor) -> list[float] | None:
    """(H, W) bool на GPU -> охватывающая рамка x, y, w, h в пикселях или None."""
    cols = torch.nonzero(mask.any(0)).flatten()
    if cols.numel() == 0:
        return None
    rows = torch.nonzero(mask.any(1)).flatten()
    x0, x1, y0, y1 = (int(v) for v in (cols[0], cols[-1], rows[0], rows[-1]))
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]


def _result(seq: str, n: int, boxes: list, t_total: list, t_enc: list, mem: list, peak: float,
            mem_before: float, wait_s: float) -> dict:  # fmt: skip
    return {
        "seq": seq,
        "n_frames": n,
        "boxes": boxes,
        "t_total_ms": t_total,
        "t_enc_ms": t_enc,
        "mem_state_mib": mem,
        "peak_mib_over_baseline": peak,
        "baseline_mib": mem_before / 2**20,
        "jpeg_wait_s": wait_s,
    }


def run_seq_pvs(predictor, ctx: dict, init: str, seq: str, init_mask: np.ndarray) -> dict:  # noqa: ANN001 — предиктор репозитория
    paths, gt = load_sequence(seq)
    n = len(paths)
    ctx["enc"] = {}
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    state = predictor.init_state(video_path=paths, offload_video_to_cpu=True)
    loader = ctx["loader"]
    try:
        if init == "box":
            x, y, w, h = gt[0]
            box = np.array([x, y, x + w, y + h], dtype=np.float32)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
        else:
            predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=init_mask)
        boxes: list[list[float] | None] = [None] * n
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
            boxes[f] = mask_to_box(pred)
            if f > 0:
                t_total[f] = dt * 1000
        peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
        predictor.reset_state(state)
    finally:
        loader.close()
    t_enc = [ctx["enc"].get(i, float("nan")) * 1000 if i > 0 else float("nan") for i in range(n)]
    return _result(seq, n, boxes, t_total, t_enc, mem, peak, mem_before, loader.wait_s)


def run_seq_vos(adapter: vos_models.VOSAdapter, ctx: dict, short_side: int, seq: str,
                init_mask: np.ndarray) -> dict:  # fmt: skip
    paths, _ = load_sequence(seq)
    n = len(paths)
    h0, w0 = init_mask.shape
    size_hw = adapter.internal_size(h0, w0, short_side)
    ctx["enc"] = {}
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    mean = torch.tensor(adapter.mean, dtype=torch.float32)[:, None, None]
    std = torch.tensor(adapter.std, dtype=torch.float32)[:, None, None]
    loader = run_pvs.PrefetchFrameLoader(paths, size_hw, run_vos.load_frame, mean, std)
    oom_frame = None
    try:
        mask_in, _ = run_vos.resize_init_mask(torch.from_numpy(init_mask).cuda(), size_hw)
        ctx["frame"] = 0
        adapter.start(loader[0].cuda(), mask_in, n)
        boxes: list[list[float] | None] = [None] * n
        boxes[0] = mask_to_box(torch.from_numpy(init_mask))
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
            boxes[f] = mask_to_box(pred)
        peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
        adapter.reset()
    except torch.OutOfMemoryError:
        oom_frame = ctx["frame"]
        peak = (torch.cuda.max_memory_allocated() - mem_before) / 2**20
    finally:
        loader.close()
    if oom_frame is not None:
        adapter.reset()  # вне except: трассировка исключения держит ссылки на тензоры
        torch.cuda.empty_cache()
        return {"seq": seq, "n_frames": n, "oom": True, "oom_frame": oom_frame,
                "peak_mib_over_baseline": peak, "baseline_mib": mem_before / 2**20}  # fmt: skip
    t_enc = [ctx["enc"].get(i, float("nan")) * 1000 if i > 0 else float("nan") for i in range(n)]
    return _result(seq, n, boxes, t_total, t_enc, mem, peak, mem_before, loader.wait_s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(pvs_models.MODELS | vos_models.MODELS))
    ap.add_argument("--init", choices=("box", "mask"), default="mask")
    ap.add_argument("--short-side", type=int, default=480, help="только VOS")
    ap.add_argument("--precision", choices=sorted(run_vos.DTYPES), default=None,
                    help="по умолчанию как в основном бенчмарке: PVS bf16, VOS fp32")  # fmt: skip
    ap.add_argument("--seqs", nargs="+", default=None, help="только эти последовательности")
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()

    is_pvs = args.model in pvs_models.MODELS
    if not is_pvs and args.init != "mask":
        ap.error("VOS-модели инициализируются только маской")
    precision = args.precision or ("bf16" if is_pvs else "fp32")
    run_pvs.set_seed(run_pvs.SEED)
    torch.cuda.set_per_process_memory_fraction(run_vos.VRAM_FRACTION)
    if is_pvs:
        tag = "" if precision == "bf16" else f"_{precision}"
        run_name = f"{args.model}{tag}__{args.init}"
    else:
        tag = "" if precision == "bf16" else f"_{precision}"
        run_name = f"{args.model}_s{args.short_side}{tag}__mask"
    out_dir = Path(args.out_root) / run_name
    (out_dir / "seqs").mkdir(parents=True, exist_ok=True)

    init_masks = json.loads(INIT_MASKS.read_text(encoding="utf-8"))
    meta = run_meta.collect(ROOT) | {
        "model": args.model,
        "init": args.init,
        "family": "pvs" if is_pvs else "vos",
        "precision": precision,
        "init_masks": str(INIT_MASKS.relative_to(ROOT)),
        "init_masks_md5": run_meta.file_md5(INIT_MASKS),
        "init_masks_meta": init_masks["meta"],
        "cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "vram_fraction": run_vos.VRAM_FRACTION,
        "seed": run_pvs.SEED,
    }
    if is_pvs:
        predictor = pvs_models.build_predictor(args.model)
        ctx = run_pvs.install_loader_and_timers(predictor)
        spec = pvs_models.MODELS[args.model]
        meta |= {"input": f"{predictor.image_size}x{predictor.image_size}",
                 "params_m": sum(p.numel() for p in predictor.parameters()) / 1e6}  # fmt: skip
    else:
        adapter = vos_models.build_adapter(args.model)
        ctx = run_vos.install_encoder_timer(adapter)
        spec = vos_models.MODELS[args.model]
        params_m = sum(p.numel() for p in adapter.network.parameters()) / 1e6
        meta |= {
            "input": f"short side {args.short_side}",
            "model_config": adapter.config(),
            "params_m": params_m,
        }
    meta |= {"checkpoint": str(spec.checkpoint.relative_to(ROOT)),
             "checkpoint_md5": run_meta.file_md5(spec.checkpoint)}  # fmt: skip
    (out_dir / f"meta_{meta['utc_start'].replace(':', '')}.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    seqs = args.seqs or sorted(p.name for p in (SOT_ROOT / "sequences").iterdir() if p.is_dir())
    dtype = run_vos.DTYPES[precision]
    autocast = torch.autocast("cuda", dtype=dtype) if dtype else contextlib.nullcontext()
    t_start = time.perf_counter()
    done = skipped = 0
    with torch.inference_mode(), autocast:
        if is_pvs:
            run_pvs.warmup(predictor, predictor.image_size)
        else:
            run_vos.warmup(adapter, adapter.internal_size(1080, 1920, args.short_side))
        for i, seq in enumerate(seqs, 1):
            out = out_dir / "seqs" / f"{seq}.json"
            if out.exists():
                skipped += 1
                continue
            mask0 = decode_rle(init_masks["sequences"][seq]["mask"]).astype(bool)
            if is_pvs:
                res = run_seq_pvs(predictor, ctx, args.init, seq, mask0)
            else:
                res = run_seq_vos(adapter, ctx, args.short_side, seq, mask0)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res), encoding="utf-8")
            tmp.replace(out)  # атомарно: недописанная последовательность не считается готовой
            done += 1
            if res.get("oom"):
                msg = f"OOM on frame {res['oom_frame']} -> skipped"
            else:
                lost = sum(b is None for b in res["boxes"][1:])
                msg = (f"{np.nanmedian(res['t_total_ms']):.1f} ms/fr median, empty masks {lost}, "
                       f"peak +{res['peak_mib_over_baseline']:.0f} MiB")  # fmt: skip
            print(f"[{run_name}] {i}/{len(seqs)} {seq} ({res['n_frames']} fr): {msg}", flush=True)
    print(
        f"done {done}, skipped (already present) {skipped}, {time.perf_counter() - t_start:.0f} s"
    )


if __name__ == "__main__":
    main()
