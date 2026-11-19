"""Проверка экспортируемости PVS-моделей в ONNX (этап 4, решающий критерий).

Для модели: 1) PyTorch-предиктор (fp32, без TF32) отслеживает объект трека AeroVIS по рамке
на первых кадрах — так получаются реальные входы частей и эталонные выходы (исходный код
модели, комплексный RoPE); 2) части из export/sam2_parts.py с вещественным RoPE сверяются
с эталоном в PyTorch (проверка разбиения); 3) части экспортируются в ONNX (torch.onnx,
dynamo, opset 18) и сверяются в ONNX Runtime CUDA EP в строгом FP32 (use_tf32=0).
Внимание к памяти проверяется на двух кадрах: 1 кадр памяти + 1 указатель (начало трека)
и полный банк (кадр подсказки + 6 последних, 9 указателей).

Выход: models/onnx/<model>/<part>.onnx, runs/export/<model>/report.json.
Запуск (WSL, из python/; PYTHONPATH — репозиторий модели):
  PYTHONPATH=third_party/sam2 uv run --no-sync python -m export.export_parts --model sam2.1_tiny
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from torch.export import Dim  # noqa: E402

from benchmark import pvs_models, run_meta, run_pvs  # noqa: E402
from export import sam2_parts as sp  # noqa: E402
from tools.rle import decode_rle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ONNX_ROOT = ROOT / "models" / "onnx"
REPORT_ROOT = ROOT / "runs" / "export"
OPSET = 18
TRACK_ANN = 77  # medium, 65 кадров, объект виден на всём треке (smoke-тесты бенчмарка)
CHECK_FRAMES = (1, 9)  # 1: память = кадр подсказки; 9: полный банк (1 + 6 кадров, 9 указателей)
MAX_POINTS, MAX_MEM_FRAMES, MAX_PTRS = 16, 7, 16


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy() if x.dtype.is_floating_point else x.cpu().numpy()


def compare(name: str, got: np.ndarray, ref: np.ndarray) -> dict:
    res = {"output": name, "shape": list(ref.shape), "max_abs": float(np.abs(got - ref).max()),
           "ref_max_abs": float(np.abs(ref).max())}  # fmt: skip
    if name == "low_res_mask":
        a, b = got > 0, ref > 0
        union = np.logical_or(a, b).sum()
        res["mask_iou"] = float(np.logical_and(a, b).sum() / union) if union else 1.0
    return res


def reference_track(model, args_model: str) -> tuple[dict, dict]:  # noqa: ANN001 — предиктор репозитория
    """Трекинг исходной моделью (fp32) -> эталонные входы/выходы частей по кадрам."""
    gt = json.loads(run_pvs.SAMPLE_GT.read_text(encoding="utf-8"))[str(TRACK_ANN)]
    paths = [run_pvs.SEQ_ROOT / rel for rel in gt["file_names"]]
    mask0 = decode_rle(gt["segmentations"][0]).astype(bool)
    h0, w0 = mask0.shape
    ys, xs = np.nonzero(mask0)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

    ctx = run_pvs.install_loader_and_timers(model)
    state = model.init_state(video_path=paths, offload_video_to_cpu=True)
    model.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
    for f, _, _ in model.propagate_in_video(state, max_frame_num_to_track=max(CHECK_FRAMES)):
        if f >= max(CHECK_FRAMES):
            break
    out_dict = state["output_dict"] if "output_dict" in state else state["output_dict_per_obj"][0]
    loader = ctx["loader"]
    size = model.image_size
    hi = bool(model.use_high_res_features_in_sam)

    def frame_feats(t: int) -> tuple:
        img = loader[t].float().unsqueeze(0).cuda()  # (1, 3, 1024, 1024); загрузчик отдаёт float64
        bo = model.forward_image(img)
        _, vf, vp, fs = model._prepare_backbone_features(bo)
        hr = [x.permute(1, 2, 0).view(1, x.size(2), *s) for x, s in zip(vf[:-1], fs[:-1], strict=True)]
        img_out = (bo["backbone_fpn"][-1], bo["vision_pos_enc"][-1]) + ((bo["backbone_fpn"][0], bo["backbone_fpn"][1]) if hi else ())  # fmt: skip
        return img, vf, vp, fs, hr, img_out

    cases: dict = {}
    with torch.inference_mode():
        # кадр 0: рамка -> одна маска; признаки + no_mem_embed (кадр подсказки без памяти)
        img, vf, vp, fs, hr, img_out = frame_feats(0)
        pix0 = model._prepare_memory_conditioned_features(
            frame_idx=0, is_init_cond_frame=True, current_vision_feats=vf[-1:],
            current_vision_pos_embeds=vp[-1:], feat_sizes=fs[-1:], output_dict=out_dict,
            num_frames=len(paths))  # fmt: skip
        coords = torch.tensor(box.reshape(1, 2, 2) / np.array([w0, h0]) * size, device="cuda")
        labels = torch.tensor([[2, 3]], dtype=torch.int32, device="cuda")
        heads = model._forward_sam_heads(pix0, {"point_coords": coords, "point_labels": labels},
                                         None, hr or None, multimask_output=False)  # fmt: skip
        cases["decoder_single"] = {
            "inputs": (pix0, coords, labels, *hr),
            "ref": {"low_res_mask": heads[3], "obj_ptr": heads[5], "object_score_logits": heads[6], "ious": heads[2]},
        }  # fmt: skip
        cases["image_encoder"] = {"inputs": (img,), "ref": dict(zip(
            ["vision_feat", "vision_pos", "high_res_0", "high_res_1"], img_out, strict=False))}  # fmt: skip

        for t in CHECK_FRAMES:
            img, vf, vp, fs, hr, img_out = frame_feats(t)
            mem = sp.gather_memory(model, out_dict, t)
            pix = model._prepare_memory_conditioned_features(
                frame_idx=t, is_init_cond_frame=False, current_vision_feats=vf[-1:],
                current_vision_pos_embeds=vp[-1:], feat_sizes=fs[-1:], output_dict=out_dict,
                num_frames=len(paths))  # fmt: skip
            cases[f"memory_attention_f{t}"] = {
                "inputs": (img_out[0], img_out[1], mem["maskmem_feats"], mem["maskmem_pos"],
                           mem["maskmem_tpos"], mem["obj_ptrs"], mem["obj_ptr_tdiff"]),
                "ref": {"pix_feat_with_mem": pix},
            }  # fmt: skip
            pad_xy = torch.zeros(1, 1, 2, device="cuda")
            pad_lb = -torch.ones(1, 1, dtype=torch.int32, device="cuda")
            heads = model._forward_sam_heads(pix, None, None, hr or None, multimask_output=True)
            cases[f"decoder_multi_f{t}"] = {
                "inputs": (pix, pad_xy, pad_lb, *hr),
                "ref": {"low_res_mask": heads[3], "obj_ptr": heads[5], "object_score_logits": heads[6], "ious": heads[2]},
            }  # fmt: skip
            feats, pos = model._encode_new_memory(vf, fs, heads[4], heads[6], is_mask_from_pts=False)
            cases[f"memory_encoder_f{t}"] = {
                "inputs": (img_out[0], heads[3], heads[6], torch.zeros(1, device="cuda")),
                "ref": {"maskmem_features": feats, "maskmem_pos_enc": pos[-1]},
            }
    info = {"track_ann": TRACK_ANN, "video_hw": [h0, w0], "n_mem_frames": {}, "n_obj_ptrs": {}}
    for t in CHECK_FRAMES:
        inp = cases[f"memory_attention_f{t}"]["inputs"]
        info["n_mem_frames"][t] = int(inp[2].shape[0])
        info["n_obj_ptrs"][t] = int(inp[5].shape[0])
    loader.close()
    return cases, info


def part_specs(model) -> dict:  # noqa: ANN001
    hi = bool(model.use_high_res_features_in_sam)
    hr_names = ["high_res_0", "high_res_1"] if hi else []
    dec_out = ["low_res_mask", "obj_ptr", "object_score_logits", "ious"]
    p = Dim("points", min=1, max=MAX_POINTS)
    nf = Dim("mem_frames", min=1, max=MAX_MEM_FRAMES)
    npt = Dim("obj_ptrs", min=1, max=MAX_PTRS)
    dec_dyn = ({}, {1: p}, {1: p}) + ({},) * len(hr_names)
    return {
        "image_encoder": (sp.ImageEncoderPart(model), ["image"],
                          ["vision_feat", "vision_pos", *hr_names], None, "image_encoder"),
        "decoder_single": (sp.DecoderPart(model, multimask=False),
                           ["pix_feat", "point_coords", "point_labels", *hr_names], dec_out, dec_dyn,
                           "decoder_single"),
        "decoder_multi": (sp.DecoderPart(model, multimask=True),
                          ["pix_feat", "point_coords", "point_labels", *hr_names], dec_out, dec_dyn,
                          "decoder_multi_f9"),
        "memory_encoder": (sp.MemoryEncoderPart(model),
                           ["vision_feat", "low_res_mask", "object_score_logits", "is_mask_from_pts"],
                           ["maskmem_features", "maskmem_pos_enc"], None, "memory_encoder_f9"),
        "memory_attention": (sp.MemoryAttentionPart(model),
                             ["vision_feat", "vision_pos", "maskmem_feats", "maskmem_pos",
                              "maskmem_tpos", "obj_ptrs", "obj_ptr_tdiff"],
                             ["pix_feat_with_mem"],
                             ({}, {}, {0: nf}, {0: nf}, {0: nf}, {0: npt}, {0: npt}),
                             "memory_attention_f9"),
    }  # fmt: skip


def ort_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", {"use_tf32": 0})])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(pvs_models.MODELS))
    ap.add_argument("--parts", nargs="+", default=None, help="только эти части")
    args = ap.parse_args()

    run_pvs.set_seed(run_pvs.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False  # эталон — строгий FP32
    torch.backends.cudnn.allow_tf32 = False
    out_dir = ONNX_ROOT / args.model
    rep_dir = REPORT_ROOT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    model = pvs_models.build_predictor(args.model)
    cases, info = reference_track(model, args.model)
    n_rope = sp.make_rope_real(model)  # после эталона: дальше — вещественный RoPE
    ort.preload_dlls()
    report = {"model": args.model, "opset": OPSET, "rope_modules_patched": n_rope, **info,
              "meta": {k: v for k, v in run_meta.collect(ROOT).items() if k != "all_packages"},
              "parts": {}}  # fmt: skip

    specs = part_specs(model)
    for name, (module, in_names, out_names, dyn, export_case) in specs.items():
        if args.parts and name not in args.parts:
            continue
        module.eval()
        res: dict = {"inputs": in_names, "outputs": out_names}
        report["parts"][name] = res
        check_cases = [c for c in cases if c.startswith(name) or c == name]
        # 1) разбиение (вещественный RoPE) против исходной модели, PyTorch
        with torch.inference_mode():
            res["torch_vs_ref"] = {}
            for c in check_cases:
                got = module(*cases[c]["inputs"])
                got = got if isinstance(got, tuple) else (got,)
                res["torch_vs_ref"][c] = [compare(k, to_np(g), to_np(r)) for g, (k, r)
                                          in zip(got, cases[c]["ref"].items(), strict=False)]  # fmt: skip
        # 2) экспорт
        path = out_dir / f"{name}.onnx"
        t0 = time.perf_counter()
        try:
            dyn_shapes = None if dyn is None else dict(zip(in_names, dyn, strict=True))
            torch.onnx.export(module, cases[export_case]["inputs"], path, input_names=in_names,
                              output_names=out_names, opset_version=OPSET, dynamo=True,
                              dynamic_shapes=dyn_shapes)  # fmt: skip
            proto = onnx.load(path)
            onnx.checker.check_model(proto)
            ops = Counter(n.op_type for n in proto.graph.node)
            res |= {"export": "ok", "export_s": round(time.perf_counter() - t0, 1),
                    "size_mb": round(path.stat().st_size / 2**20, 1), "ir_version": proto.ir_version,
                    "n_nodes": len(proto.graph.node), "op_types": dict(sorted(ops.items()))}  # fmt: skip
        except Exception as e:  # noqa: BLE001 — фиксируем причину отказа экспорта в отчёте
            res |= {"export": "failed", "error": f"{type(e).__name__}: {e}"[:3000],
                    "traceback": traceback.format_exc()[-3000:]}  # fmt: skip
            print(f"[{args.model}] {name}: EXPORT FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            continue
        # 3) ONNX Runtime (CUDA EP, FP32 без TF32) против исходной модели
        try:
            sess = ort_session(path)
            res["ort_vs_ref"] = {}
            for c in check_cases:
                feed = {n: to_np(x) for n, x in zip(in_names, cases[c]["inputs"], strict=True)}
                outs = sess.run(None, feed)
                res["ort_vs_ref"][c] = [compare(k, o, to_np(r)) for o, (k, r)
                                        in zip(outs, cases[c]["ref"].items(), strict=False)]  # fmt: skip
            times = []
            for _ in range(20):
                t1 = time.perf_counter()
                sess.run(None, feed)
                times.append((time.perf_counter() - t1) * 1000)
            res["ort_ms_median_incl_copies"] = float(np.median(times[5:]))
        except Exception as e:  # noqa: BLE001
            res["ort"] = f"failed: {type(e).__name__}: {e}"[:3000]
        worst = max((r["max_abs"] for v in res.get("ort_vs_ref", {}).values() for r in v), default=None)
        print(f"[{args.model}] {name}: export ok ({res['size_mb']} MB, {res['n_nodes']} nodes), "
              f"ORT max_abs vs ref {worst}", flush=True)  # fmt: skip

    (rep_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"report: {rep_dir / 'report.json'}")


if __name__ == "__main__":
    main()
