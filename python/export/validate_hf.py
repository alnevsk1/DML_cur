"""Validate the pinned HF SAM video bundle on real frames and prepare TensorRT inputs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "models" / "onnx" / "sam2.1_tiny_hf"
OUT = ROOT / "runs" / "validation" / "sam2.1_tiny_hf"
PARTS = ("vision_encoder", "mask_decoder", "memory_encoder", "pointer_tpos", "memory_attention")


def as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def difference(actual: np.ndarray, reference: np.ndarray) -> dict:
    if actual.shape != reference.shape:
        return {
            "shape_match": False,
            "actual": list(actual.shape),
            "reference": list(reference.shape),
        }
    delta = np.abs(actual - reference)
    return {
        "shape_match": True,
        "finite": bool(np.isfinite(actual).all() and np.isfinite(reference).all()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "allclose_1e3": bool(np.allclose(actual, reference, atol=1e-3, rtol=1e-3)),
    }


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def prepare(frames: int) -> None:
    """Exercise the five CUDA graphs, compare parts with upstream and track a short clip.

    The fixed-bank padding follows the HF card. Video parity against the upstream
    dynamic-bank predictor is reported separately; the two bank policies can differ.
    """
    import onnx
    import onnxruntime as ort
    import pandas as pd
    import torch
    from PIL import Image
    from sam2.modeling.sam2_utils import get_1d_sine_pe

    from benchmark import pvs_models, run_pvs

    OUT.mkdir(parents=True, exist_ok=True)
    run_pvs.set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = {}
    report: dict = {"seed": 42, "frames": frames, "parts": {}, "video": []}
    for part in PARTS:
        path = BUNDLE / f"{part}.onnx"
        proto = onnx.load(path)
        onnx.checker.check_model(proto)
        sessions[part] = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": 0})],
        )
        if sessions[part].get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider unavailable for {part}")
        report["parts"][part] = {
            "ir": proto.ir_version,
            "opsets": {o.domain: o.version for o in proto.opset_import},
            "inputs": [
                {"name": x.name, "shape": x.shape, "type": x.type}
                for x in sessions[part].get_inputs()
            ],
        }
        print(f"checked and loaded: {part}", flush=True)
    report["versions"] = {
        "onnx": onnx.__version__,
        "ort": ort.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    constants = json.loads((BUNDLE / "constants.json").read_text(encoding="utf-8"))
    tpos = np.asarray(constants["memory_temporal_positional_encoding"], dtype=np.float32)
    gt = json.loads(run_pvs.SAMPLE_GT.read_text(encoding="utf-8"))["77"]
    paths = [run_pvs.SEQ_ROOT / name for name in gt["file_names"][:frames]]
    table = pd.read_csv(run_pvs.SAMPLE)
    track = table.loc[table.ann_id == 77].iloc[0]
    model = pvs_models.build_predictor("sam2.1_tiny")
    upstream_ctx = run_pvs.install_loader_and_timers(model)
    state = model.init_state(video_path=paths, offload_video_to_cpu=True)
    model.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=np.array([[track.click_x, track.click_y]], dtype=np.float32),
        labels=np.array([1], dtype=np.int32),
    )
    upstream_masks = {}
    with torch.inference_mode():
        for index, _, logits in model.propagate_in_video(state):
            upstream_masks[index] = as_numpy(logits[0, 0])
    upstream_ctx["loader"].close()
    del state
    gc.collect()
    captured = {}

    def run(part: str, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        feed = {
            name: np.ascontiguousarray(value) if value.ndim else value
            for name, value in feed.items()
        }
        values = sessions[part].run(None, feed)
        result = dict(zip([x.name for x in sessions[part].get_outputs()], values, strict=True))
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError(f"Nonfinite output in {part}")
        captured[part] = (feed, result)
        return result

    memories = []
    predicted_masks = []
    with torch.inference_mode():
        for index, path in enumerate(paths):
            # Match the existing benchmark's PIL preprocessing exactly.
            tensor = upstream_ctx["loader"]._load(index).float().unsqueeze(0)
            pixel_values = tensor.numpy()
            features = run("vision_encoder", {"pixel_values": pixel_values})
            image = Image.open(path).convert("RGB")
            width, height = image.size
            if index == 0:
                conditioned = features["feats2_no_mem"]
                points = np.array(
                    [[[[track.click_x * 1024 / width, track.click_y * 1024 / height]]]],
                    dtype=np.float32,
                )
                labels = np.ones((1, 1, 1), dtype=np.int32)
            else:
                selected = [memories[0], *memories[max(1, index - 6) : index]]
                memory = [m["memory_tokens"] for m in selected]
                positions = [
                    m["memory_pos"] + tpos[6 if m["index"] == 0 else index - m["index"] - 1]
                    for m in selected
                ]
                while len(memory) < 7:
                    memory.append(memory[-1])
                    positions.append(positions[-1])
                pointers = [memories[0], *reversed(memories[max(1, index - 15) : index])]
                while len(pointers) < 16:
                    pointers.append(pointers[-1])
                diffs = np.array(
                    [(index - m["index"]) / max(min(len(paths), 16) - 1, 1) for m in pointers],
                    dtype=np.float32,
                )
                pointer_pos = run("pointer_tpos", {"normalized_diffs": diffs})["pointer_pos"]
                pointer_tokens = np.concatenate(
                    [m["object_pointer"].reshape(4, 1, 64) for m in pointers], axis=0
                )
                attention = run(
                    "memory_attention",
                    {
                        "current_vision_features": features["feats2"]
                        .reshape(1, 256, 4096)
                        .transpose(2, 0, 1),
                        "current_vision_position_embeddings": features["vision_pos_embed"]
                        .reshape(1, 256, 4096)
                        .transpose(2, 0, 1),
                        "memory": np.concatenate([*memory, pointer_tokens], axis=0),
                        "memory_pos": np.concatenate(
                            [*positions, np.repeat(pointer_pos, 4, axis=0)[:, None]], axis=0
                        ),
                    },
                )
                conditioned = attention["conditioned_feats"]
                points = np.zeros((1, 1, 1, 2), dtype=np.float32)
                labels = np.full((1, 1, 1), -1, dtype=np.int32)
            decoded = run(
                "mask_decoder",
                {
                    "feats0": features["feats0"],
                    "feats1": features["feats1"],
                    "feats2_cond": conditioned,
                    "input_points": points,
                    "input_labels": labels,
                },
            )
            encoded = run(
                "memory_encoder",
                {
                    "feats2": features["feats2"],
                    "high_res_mask": decoded["high_res_mask"],
                    "object_score_logits": decoded["object_score_logits"].reshape(1, 1),
                    "binarize": np.array(float(index == 0), dtype=np.float32),
                },
            )
            memories.append(
                {"index": index, **encoded, "object_pointer": decoded["object_pointer"]}
            )
            high_res = torch.from_numpy(decoded["high_res_mask"])
            mask = torch.nn.functional.interpolate(
                high_res, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0].numpy()
            predicted_masks.append(mask > 0)
            iou = mask_iou(mask, upstream_masks[index])
            report["video"].append(
                {
                    "frame": index,
                    "iou_vs_upstream": iou,
                    "positive_pixels": int((mask > 0).sum()),
                    "object_score": float(decoded["object_score_logits"].reshape(-1)[0]),
                }
            )
            print(f"frame {index}: IoU vs upstream = {iou:.6f}", flush=True)

        # Isolate graph parity from differences in fixed/dynamic video memory policy.
        for part, (feed, expected) in captured.items():
            x = {name: torch.from_numpy(np.asarray(value)).cuda() for name, value in feed.items()}
            if part == "vision_encoder":
                output = model.forward_image(x["pixel_values"])
                f0, f1, f2 = output["backbone_fpn"]
                refs = {
                    "feats0": f0,
                    "feats1": f1,
                    "feats2": f2,
                    "feats2_no_mem": f2 + model.no_mem_embed.reshape(1, 256, 1, 1),
                    "vision_pos_embed": output["vision_pos_enc"][-1],
                }
            elif part == "mask_decoder":
                h = model._forward_sam_heads(
                    x["feats2_cond"],
                    {
                        "point_coords": x["input_points"].reshape(1, -1, 2),
                        "point_labels": x["input_labels"].reshape(1, -1),
                    },
                    None,
                    [x["feats0"], x["feats1"]],
                    multimask_output=True,
                )
                refs = {
                    "low_res_mask": h[3],
                    "high_res_mask": h[4],
                    "iou": h[2].max(dim=1, keepdim=True).values,
                    "object_pointer": h[5].reshape(1, 1, 256),
                    "object_score_logits": h[6].reshape(1, 1, 1),
                }
            elif part == "memory_encoder":
                f, p = model._encode_new_memory(
                    [x["feats2"].flatten(2).permute(2, 0, 1)],
                    [(64, 64)],
                    x["high_res_mask"],
                    x["object_score_logits"],
                    bool(x["binarize"].item()),
                )
                refs = {
                    "memory_tokens": f.flatten(2).permute(2, 0, 1),
                    "memory_pos": p[-1].flatten(2).permute(2, 0, 1),
                }
            elif part == "pointer_tpos":
                refs = {
                    "pointer_pos": model.obj_ptr_tpos_proj(
                        get_1d_sine_pe(x["normalized_diffs"], dim=256)
                    )
                }
            else:
                value = model.memory_attention(
                    curr=[x["current_vision_features"]],
                    curr_pos=[x["current_vision_position_embeddings"]],
                    memory=x["memory"],
                    memory_pos=x["memory_pos"],
                    num_obj_ptr_tokens=64,
                )
                refs = {"conditioned_feats": value.permute(1, 2, 0).reshape(1, 256, 64, 64)}
            checks = {
                name: difference(value, as_numpy(refs[name])) for name, value in expected.items()
            }
            if part == "mask_decoder":
                checks["mask_iou_vs_upstream_part"] = mask_iou(
                    expected["low_res_mask"], as_numpy(refs["low_res_mask"])
                )
            report["parts"][part]["ort_vs_upstream_same_inputs"] = checks
            case_dir = OUT / "cases" / part
            case_dir.mkdir(parents=True, exist_ok=True)
            for name, value in feed.items():
                value.tofile(case_dir / f"{name}.bin")
            for name, value in expected.items():
                np.save(case_dir / f"expected_{name}.npy", value)
            report["parts"][part]["output_shapes"] = {
                name: list(value.shape) for name, value in expected.items()
            }
            print(f"part parity: {part}: {checks}", flush=True)
    report["video_min_iou"] = min(row["iou_vs_upstream"] for row in report["video"])
    (OUT / "ort_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "video_masks.npz", **{f"frame_{i}": m for i, m in enumerate(predicted_masks)}
    )
    print(f"report: {OUT / 'ort_report.json'}", flush=True)


def compare_trt() -> None:
    """Compare native TensorRT raw outputs with the recorded CUDA ORT outputs."""
    results: dict = {}
    for precision in ("fp32", "fp16"):
        results[precision] = {}
        for part in PARTS:
            outputs = {}
            for reference_path in sorted((OUT / "cases" / part).glob("expected_*.npy")):
                name = reference_path.stem.removeprefix("expected_")
                candidates = list(
                    (OUT / "trt_outputs" / f"{part}_{precision}").glob(f"{name}.output.*.fp32.raw")
                )
                if len(candidates) != 1:
                    raise RuntimeError(
                        f"Expected one TensorRT output for {precision}/{part}/{name}"
                    )
                reference = np.load(reference_path)
                actual = np.fromfile(candidates[0], dtype=np.float32).reshape(reference.shape)
                check = difference(actual, reference)
                check["relative_l2"] = float(
                    np.linalg.norm(actual - reference)
                    / max(float(np.linalg.norm(reference)), 1e-12)
                )
                if "mask" in name:
                    check["mask_iou"] = mask_iou(actual, reference)
                outputs[name] = check
            results[precision][part] = outputs
            worst = max(x["max_abs"] for x in outputs.values())
            finite = all(x["finite"] for x in outputs.values())
            print(f"{precision}/{part}: finite={finite}, max_abs={worst:.6g}", flush=True)
            if part == "mask_decoder":
                print(
                    f"  low/high mask IoU: {outputs['low_res_mask']['mask_iou']:.6f} / "
                    f"{outputs['high_res_mask']['mask_iou']:.6f}",
                    flush=True,
                )
    results["criteria"] = {
        "fp32": "finite and allclose(atol=1e-3, rtol=1e-3)",
        "fp16": "finite and relative L2 <= 0.01; decoder mask IoU >= 0.99",
        "scope": "same-input isolated graphs; not a chained TensorRT video evaluation",
    }
    results["passed"] = all(
        check["finite"]
        and (
            check["allclose_1e3"]
            if precision == "fp32"
            else check["relative_l2"] <= 0.01 and check.get("mask_iou", 1.0) >= 0.99
        )
        for precision in ("fp32", "fp16")
        for part in results[precision].values()
        for check in part.values()
    )
    (OUT / "trt_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ort_path = OUT / "ort_report.json"
    ort_report = json.loads(ort_path.read_text(encoding="utf-8"))
    ort_report["criteria"] = {
        "parts": "finite and allclose(atol=1e-3, rtol=1e-3) against upstream on same inputs",
        "video_smoke": "minimum mask IoU >= 0.99 on this ten-frame clip",
        "scope": "AeroVIS ann 77, click; fixed-bank ONNX versus dynamic-bank upstream",
    }
    ort_report["parts_passed"] = all(
        check["finite"] and check["allclose_1e3"]
        for part in ort_report["parts"].values()
        for check in part["ort_vs_upstream_same_inputs"].values()
        if isinstance(check, dict)
    )
    ort_report["video_smoke_passed"] = ort_report["video_min_iou"] >= 0.99
    ort_path.write_text(json.dumps(ort_report, indent=2), encoding="utf-8")
    if (
        not results["passed"]
        or not ort_report["parts_passed"]
        or not ort_report["video_smoke_passed"]
    ):
        raise SystemExit("Validation criteria failed; inspect the JSON reports")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--compare-trt", action="store_true")
    args = parser.parse_args()
    if args.compare_trt:
        compare_trt()
        return
    if args.frames < 2:
        raise ValueError("At least two frames are required to exercise memory attention")
    prepare(args.frames)


if __name__ == "__main__":
    main()
