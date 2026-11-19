"""Prepare independent fixed-bank ORT video references and validate the C++ tracker.

Run only through the authorized WSL uv environment. Existing benchmark/validation
artifacts are read-only inputs; every new output directory must initially be empty.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from export.validate_hf import PARTS, difference, mask_iou

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "validation" / "sam2.1_tiny_cpp"
BUNDLE = ROOT / "models" / "onnx" / "sam2.1_tiny_hf"


def set_seed(seed: int) -> None:
    # ORT-only inference uses no PyTorch RNG; avoid loading both CUDA runtimes.
    random.seed(seed)
    np.random.seed(seed)


def environment() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "git_sha": None
        if not (ROOT / ".git").exists()
        else subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip(),
        "gpu_driver": gpu.stdout.strip(),
        "all_packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions()),
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def fresh(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def windows_path(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) < 4 or parts[1] != "mnt" or len(parts[2]) != 1:
        raise ValueError(f"Expected WSL drive path: {path}")
    return f"{parts[2].upper()}:/{'/'.join(parts[3:])}"


def pixels(rgb: np.ndarray, constants: dict) -> np.ndarray:
    image = Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.BICUBIC)
    mean = np.asarray(constants["image_mean"], dtype=np.float32)
    std = np.asarray(constants["image_std"], dtype=np.float32)
    normalized = (np.asarray(image).astype(np.float64) / 255.0 - mean) / std
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], dtype=np.float32)


def prepare(frames: int) -> None:
    """Run an ORT-only video chain with its own outputs, independent of C++ state."""
    import cv2
    import onnxruntime as ort
    import pandas as pd

    set_seed(42)
    constants = json.loads((BUNDLE / "constants.json").read_text(encoding="utf-8"))
    temporal = np.asarray(constants["memory_temporal_positional_encoding"], dtype=np.float32)
    out = BASE / f"clip_{frames}"
    fresh(out)
    gt = json.loads((ROOT / "data" / "annotations" / "aerovis_sample_v1_gt.json").read_text())["77"]
    paths = [
        ROOT / "data" / "external" / "AeroVIS" / "sequences" / n for n in gt["file_names"][:frames]
    ]
    if len(paths) != frames:
        raise ValueError("Not enough source frames")
    table = pd.read_csv(ROOT / "data" / "annotations" / "aerovis_sample_v1.csv")
    track = table.loc[table.ann_id == 77].iloc[0]
    inputs = out / "frames"
    inputs.mkdir()
    manifest = {"frames": [], "click_x": float(track.click_x), "click_y": float(track.click_y)}
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            destination = inputs / f"{i:04d}.png"
            rgb.save(destination)
            manifest["frames"].append(windows_path(destination))
    write_json(out / "manifest.json", manifest)
    metadata = environment()
    metadata.update(
        {
            "seed": 42,
            "scope": "fixed-bank ORT versus C++ chain, not GT quality or benchmark",
            "frames": frames,
            "ann_id": 77,
            "postprocess_backend": "opencv",
            "command": sys.argv,
            "uv_lock_sha256": hashlib.sha256(
                (ROOT / "python" / "uv.lock").read_bytes()
            ).hexdigest(),
            "source_frames": [str(p.relative_to(ROOT)) for p in paths],
            "sources_sha256": [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],
            "manifest": manifest,
            "trace_shapes": {},
        }
    )
    write_json(out / "reference_metadata.json", metadata)
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    sessions = {
        part: ort.InferenceSession(
            str(BUNDLE / f"{part}.onnx"),
            sess_options=options,
            providers=[("CUDAExecutionProvider", {"use_tf32": 0})],
        )
        for part in PARTS
    }
    if any(s.get_providers()[0] != "CUDAExecutionProvider" for s in sessions.values()):
        raise RuntimeError("ORT CUDA unavailable")
    memories = []
    rows = []
    ref = out / "ort"
    for index in range(frames):
        directory = ref / f"frame_{index:04d}"
        directory.mkdir(parents=True)

        def save(name: str, value: np.ndarray, directory: Path = directory) -> None:
            np.save(directory / f"{name}.npy", value)
            metadata["trace_shapes"][name] = list(value.shape)

        def run(part: str, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            values = sessions[part].run(
                None, {k: np.ascontiguousarray(v) if v.ndim else v for k, v in feed.items()}
            )
            result = dict(zip([o.name for o in sessions[part].get_outputs()], values, strict=True))
            for name, value in result.items():
                if not np.isfinite(value).all():
                    raise RuntimeError(f"Nonfinite ORT output: {part}/{name}")
                save(f"{part}__{name}", value)
            return result

        with Image.open(inputs / f"{index:04d}.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        height, width = rgb.shape[:2]
        pixel_values = pixels(rgb, constants)
        save("pixel_values", pixel_values)
        features = run("vision_encoder", {"pixel_values": pixel_values})
        if index == 0:
            conditioned = features["feats2_no_mem"]
            points = np.asarray(
                [[[[track.click_x * 1024 / width, track.click_y * 1024 / height]]]],
                dtype=np.float32,
            )
            labels = np.ones((1, 1, 1), dtype=np.int32)
        else:
            selected = [memories[0], *memories[max(1, index - 6) : index]]
            bank = [m["memory_tokens"] for m in selected]
            positions = [
                m["memory_pos"] + temporal[6 if m["index"] == 0 else index - m["index"] - 1]
                for m in selected
            ]
            while len(bank) < 7:
                bank.append(bank[-1])
                positions.append(positions[-1])
            pointers = [memories[0], *reversed(memories[max(1, index - 15) : index])]
            while len(pointers) < 16:
                pointers.append(pointers[-1])
            diffs = np.asarray(
                [(index - m["index"]) / max(min(frames, 16) - 1, 1) for m in pointers],
                dtype=np.float32,
            )
            tpos = run("pointer_tpos", {"normalized_diffs": diffs})["pointer_pos"]
            pointer_tokens = np.concatenate(
                [m["object_pointer"].reshape(4, 1, 64) for m in pointers]
            )
            conditioned = run(
                "memory_attention",
                {
                    "current_vision_features": features["feats2"]
                    .reshape(1, 256, 4096)
                    .transpose(2, 0, 1),
                    "current_vision_position_embeddings": features["vision_pos_embed"]
                    .reshape(1, 256, 4096)
                    .transpose(2, 0, 1),
                    "memory": np.concatenate([*bank, pointer_tokens]),
                    "memory_pos": np.concatenate([*positions, np.repeat(tpos, 4, axis=0)[:, None]]),
                },
            )["conditioned_feats"]
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
                "binarize": np.asarray(float(index == 0), dtype=np.float32),
            },
        )
        memories.append({"index": index, **encoded, "object_pointer": decoded["object_pointer"]})
        restored = cv2.resize(
            decoded["high_res_mask"][0, 0], (width, height), interpolation=cv2.INTER_LINEAR
        )
        np.save(directory / "restored_logits.npy", restored)
        Image.fromarray((restored > 0).astype(np.uint8) * 255).save(directory / "mask.png")
        rows.append(
            {
                "frame": index,
                "positive_pixels": int((restored > 0).sum()),
                "object_score": float(decoded["object_score_logits"].item()),
            }
        )
        print(f"ORT frame {index}/{frames}: {rows[-1]}", flush=True)
    sessions.clear()
    print(f"ORT sessions released ({frames} frames)", flush=True)
    metadata["video"] = rows
    metadata["ort_completed"] = True
    write_json(out / "reference_metadata.json", metadata)
    if frames == 10:
        sample_dir = BASE / "preprocess_samples"
        fresh(sample_dir)
        rng = np.random.default_rng(42)
        for height, width in [(37, 53), (1080, 1920), (1024, 1024), (1024, 511), (499, 1024)]:
            Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(
                sample_dir / f"noise_{width}x{height}.png"
            )
        with Image.open(inputs / "0000.png") as image:
            image.save(sample_dir / "aerovis.png")


def compare_parts(precision: str) -> None:
    old = ROOT / "runs" / "validation" / "sam2.1_tiny_hf" / "cases"
    actual = BASE / f"parts_{precision}"
    rows = {}
    for part in PARTS:
        rows[part] = {}
        for path in (old / part).glob("expected_*.npy"):
            name = path.stem.removeprefix("expected_")
            ref = np.load(path)
            value = np.fromfile(actual / part / f"{name}.bin", dtype=np.float32).reshape(ref.shape)
            check = difference(value, ref)
            check["relative_l2"] = float(
                np.linalg.norm(value - ref) / max(np.linalg.norm(ref), 1e-12)
            )
            if "mask" in name:
                check["mask_iou"] = mask_iou(value, ref)
            rows[part][name] = check
    passed = all(
        c["finite"]
        and (
            c["allclose_1e3"]
            if precision == "fp32"
            else c["relative_l2"] <= 0.01 and c.get("mask_iou", 1) >= 0.99
        )
        for p in rows.values()
        for c in p.values()
    )
    write_json(actual / "report.json", {"precision": precision, "passed": passed, "parts": rows})
    print(f"C++ isolated {precision}: passed={passed}")
    if not passed:
        raise SystemExit(1)


def compare_preprocess() -> None:
    constants = json.loads((BUNDLE / "constants.json").read_text())
    rows = []
    for path in sorted((BASE / "preprocess_samples").glob("*.png")):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        expected = np.asarray(Image.fromarray(rgb).resize((1024, 1024), Image.Resampling.BICUBIC))
        with Image.open(BASE / "preprocess_cpp" / path.name) as image:
            actual = np.asarray(image.convert("RGB"))
        tensor = np.fromfile(
            BASE / "preprocess_cpp" / f"{path.stem}.bin", dtype=np.float32
        ).reshape(1, 3, 1024, 1024)
        rows.append(
            {
                "sample": path.name,
                "pixels_equal": bool(np.array_equal(actual, expected)),
                "tensor": difference(tensor, pixels(rgb, constants)),
            }
        )
    passed = bool(rows) and all(r["pixels_equal"] and r["tensor"]["max_abs"] == 0 for r in rows)
    write_json(BASE / "preprocess_cpp" / "report.json", {"passed": passed, "samples": rows})
    print(f"PIL vs C++ preprocessing: passed={passed}")
    if not passed:
        raise SystemExit(1)


def compare_video(frames: int, precision: str) -> None:
    """Compare two closed video chains, never injecting ORT memory into TensorRT."""
    clip = BASE / f"clip_{frames}"
    actual_root = clip / precision
    metadata = json.loads((clip / "reference_metadata.json").read_text())
    if not metadata.get("ort_completed"):
        raise RuntimeError("Incomplete ORT reference")
    rows = []
    for index in range(frames):
        name = f"frame_{index:04d}"
        ref_dir, actual_dir = clip / "ort" / name, actual_root / name
        traces = {}
        for path in sorted(ref_dir.glob("*.npy")):
            if path.stem == "restored_logits":
                continue
            ref = np.load(path)
            value = np.fromfile(actual_dir / f"{path.stem}.bin", dtype=np.float32).reshape(
                ref.shape
            )
            check = difference(value, ref)
            check["relative_l2"] = float(
                np.linalg.norm(value - ref) / max(float(np.linalg.norm(ref)), 1e-12)
            )
            if "mask" in path.stem:
                check["mask_iou"] = mask_iou(value, ref)
            traces[path.stem] = check
        ref_logits = np.load(ref_dir / "restored_logits.npy")
        with Image.open(actual_dir / "mask.png") as image:
            actual_mask = np.asarray(image) > 0
        iou = mask_iou(actual_mask, ref_logits)
        high = traces["mask_decoder__high_res_mask"]
        row = {
            "frame": index,
            "iou_restored": iou,
            "iou_1024": high["mask_iou"],
            "positive_pixels": int(actual_mask.sum()),
            "traces": traces,
        }
        # Closed-chain mask parity is the gate; isolated allclose criteria do not
        # transfer to feedback trajectories. Record every tensor error for audit.
        row["passed"] = (
            iou >= 0.99
            and high["mask_iou"] >= 0.99
            and all(c["finite"] for c in traces.values())
            and traces["pixel_values"]["max_abs"] == 0
        )
        rows.append(row)
        print(
            f"{precision} frame {index}: IoU={iou:.8f}, high_logits_max_abs={high['max_abs']:.6g}"
        )
    passed = all(r["passed"] for r in rows)
    report = {
        "precision": precision,
        "frames": frames,
        "passed": passed,
        "minimum_iou": min(r["iou_restored"] for r in rows),
        "criteria": "finite traces; exact pixels; IoU >= 0.99 at both resolutions",
        "scope": "closed fixed-bank chains; diagnostic timing includes trace I/O",
        "video": rows,
    }
    write_json(actual_root / "report.json", report)
    (actual_root / "metrics.csv").write_text(
        "frame,iou_restored,iou_1024,positive_pixels\n"
        + "".join(
            f"{r['frame']},{r['iou_restored']},{r['iou_1024']},{r['positive_pixels']}\n"
            for r in rows
        )
    )
    print(f"Video {precision}/{frames}: passed={passed}; minimum IoU={report['minimum_iou']}")
    if not passed:
        raise SystemExit(1)


def make_demo() -> None:
    """Create a small local GUI smoke video from the validation frames."""
    import cv2

    source = BASE / "clip_10" / "frames"
    paths = sorted(source.glob("*.png"))
    if not paths:
        raise RuntimeError("Prepare ten-frame fixture first")
    first = cv2.imread(str(paths[0]))
    destination = BASE / "demo.avi"
    if destination.exists():
        raise FileExistsError(destination)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"MJPG"), 5, (first.shape[1], first.shape[0])
    )
    if not writer.isOpened():
        raise RuntimeError("Existing OpenCV cannot encode MJPG demo")
    try:
        for path in paths:
            writer.write(cv2.imread(str(path)))
    finally:
        writer.release()
    print(f"GUI demo: {destination}")


def diagnose_decoder() -> None:
    """Replay the decoder on recorded FP16 inputs to isolate arithmetic from feedback."""
    import onnxruntime as ort

    out = BASE / "decoder_frame5"
    fresh(out)
    actual = BASE / "clip_10" / "fp16" / "frame_0005"
    reference = BASE / "clip_10" / "ort" / "frame_0005"
    sources = {
        "feats0": "vision_encoder__feats0",
        "feats1": "vision_encoder__feats1",
        "feats2_cond": "memory_attention__conditioned_feats",
    }
    feed = {}
    for name, trace in sources.items():
        shape = np.load(reference / f"{trace}.npy").shape
        feed[name] = np.fromfile(actual / f"{trace}.bin", dtype=np.float32).reshape(shape)
    feed["input_points"] = np.zeros((1, 1, 1, 2), dtype=np.float32)
    feed["input_labels"] = np.full((1, 1, 1), -1, dtype=np.int32)
    case = out / "inputs" / "mask_decoder"
    case.mkdir(parents=True)
    for name, value in feed.items():
        value.tofile(case / f"{name}.bin")
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(BUNDLE / "mask_decoder.onnx"),
        sess_options=options,
        providers=[("CUDAExecutionProvider", {"use_tf32": 0})],
    )
    values = session.run(None, feed)
    checks = {}
    for output, value in zip(session.get_outputs(), values, strict=True):
        np.save(out / f"expected_{output.name}.npy", value)
        ref = np.load(reference / f"mask_decoder__{output.name}.npy")
        native = np.fromfile(actual / f"mask_decoder__{output.name}.bin", dtype=np.float32).reshape(
            ref.shape
        )
        checks[output.name] = {
            "ort_replay_vs_ort_chain": difference(value, ref),
            "ort_replay_vs_trt_fp16_same_inputs": difference(value, native),
        }
        if "mask" in output.name:
            checks[output.name]["iou_vs_ort_chain"] = mask_iou(value, ref)
            checks[output.name]["iou_vs_trt_fp16_same_inputs"] = mask_iou(value, native)
    write_json(out / "report.json", {"ort": ort.__version__, "checks": checks})
    print(json.dumps(checks, indent=2), flush=True)
    del session
    print("Decoder replay released", flush=True)


def compare_postprocess() -> None:
    """Check OpenCV restoration against already recorded Torch align_corners=False outputs."""
    import cv2

    rows = []
    for frames in (10, 24):
        for directory in sorted((BASE / f"clip_{frames}" / "ort").glob("frame_*")):
            reference = np.load(directory / "restored_logits.npy")
            high = np.load(directory / "mask_decoder__high_res_mask.npy")[0, 0]
            restored = cv2.resize(
                high, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR
            )
            rows.append(
                {
                    "clip": frames,
                    "frame": directory.name,
                    "max_abs": float(np.abs(restored - reference).max()),
                    "mask_iou": mask_iou(restored, reference),
                }
            )
    passed = len(rows) == 34 and all(r["mask_iou"] == 1 for r in rows)
    write_json(BASE / "postprocess_report.json", {"passed": passed, "frames": rows})
    print(f"OpenCV versus recorded Torch restoration: passed={passed}")
    if not passed:
        raise SystemExit(1)


def summarize() -> None:
    """Collect the saved checks without rerunning inference or rewriting prior reports."""
    replay = BASE / "decoder_frame5"
    decoder_checks = {}
    for path in sorted(replay.glob("expected_*.npy")):
        name = path.stem.removeprefix("expected_")
        ref = np.load(path)
        value = np.fromfile(
            replay / "trt_fp32" / "mask_decoder" / f"{name}.bin", dtype=np.float32
        ).reshape(ref.shape)
        check = difference(value, ref)
        if "mask" in name:
            check["mask_iou"] = mask_iou(value, ref)
        decoder_checks[name] = check
    decoder_passed = bool(decoder_checks) and all(
        c["finite"] and c["allclose_1e3"] for c in decoder_checks.values()
    )
    write_json(
        replay / "trt_fp32" / "report.json", {"passed": decoder_passed, "outputs": decoder_checks}
    )
    videos = []
    for frames, precision in [
        (10, "fp32"),
        (24, "fp32"),
        (10, "fp16"),
        (10, "mixed"),
        (24, "mixed"),
    ]:
        report = json.loads((BASE / f"clip_{frames}" / precision / "report.json").read_text())
        videos.append({k: report[k] for k in ("frames", "precision", "passed", "minimum_iou")})
    sources = [ROOT / "cpp" / "CMakeLists.txt", Path(__file__)]
    sources.extend(sorted((ROOT / "cpp" / "src").rglob("*.hpp")))
    sources.extend(
        p for p in sorted((ROOT / "cpp" / "src").rglob("*.cpp")) if p.name != "env_check.cpp"
    )
    summary = {
        "videos": videos,
        "preprocessing_exact": json.loads((BASE / "preprocess_cpp" / "report.json").read_text())[
            "passed"
        ],
        "postprocessing_masks_exact": json.loads((BASE / "postprocess_report.json").read_text())[
            "passed"
        ],
        "isolated_parts": {
            p: json.loads((BASE / f"parts_{p}" / "report.json").read_text())["passed"]
            for p in ("fp32", "fp16")
        },
        "decoder_fp32_same_input_passed": decoder_passed,
        "scripted_gui_passed": bool(
            json.loads((BASE / "ui_smoke" / "report.json").read_text())["passed"]
        ),
        "binary_sha256": hashlib.sha256(
            (ROOT / "cpp" / "build" / "release" / "uav_tracker.exe").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "engines_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "models" / "engines" / "sam2.1_tiny_hf").glob("*.engine"))
        },
        "limits": [
            "One AeroVIS track, 10/24 frames; parity versus ORT, not GT quality.",
            "FP16 decoder fails on frame 5; mixed uses FP32 decoder and four FP16 engines.",
            "Physical mouse/keyboard input unverified: computer-use sandbox error 267.",
            "ORT workers stalled after saving results; interrupted. Native C++ exits normally.",
            "Diagnostic timing includes trace I/O; no application benchmark.",
        ],
    }
    write_json(BASE / "summary.json", summary)
    print(
        json.dumps({"videos": videos, "decoder_fp32_same_input_passed": decoder_passed}, indent=2)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "compare-parts",
            "compare-preprocess",
            "compare-video",
            "make-demo",
            "diagnose-decoder",
            "compare-postprocess",
            "summarize",
        ),
    )
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--precision", choices=("fp32", "fp16", "mixed"), default="fp32")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("At least two frames required")
    if args.action == "prepare":
        prepare(args.frames)
    elif args.action == "compare-parts":
        compare_parts(args.precision)
    elif args.action == "compare-preprocess":
        compare_preprocess()
    elif args.action == "diagnose-decoder":
        diagnose_decoder()
    elif args.action == "compare-postprocess":
        compare_postprocess()
    elif args.action == "summarize":
        summarize()
    elif args.action == "make-demo":
        make_demo()
    else:
        compare_video(args.frames, args.precision)


if __name__ == "__main__":
    main()
