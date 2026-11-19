"""Сквозная проверка цепочки PyTorch -> ONNX -> ONNX Runtime на малой модели (этап 2).

ResNet-18 со случайными весами (без скачивания) экспортируется в ONNX с opset <= 24
(предел парсера TensorRT 10.16), затем выходы ONNX Runtime (CUDA EP) сверяются с PyTorch.
Полученный .onnx дальше собирается в engine через trtexec на Windows.

Запуск (WSL):  uv run --no-sync python export/smoke_export.py
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

OPSET = 18
SEED = 42
INPUT_SHAPE = (1, 3, 224, 224)  # (B, C, H, W)
OUT_PATH = Path(__file__).resolve().parents[2] / "models" / "smoke" / "resnet18_rand.onnx"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    set_seed(SEED)
    model = torchvision.models.resnet18(weights=None).eval()
    x = torch.randn(*INPUT_SHAPE)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (x,),
        OUT_PATH,
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=True,
    )
    proto = onnx.load(OUT_PATH)
    onnx.checker.check_model(proto)
    opset = next(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx"))
    print(f"Exported {OUT_PATH} | IR {proto.ir_version} | opset {opset}")

    with torch.no_grad():
        ref = model(x).numpy()  # (1, 1000)

    ort.preload_dlls()
    # Сверка экспорта — в строгом FP32: по умолчанию CUDA EP на Ampere+ считает свёртки в TF32
    # (10 бит мантиссы), что даёт ~1e-3 расхождения и маскирует ошибки экспорта.
    sess = ort.InferenceSession(
        OUT_PATH, providers=[("CUDAExecutionProvider", {"use_tf32": 0})]
    )
    out = sess.run(None, {"input": x.numpy()})[0]
    max_abs = float(np.abs(out - ref).max())
    same_top1 = int(out.argmax()) == int(ref.argmax())
    print(f"ORT providers: {sess.get_providers()}")
    print(f"max |ORT(FP32, no TF32) - torch| = {max_abs:.2e}, top-1 match: {same_top1}")

    sess_tf32 = ort.InferenceSession(OUT_PATH, providers=["CUDAExecutionProvider"])
    out_tf32 = sess_tf32.run(None, {"input": x.numpy()})[0]
    print(f"max |ORT(TF32 default) - torch| = {float(np.abs(out_tf32 - ref).max()):.2e} (for info)")

    ok = max_abs < 1e-4 and same_top1
    print("SMOKE EXPORT: OK" if ok else "SMOKE EXPORT: MISMATCH")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
