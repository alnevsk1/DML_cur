"""Обвязка VOS-моделей (XMem, Cutie, DeAOT) с единым интерфейсом для бенчмарка (этап 4).

Инициализация — только маской первого кадра трека (так эти модели обучены и оцениваются).
Параметры памяти — как в официальной оценке авторов на DAVIS 2017 / YouTube-VOS:
  * XMem (eval.py по умолчанию): mem_every=5, top_k=30, долговременная память включена,
    T_min/T_max = 5/10, P = 128, LT_max = 10 000; count_usage — по длине трека, как в eval.py;
  * Cutie-base (eval_config.yaml, набор d17-val): mem_every=5, без долговременной памяти,
    рабочая память — 5 кадров + постоянная память кадра инициализации;
  * DeAOT (configs/models/*): долговременная память — кадр инициализации плюс каждый
    TEST_LONG_TERM_MEM_GAP-й кадр (T/S: 9999 = только кадр инициализации; R50-L: 5 —
    память растёт с длиной трека), кратковременная — предыдущий кадр.

Размер входа задаёт вызывающий код (короткая сторона кадра); у DeAOT стороны дополнительно
выравниваются до 16k + 1, как в его MultiRestrictSize (align_corners=True).
Выход каждого шага — бинарная маска объекта на GPU в исходном разрешении кадра: вероятности
(логиты у DeAOT) интерполируются билинейно, как в eval-скриптах авторов, затем argmax.

Каждая модель — в отдельном процессе с PYTHONPATH=third_party/<repo>: у XMem пакеты верхнего
уровня `model`, `inference`, `util`, у aot-benchmark — `networks`, `utils`, `configs`.
DeAOT: CUDA-расширение `spatial_correlation_sampler` не установлено, используется
реализация на PyTorch из репозитория (медленнее, чем в статье).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "models" / "checkpoints"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VOSModel:
    repo: str  # каталог в python/third_party/
    checkpoint: Path
    variant: str = ""  # имя конфига модели (только DeAOT)


MODELS: dict[str, VOSModel] = {
    "xmem": VOSModel("XMem", CKPT / "xmem" / "XMem.pth"),
    "cutie_base": VOSModel("Cutie", CKPT / "cutie" / "cutie-base-mega.pth"),
    "deaot_t": VOSModel("aot-benchmark", CKPT / "deaot" / "DeAOT-T_PRE_YTB_DAV.pth", "deaott"),
    "deaot_s": VOSModel("aot-benchmark", CKPT / "deaot" / "DeAOT-S_PRE_YTB_DAV.pth", "deaots"),
    "r50_deaot_l": VOSModel(
        "aot-benchmark", CKPT / "deaot" / "R50_DeAOT-L_PRE_YTB_DAV.pth", "r50_deaotl"
    ),
}  # fmt: skip


def _check_repo(module: str, repo: str) -> None:
    mod = importlib.import_module(module)
    path = mod.__file__ or next(iter(mod.__path__))  # cutie — namespace-пакет без __init__.py
    origin = Path(path).resolve().as_posix()
    if f"third_party/{repo}/" not in origin:
        raise RuntimeError(f"wrong package on PYTHONPATH for {repo}: {origin}")


def _disable_imagenet_download(module: str) -> None:
    """resnet18/50(pretrained=True) в XMem и Cutie качают ImageNet-веса из torch.hub.

    Эти веса всё равно перезаписываются чекпойнтом модели (загрузка строгая), поэтому
    функции подменяются на вызов с pretrained=False (как timm.create_model у EdgeTAM).
    """
    mod = importlib.import_module(module)
    for name in ("resnet18", "resnet50"):
        original = getattr(mod, name)

        def no_download(*args, _original=original, **kwargs):  # noqa: ANN002, ANN003, ANN202 — прокси к resnet*
            kwargs["pretrained"] = False
            return _original(*args, **kwargs)

        setattr(mod, name, no_download)


def short_side_size(h: int, w: int, short_side: int) -> tuple[int, int]:
    """Короткая сторона -> short_side с сохранением пропорций (как Resize(size) у XMem)."""
    if short_side <= 0:
        return h, w
    s = short_side / min(h, w)
    return int(h * s), int(w * s)


def _to_out(prob: torch.Tensor, out_hw: tuple[int, int], align_corners: bool) -> torch.Tensor:
    """(C, h, w) вероятности/логиты, канал 0 — фон -> бинарная маска (H, W) объекта 1."""
    if tuple(prob.shape[-2:]) != out_hw:
        prob = prob[None].float()
        prob = F.interpolate(prob, out_hw, mode="bilinear", align_corners=align_corners)[0]
    return prob.argmax(0) == 1


class VOSAdapter:
    """Общий интерфейс: start(кадр, маска) -> step(кадр) -> ... -> reset()."""

    mean: tuple[float, float, float] = IMAGENET_MEAN  # нормализация входа (в загрузчике)
    std: tuple[float, float, float] = IMAGENET_STD
    encoder_attrs: tuple[str, ...] = ()  # методы self.network, время которых = «энкодер»

    network: torch.nn.Module

    def internal_size(self, h: int, w: int, short_side: int) -> tuple[int, int]:
        return short_side_size(h, w, short_side)

    def start(self, image: torch.Tensor, mask: torch.Tensor, n_frames: int) -> None:
        """image: (3, h, w) на GPU; mask: (h, w) bool на GPU во внутреннем разрешении."""
        raise NotImplementedError

    def step(self, image: torch.Tensor, out_hw: tuple[int, int], last: bool) -> torch.Tensor:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def config(self) -> dict:
        return {}


class XMemAdapter(VOSAdapter):
    encoder_attrs = ("encode_key",)  # ResNet-50 + проекция ключа

    def __init__(self, spec: VOSModel) -> None:
        _check_repo("model.network", spec.repo)
        _disable_imagenet_download("model.resnet")
        from inference.inference_core import InferenceCore
        from model.network import XMem

        self._core_cls = InferenceCore
        # значения по умолчанию eval.py (протокол статьи)
        self.cfg = {
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
            "enable_long_term": True,
            "max_mid_term_frames": 10,
            "min_mid_term_frames": 5,
            "max_long_term_elements": 10000,
            "num_prototypes": 128,
            "single_object": False,
        }
        # XMem.__init__ грузит веса через load_state_dict (strict=True по умолчанию)
        self.network = XMem(self.cfg, str(spec.checkpoint), map_location="cpu").cuda().eval()
        self.core = None

    def start(self, image: torch.Tensor, mask: torch.Tensor, n_frames: int) -> None:
        cfg = dict(self.cfg)
        span = cfg["max_mid_term_frames"] - cfg["min_mid_term_frames"]
        cfg["enable_long_term_count_usage"] = cfg["enable_long_term"] and (
            n_frames / span * cfg["num_prototypes"] >= cfg["max_long_term_elements"]
        )
        self.core = self._core_cls(self.network, config=cfg)
        self.core.set_all_labels([1])
        self.core.step(image, mask[None].float(), [1], end=n_frames == 1)

    def step(self, image: torch.Tensor, out_hw: tuple[int, int], last: bool) -> torch.Tensor:
        prob = self.core.step(image, end=last)  # (2, h, w), с фоном
        return _to_out(prob, out_hw, align_corners=False)

    def reset(self) -> None:
        self.core = None

    def config(self) -> dict:
        return dict(self.cfg)


class CutieAdapter(VOSAdapter):
    mean = (0.0, 0.0, 0.0)  # Cutie нормализует вход внутри сети
    std = (1.0, 1.0, 1.0)
    encoder_attrs = ("encode_image", "transform_key")  # то, что делает ImageFeatureStore

    def __init__(self, spec: VOSModel) -> None:
        _check_repo("cutie", spec.repo)
        _disable_imagenet_download("cutie.model.utils.resnet")
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict

        cutie_pkg = Path(importlib.import_module("cutie.model").__file__).parents[1]
        with initialize_config_dir(
            version_base="1.3.2", config_dir=str(cutie_pkg / "config"), job_name="bench"
        ):
            cfg = compose(config_name="eval_config")
        with open_dict(cfg):
            cfg.weights = str(spec.checkpoint)
            cfg.dataset = (
                "d17-val"  # mem_every=5, use_long_term=False (как в статье на DAVIS/YT/MOSE)
            )
            cfg.max_internal_size = -1  # ресайз делает загрузчик бенчмарка
            get_dataset_cfg(cfg)
        self.cfg = cfg
        self._core_cls = InferenceCore
        self.network = CUTIE(cfg).cuda().eval()
        state = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
        self.network.load_weights(state)  # load_state_dict(strict=False) -> проверяем ключи сами
        mismatch = set(state) ^ set(self.network.state_dict())
        if mismatch:
            raise RuntimeError(f"Cutie checkpoint keys mismatch: {sorted(mismatch)[:10]}")
        self.core = None

    def start(self, image: torch.Tensor, mask: torch.Tensor, n_frames: int) -> None:
        self.core = self._core_cls(self.network, cfg=self.cfg)
        self.core.step(image, mask.long(), objects=[1], end=n_frames == 1)

    def step(self, image: torch.Tensor, out_hw: tuple[int, int], last: bool) -> torch.Tensor:
        prob = self.core.step(image, end=last)  # (2, h, w), с фоном
        return _to_out(prob, out_hw, align_corners=False)

    def reset(self) -> None:
        self.core = None

    def config(self) -> dict:
        keys = ("mem_every", "use_long_term", "max_mem_frames", "top_k", "stagger_updates")
        return {k: self.cfg[k] for k in keys}


class DeAOTAdapter(VOSAdapter):
    encoder_attrs = ("encode_image",)  # энкодер + проекция в пространство LSTT

    def __init__(self, spec: VOSModel) -> None:
        _check_repo("networks", spec.repo)
        from configs.default import DefaultEngineConfig
        from networks.engines import build_engine
        from networks.models import build_vos_model

        # DefaultEngineConfig, а не configs.pre_ytb_dav: подклассы вызывают init_dir(),
        # который создаёт каталоги result/, img_logs/, results/ в текущей папке.
        # Для инференса нужны только поля модели (MODEL_*, TEST_*).
        self.cfg = DefaultEngineConfig("bench", spec.variant)
        self._build_engine = build_engine
        self.network = build_vos_model(self.cfg.MODEL_VOS, self.cfg).cuda().eval()
        # в чекпойнтах DeAOT-T и R50-DeAOT-L кроме тензоров есть numpy-скаляр (pickle
        # numpy.core.multiarray.scalar + numpy.dtype): разрешаем только эти имена
        allowed = [
            (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            *(
                getattr(np.dtypes, n)
                for n in ("Float64DType", "Int64DType", "Float32DType", "Int32DType")
            ),
        ]
        with torch.serialization.safe_globals(allowed):
            ckpt = torch.load(spec.checkpoint, map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
        self.network.load_state_dict(state, strict=True)  # utils.checkpoint молча пропускает ключи
        self.engine = None

    def internal_size(self, h: int, w: int, short_side: int) -> tuple[int, int]:
        nh, nw = short_side_size(h, w, short_side)
        # MultiRestrictSize, align_corners=True: стороны вида 16k + 1
        return round((nh - 1) / 16) * 16 + 1, round((nw - 1) / 16) * 16 + 1

    def start(self, image: torch.Tensor, mask: torch.Tensor, n_frames: int) -> None:
        self.engine = self._build_engine(
            self.cfg.MODEL_ENGINE,
            phase="eval",
            aot_model=self.network,
            gpu_id=0,
            long_term_mem_gap=self.cfg.TEST_LONG_TERM_MEM_GAP,
            short_term_mem_skip=self.cfg.TEST_SHORT_TERM_MEM_SKIP,
        ).eval()
        label = mask[None, None].float()  # (1, 1, h, w), id объекта = 1
        self.engine.add_reference_frame(image[None], label, frame_step=0, obj_nums=[1])

    def step(self, image: torch.Tensor, out_hw: tuple[int, int], last: bool) -> torch.Tensor:
        # как в networks/managers/evaluator.py: логиты сразу в исходное разрешение,
        # в память — argmax, возвращённый nearest-ом во внутреннее разрешение
        self.engine.match_propogate_one_frame(image[None])
        logit = self.engine.decode_current_logits(out_hw)  # (1, 1 + max_obj, H, W)
        label = logit.argmax(1, keepdim=True).float()
        if not last:
            self.engine.update_memory(
                F.interpolate(label, self.engine.input_size_2d, mode="nearest")
            )
        return label[0, 0] == 1

    def reset(self) -> None:
        self.engine = None

    def config(self) -> dict:
        keys = (
            "MODEL_NAME", "MODEL_ENCODER", "MODEL_LSTT_NUM", "MODEL_ALIGN_CORNERS",
            "TEST_LONG_TERM_MEM_GAP", "TEST_SHORT_TERM_MEM_SKIP", "MODEL_MAX_OBJ_NUM",
        )  # fmt: skip
        cfg = {k: getattr(self.cfg, k) for k in keys}
        corr = importlib.import_module("networks.layers.attention").enable_corr
        return cfg | {"spatial_correlation_sampler": corr}


_ADAPTERS: dict[str, Callable[[VOSModel], VOSAdapter]] = {
    "XMem": XMemAdapter,
    "Cutie": CutieAdapter,
    "aot-benchmark": DeAOTAdapter,
}


def build_adapter(name: str) -> VOSAdapter:
    spec = MODELS[name]
    return _ADAPTERS[spec.repo](spec)
