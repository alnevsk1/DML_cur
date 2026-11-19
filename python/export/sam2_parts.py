"""Разбиение PVS-моделей семейства SAM 2 (SAM 2.1, EfficientTAM, EdgeTAM) на экспортируемые части.

Части (вход 1024x1024, батч 1, один объект; состояние — банк памяти — хранит вызывающий код):
  * ImageEncoderPart — кадр -> признаки 64x64 (+ позиционное кодирование, + признаки высокого
    разрешения 256²/128² для SAM 2.1 и EdgeTAM);
  * DecoderPart — энкодер подсказок + декодер масок + указатель объекта (`_forward_sam_heads`),
    отдельные графы для multimask (клик / кадры трекинга) и одной маски (рамка);
  * MemoryEncoderPart — маска + признаки кадра -> память кадра (`_encode_new_memory`;
    у EdgeTAM — со сжатием перцептором);
  * MemoryAttentionPart — признаки кадра + банк памяти + указатели -> признаки, обусловленные
    памятью (`_prepare_memory_conditioned_features` без логики выбора кадров).
Выбор кадров памяти (кадр подсказки + 6 последних) и указателей — gather_memory(): это прототип
логики банка памяти для C++.

ONNX не поддерживает комплексные числа, а RoPE во внимании к памяти считается через
torch.polar / view_as_complex. make_rope_real() заменяет частоты на вещественное представление
(cos, sin) и подменяет apply_rotary_enc / apply_rotary_enc_v2 вещественными версиями — та же
математика (поворот пар соседних каналов), репозитории моделей не меняются.
"""

from __future__ import annotations

import functools
import inspect
import sys

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROPE_ATTRS = ("freqs_cis", "freqs_cis_q", "freqs_cis_k")


# --- RoPE без комплексных чисел ---------------------------------------------------------------


def _rope_rotate(x: Tensor, freqs: Tensor) -> Tensor:
    """x: (..., L, d); freqs: (L, d/2, 2) = (cos, sin). Поворот пар каналов (2i, 2i+1)."""
    xr = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    cos, sin = freqs[..., 0], freqs[..., 1]
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


def apply_rotary_enc_real(
    xq: Tensor, xk: Tensor, freqs_cis: Tensor, repeat_freqs_k: bool = False
) -> tuple[Tensor, Tensor]:
    """Вещественный аналог position_encoding.apply_rotary_enc (freqs_cis: (L, d/2, 2))."""
    xq_out = _rope_rotate(xq, freqs_cis)
    if xk.shape[-2] == 0:
        return xq_out, xk
    if repeat_freqs_k:  # ключи — r кадров памяти по L токенов, частоты повторяются по кадрам
        n = freqs_cis.shape[0]
        k = xk.reshape(*xk.shape[:-2], -1, n, xk.shape[-1])  # (B, heads, r, L, d)
        return xq_out, _rope_rotate(k, freqs_cis).reshape(xk.shape)
    return xq_out, _rope_rotate(xk, freqs_cis)


def apply_rotary_enc_v2_real(x: Tensor, freqs_cis: Tensor, repeat_freqs: int) -> Tensor:
    """Вещественный аналог apply_rotary_enc_v2 (EdgeTAM): RoPE только на 2D-латентах кадра."""
    if x.shape[-2] == 0:
        return x
    b, h, n, c = x.shape
    rope_tokens = freqs_cis.shape[0]
    per_frame = n // repeat_freqs
    x = x.view(b, h, repeat_freqs, per_frame, c)
    no_rope = per_frame - rope_tokens
    rotated = _rope_rotate(x[..., no_rope:, :], freqs_cis)
    return torch.cat((x[..., :no_rope, :], rotated), dim=3).view(b, h, n, c)


def make_rope_real(model: nn.Module, q_hw: tuple[int, int] = (64, 64)) -> int:
    """Переводит все RoPE-модули модели на вещественные частоты. Возвращает число модулей."""
    pkg = type(model).__module__.split(".")[0]
    pe = sys.modules[f"{pkg}.modeling.position_encoding"]
    tr = sys.modules[f"{pkg}.modeling.sam.transformer"]
    original_cis = pe.compute_axial_cis
    device = next(model.parameters()).device  # чтобы .to(q.device) в forward был пустым

    def compute_axial_cis_real(dim: int, end_x: int, end_y: int, theta: float = 10000.0) -> Tensor:
        return torch.view_as_real(original_cis(dim, int(end_x), int(end_y), theta)).to(device)

    for mod in (pe, tr):
        mod.apply_rotary_enc = apply_rotary_enc_real
        if hasattr(pe, "apply_rotary_enc_v2"):
            mod.apply_rotary_enc_v2 = apply_rotary_enc_v2_real
    n = 0
    for m in model.modules():
        if not any(hasattr(m, a) for a in ROPE_ATTRS):
            continue
        if hasattr(m, "compute_cis"):
            m.compute_cis = functools.partial(compute_axial_cis_real, **m.compute_cis.keywords)
        if hasattr(m, "freqs_cis"):  # RoPEAttention: q — всегда кадр 64x64; пересчёт заранее
            m.freqs_cis = m.compute_cis(end_x=q_hw[0], end_y=q_hw[1])
        for a in ("freqs_cis_q", "freqs_cis_k"):  # RoPEAttentionv2 (EdgeTAM): размеры из конфига
            t = getattr(m, a, None)
            if t is not None and t.is_complex():
                setattr(m, a, torch.view_as_real(t).to(device))
        n += 1
    return n


# --- Части модели -----------------------------------------------------------------------------


def base_module(model: nn.Module):  # noqa: ANN201 — модуль репозитория (sam2_base / efficienttam_base)
    return inspect.getmodule(type(model)._prepare_memory_conditioned_features)


class ImageEncoderPart(nn.Module):
    """(1, 3, 1024, 1024) нормализованный кадр -> vision_feat, vision_pos [, hr0, hr1]."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.high_res = bool(model.use_high_res_features_in_sam)

    def forward(self, image: Tensor) -> tuple[Tensor, ...]:
        out = self.model.forward_image(image)
        fpn, pos = out["backbone_fpn"], out["vision_pos_enc"]
        res = (fpn[-1], pos[-1])  # (1, 256, 64, 64) x 2
        if self.high_res:
            res += (fpn[0], fpn[1])  # (1, 32, 256, 256), (1, 64, 128, 128) — после conv_s0/s1
        return res


class DecoderPart(nn.Module):
    """Энкодер подсказок + декодер масок + указатель объекта.

    Точки — в координатах входа 1024x1024; метки: 1/0 — клик +/-, 2/3 — углы рамки,
    -1 — пустая точка (кадры трекинга без подсказки).
    Выход: low_res_mask (1, 1, 256, 256) — логиты лучшей маски, obj_ptr (1, 256),
    object_score_logits (1, 1), ious (1, M).
    """

    def __init__(self, model: nn.Module, multimask: bool) -> None:
        super().__init__()
        self.model = model
        self.multimask = multimask

    def forward(self, pix_feat: Tensor, point_coords: Tensor, point_labels: Tensor,
                *high_res: Tensor) -> tuple[Tensor, ...]:  # fmt: skip
        _, _, ious, low_res_mask, _, obj_ptr, obj_score = self.model._forward_sam_heads(
            backbone_features=pix_feat,
            point_inputs={"point_coords": point_coords, "point_labels": point_labels},
            mask_inputs=None,
            high_res_features=list(high_res) if high_res else None,
            multimask_output=self.multimask,
        )
        return low_res_mask, obj_ptr, obj_score, ious


class MemoryEncoderPart(nn.Module):
    """Память кадра по его признакам и предсказанной маске (как _encode_new_memory).

    is_mask_from_pts (1,): 1.0 — на кадре были клики/рамка (маска бинаризуется), 0.0 — трекинг.
    Выход: maskmem_features, maskmem_pos_enc — (1, 64, 64, 64) или (1, T, 64) у EdgeTAM.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, vision_feat: Tensor, low_res_mask: Tensor, object_score_logits: Tensor,
                is_mask_from_pts: Tensor) -> tuple[Tensor, Tensor]:  # fmt: skip
        m = self.model
        high_res = F.interpolate(low_res_mask, size=(m.image_size, m.image_size),
                                 mode="bilinear", align_corners=False)  # fmt: skip
        mask = torch.sigmoid(high_res)
        if m.binarize_mask_from_pts_for_mem_enc:
            mask = torch.where(is_mask_from_pts.view(1, 1, 1, 1) > 0.5, (high_res > 0).float(), mask)
        if m.sigmoid_scale_for_mem_enc != 1.0:
            mask = mask * m.sigmoid_scale_for_mem_enc
        if m.sigmoid_bias_for_mem_enc != 0.0:
            mask = mask + m.sigmoid_bias_for_mem_enc
        out = m.memory_encoder(vision_feat, mask, skip_mask_sigmoid=True)
        feats, pos = out["vision_features"], out["vision_pos_enc"][-1]
        if getattr(m, "no_obj_embed_spatial", None) is not None:
            appearing = (object_score_logits > 0).float()  # (1, 1)
            feats = feats + (1 - appearing[..., None, None]) * m.no_obj_embed_spatial[..., None, None]
        if getattr(m, "spatial_perceiver", None) is not None:
            feats, pos = m.spatial_perceiver(feats, pos)  # (1, T, 64)
        return feats, pos


class MemoryAttentionPart(nn.Module):
    """Признаки кадра + банк памяти -> признаки, обусловленные памятью (1, 256, 64, 64).

    maskmem_feats / maskmem_pos: (Nf, 64, 64, 64) или (Nf, T, 64) — память Nf кадров;
    maskmem_tpos: (Nf,) int64 — временная позиция кадра памяти (0 — кадр подсказки,
    1..num_maskmem-1 — от дальнего к ближайшему); obj_ptrs: (Np, 256) указатели объекта,
    obj_ptr_tdiff: (Np,) float — расстояние в кадрах до кадра указателя.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.get_1d_sine_pe = base_module(model).get_1d_sine_pe
        self.edgetam = "num_spatial_mem" in inspect.signature(model.memory_attention.forward).parameters

    def forward(self, vision_feat: Tensor, vision_pos: Tensor, maskmem_feats: Tensor,
                maskmem_pos: Tensor, maskmem_tpos: Tensor, obj_ptrs: Tensor,
                obj_ptr_tdiff: Tensor) -> Tensor:  # fmt: skip
        m = self.model
        c, mem_dim = m.hidden_dim, m.mem_dim
        nf = maskmem_feats.shape[0]
        curr = vision_feat.flatten(2).permute(2, 0, 1)  # (HW, 1, C)
        curr_pos = vision_pos.flatten(2).permute(2, 0, 1)
        if maskmem_feats.dim() == 4:  # (Nf, 64, H, W) -> (Nf, HW, 64)
            maskmem_feats = maskmem_feats.flatten(2).permute(0, 2, 1)
            maskmem_pos = maskmem_pos.flatten(2).permute(0, 2, 1)
        tpos = m.maskmem_tpos_enc[m.num_maskmem - 1 - maskmem_tpos].reshape(nf, 1, mem_dim)
        mem = maskmem_feats.reshape(-1, 1, mem_dim)  # (Nf * T, 1, 64), кадр за кадром
        mem_pos = (maskmem_pos + tpos).reshape(-1, 1, mem_dim)

        if m.add_tpos_enc_to_obj_ptrs:
            t_diff_max = m.max_obj_ptrs_in_encoder - 1
            tpos_dim = c if m.proj_tpos_enc_in_obj_ptrs else mem_dim
            obj_pos = self.get_1d_sine_pe(obj_ptr_tdiff / t_diff_max, dim=tpos_dim)
            obj_pos = m.obj_ptr_tpos_proj(obj_pos).unsqueeze(1)  # (Np, 1, 64)
        else:
            obj_pos = obj_ptrs.new_zeros(obj_ptrs.shape[0], 1, mem_dim)
        ptrs = obj_ptrs.unsqueeze(1)  # (Np, 1, C)
        if mem_dim < c:  # указатель C=256 -> C/64 токенов по 64
            ptrs = ptrs.reshape(-1, 1, c // mem_dim, mem_dim).permute(0, 2, 1, 3).flatten(0, 1)
            obj_pos = obj_pos.repeat_interleave(c // mem_dim, dim=0)

        kwargs = {"num_spatial_mem": nf} if self.edgetam else {}
        out = m.memory_attention(
            curr=[curr],
            curr_pos=[curr_pos],
            memory=torch.cat([mem, ptrs], dim=0),
            memory_pos=torch.cat([mem_pos, obj_pos], dim=0),
            num_obj_ptr_tokens=ptrs.shape[0],
            **kwargs,
        )
        return out.permute(1, 2, 0).reshape(1, c, *vision_feat.shape[-2:])


# --- Банк памяти (прототип логики C++) ---------------------------------------------------------


def gather_memory(model: nn.Module, output_dict: dict, frame_idx: int) -> dict[str, Tensor]:
    """Входы MemoryAttentionPart для кадра frame_idx — как _prepare_memory_conditioned_features.

    Онлайн-трекинг вперёд, шаг памяти 1, число указателей — max_obj_ptrs_in_encoder
    (в потоковом режиме длина видео неизвестна, поэтому без min(num_frames, ...)).
    """
    cond = output_dict["cond_frame_outputs"]
    non_cond = output_dict["non_cond_frame_outputs"]
    selected, unselected = base_module(model).select_closest_cond_frames(
        frame_idx, cond, model.max_cond_frames_in_attn
    )
    frames = [(0, out) for out in selected.values()]
    for t_pos in range(1, model.num_maskmem):
        prev = frame_idx - (model.num_maskmem - t_pos)
        out = non_cond.get(prev, unselected.get(prev))
        if out is not None:
            frames.append((t_pos, out))
    ptrs = [(frame_idx - t, out["obj_ptr"]) for t, out in selected.items() if t <= frame_idx]
    for t_diff in range(1, model.max_obj_ptrs_in_encoder):
        t = frame_idx - t_diff
        if t < 0:
            break
        out = non_cond.get(t, unselected.get(t))
        if out is not None:
            ptrs.append((t_diff, out["obj_ptr"]))
    dev = frames[0][1]["obj_ptr"].device
    return {
        "maskmem_feats": torch.cat([o["maskmem_features"].float() for _, o in frames]),
        "maskmem_pos": torch.cat([o["maskmem_pos_enc"][-1].float() for _, o in frames]),
        "maskmem_tpos": torch.tensor([t for t, _ in frames], dtype=torch.int64, device=dev),
        "obj_ptrs": torch.cat([p.float() for _, p in ptrs]),
        "obj_ptr_tdiff": torch.tensor([float(d) for d, _ in ptrs], device=dev),
    }
