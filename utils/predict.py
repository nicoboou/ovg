from typing import Optional, Union

import torch

from ovg.utils.helpers import unwrap_model


def model_forward(model, x: torch.Tensor, t: torch.Tensor, cond=None):
    unwrapped = unwrap_model(model)

    config = getattr(unwrapped, "config", None)
    model_class_name = unwrapped.__class__.__name__
    kw = {}

    is_dit = "DiT" in model_class_name or "Transformer" in model_class_name

    if is_dit:
        kw["hidden_states"] = x
    else:
        kw["sample"] = x

    if torch.is_tensor(t):
        t = t.to(device=x.device)
    else:
        t = torch.tensor(t, device=x.device)

    if is_dit:
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(x.shape[0])
        elif t.ndim == 1 and t.shape[0] == 1 and x.shape[0] > 1:
            t = t.expand(x.shape[0])

    kw["timestep"] = t
    if is_dit and config is not None:
        is_continuous_time = (
            torch.is_floating_point(t) if torch.is_tensor(t) else isinstance(t, float)
        )

        n = getattr(config, "num_embeds_ada_norm", None)
        if n is not None and not is_continuous_time:
            if torch.is_tensor(t):
                t_float = t.to(device=x.device, dtype=torch.float32)
            else:
                t_float = torch.tensor(t, device=x.device, dtype=torch.float32)

            source_max = getattr(unwrapped, "_ovg_timestep_source_max", None)
            current_max = float(t_float.max().item()) if t_float.numel() else 0.0
            if source_max is None or current_max > source_max:
                source_max = max(current_max, 1.0)
                setattr(unwrapped, "_ovg_timestep_source_max", source_max)

            if source_max <= 0:
                mapped = torch.zeros_like(t_float, dtype=torch.long)
            else:
                mapped = (
                    torch.round(t_float / source_max * (n - 1))
                    .clamp_(0, n - 1)
                    .to(dtype=torch.long)
                )

            kw["timestep"] = mapped
        elif not is_continuous_time:
            if torch.is_tensor(t) and t.dtype != torch.long:
                kw["timestep"] = t.to(dtype=torch.long)
            elif not torch.is_tensor(t):
                t_tensor = torch.tensor(t, device=x.device, dtype=torch.long)
                if t_tensor.ndim == 0:
                    t_tensor = t_tensor.unsqueeze(0).expand(x.shape[0])
                kw["timestep"] = t_tensor

    if cond is not None:
        if is_dit:
            if torch.is_tensor(cond):
                cond = cond.to(dtype=torch.long)
            kw["class_labels"] = cond
        else:
            num_class_embeds = (
                getattr(config, "num_class_embeds", None) if config else None
            )
            if num_class_embeds is not None:
                if torch.is_tensor(cond):
                    cond = cond.to(dtype=torch.long)
                kw["class_labels"] = cond
            elif config is not None and hasattr(config, "cross_attention_dim"):
                if cond.ndim == 2:
                    cond = cond.unsqueeze(1)
                kw["encoder_hidden_states"] = cond

    out = model(**kw)
    return out.sample if hasattr(out, "sample") else out


def predict_denoiser(
    denoiser,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: Optional[Union[torch.Tensor, torch.LongTensor]] = None,
):
    return model_forward(denoiser, x_t, t, cond)


def predict_with_cfg(
    denoiser,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: Union[torch.Tensor, torch.LongTensor],
    cfg_scale: float,
    null_cond: Optional[Union[torch.Tensor, torch.LongTensor]] = None,
    batched: bool = True,
):
    if cfg_scale == 0.0:
        return predict_denoiser(denoiser, x_t, t, cond)

    if null_cond is None:
        if torch.is_tensor(cond):
            if torch.is_floating_point(cond):
                null_cond = torch.zeros_like(cond)
            else:
                unwrapped = unwrap_model(denoiser)
                null_id = getattr(unwrapped, "_ovg_null_class_label", None)
                if null_id is None:
                    config = getattr(unwrapped, "config", None)
                    null_id = getattr(config, "num_classes", None)
                if null_id is None:
                    raise RuntimeError(
                        "Unable to determine unconditional label for DiT classifier-free guidance."
                    )
                null_cond = torch.full_like(cond, fill_value=int(null_id))
        else:
            null_cond = None

    if not batched:
        pred_uncond = predict_denoiser(denoiser, x_t, t, null_cond)
        pred_cond = predict_denoiser(denoiser, x_t, t, cond)
        return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

    x_cat = torch.cat([x_t, x_t], dim=0)

    if t.ndim == 0:
        t_cat = t.unsqueeze(0).expand(x_cat.shape[0])
    else:
        t_cat = torch.cat([t, t], dim=0)

    if torch.is_tensor(cond):
        cond_cat = torch.cat([null_cond, cond], dim=0)
        pred = predict_denoiser(denoiser, x_cat, t_cat, cond_cat)
        pred_uncond, pred_cond = pred.chunk(2, dim=0)
    else:
        if null_cond is None:
            pred_uncond = predict_denoiser(denoiser, x_t, t, None)
            pred_cond = predict_denoiser(denoiser, x_t, t, cond)
        else:
            cond_cat = torch.cat([null_cond, cond], dim=0)
            pred = predict_denoiser(denoiser, x_cat, t_cat, cond_cat)
            pred_uncond, pred_cond = pred.chunk(2, dim=0)

    return pred_uncond + cfg_scale * (pred_cond - pred_uncond)
