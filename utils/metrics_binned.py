import torch
from torch import nn
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import matplotlib

from ovg.utils.predict import predict_denoiser

_LABEL_KEYS = (
    "class_label",
    "labels",
    "label",
    "y",
    "condition",
    "res_class",
    "cond_class",
    "class_idx",
)
_EPS = 1e-12


@dataclass
class BinnedMetrics:
    bins_mid_t: torch.Tensor
    mse_eps_all: torch.Tensor
    mse_eps_class_0: torch.Tensor
    mse_eps_class_1: torch.Tensor
    eps2_mean_all: torch.Tensor
    eps2_var_all: torch.Tensor
    eps2_hist_edges: torch.Tensor
    eps2_hist_counts: torch.Tensor

    eps2_hist_counts_top: Optional[torch.Tensor] = None
    mse_x0_all: Optional[torch.Tensor] = None
    class_flip_delta: Optional[torch.Tensor] = None
    cond_influence_ratio: Optional[torch.Tensor] = None
    ema_delta: Optional[torch.Tensor] = None
    failure_rate_lastbin: Optional[float] = None


@dataclass
class _SchedulerInfo:
    is_ddpm_like: bool
    is_edm_like: bool
    alphas_cumprod: Optional[torch.Tensor]
    sigmas_sched: Optional[torch.Tensor]
    T: int
    noise_scheduler: Any


@dataclass
class _SetupInfo:
    denoiser: nn.Module
    vae: Optional[nn.Module]
    class_embedding: Optional[nn.Module]
    unet_dtype: torch.dtype
    vae_dtype: torch.dtype
    t_idx: torch.Tensor
    bins_mid_t: torch.Tensor
    hist_edges: torch.Tensor
    scheduler_info: _SchedulerInfo


class _MetricAccumulator:
    def __init__(
        self,
        K: int,
        H_eps2: int,
        device: torch.device,
        compute_flags: Dict[str, bool],
        hist_edges_eps2: torch.Tensor,
        top_k_for_top_hist: int,
    ):
        self.K = K
        self.H_eps2 = H_eps2
        self.device = device
        self.compute_flags = compute_flags
        self.hist_edges_eps2 = hist_edges_eps2
        self.top_k_for_top_hist = int(top_k_for_top_hist)

        z = torch.zeros(K, device=device)
        self.sums_mse = z.clone()
        self.cnts_mse = z.clone()
        self.sums_mse_class_0 = z.clone()
        self.cnts_mse_class_0 = z.clone()
        self.sums_mse_class_1 = z.clone()
        self.cnts_mse_class_1 = z.clone()
        self.sums_eps2 = z.clone()
        self.sums_eps2_sq = z.clone()
        self.cnts_eps2 = z.clone()
        self.hist_counts = torch.zeros(K, H_eps2 - 1, device=device)

        self.sums_mse_x0 = z.clone() if compute_flags.get("compute_x0_mse") else None
        self.cnts_mse_x0 = z.clone() if compute_flags.get("compute_x0_mse") else None
        self.class_flip_delta = (
            z.clone() if compute_flags.get("compute_class_flip") else None
        )
        self.cond_influence = (
            z.clone() if compute_flags.get("compute_cond_influence") else None
        )
        self.ema_delta = z.clone() if compute_flags.get("compute_ema_delta") else None

        self.batch_cnt = z.clone()

    def update(
        self,
        k: int,
        eps_pred: torch.Tensor,
        eps_tgt: torch.Tensor,
        labels: Optional[torch.Tensor],
        mse_x0_b: Optional[torch.Tensor] = None,
        class_flip_delta_b: Optional[torch.Tensor] = None,
        cond_influence_b: Optional[torch.Tensor] = None,
        ema_delta_b: Optional[torch.Tensor] = None,
    ):
        B = eps_pred.shape[0]

        mse_b = torch.mean(
            (eps_pred - eps_tgt) ** 2, dim=tuple(range(1, eps_pred.ndim))
        )
        self.sums_mse[k] += mse_b.sum()
        self.cnts_mse[k] += B

        eps2 = (eps_pred**2).flatten(1).sum(1)
        self.sums_eps2[k] += eps2.sum()
        self.sums_eps2_sq[k] += (eps2**2).sum()
        self.cnts_eps2[k] += B

        if labels is not None:
            m0 = labels == 0
            m1 = labels == 1
            if m0.any():
                self.sums_mse_class_0[k] += mse_b[m0].sum()
                self.cnts_mse_class_0[k] += m0.sum()
            if m1.any():
                self.sums_mse_class_1[k] += mse_b[m1].sum()
                self.cnts_mse_class_1[k] += m1.sum()

        e2_clamped = eps2.clamp_min(self.hist_edges_eps2[0]).clamp_max(
            self.hist_edges_eps2[-1] - 1e-12
        )
        bins = torch.bucketize(e2_clamped, self.hist_edges_eps2) - 1
        for bi in range(self.H_eps2 - 1):
            self.hist_counts[k, bi] += (bins == bi).sum()

        if self.sums_mse_x0 is not None and mse_x0_b is not None:
            self.sums_mse_x0[k] += mse_x0_b.sum()
            self.cnts_mse_x0[k] += B

        if self.class_flip_delta is not None and class_flip_delta_b is not None:
            self.class_flip_delta[k] += class_flip_delta_b
        if self.cond_influence is not None and cond_influence_b is not None:
            self.cond_influence[k] += cond_influence_b
        if self.ema_delta is not None and ema_delta_b is not None:
            self.ema_delta[k] += ema_delta_b

        self.batch_cnt[k] += 1

    def finalize(self, bins_mid_t: torch.Tensor) -> BinnedMetrics:
        mse_all = self.sums_mse / (self.cnts_mse + _EPS)
        mse_class_0 = torch.where(
            self.cnts_mse_class_0 > 0,
            self.sums_mse_class_0 / (self.cnts_mse_class_0 + _EPS),
            torch.nan,
        )
        mse_class_1 = torch.where(
            self.cnts_mse_class_1 > 0,
            self.sums_mse_class_1 / (self.cnts_mse_class_1 + _EPS),
            torch.nan,
        )

        mean_eps2 = self.sums_eps2 / (self.cnts_eps2 + _EPS)
        var_eps2 = (self.sums_eps2_sq / (self.cnts_eps2 + _EPS)) - mean_eps2**2
        var_eps2 = torch.clamp(var_eps2, min=0.0)

        mse_x0_out = None
        if self.sums_mse_x0 is not None:
            mse_x0_out = torch.where(
                self.cnts_mse_x0 > 0,
                self.sums_mse_x0 / (self.cnts_mse_x0 + _EPS),
                torch.nan,
            )

        batch_cnt_safe = self.batch_cnt.clamp_min(1.0)
        class_flip_out = (
            self.class_flip_delta / batch_cnt_safe
            if self.class_flip_delta is not None
            else None
        )
        cond_inf_out = (
            self.cond_influence / batch_cnt_safe
            if self.cond_influence is not None
            else None
        )
        ema_delta_out = (
            self.ema_delta / batch_cnt_safe if self.ema_delta is not None else None
        )

        failure = None
        if self.compute_flags.get("failure_rate_lastbin"):
            last_k = self.K - 1
            topc = self.hist_counts[last_k, -1].item()
            tot = self.hist_counts[last_k].sum().item()
            failure = float(topc / max(1, tot))

        eps2_hist_counts_top = self.hist_counts[self.top_k_for_top_hist].detach().cpu()

        return BinnedMetrics(
            bins_mid_t=bins_mid_t.cpu(),
            mse_eps_all=mse_all.cpu(),
            mse_eps_class_0=mse_class_0.cpu(),
            mse_eps_class_1=mse_class_1.cpu(),
            eps2_mean_all=mean_eps2.cpu(),
            eps2_var_all=var_eps2.cpu(),
            eps2_hist_edges=self.hist_edges_eps2.detach().cpu(),
            eps2_hist_counts=self.hist_counts.detach().cpu(),
            eps2_hist_counts_top=eps2_hist_counts_top,
            mse_x0_all=(mse_x0_out.cpu() if mse_x0_out is not None else None),
            class_flip_delta=(
                class_flip_out.cpu() if class_flip_out is not None else None
            ),
            cond_influence_ratio=(
                cond_inf_out.cpu() if cond_inf_out is not None else None
            ),
            ema_delta=(ema_delta_out.cpu() if ema_delta_out is not None else None),
            failure_rate_lastbin=failure,
        )


def _get_labels(batch: Dict[str, Any]) -> Optional[torch.Tensor]:
    for k in _LABEL_KEYS:
        if k in batch:
            return batch[k]
    return None


def _setup_evaluation(
    denoiser,
    vae,
    class_embedding,
    noise_scheduler,
    device,
    n_bins,
    epsilon_hist_num_bins,
    epsilon_hist_logspace,
    epsilon_hist_min_exp,
    epsilon_hist_max_exp,
    prediction_type,
) -> _SetupInfo:
    denoiser.eval()
    if class_embedding is not None:
        class_embedding.eval()
    if vae is not None:
        vae.eval()

    unet_dtype = next(denoiser.parameters()).dtype
    vae_dtype = next(vae.parameters()).dtype if vae is not None else unet_dtype

    ddpm_like = hasattr(noise_scheduler, "alphas_cumprod")

    edm_like = (not ddpm_like) and hasattr(noise_scheduler, "inversion_timesteps")

    fm_like = (not ddpm_like) and (not edm_like)
    if not ddpm_like and not edm_like and not fm_like:
        raise RuntimeError(
            "eval_binned_metrics requires a DDPM/DDIM-style scheduler (alphas_cumprod), "
            "an EDM-like scheduler with `inversion_timesteps`, or a flow-matching scheduler."
        )

    T = int(getattr(noise_scheduler.config, "num_train_timesteps", 1000))
    alphas_cumprod = (
        noise_scheduler.alphas_cumprod.to(device, dtype=unet_dtype)
        if ddpm_like
        else None
    )
    sigmas_sched = (
        torch.as_tensor(
            getattr(noise_scheduler, "sigmas"), device=device, dtype=torch.float32
        )
        if edm_like
        else None
    )
    scheduler_info = _SchedulerInfo(
        ddpm_like, edm_like, alphas_cumprod, sigmas_sched, T, noise_scheduler
    )

    K = int(n_bins)
    t_idx = torch.linspace(0, T - 1, steps=K, device=device).long()
    bins_mid_t = (t_idx.float() + 0.5) / T

    if epsilon_hist_logspace:
        edges = torch.logspace(
            epsilon_hist_min_exp,
            epsilon_hist_max_exp,
            steps=epsilon_hist_num_bins,
            device=device,
        )
    else:
        edges = torch.linspace(0.0, 1.0, steps=epsilon_hist_num_bins, device=device)

    return _SetupInfo(
        denoiser=denoiser,
        vae=vae,
        class_embedding=class_embedding,
        unet_dtype=unet_dtype,
        vae_dtype=vae_dtype,
        t_idx=t_idx,
        bins_mid_t=bins_mid_t,
        hist_edges=edges,
        scheduler_info=scheduler_info,
    )


def _prepare_batch(
    batch: Dict[str, Any], setup: _SetupInfo, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    imgs = batch.get("image", batch.get("hr", None))
    if imgs is None:
        raise KeyError("Batch must contain 'image' or 'hr' tensor.")
    imgs = imgs.to(device, non_blocking=True)

    if setup.vae is not None:
        imgs_for_vae = imgs.to(device, dtype=setup.vae_dtype)
        x0_vae = setup.vae.encode(imgs_for_vae).latent_dist.sample()
        x0_vae = x0_vae * setup.vae.config.scaling_factor
        x0 = x0_vae.to(device, dtype=setup.unet_dtype)
    else:
        x0 = imgs.to(device, dtype=setup.unet_dtype)

    noise = torch.randn_like(x0, dtype=setup.unet_dtype)

    labels = _get_labels(batch)
    if labels is not None:
        labels = labels.to(device).long()

        num_classes = None
        if setup.class_embedding is not None:
            emb_module = getattr(setup.class_embedding, "module", setup.class_embedding)
            num_classes = getattr(emb_module, "num_classes", None)
        elif setup.class_embedding is None:
            base_model = getattr(setup.denoiser, "module", setup.denoiser)
            model_cfg = getattr(base_model, "config", None)
            num_classes = getattr(model_cfg, "num_classes", None)

        if isinstance(num_classes, int) and num_classes > 0:
            labels = labels.clamp(min=0, max=num_classes - 1)

    cond = None
    if setup.class_embedding is not None and labels is not None:
        cond_raw = setup.class_embedding(labels)
        cond = cond_raw.to(device, dtype=setup.unet_dtype)

    return x0, noise, cond, labels


def _get_timestep_inputs(
    x0: torch.Tensor,
    noise: torch.Tensor,
    ti: torch.Tensor,
    sched_info: _SchedulerInfo,
    device: torch.device,
    unet_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    B = x0.shape[0]
    t = ti.repeat(B).to(device)

    if sched_info.is_ddpm_like:
        x_t = sched_info.noise_scheduler.add_noise(x0, noise, t).to(
            device, dtype=unet_dtype
        )
        a = sched_info.alphas_cumprod[t].view(B, *([1] * (x0.ndim - 1))).clamp_min(1e-8)
        alpha_t, sigma_t = a.sqrt(), (1 - a).sqrt()

    elif sched_info.is_edm_like:
        ti_clamped = int(
            min(max(int(ti.item()), 0), sched_info.sigmas_sched.numel() - 1)
        )
        sigma_scalar = sched_info.sigmas_sched[ti_clamped].to(dtype=unet_dtype)
        sigma_t = sigma_scalar.view(1, *([1] * (x0.ndim - 1))).expand_as(x0)
        alpha_t = torch.ones_like(sigma_t)
        x_t = (x0 + sigma_t * noise).to(device, dtype=unet_dtype)

    else:
        t_cont = float(ti.item()) / max(1, sched_info.T - 1)
        t_scalar = torch.tensor(t_cont, device=device, dtype=unet_dtype)
        t_view = t_scalar.view(1, *([1] * (x0.ndim - 1)))
        x_t = ((1.0 - t_view) * noise + t_view * x0).to(device, dtype=unet_dtype)
        alpha_t, sigma_t = None, None

    return x_t, t, alpha_t, sigma_t


def _get_eps_from_prediction(
    raw_pred: torch.Tensor,
    prediction_type: str,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    alpha_t: Optional[torch.Tensor],
    sigma_t: Optional[torch.Tensor],
    is_ddpm_like: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prediction_type == "epsilon":
        eps_pred = raw_pred
        eps_tgt = noise

    elif prediction_type == "v_prediction":
        if not is_ddpm_like:
            vel_pred, vel_tgt = raw_pred, (x0 - noise)
            eps_pred, eps_tgt = vel_pred, vel_tgt
        else:
            eps_pred = (raw_pred + sigma_t * x0) / alpha_t
            eps_tgt = noise

    elif prediction_type == "sample":
        if not is_ddpm_like:
            eps_pred = (x_t - raw_pred) / (sigma_t + _EPS)
            eps_tgt = noise
        else:
            eps_pred = (x_t - alpha_t * raw_pred) / sigma_t
            eps_tgt = noise
    else:
        raise RuntimeError(f"Unsupported prediction_type '{prediction_type}'.")

    return eps_pred, eps_tgt


def _compute_x0_mse_batch(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    eps_pred: torch.Tensor,
    alpha_t: Optional[torch.Tensor],
    sigma_t: Optional[torch.Tensor],
    prediction_type: str,
    sched_info: _SchedulerInfo,
    ti: torch.Tensor,
) -> Optional[torch.Tensor]:
    x0_hat = None
    if sched_info.is_ddpm_like:
        x0_hat = (x_t - sigma_t * eps_pred) / alpha_t
    elif sched_info.is_edm_like:
        x0_hat = x_t - sigma_t * eps_pred
    elif prediction_type == "v_prediction":
        t_cont = float(ti.item()) / max(1, sched_info.T - 1)
        one_minus_t = x0.new_tensor(1.0 - t_cont).view(1, *([1] * (x0.ndim - 1)))
        vel_pred = eps_pred
        x0_hat = x_t + one_minus_t * vel_pred

    if x0_hat is not None:
        return torch.mean((x0_hat - x0) ** 2, dim=tuple(range(1, x0.ndim)))
    return None


def _compute_class_flip_batch(
    denoiser: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    eps_pred: torch.Tensor,
    class_embedding: Optional[nn.Module],
    prediction_type: str,
    x0: torch.Tensor,
    alpha_t: Optional[torch.Tensor],
    sigma_t: Optional[torch.Tensor],
    is_ddpm_like: bool,
) -> Optional[torch.Tensor]:
    try:
        num_classes = None
        if class_embedding is not None:
            emb_module = getattr(class_embedding, "module", class_embedding)
            num_classes = getattr(emb_module, "num_classes", None)

        if isinstance(num_classes, int) and num_classes > 1:
            lbl_flip = (labels + 1) % num_classes
        else:
            max_label = int(labels.max().item())
            if max_label == 0:
                return None
            lbl_flip = (labels + 1).clamp(max=max_label)

        cond_arg_flip = lbl_flip
        if class_embedding is not None:
            cond_arg_flip = class_embedding(lbl_flip).to(x_t.dtype)

        raw_flip = predict_denoiser(denoiser, x_t, t, cond_arg_flip)

        eps_flip, _ = _get_eps_from_prediction(
            raw_flip,
            prediction_type,
            x_t,
            x0,
            noise=None,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            is_ddpm_like=is_ddpm_like,
        )

        return (eps_pred - eps_flip).flatten(1).norm(2, dim=1).mean()

    except Exception:
        return None


def _compute_cond_influence_batch(
    denoiser: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    eps_pred: torch.Tensor,
    prediction_type: str,
    x0: torch.Tensor,
    alpha_t: Optional[torch.Tensor],
    sigma_t: Optional[torch.Tensor],
    is_ddpm_like: bool,
) -> Optional[torch.Tensor]:
    try:
        cond_zero = torch.zeros_like(cond)
        raw_zero = predict_denoiser(denoiser, x_t, t, cond_zero)

        eps_zero, _ = _get_eps_from_prediction(
            raw_zero,
            prediction_type,
            x_t,
            x0,
            noise=None,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            is_ddpm_like=is_ddpm_like,
        )

        num = (eps_pred - eps_zero).flatten(1).norm(2, dim=1).mean()
        den = eps_pred.flatten(1).norm(2, dim=1).mean().clamp_min(_EPS)
        return num / den
    except Exception:
        return None


def _compute_ema_delta_batch(
    ema_denoiser: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    eps_pred: torch.Tensor,
    prediction_type: str,
    x0: torch.Tensor,
    alpha_t: Optional[torch.Tensor],
    sigma_t: Optional[torch.Tensor],
    is_ddpm_like: bool,
) -> Optional[torch.Tensor]:
    try:
        ema_dtype = next(ema_denoiser.parameters()).dtype
        x_t_ema = x_t.to(ema_dtype)

        cond_arg_ema = None
        if cond is not None:
            cond_arg_ema = cond.to(ema_dtype)
        elif labels is not None:
            cond_arg_ema = labels

        raw_ema = predict_denoiser(ema_denoiser, x_t_ema, t, cond_arg_ema).to(x_t.dtype)

        eps_ema, _ = _get_eps_from_prediction(
            raw_ema,
            prediction_type,
            x_t,
            x0,
            noise=None,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            is_ddpm_like=is_ddpm_like,
        )

        return (eps_pred - eps_ema).flatten(1).norm(2, dim=1).mean()
    except Exception:
        return None


@torch.no_grad()
def eval_binned_metrics(
    *,
    vae,
    denoiser,
    class_embedding,
    noise_scheduler,
    dataloader,
    device: torch.device,
    n_bins: int = 20,
    max_batches: int = 10,
    prediction_type: str = "epsilon",
    compute_x0_mse: bool = True,
    compute_class_flip: bool = True,
    compute_cond_influence: bool = True,
    compute_ema_delta: bool = False,
    ema_denoiser=None,
    epsilon_hist_num_bins: int = 33,
    epsilon_hist_logspace: bool = True,
    epsilon_hist_min_exp: float = -2.0,
    epsilon_hist_max_exp: float = 2.0,
    failure_rate_lastbin: bool = False,
) -> BinnedMetrics:
    setup = _setup_evaluation(
        denoiser,
        vae,
        class_embedding,
        noise_scheduler,
        device,
        n_bins,
        epsilon_hist_num_bins,
        epsilon_hist_logspace,
        epsilon_hist_min_exp,
        epsilon_hist_max_exp,
        prediction_type,
    )
    sched_info = setup.scheduler_info

    compute_flags = {
        "compute_x0_mse": compute_x0_mse,
        "compute_class_flip": compute_class_flip and class_embedding is not None,
        "compute_cond_influence": compute_cond_influence
        and class_embedding is not None,
        "compute_ema_delta": compute_ema_delta and ema_denoiser is not None,
        "failure_rate_lastbin": failure_rate_lastbin,
    }

    if sched_info.is_ddpm_like:
        top_k = n_bins - 1
    elif sched_info.is_edm_like:
        ti_vals = setup.t_idx.clamp(0, sched_info.sigmas_sched.numel() - 1)
        sigmas_bins = sched_info.sigmas_sched[ti_vals]
        top_k = int(torch.argmax(sigmas_bins).item())
    else:
        top_k = n_bins - 1

    accumulator = _MetricAccumulator(
        K=n_bins,
        H_eps2=int(setup.hist_edges.numel()),
        device=device,
        compute_flags=compute_flags,
        hist_edges_eps2=setup.hist_edges,
        top_k_for_top_hist=top_k,
    )

    for b, batch in enumerate(dataloader):
        if b >= max_batches:
            break

        x0, noise, cond, labels = _prepare_batch(batch, setup, device)
        B = x0.shape[0]

        for k, ti in enumerate(setup.t_idx):
            x_t, t, alpha_t, sigma_t = _get_timestep_inputs(
                x0, noise, ti, sched_info, device, setup.unet_dtype
            )

            cond_arg = cond if cond is not None else labels
            raw_pred = predict_denoiser(denoiser, x_t, t, cond_arg)

            eps_pred, eps_tgt = _get_eps_from_prediction(
                raw_pred,
                prediction_type,
                x_t,
                x0,
                noise,
                alpha_t,
                sigma_t,
                sched_info.is_ddpm_like,
            )

            mse_x0_b = (
                _compute_x0_mse_batch(
                    x_t, x0, eps_pred, alpha_t, sigma_t, prediction_type, sched_info, ti
                )
                if compute_flags["compute_x0_mse"]
                else None
            )

            class_flip_b = (
                _compute_class_flip_batch(
                    denoiser,
                    x_t,
                    t,
                    labels,
                    eps_pred,
                    class_embedding,
                    prediction_type,
                    x0,
                    alpha_t,
                    sigma_t,
                    sched_info.is_ddpm_like,
                )
                if compute_flags["compute_class_flip"] and labels is not None
                else None
            )

            cond_inf_b = (
                _compute_cond_influence_batch(
                    denoiser,
                    x_t,
                    t,
                    cond,
                    eps_pred,
                    prediction_type,
                    x0,
                    alpha_t,
                    sigma_t,
                    sched_info.is_ddpm_like,
                )
                if compute_flags["compute_cond_influence"] and cond is not None
                else None
            )

            ema_delta_b = (
                _compute_ema_delta_batch(
                    ema_denoiser,
                    x_t,
                    t,
                    cond,
                    labels,
                    eps_pred,
                    prediction_type,
                    x0,
                    alpha_t,
                    sigma_t,
                    sched_info.is_ddpm_like,
                )
                if compute_flags["compute_ema_delta"]
                else None
            )

            accumulator.update(
                k,
                eps_pred,
                eps_tgt,
                labels,
                mse_x0_b=mse_x0_b,
                class_flip_delta_b=class_flip_b,
                cond_influence_b=cond_inf_b,
                ema_delta_b=ema_delta_b,
            )

    return accumulator.finalize(setup.bins_mid_t)


def metrics_to_dict(m: BinnedMetrics) -> Dict[str, Any]:
    out = asdict(m)
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.numpy().tolist()
        elif v is None:
            del out[k]
    return out


def quickplot_binned(m: BinnedMetrics, title: str = "Binned Metrics"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib not installed. Skipping quickplot_binned.")
        return

    t = m.bins_mid_t.numpy()

    plt.figure()
    plt.plot(t, m.mse_eps_all.numpy())
    plt.title(f"{title}: MSE(ε)")
    plt.xlabel("t/T")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)

    plt.figure()
    plt.plot(t, m.eps2_mean_all.numpy())
    plt.title(f"{title}: mean ||ε||²")
    plt.xlabel("t/T")
    plt.ylabel("mean")
    plt.grid(True, alpha=0.3)

    if m.mse_x0_all is not None:
        plt.figure()
        plt.plot(t, m.mse_x0_all.numpy())
        plt.title(f"{title}: MSE(x₀)")
        plt.xlabel("t/T")
        plt.ylabel("MSE")
        plt.grid(True, alpha=0.3)

    plt.show()


def render_binned_plots(
    m: BinnedMetrics, title_prefix: str = "Binned Metrics"
) -> Dict[str, "matplotlib.figure.Figure"]:
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib/numpy not installed. Skipping render_binned_plots.")
        return {}

    figs = {}
    t = m.bins_mid_t.numpy()

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)
    ax1.plot(t, m.mse_eps_all.numpy(), label="all")
    if hasattr(m, "mse_eps_class_0") and np.isfinite(m.mse_eps_class_0.numpy()).any():
        ax1.plot(
            t, m.mse_eps_class_0.numpy(), label="class_0", alpha=0.7, linestyle="--"
        )
    if hasattr(m, "mse_eps_class_1") and np.isfinite(m.mse_eps_class_1.numpy()).any():
        ax1.plot(
            t, m.mse_eps_class_1.numpy(), label="class_1", alpha=0.7, linestyle="--"
        )
    ax1.set_title(f"{title_prefix}: MSE(ε)")
    ax1.set_xlabel("t/T")
    ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    figs["eval/plots/mse_eps"] = fig1

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(t, m.eps2_mean_all.numpy())
    ax2.set_title(f"{title_prefix}: mean ||ε||²")
    ax2.set_xlabel("t/T")
    ax2.set_ylabel("mean")
    ax2.grid(True, alpha=0.3)
    figs["eval/plots/eps2_mean"] = fig2

    if getattr(m, "eps2_var_all", None) is not None:
        try:
            arr = m.eps2_var_all.numpy()
            if np.isfinite(arr).any():
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(1, 1, 1)
                ax3.plot(t, arr)
                ax3.set_title(f"{title_prefix}: Var(||ε||²)")
                ax3.set_xlabel("t/T")
                ax3.set_ylabel("variance")
                ax3.grid(True, alpha=0.3)
                figs["eval/plots/eps2_var"] = fig3
        except Exception:
            pass

    if m.mse_x0_all is not None and np.isfinite(m.mse_x0_all.numpy()).any():
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1, 1, 1)
        ax4.plot(t, m.mse_x0_all.numpy())
        ax4.set_title(f"{title_prefix}: MSE(x₀)")
        ax4.set_xlabel("t/T")
        ax4.set_ylabel("MSE")
        ax4.grid(True, alpha=0.3)
        figs["eval/plots/mse_x0"] = fig4

    counts = m.eps2_hist_counts.numpy()
    edges = m.eps2_hist_edges.numpy()
    if counts.sum() > 0:
        fig4 = plt.figure()
        ax4 = fig4.add_subplot(1, 1, 1)
        im = ax4.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            extent=[0.0, 1.0, float(edges[0]), float(edges[-1])],
            interpolation="nearest",
            cmap="viridis",
        )
        fig4.colorbar(im, ax=ax4, label="count")
        ax4.set_title(f"{title_prefix}: hist ||ε||² vs t")
        ax4.set_xlabel("t/T")
        ax4.set_ylabel("||ε||²")
        if m.eps2_hist_edges.shape[0] < 20:
            ax4.set_yticks(edges)
        figs["eval/plots/eps2_hist2d"] = fig4

    if getattr(m, "eps2_hist_counts_top", None) is not None:
        edges = m.eps2_hist_edges.numpy()
        counts = m.eps2_hist_counts_top.numpy()
        if counts.sum() > 0:
            centers = 0.5 * (edges[:-1] + edges[1:])
            fig5 = plt.figure()
            ax5 = fig5.add_subplot(1, 1, 1)
            ax5.bar(centers, counts, width=(edges[1:] - edges[:-1]), align="center")
            ax5.set_title(f"{title_prefix}: hist ||ε||² @ top step")
            ax5.set_xlabel("||ε||²")
            ax5.set_ylabel("count")
            figs["eval/plots/eps2_hist_top"] = fig5

    if getattr(m, "class_flip_delta", None) is not None:
        arr = m.class_flip_delta.numpy()
        if np.isfinite(arr).any():
            fig6 = plt.figure()
            ax6 = fig6.add_subplot(1, 1, 1)
            ax6.plot(t, arr)
            ax6.set_title(f"{title_prefix}: class flip delta")
            ax6.set_xlabel("t/T")
            ax6.set_ylabel("delta")
            ax6.grid(True, alpha=0.3)
            figs["eval/plots/class_flip"] = fig6

    if getattr(m, "cond_influence_ratio", None) is not None:
        arr = m.cond_influence_ratio.numpy()
        if np.isfinite(arr).any():
            fig7 = plt.figure()
            ax7 = fig7.add_subplot(1, 1, 1)
            ax7.plot(t, arr)
            ax7.set_title(f"{title_prefix}: cond influence ratio")
            ax7.set_xlabel("t/T")
            ax7.set_ylabel("ratio")
            ax7.grid(True, alpha=0.3)
            figs["eval/plots/cond_influence"] = fig7

    return figs
