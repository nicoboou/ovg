from abc import ABC
from dataclasses import dataclass
import time
import torch
import warnings
from tqdm.auto import tqdm

from ovg.utils.predict import predict_denoiser
from ovg.utils.metrics_binned import _get_eps_from_prediction

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class LiveBinnedMetrics:
    t_norm: list
    eps2_mean: list
    eps_mse: list
    mse_x0: list
    cond_influence: list
    class_flip: list
    eps2_var: list
    eps2_hist_edges: list | None
    eps2_hist_counts: list | None
    errors: list

    def __init__(self):
        self.t_norm = []
        self.eps2_mean = []
        self.eps_mse = []
        self.mse_x0 = []
        self.cond_influence = []
        self.class_flip = []
        self.eps2_var = []
        self.eps2_hist_edges = None
        self.eps2_hist_counts = None
        self.errors = []

    def to_payload(self):
        return {
            "t_norm": self.t_norm,
            "eps2_mean": self.eps2_mean,
            "eps_mse": self.eps_mse,
            "mse_x0": self.mse_x0,
            "cond_influence": self.cond_influence,
            "class_flip": self.class_flip,
            "eps2_var": self.eps2_var,
            "eps2_hist_edges": self.eps2_hist_edges,
            "eps2_hist_counts": self.eps2_hist_counts,
            "errors": self.errors,
        }


class BaseInversionMethod(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def get_config(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
        }

    @staticmethod
    def _denoiser_device_dtype(
        denoiser: torch.nn.Module,
    ) -> tuple[torch.device, torch.dtype]:
        ref_param = next(denoiser.parameters())
        return ref_param.device, ref_param.dtype

    @classmethod
    def _match_denoiser(cls, tensor: torch.Tensor, denoiser: torch.nn.Module, copy: bool = False) -> torch.Tensor:
        device, dtype = cls._denoiser_device_dtype(denoiser)
        if copy:
            tensor = tensor.clone()
        return tensor.to(device=device, dtype=dtype)

    @classmethod
    def _ensure_condition_dtype(
        cls,
        conditioning: torch.Tensor | None,
        denoiser: torch.nn.Module,
    ) -> torch.Tensor | None:
        if conditioning is None:
            return None
        return cls._match_denoiser(conditioning, denoiser)

    @staticmethod
    def _sched_kind(scheduler):
        ddpm_like = hasattr(scheduler, "alphas_cumprod")

        edm_like = (not ddpm_like) and hasattr(scheduler, "inversion_timesteps")

        fm_like = (not ddpm_like) and (not edm_like)

        return ddpm_like, edm_like, fm_like

    @staticmethod
    def _get_prediction_type(scheduler) -> str:
        cfg_obj = getattr(scheduler, "config", None)
        if isinstance(cfg_obj, dict):
            return str(cfg_obj.get("prediction_type", "epsilon")).lower()
        return str(getattr(cfg_obj, "prediction_type", "epsilon")).lower()

    @staticmethod
    def _convert_model_output(
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        alpha_prod_t: torch.Tensor,
        prediction_type: str = "epsilon",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta_prod_t = 1 - alpha_prod_t

        if prediction_type == "sample":
            pred_original_sample = model_output

            pred_epsilon = (x_t - alpha_prod_t**0.5 * pred_original_sample) / (beta_prod_t**0.5 + 1e-8)
        elif prediction_type == "v_prediction":
            sqrt_alpha = alpha_prod_t**0.5
            sqrt_beta = beta_prod_t**0.5
            pred_original_sample = sqrt_alpha * x_t - sqrt_beta * model_output
            pred_epsilon = sqrt_alpha * model_output + sqrt_beta * x_t
        elif prediction_type == "epsilon":
            pred_epsilon = model_output

            pred_original_sample = (x_t - beta_prod_t**0.5 * pred_epsilon) / alpha_prod_t**0.5
        else:
            raise RuntimeError(f"Unsupported prediction_type '{prediction_type}'.")

        return pred_original_sample, pred_epsilon

    def _predict_noise_for_capture(
        self,
        x_t: torch.Tensor,
        t: float,
        denoiser: torch.nn.Module,
        scheduler,
        encoder_hidden_states: torch.Tensor,
        ddpm_like: bool,
        edm_like: bool,
    ) -> torch.Tensor:
        B = x_t.shape[0]

        if edm_like:
            cfg_obj = getattr(scheduler, "config", None)
            if isinstance(cfg_obj, dict):
                sigma_data = cfg_obj.get("edm_sigma_data", 0.5)
            else:
                sigma_data = getattr(cfg_obj, "edm_sigma_data", 0.5)

            sigma_cur = torch.as_tensor(t, device=x_t.device, dtype=x_t.dtype)
            sigma_data_t = torch.as_tensor(sigma_data, device=x_t.device, dtype=x_t.dtype)
            c_in = 1.0 / torch.sqrt(sigma_cur**2 + sigma_data_t**2)

            unet_in = c_in * x_t
            t_batch = sigma_cur.expand(B)
            F_theta = predict_denoiser(denoiser, unet_in, t_batch, encoder_hidden_states)
            return F_theta
        else:
            t_tensor = torch.tensor(t, device=x_t.device, dtype=torch.long).repeat(B)
            noise_pred = predict_denoiser(denoiser, x_t, t_tensor, encoder_hidden_states)
            return noise_pred

    def invert(self, image, denoiser, scheduler, encoder_hidden_states, num_steps=50, **kwargs):
        latents = self._match_denoiser(image, denoiser, copy=True)

        original_x0 = latents.clone().detach()
        encoder_hidden_states = self._ensure_condition_dtype(encoder_hidden_states, denoiser)
        store_intermediates = kwargs.pop("store_intermediates", False)
        store_predicted_noise = kwargs.pop("store_predicted_noise", False)
        store_timesteps = kwargs.pop("store_timesteps", False)
        intermediate_latents = [latents.clone().detach().cpu()] if store_intermediates else None
        predicted_noises = [] if store_predicted_noise else None
        timesteps_list = [] if store_timesteps else None
        x0 = kwargs.pop("x0", None)

        collect_live_binned: bool = bool(kwargs.pop("collect_live_binned", False))
        inversion_live_flip_embeddings = kwargs.pop("inversion_live_flip_embeddings", None)
        _ = kwargs.pop("sample_live_flip_embeddings", None)
        live_metrics = LiveBinnedMetrics() if collect_live_binned else None

        scheduler.set_timesteps(num_steps)

        ddpm_like, edm_like, fm_like = self._sched_kind(scheduler)

        cfg_obj = getattr(scheduler, "config", None)
        if isinstance(cfg_obj, dict):
            T = int(cfg_obj.get("num_train_timesteps", 1000))
            pred_type = str(cfg_obj.get("prediction_type", "epsilon")).lower()
        else:
            T = int(getattr(cfg_obj, "num_train_timesteps", 1000)) if cfg_obj is not None else 1000
            pred_type = str(getattr(cfg_obj, "prediction_type", "epsilon")).lower()

        if edm_like:
            sigma0 = scheduler.inversion_timesteps[0]
            noise = torch.randn_like(latents)
            latents = latents + sigma0 * noise

        if ddpm_like:
            reversed_timesteps = list(reversed(scheduler.timesteps))
            num_inf = int(getattr(scheduler, "num_inference_steps", num_steps))
            step = max(1, T // max(1, num_inf))
            loop_iter = enumerate(tqdm(reversed_timesteps, desc="Inversion"))
        elif edm_like:
            inv_ts = scheduler.inversion_timesteps
            loop_iter = ((i, inv_ts[i + 1]) for i in range(min(num_steps, inv_ts.shape[0] - 1)))
        else:
            reversed_timesteps = list(reversed(getattr(scheduler, "timesteps", list(range(num_steps)))))
            num_inf = int(getattr(scheduler, "num_inference_steps", num_steps))
            step = max(1, T // max(1, num_inf))
            loop_iter = enumerate(tqdm(reversed_timesteps, desc="Inversion"))

        for i, next_t in loop_iter:
            if ddpm_like:
                current_t = max(int(next_t) - step, 0)
            elif edm_like:
                current_t = scheduler.inversion_timesteps[i]
            else:
                current_t = max(int(next_t) - step, 0)

            step_kwargs = kwargs.copy()
            if self.name == "ours" and x0 is not None:
                step_kwargs["x0"] = self._match_denoiser(tensor=x0, denoiser=denoiser)

            if store_timesteps:
                timesteps_list.append(float(current_t) if not edm_like else float(next_t))

            if store_predicted_noise:
                noise_pred = self._predict_noise_for_capture(
                    latents,
                    current_t,
                    denoiser,
                    scheduler,
                    encoder_hidden_states,
                    ddpm_like,
                    edm_like,
                )
                predicted_noises.append(noise_pred.clone().detach().cpu())

            if collect_live_binned:
                try:
                    B = latents.shape[0]

                    t_dtype = torch.long if ddpm_like else latents.dtype
                    t_tensor = torch.tensor(current_t, device=latents.device, dtype=t_dtype).repeat(B)

                    alpha_t = None
                    sigma_t = None

                    if ddpm_like:
                        raw_pred = predict_denoiser(denoiser, latents, t_tensor, encoder_hidden_states)
                        a_buf = scheduler.alphas_cumprod
                        t_idx = t_tensor.to(device=a_buf.device)
                        a = a_buf[t_idx].to(device=latents.device, dtype=latents.dtype)
                        a = a.view(B, *([1] * (latents.ndim - 1))).clamp_min(1e-8)
                        alpha_t = a.sqrt()
                        sigma_t = (1 - a).sqrt()
                    elif edm_like:
                        sigma_scalar = torch.as_tensor(current_t, device=latents.device, dtype=latents.dtype)
                        sigma_t = sigma_scalar.view(1, *([1] * (latents.ndim - 1))).expand_as(latents)

                        cfg_obj = getattr(scheduler, "config", None)
                        if isinstance(cfg_obj, dict):
                            sigma_data = cfg_obj.get("edm_sigma_data", 0.5)
                        else:
                            sigma_data = getattr(cfg_obj, "edm_sigma_data", 0.5)
                        sigma_data_t = torch.as_tensor(sigma_data, device=latents.device, dtype=latents.dtype)
                        c_in = 1.0 / torch.sqrt(sigma_scalar**2 + sigma_data_t**2)
                        c_skip = (sigma_data_t**2) / (sigma_scalar**2 + sigma_data_t**2)
                        c_out = sigma_scalar * sigma_data_t / torch.sqrt(sigma_scalar**2 + sigma_data_t**2)
                        unet_in = c_in * latents
                        F_theta = predict_denoiser(denoiser, unet_in, t_tensor, encoder_hidden_states)

                        raw_pred = c_skip * latents + c_out * F_theta

                    x0_ref = self._match_denoiser(x0, denoiser) if x0 is not None else original_x0

                    used_pred_type = "sample" if edm_like else pred_type
                    eps_pred, _ = _get_eps_from_prediction(
                        raw_pred,
                        used_pred_type,
                        latents,
                        x0_ref,
                        noise=latents.new_zeros(latents.shape),
                        alpha_t=alpha_t,
                        sigma_t=sigma_t,
                        is_ddpm_like=ddpm_like,
                    )

                    eps2_vals = (eps_pred**2).flatten(1).sum(1)
                    eps2_vals64 = eps2_vals.to(torch.float64)
                    eps2_mean_val = float(eps2_vals64.mean().item())
                    var_eps2_val = float(eps2_vals64.var(unbiased=False).item())
                    live_metrics.eps2_mean.append(float(eps2_mean_val))
                    live_metrics.eps2_var.append(var_eps2_val)

                    if x0 is not None or original_x0 is not None:
                        try:
                            if ddpm_like and alpha_t is not None and sigma_t is not None:
                                eps_true = torch.randn_like(latents)
                                x_t_fwd = alpha_t * x0_ref + sigma_t * eps_true
                                raw_fwd = predict_denoiser(denoiser, x_t_fwd, t_tensor, encoder_hidden_states)
                                eps_pred_fwd, _ = _get_eps_from_prediction(
                                    raw_fwd,
                                    pred_type,
                                    x_t_fwd,
                                    x0_ref,
                                    noise=None,
                                    alpha_t=alpha_t,
                                    sigma_t=sigma_t,
                                    is_ddpm_like=True,
                                )
                                mse_eps = (
                                    ((eps_pred_fwd - eps_true) ** 2)
                                    .mean(dim=tuple(range(1, eps_pred_fwd.ndim)))
                                    .mean()
                                )
                                live_metrics.eps_mse.append(float(mse_eps.item()))
                            elif edm_like and sigma_t is not None:
                                eps_true = torch.randn_like(latents)
                                x_fwd = x0_ref + sigma_t * eps_true

                                F_fwd = predict_denoiser(
                                    denoiser,
                                    c_in * x_fwd,
                                    t_tensor,
                                    encoder_hidden_states,
                                )
                                x0_hat_fwd = c_skip * x_fwd + c_out * F_fwd
                                eps_pred_fwd, _ = _get_eps_from_prediction(
                                    x0_hat_fwd,
                                    "sample",
                                    x_fwd,
                                    x0_ref,
                                    noise=None,
                                    alpha_t=None,
                                    sigma_t=sigma_t,
                                    is_ddpm_like=False,
                                )
                                mse_eps = (
                                    ((eps_pred_fwd - eps_true) ** 2)
                                    .mean(dim=tuple(range(1, eps_pred_fwd.ndim)))
                                    .mean()
                                )
                                live_metrics.eps_mse.append(float(mse_eps.item()))
                            else:
                                live_metrics.eps_mse.append(float("nan"))
                        except Exception:
                            live_metrics.eps_mse.append(float("nan"))
                    else:
                        live_metrics.eps_mse.append(float("nan"))

                    if x0 is not None or original_x0 is not None:
                        if ddpm_like and alpha_t is not None and sigma_t is not None:
                            x0_hat = (latents - sigma_t * eps_pred) / alpha_t
                        elif edm_like and sigma_t is not None:
                            x0_hat = latents - sigma_t * eps_pred
                        elif pred_type == "v_prediction":
                            t_cont = float(current_t) / max(1, T - 1)
                            one_minus_t = torch.tensor(1.0 - t_cont, device=latents.device, dtype=latents.dtype)
                            one_minus_t = one_minus_t.view(1, *([1] * (latents.ndim - 1)))
                            vel_pred = eps_pred
                            x0_hat = latents + one_minus_t * vel_pred
                        else:
                            x0_hat = None
                        if x0_hat is not None:
                            x0_ref_batch = x0_ref
                            mse_x0 = torch.mean(
                                (x0_hat - x0_ref_batch) ** 2,
                                dim=tuple(range(1, x0_hat.ndim)),
                            ).mean()
                            live_metrics.mse_x0.append(float(mse_x0.item()))
                        else:
                            live_metrics.mse_x0.append(float("nan"))
                    else:
                        live_metrics.mse_x0.append(float("nan"))

                    ci_val = float("nan")
                    if encoder_hidden_states is not None and torch.is_floating_point(encoder_hidden_states):
                        cond_zero = torch.zeros_like(encoder_hidden_states)
                        if edm_like:
                            cfg_obj = getattr(scheduler, "config", None)
                            if isinstance(cfg_obj, dict):
                                sigma_data = cfg_obj.get("edm_sigma_data", 0.5)
                            else:
                                sigma_data = getattr(cfg_obj, "edm_sigma_data", 0.5)
                            sigma_scalar = torch.as_tensor(current_t, device=latents.device, dtype=latents.dtype)
                            sigma_data_t = torch.as_tensor(sigma_data, device=latents.device, dtype=latents.dtype)
                            c_in = 1.0 / torch.sqrt(sigma_scalar**2 + sigma_data_t**2)
                            c_skip = (sigma_data_t**2) / (sigma_scalar**2 + sigma_data_t**2)
                            c_out = sigma_scalar * sigma_data_t / torch.sqrt(sigma_scalar**2 + sigma_data_t**2)
                            unet_in = c_in * latents
                            F_zero = predict_denoiser(denoiser, unet_in, t_tensor, cond_zero)
                            raw_zero = c_skip * latents + c_out * F_zero
                            used_pred_type_zero = "sample"
                        else:
                            raw_zero = predict_denoiser(denoiser, latents, t_tensor, cond_zero)
                            used_pred_type_zero = pred_type
                        eps_zero, _ = _get_eps_from_prediction(
                            raw_zero,
                            used_pred_type_zero,
                            latents,
                            x0_ref,
                            noise=latents.new_zeros(latents.shape),
                            alpha_t=alpha_t,
                            sigma_t=sigma_t,
                            is_ddpm_like=ddpm_like,
                        )
                        num = (eps_pred - eps_zero).flatten(1).norm(2, dim=1).mean()
                        den = eps_pred.flatten(1).norm(2, dim=1).mean().clamp_min(1e-12)
                        ci_val = float((num / den).item())
                    live_metrics.cond_influence.append(ci_val)

                    if inversion_live_flip_embeddings is not None:
                        if edm_like:
                            cfg_obj = getattr(scheduler, "config", None)
                            if isinstance(cfg_obj, dict):
                                sigma_data = cfg_obj.get("edm_sigma_data", 0.5)
                            else:
                                sigma_data = getattr(cfg_obj, "edm_sigma_data", 0.5)
                            sigma_scalar = torch.as_tensor(current_t, device=latents.device, dtype=latents.dtype)
                            sigma_data_t = torch.as_tensor(sigma_data, device=latents.device, dtype=latents.dtype)
                            c_in = 1.0 / torch.sqrt(sigma_scalar**2 + sigma_data_t**2)
                            c_skip = (sigma_data_t**2) / (sigma_scalar**2 + sigma_data_t**2)
                            c_out = sigma_scalar * sigma_data_t / torch.sqrt(sigma_scalar**2 + sigma_data_t**2)
                            unet_in = c_in * latents
                            F_flip = predict_denoiser(
                                denoiser,
                                unet_in,
                                t_tensor,
                                inversion_live_flip_embeddings,
                            )
                            raw_flip = c_skip * latents + c_out * F_flip
                            used_pred_type_flip = "sample"
                        else:
                            raw_flip = predict_denoiser(
                                denoiser,
                                latents,
                                t_tensor,
                                inversion_live_flip_embeddings,
                            )
                            used_pred_type_flip = pred_type
                        eps_flip, _ = _get_eps_from_prediction(
                            raw_flip,
                            used_pred_type_flip,
                            latents,
                            x0_ref,
                            noise=None,
                            alpha_t=alpha_t,
                            sigma_t=sigma_t,
                            is_ddpm_like=ddpm_like,
                        )
                        cf = (eps_pred - eps_flip).flatten(1).norm(2, dim=1).mean()
                        live_metrics.class_flip.append(float(cf.item()))
                    else:
                        live_metrics.class_flip.append(float("nan"))

                    if live_metrics.eps2_hist_counts is None:
                        try:
                            edges = torch.logspace(
                                -2.0,
                                2.0,
                                steps=33,
                                device=latents.device,
                                dtype=latents.dtype,
                            )
                            e2 = eps2_vals.clamp_min(edges[0]).clamp_max(edges[-1] - 1e-12)
                            bins_r = torch.bucketize(e2, edges) - 1
                            counts = torch.stack([(bins_r == bi).sum() for bi in range(edges.numel() - 1)])
                            live_metrics.eps2_hist_edges = edges.detach().cpu().tolist()
                            live_metrics.eps2_hist_counts = counts.detach().cpu().tolist()
                        except Exception:
                            live_metrics.eps2_hist_edges = None
                            live_metrics.eps2_hist_counts = None

                    if edm_like:
                        total = min(
                            num_steps,
                            getattr(scheduler, "inversion_timesteps", torch.tensor([0])).shape[0] - 1,
                        )
                        live_metrics.t_norm.append(float((i + 0.5) / max(1, total)))
                    else:
                        live_metrics.t_norm.append(float((float(current_t) + 0.5) / max(1, T)))

                except Exception as e:
                    live_metrics.eps2_mean.append(float("nan"))
                    live_metrics.mse_x0.append(float("nan"))
                    live_metrics.cond_influence.append(float("nan"))
                    live_metrics.eps2_var.append(float("nan"))
                    live_metrics.t_norm.append(float((float(current_t) + 0.5) / max(1, T)))
                    try:
                        live_metrics.errors.append(f"invert_live_failed@t={current_t}: {type(e).__name__}: {e}")
                    except Exception:
                        pass

            latents = self.invert_step(
                x_t=latents,
                t_next=next_t,
                t_curr=current_t,
                denoiser=denoiser,
                scheduler=scheduler,
                encoder_hidden_states=encoder_hidden_states,
                **step_kwargs,
            )

            if store_intermediates:
                intermediate_latents.append(latents.clone().detach().cpu())

            if i % 10 == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        return {
            "latents": latents,
            "intermediate_latents": intermediate_latents,
            "predicted_noises": predicted_noises,
            "timesteps": timesteps_list,
            "live_metrics": (live_metrics.to_payload() if live_metrics is not None else None),
        }

    @torch.no_grad()
    def sample(
        self,
        latents,
        denoiser,
        scheduler,
        encoder_hidden_states=None,
        uncond_embeddings=None,
        guidance_scale=1.0,
        num_steps=50,
        batch_size=None,
        **kwargs,
    ):
        n = latents.shape[0]
        latents = self._match_denoiser(latents, denoiser, copy=True)
        encoder_hidden_states = self._ensure_condition_dtype(encoder_hidden_states, denoiser)
        uncond_embeddings = self._ensure_condition_dtype(uncond_embeddings, denoiser)

        if batch_size is None or n <= batch_size:
            return self._sample_batch(
                latents,
                denoiser,
                scheduler,
                encoder_hidden_states,
                uncond_embeddings,
                guidance_scale,
                num_steps,
                **kwargs,
            )

        parts = []
        for i in range(0, n, batch_size):
            part = latents[i : i + batch_size]
            enc = encoder_hidden_states[i : i + batch_size] if encoder_hidden_states is not None else None
            unc = uncond_embeddings[i : i + batch_size] if uncond_embeddings is not None else None
            parts.append(
                self._sample_batch(
                    part,
                    denoiser,
                    scheduler,
                    enc,
                    unc,
                    guidance_scale,
                    num_steps,
                    **kwargs,
                )
            )
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _sample_batch(
        self,
        latents,
        denoiser,
        scheduler,
        encoder_hidden_states=None,
        uncond_embeddings=None,
        guidance_scale=1.0,
        num_steps=50,
        **kwargs,
    ):
        image = self._match_denoiser(latents, denoiser, copy=True)
        scheduler.set_timesteps(num_steps)
        do_cfg = guidance_scale > 1.0 and uncond_embeddings is not None

        ddpm_like, edm_like, fm_like = self._sched_kind(scheduler)

        collect_live_binned = bool(kwargs.pop("collect_live_binned", False))
        sample_live_flip_embeddings = kwargs.pop("sample_live_flip_embeddings", None)
        live_rec: LiveBinnedMetrics | None = kwargs.pop("live_rec", None)
        Tcfg = (
            int(getattr(getattr(scheduler, "config", object()), "num_train_timesteps", 1000))
            if ddpm_like
            else num_steps
        )

        for i, t in enumerate(tqdm(scheduler.timesteps, desc="Sampling")):
            if ddpm_like:
                if do_cfg:
                    image_input = torch.cat([image, image])
                    encoder_hidden_states_input = torch.cat([uncond_embeddings, encoder_hidden_states])
                    noise_pred = predict_denoiser(
                        denoiser,
                        image_input,
                        t,
                        cond=encoder_hidden_states_input,
                    )
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = predict_denoiser(denoiser, image, t, cond=encoder_hidden_states)

            elif edm_like:
                sigma_data = (
                    scheduler.config["edm_sigma_data"]
                    if isinstance(getattr(scheduler, "config", {}), dict)
                    else getattr(scheduler.config, "edm_sigma_data", 0.5)
                )
                sigma_cur = t if torch.is_tensor(t) else torch.tensor(t, device=image.device, dtype=image.dtype)
                c_in = 1.0 / torch.sqrt(
                    sigma_cur**2 + torch.as_tensor(sigma_data, device=image.device, dtype=image.dtype) ** 2
                )
                if do_cfg:
                    image_input = torch.cat([image, image])
                    encoder_hidden_states_input = torch.cat([uncond_embeddings, encoder_hidden_states])
                    unet_in = c_in * image_input
                    pred = predict_denoiser(
                        denoiser,
                        unet_in,
                        sigma_cur.expand(unet_in.shape[0]),
                        cond=encoder_hidden_states_input,
                    )
                    pred_uncond, pred_cond = pred.chunk(2)
                    noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    unet_in = c_in * image
                    noise_pred = predict_denoiser(
                        denoiser,
                        unet_in,
                        sigma_cur.expand(image.shape[0]),
                        cond=encoder_hidden_states,
                    )

            else:
                if do_cfg:
                    image_input = torch.cat([image, image])
                    encoder_hidden_states_input = torch.cat([uncond_embeddings, encoder_hidden_states])
                    noise_pred = predict_denoiser(
                        denoiser,
                        image_input,
                        t,
                        cond=encoder_hidden_states_input,
                    )
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = predict_denoiser(denoiser, image, t, cond=encoder_hidden_states)

            if collect_live_binned and live_rec is not None:
                try:
                    B = image.shape[0]
                    if ddpm_like:
                        t_tensor = t if torch.is_tensor(t) else torch.tensor(t, device=image.device, dtype=torch.long)
                        t_tensor = t_tensor.view(1).repeat(B)
                        a = scheduler.alphas_cumprod[t_tensor]
                        a = a.view(B, *([1] * (image.ndim - 1))).clamp_min(1e-8)
                        alpha_t = a.sqrt()
                        sigma_t = (1 - a).sqrt()
                        eps_pred = noise_pred
                    else:
                        sigma_cur = (
                            t if torch.is_tensor(t) else torch.tensor(t, device=image.device, dtype=image.dtype)
                        )
                        alpha_t = None
                        sigma_t = sigma_cur.view(1, *([1] * (image.ndim - 1))).expand_as(image)
                        if edm_like:
                            sigma_data = (
                                scheduler.config["edm_sigma_data"]
                                if isinstance(getattr(scheduler, "config", {}), dict)
                                else getattr(scheduler.config, "edm_sigma_data", 0.5)
                            )
                            sigma_data_t = torch.as_tensor(sigma_data, device=image.device, dtype=image.dtype)
                            c_skip = (sigma_data_t**2) / (sigma_cur**2 + sigma_data_t**2)
                            c_out = sigma_cur * sigma_data_t / torch.sqrt(sigma_cur**2 + sigma_data_t**2)
                            x0_hat_pred = c_skip * image + c_out * noise_pred
                            eps_pred = (image - x0_hat_pred) / (sigma_t + 1e-12)
                        else:
                            eps_pred = noise_pred

                    eps2_vals = (eps_pred**2).flatten(1).sum(1)
                    eps2_mean = float(eps2_vals.mean().item())
                    eps2_var = float(max(0.0, (eps2_vals**2).mean().item() - eps2_mean**2))
                    live_rec.eps2_mean.append(eps2_mean)
                    live_rec.eps2_var.append(eps2_var)

                    d_dim = eps_pred.flatten(1).shape[1]
                    eps_norm_sq = (eps_pred**2).flatten(1).sum(1)
                    mse_eps_whiteness = ((eps_norm_sq - d_dim) ** 2).mean().item()
                    live_rec.eps_mse.append(float(mse_eps_whiteness))

                    if sample_live_flip_embeddings is not None:
                        if edm_like:
                            sigma_data = (
                                scheduler.config["edm_sigma_data"]
                                if isinstance(getattr(scheduler, "config", {}), dict)
                                else getattr(scheduler.config, "edm_sigma_data", 0.5)
                            )
                            sigma_cur = (
                                t if torch.is_tensor(t) else torch.tensor(t, device=image.device, dtype=image.dtype)
                            )
                            c_in = 1.0 / torch.sqrt(
                                sigma_cur**2 + torch.as_tensor(sigma_data, device=image.device, dtype=image.dtype) ** 2
                            )
                            F_flip = predict_denoiser(
                                denoiser,
                                c_in * image,
                                sigma_cur.expand(B),
                                cond=sample_live_flip_embeddings,
                            )

                            sigma_data_t = torch.as_tensor(sigma_data, device=image.device, dtype=image.dtype)
                            c_skip = (sigma_data_t**2) / (sigma_cur**2 + sigma_data_t**2)
                            c_out = sigma_cur * sigma_data_t / torch.sqrt(sigma_cur**2 + sigma_data_t**2)
                            x0_hat_flip = c_skip * image + c_out * F_flip
                            eps_flip = (image - x0_hat_flip) / (sigma_t + 1e-12)
                        else:
                            raw_flip = predict_denoiser(denoiser, image, t, cond=sample_live_flip_embeddings)

                            cfg_obj = getattr(scheduler, "config", None)
                            pred_type = (
                                str(getattr(cfg_obj, "prediction_type", "epsilon")).lower() if cfg_obj else "epsilon"
                            )
                            eps_flip, _ = _get_eps_from_prediction(
                                raw_flip,
                                pred_type,
                                image,
                                image.new_zeros(image.shape),
                                image.new_zeros(image.shape),
                                alpha_t if alpha_t is not None else torch.ones_like(sigma_t),
                                sigma_t if sigma_t is not None else torch.ones_like(image[:, :1, :, :]),
                                is_ddpm_like=ddpm_like,
                            )
                        cf = (eps_pred - eps_flip).flatten(1).norm(2, dim=1).mean()
                        live_rec.class_flip.append(float(cf.item()))
                    else:
                        live_rec.class_flip.append(float("nan"))

                    ci_val = float("nan")
                    if encoder_hidden_states is not None and torch.is_floating_point(encoder_hidden_states):
                        if edm_like:
                            sigma_data = (
                                scheduler.config["edm_sigma_data"]
                                if isinstance(getattr(scheduler, "config", {}), dict)
                                else getattr(scheduler.config, "edm_sigma_data", 0.5)
                            )
                            sigma_cur = (
                                t if torch.is_tensor(t) else torch.tensor(t, device=image.device, dtype=image.dtype)
                            )
                            c_in = 1.0 / torch.sqrt(
                                sigma_cur**2 + torch.as_tensor(sigma_data, device=image.device, dtype=image.dtype) ** 2
                            )
                            F_zero = predict_denoiser(
                                denoiser,
                                c_in * image,
                                sigma_cur.expand(B),
                                cond=torch.zeros_like(encoder_hidden_states),
                            )
                            sigma_data_t = torch.as_tensor(sigma_data, device=image.device, dtype=image.dtype)
                            c_skip = (sigma_data_t**2) / (sigma_cur**2 + sigma_data_t**2)
                            c_out = sigma_cur * sigma_data_t / torch.sqrt(sigma_cur**2 + sigma_data_t**2)
                            x0_hat_zero = c_skip * image + c_out * F_zero
                            eps_zero = (image - x0_hat_zero) / (sigma_t + 1e-12)
                        else:
                            raw_zero = predict_denoiser(
                                denoiser,
                                image,
                                t,
                                cond=torch.zeros_like(encoder_hidden_states),
                            )

                            cfg_obj = getattr(scheduler, "config", None)
                            pred_type = (
                                str(getattr(cfg_obj, "prediction_type", "epsilon")).lower() if cfg_obj else "epsilon"
                            )
                            eps_zero, _ = _get_eps_from_prediction(
                                raw_zero,
                                pred_type,
                                image,
                                image.new_zeros(image.shape),
                                image.new_zeros(image.shape),
                                alpha_t if alpha_t is not None else torch.ones_like(sigma_t),
                                sigma_t if sigma_t is not None else torch.ones_like(image[:, :1, :, :]),
                                is_ddpm_like=ddpm_like,
                            )
                        num = (eps_pred - eps_zero).flatten(1).norm(2, dim=1).mean()
                        den = eps_pred.flatten(1).norm(2, dim=1).mean().clamp_min(1e-12)
                        ci_val = float((num / den).item())
                    live_rec.cond_influence.append(ci_val)

                    if live_rec.eps2_hist_counts is None:
                        edges = torch.logspace(-2.0, 2.0, steps=33, device=image.device, dtype=image.dtype)
                        e2 = eps2_vals.clamp_min(edges[0]).clamp_max(edges[-1] - 1e-12)
                        bins = torch.bucketize(e2, edges) - 1
                        counts = torch.stack([(bins == bi).sum() for bi in range(edges.numel() - 1)])
                        live_rec.eps2_hist_edges = edges.detach().cpu().tolist()
                        live_rec.eps2_hist_counts = counts.detach().cpu().tolist()

                    if ddpm_like:
                        live_rec.t_norm.append(float((float(int(t)) + 0.5) / max(1, Tcfg)))
                    else:
                        live_rec.t_norm.append(float((i + 0.5) / max(1, num_steps)))
                except Exception as e:
                    live_rec.eps2_mean.append(float("nan"))
                    live_rec.eps2_var.append(float("nan"))
                    live_rec.class_flip.append(float("nan"))
                    live_rec.cond_influence.append(float("nan"))
                    live_rec.t_norm.append(float("nan"))
                    live_rec.eps_mse.append(float("nan"))
                    try:
                        live_rec.errors.append(
                            f"sampling_live_failed@t={int(t) if torch.is_tensor(t) else t}: {type(e).__name__}: {e}"
                        )
                    except Exception:
                        pass

            image = scheduler.step(noise_pred, t, image).prev_sample

        return image

    def run(
        self,
        image,
        denoiser,
        scheduler,
        encoder_hidden_states,
        sample_encoder_hidden_states=None,
        sample_uncond_embeddings=None,
        guidance_scale=1.0,
        num_steps=50,
        batch_size=None,
        store_intermediates=False,
        store_predicted_noise=False,
        store_timesteps=False,
        **kwargs,
    ):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inversion_start_ns = time.perf_counter_ns()

        invert_out = self.invert(
            image=image,
            denoiser=denoiser,
            scheduler=scheduler,
            encoder_hidden_states=encoder_hidden_states,
            num_steps=num_steps,
            store_intermediates=store_intermediates,
            store_predicted_noise=store_predicted_noise,
            store_timesteps=store_timesteps,
            **kwargs,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inversion_end_ns = time.perf_counter_ns()
        inversion_time_ns = inversion_end_ns - inversion_start_ns

        live_metrics = None
        intermediate_latents = []
        predicted_noises = []
        timesteps_list = []

        if isinstance(invert_out, dict):
            inverted_latents = invert_out["latents"]
            live_metrics = invert_out.get("live_metrics")
            intermediate_latents = invert_out.get("intermediate_latents", [])
            predicted_noises = invert_out.get("predicted_noises", [])
            timesteps_list = invert_out.get("timesteps", [])
        elif isinstance(invert_out, tuple):
            if len(invert_out) == 3:
                inverted_latents, _, live_metrics = invert_out
            elif len(invert_out) == 2:
                inverted_latents, _ = invert_out
            elif len(invert_out) == 1:
                inverted_latents = invert_out[0]
            else:
                inverted_latents = invert_out[0]
        else:
            inverted_latents = invert_out
        self.live_binned = live_metrics

        if sample_encoder_hidden_states is not None:
            sampling_embeddings = sample_encoder_hidden_states
        else:
            sampling_embeddings = encoder_hidden_states

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sampling_start_ns = time.perf_counter_ns()

        generated = self.sample(
            latents=inverted_latents,
            denoiser=denoiser,
            scheduler=scheduler,
            encoder_hidden_states=sampling_embeddings,
            uncond_embeddings=sample_uncond_embeddings,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            batch_size=batch_size,
            **kwargs,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        sampling_end_ns = time.perf_counter_ns()
        sampling_time_ns = sampling_end_ns - sampling_start_ns

        if store_intermediates or store_predicted_noise or store_timesteps:
            return {
                "generated": generated,
                "inverted_latents": inverted_latents,
                "intermediate_latents": intermediate_latents,
                "predicted_noises": predicted_noises,
                "timesteps": timesteps_list,
                "inversion_time_ns": inversion_time_ns,
                "sampling_time_ns": sampling_time_ns,
            }

        return {
            "generated": generated,
            "inverted_latents": inverted_latents,
            "inversion_time_ns": inversion_time_ns,
            "sampling_time_ns": sampling_time_ns,
        }


class StandardInversion(BaseInversionMethod):
    def __init__(self):
        super().__init__(name="standard_inversion", description="Standard DDIM Inversion")

    def get_config(self) -> dict:
        return super().get_config()

    def invert_step(self, x_t, t_next, t_curr, denoiser, scheduler, encoder_hidden_states):
        if not hasattr(scheduler, "alphas_cumprod") and hasattr(scheduler, "inversion_timesteps"):
            with torch.no_grad():
                sigma_data = (
                    scheduler.config["edm_sigma_data"]
                    if isinstance(getattr(scheduler, "config", {}), dict)
                    else getattr(scheduler.config, "edm_sigma_data", 0.5)
                )
                sigma_cur = (
                    t_curr if torch.is_tensor(t_curr) else torch.tensor(t_curr, device=x_t.device, dtype=x_t.dtype)
                )

                c_in = 1.0 / torch.sqrt(
                    sigma_cur**2 + torch.as_tensor(sigma_data, device=x_t.device, dtype=x_t.dtype) ** 2
                )
                unet_input = c_in * x_t
                B = x_t.shape[0]
                t_batch = sigma_cur.expand(B)
                F_theta = predict_denoiser(denoiser, unet_input, t_batch, cond=encoder_hidden_states)
                return scheduler.invert_step(F_theta, sigma_cur, t_next, x_t).next_sample

        with torch.no_grad():
            model_output = predict_denoiser(denoiser, x_t, t_curr, cond=encoder_hidden_states)

        with torch.no_grad():
            alpha_prod_t_next = scheduler.alphas_cumprod[t_next] if t_next >= 0 else scheduler.final_alpha_cumprod
            alpha_prod_t_curr = scheduler.alphas_cumprod[t_curr]

            prediction_type = self._get_prediction_type(scheduler)
            pred_original_sample, pred_epsilon = self._convert_model_output(
                model_output, x_t, alpha_prod_t_curr, prediction_type
            )

            x_next_standard = (
                alpha_prod_t_next**0.5 * pred_original_sample + (1 - alpha_prod_t_next) ** 0.5 * pred_epsilon
            )

        return x_next_standard


class ControlledGaussianizationInversion(BaseInversionMethod):
    def __init__(
        self,
        use_cgd: bool = True,
        use_scp: bool = True,
        cgd_eta: float = 1.0,
        scp_eta: float = 0.5,
        forward_fraction: float = 0.00,
        min_forward_steps: int = 0,
    ):
        super().__init__(name="ours", description="Orthogonal Variance Guidance Inversion")
        self.use_cgd = use_cgd
        self.use_scp = use_scp
        self.cgd_eta = cgd_eta
        self.scp_eta = scp_eta
        fraction = max(0.00, min(float(forward_fraction), 1.00))
        self.forward_fraction = fraction
        self.min_forward_steps = max(0, int(min_forward_steps))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "use_cgd": self.use_cgd,
                "use_scp": self.use_scp,
                "cgd_eta": self.cgd_eta,
                "scp_eta": self.scp_eta,
                "forward_fraction": self.forward_fraction,
                "min_forward_steps": self.min_forward_steps,
            }
        )
        return config

    @torch.enable_grad()
    def invert_step(
        self,
        x_t,
        t_next,
        t_curr,
        denoiser,
        scheduler,
        encoder_hidden_states,
        use_cgd=True,
        cgd_eta=1.0,
        use_scp=True,
        scp_eta=0.5,
        x0=None,
        clip_frac=0.2,
        epsilon=1e-6,
    ):
        x_t = x_t.detach().requires_grad_(True)

        ddpm_like, edm_like, fm_like = self._sched_kind(scheduler)

        if edm_like:
            sigma_data = (
                scheduler.config["edm_sigma_data"]
                if isinstance(getattr(scheduler, "config", {}), dict)
                else getattr(scheduler.config, "edm_sigma_data", 0.5)
            )

            sigma_cur = t_curr if torch.is_tensor(t_curr) else torch.tensor(t_curr, device=x_t.device, dtype=x_t.dtype)
            B = x_t.shape[0]
            t_batch = sigma_cur.expand(B)
            c_in = 1.0 / torch.sqrt(
                sigma_cur**2 + torch.as_tensor(sigma_data, device=x_t.device, dtype=x_t.dtype) ** 2
            )
            unet_input = c_in * x_t
            F_theta = predict_denoiser(denoiser, unet_input, t_batch, cond=encoder_hidden_states)

            sigma_data_t = torch.as_tensor(sigma_data, device=x_t.device, dtype=x_t.dtype)
            c_skip = (sigma_data_t**2) / (sigma_cur**2 + sigma_data_t**2)
            c_out = sigma_cur * sigma_data_t / torch.sqrt(sigma_cur**2 + sigma_data_t**2)
            x0_pred = c_skip * x_t + c_out * F_theta

            x_next_standard = scheduler.invert_step(F_theta, sigma_cur, t_next, x_t).next_sample

            norm_dims = tuple(range(1, x_t.ndim))
            total_correction = torch.zeros_like(x_t)

            dt_normalized = torch.log(t_next / (sigma_cur + epsilon))
            d_dim = x_t.numel() / x_t.shape[0]

            if (use_cgd and cgd_eta > 0) or (use_scp and scp_eta > 0):
                if x0 is None:
                    raise ValueError("x0 must be provided for orthogonal corrections")

                r = x_t - x0_pred
                eps_equiv = r / (sigma_cur + epsilon)
                eps_norm_sq = torch.sum(eps_equiv**2, dim=norm_dims, keepdim=True)
                target = torch.as_tensor(d_dim, device=x_t.device, dtype=x_t.dtype)
                noise_loss = (eps_norm_sq - target) ** 2
                g_n = torch.autograd.grad(noise_loss.sum(), x_t, retain_graph=True)[0]

                content_loss = torch.sum((x0_pred - x0) ** 2, dim=norm_dims, keepdim=True)
                g_c = torch.autograd.grad(content_loss.sum(), x_t, retain_graph=False)[0]

                with torch.no_grad():
                    g_n_norm_sq = torch.sum(g_n**2, dim=norm_dims, keepdim=True) + epsilon
                    g_c_norm_sq = torch.sum(g_c**2, dim=norm_dims, keepdim=True) + epsilon
                    g_n_dot_g_c = torch.sum(g_n * g_c, dim=norm_dims, keepdim=True)
                    cos_angle = g_n_dot_g_c / (g_n_norm_sq.sqrt() * g_c_norm_sq.sqrt() + epsilon)
                    is_conflicting = cos_angle < 0

                    ref_drift = x_next_standard.detach() - x_t.detach()
                    norm_ref = torch.norm(ref_drift, p=2, dim=norm_dims, keepdim=True)

                    if use_cgd and cgd_eta > 0:
                        proj_n = (g_n_dot_g_c / g_c_norm_sq) * g_c
                        u_cgd_orth = -(cgd_eta) * (g_n - proj_n)
                        u_cgd_full = -(cgd_eta) * g_n
                        u_cgd = torch.where(is_conflicting, u_cgd_orth, u_cgd_full)
                        u_cgd_scaled = u_cgd * dt_normalized
                        norm_u = torch.norm(u_cgd_scaled, p=2, dim=norm_dims, keepdim=True)
                        max_norm = clip_frac * (norm_ref + epsilon)
                        scale = (max_norm / (norm_u + epsilon)).clamp(max=1.0)
                        total_correction += u_cgd_scaled * scale

                    if use_scp and scp_eta > 0:
                        proj_c = (g_n_dot_g_c / g_n_norm_sq) * g_n
                        u_scp_orth = -(scp_eta) * (g_c - proj_c)
                        u_scp_full = -(scp_eta) * g_c
                        u_scp = torch.where(is_conflicting, u_scp_orth, u_scp_full)
                        u_scp_scaled = u_scp * dt_normalized
                        norm_u = torch.norm(u_scp_scaled, p=2, dim=norm_dims, keepdim=True)
                        max_norm = clip_frac * (norm_ref + epsilon)
                        scale = (max_norm / (norm_u + epsilon)).clamp(max=1.0)
                        total_correction += u_scp_scaled * scale

            return (x_next_standard + total_correction).detach()

        model_output = predict_denoiser(denoiser, x_t, t_curr, cond=encoder_hidden_states)

        alpha_prod_t_next = scheduler.alphas_cumprod[t_next] if t_next >= 0 else scheduler.final_alpha_cumprod
        alpha_prod_t_curr = scheduler.alphas_cumprod[t_curr]

        prediction_type = self._get_prediction_type(scheduler)
        pred_original_sample, pred_epsilon = self._convert_model_output(
            model_output, x_t, alpha_prod_t_curr, prediction_type
        )

        x_next_standard = alpha_prod_t_next**0.5 * pred_original_sample + (1 - alpha_prod_t_next) ** 0.5 * pred_epsilon

        with torch.no_grad():
            original_drift_detached = x_next_standard.detach() - x_t.detach()
            norm_dims = tuple(range(1, original_drift_detached.ndim))
            norm_original_drift = torch.norm(original_drift_detached, p=2, dim=norm_dims, keepdim=True)

        total_correction = torch.zeros_like(x_t)
        dt = t_next - t_curr
        d = x_t.numel() / x_t.shape[0]

        if (use_cgd and cgd_eta > 0) or (use_scp and scp_eta > 0):
            if x0 is None:
                raise ValueError("x0 must be provided for orthogonal corrections")

            noise_norm_sq = torch.sum(pred_epsilon**2, dim=norm_dims, keepdim=True)
            noise_loss = (noise_norm_sq - d) ** 2
            g_n = torch.autograd.grad(noise_loss.sum(), x_t, retain_graph=True)[0]

            content_loss = torch.sum((pred_original_sample - x0) ** 2, dim=norm_dims, keepdim=True)
            g_c = torch.autograd.grad(content_loss.sum(), x_t, retain_graph=False)[0]

            with torch.no_grad():
                g_n_norm_sq = torch.sum(g_n**2, dim=norm_dims, keepdim=True) + epsilon
                g_c_norm_sq = torch.sum(g_c**2, dim=norm_dims, keepdim=True) + epsilon
                g_n_dot_g_c = torch.sum(g_n * g_c, dim=norm_dims, keepdim=True)

                cos_angle = g_n_dot_g_c / (g_n_norm_sq.sqrt() * g_c_norm_sq.sqrt() + epsilon)

                is_conflicting = cos_angle < 0

                if use_cgd and cgd_eta > 0:
                    projection_n = (g_n_dot_g_c / g_c_norm_sq) * g_c
                    u_cgd_orthogonal = -cgd_eta * (g_n - projection_n)
                    u_cgd_full = -cgd_eta * g_n
                    u_cgd = torch.where(is_conflicting, u_cgd_orthogonal, u_cgd_full)
                    u_cgd_scaled = u_cgd * dt
                    norm_u_cgd = torch.norm(u_cgd_scaled, p=2, dim=norm_dims, keepdim=True)
                    max_norm = clip_frac * (norm_original_drift + epsilon)
                    scale = (max_norm / (norm_u_cgd + epsilon)).clamp(max=1.0)
                    total_correction += u_cgd_scaled * scale

                if use_scp and scp_eta > 0:
                    projection_c = (g_n_dot_g_c / g_n_norm_sq) * g_n
                    u_scp_orthogonal = -scp_eta * (g_c - projection_c)
                    u_scp_full = -scp_eta * g_c
                    u_scp = torch.where(is_conflicting, u_scp_orthogonal, u_scp_full)
                    u_scp_scaled = u_scp * dt
                    norm_u_scp = torch.norm(u_scp_scaled, p=2, dim=norm_dims, keepdim=True)
                    max_norm = clip_frac * (norm_original_drift + epsilon)
                    scale = (max_norm / (norm_u_scp + epsilon)).clamp(max=1.0)
                    total_correction += u_scp_scaled * scale

        return (x_next_standard + total_correction).detach()

    def invert(self, image, denoiser, scheduler, encoder_hidden_states, num_steps=50, **kwargs):
        latents = self._match_denoiser(image, denoiser, copy=True)
        reference_image = self._match_denoiser(image, denoiser)
        encoder_hidden_states = self._ensure_condition_dtype(encoder_hidden_states, denoiser)

        collect_live_binned: bool = bool(kwargs.pop("collect_live_binned", False))
        live_metrics = LiveBinnedMetrics() if collect_live_binned else None

        store_intermediates = kwargs.pop("store_intermediates", False)
        store_predicted_noise = kwargs.pop("store_predicted_noise", False)
        store_timesteps = kwargs.pop("store_timesteps", False)
        intermediate_latents = [latents.clone().detach().cpu()] if store_intermediates else None
        predicted_noises = [] if store_predicted_noise else None
        timesteps_list = [] if store_timesteps else None
        noise_generator = kwargs.pop("generator", None)

        forward_fraction_override = kwargs.pop("forward_fraction", None)
        min_forward_steps_override = kwargs.pop("min_forward_steps", None)
        forward_steps_override = kwargs.pop("forward_steps", None)

        if forward_fraction_override is not None:
            forward_fraction = max(0.0, min(float(forward_fraction_override), 1.0))
        else:
            forward_fraction = self.forward_fraction

        if min_forward_steps_override is not None:
            min_forward_steps = max(0, int(min_forward_steps_override))
        else:
            min_forward_steps = self.min_forward_steps

        scheduler.set_timesteps(num_steps)

        ddpm_like, edm_like, fm_like = self._sched_kind(scheduler)

        if edm_like:
            sigma0 = scheduler.inversion_timesteps[0]
            if noise_generator is not None:
                noise = torch.randn(
                    latents.shape,
                    device=latents.device,
                    dtype=latents.dtype,
                    generator=noise_generator,
                )
            else:
                noise = torch.randn_like(latents)
            latents = latents + sigma0 * noise

            inv_ts = scheduler.inversion_timesteps
            n_steps = min(num_steps, inv_ts.shape[0] - 1)
            for i in range(n_steps):
                t_curr = inv_ts[i]
                t_next = inv_ts[i + 1]

                if store_timesteps:
                    timesteps_list.append(float(t_next))

                if store_predicted_noise:
                    noise_pred = self._predict_noise_for_capture(
                        latents,
                        t_curr,
                        denoiser,
                        scheduler,
                        encoder_hidden_states,
                        False,
                        True,
                    )
                    predicted_noises.append(noise_pred.clone().detach().cpu())

                if "collect_live_binned" in locals() and collect_live_binned:
                    try:
                        B = latents.shape[0]
                        sigma_cur = (
                            t_curr
                            if torch.is_tensor(t_curr)
                            else torch.tensor(t_curr, device=latents.device, dtype=latents.dtype)
                        )
                        sigma_data = (
                            scheduler.config["edm_sigma_data"]
                            if isinstance(getattr(scheduler, "config", {}), dict)
                            else getattr(scheduler.config, "edm_sigma_data", 0.5)
                        )
                        c_in = 1.0 / torch.sqrt(
                            sigma_cur**2 + torch.as_tensor(sigma_data, device=latents.device, dtype=latents.dtype) ** 2
                        )
                        unet_in = c_in * latents
                        F_theta = denoiser(
                            unet_in,
                            sigma_cur.expand(B),
                            encoder_hidden_states=encoder_hidden_states,
                        ).sample

                        sigma_data_t = torch.as_tensor(sigma_data, device=latents.device, dtype=latents.dtype)
                        c_skip = (sigma_data_t**2) / (sigma_cur**2 + sigma_data_t**2)
                        c_out = sigma_cur * sigma_data_t / torch.sqrt(sigma_cur**2 + sigma_data_t**2)
                        x0_hat = c_skip * latents + c_out * F_theta
                        eps_pred = (latents - x0_hat) / (
                            sigma_cur.view(1, *([1] * (latents.ndim - 1))).expand_as(latents) + 1e-12
                        )
                        eps2_vals = (eps_pred**2).flatten(1).sum(1)
                        live_metrics.eps2_mean.append(float(eps2_vals.mean().item()))
                        x0_pred = latents - sigma_cur.view(1, *([1] * (latents.ndim - 1))) * eps_pred
                        mse_x0 = torch.mean(
                            (x0_pred - reference_image) ** 2,
                            dim=tuple(range(1, x0_pred.ndim)),
                        ).mean()
                        live_metrics.mse_x0.append(float(mse_x0.item()))
                        ci_val = float("nan")
                        if encoder_hidden_states is not None and torch.is_floating_point(encoder_hidden_states):
                            F_zero = denoiser(
                                unet_in,
                                sigma_cur.expand(B),
                                encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                            ).sample
                            x0_hat_zero = c_skip * latents + c_out * F_zero
                            eps_zero = (latents - x0_hat_zero) / (
                                sigma_cur.view(1, *([1] * (latents.ndim - 1))).expand_as(latents) + 1e-12
                            )
                            num = (eps_pred - eps_zero).flatten(1).norm(2, dim=1).mean()
                            den = eps_pred.flatten(1).norm(2, dim=1).mean().clamp_min(1e-12)
                            ci_val = float((num / den).item())
                        live_metrics.cond_influence.append(ci_val)
                        eps2_sq_mean = (eps2_vals**2).mean().item()
                        var_eps2 = float(max(0.0, eps2_sq_mean - eps2_vals.mean().item() ** 2))
                        live_metrics.eps2_var.append(var_eps2)
                        if live_metrics.eps2_hist_counts is None:
                            edges = torch.logspace(
                                -2.0,
                                2.0,
                                steps=33,
                                device=latents.device,
                                dtype=latents.dtype,
                            )
                            e2 = eps2_vals.clamp_min(edges[0]).clamp_max(edges[-1] - 1e-12)
                            bins_r = torch.bucketize(e2, edges) - 1
                            counts = torch.stack([(bins_r == bi).sum() for bi in range(edges.numel() - 1)])
                            live_metrics.eps2_hist_edges = edges.detach().cpu().tolist()
                            live_metrics.eps2_hist_counts = counts.detach().cpu().tolist()
                        live_metrics.t_norm.append(float((i + 0.5) / max(1, n_steps)))
                    except Exception:
                        live_metrics.eps2_mean.append(float("nan"))
                        live_metrics.mse_x0.append(float("nan"))
                        live_metrics.cond_influence.append(float("nan"))
                        live_metrics.eps2_var.append(float("nan"))
                        live_metrics.t_norm.append(float("nan"))
                latents = self.invert_step(
                    x_t=latents,
                    t_next=t_next,
                    t_curr=t_curr,
                    denoiser=denoiser,
                    scheduler=scheduler,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cgd=self.use_cgd,
                    cgd_eta=self.cgd_eta,
                    use_scp=self.use_scp,
                    scp_eta=self.scp_eta,
                    x0=reference_image,
                    clip_frac=1.0,
                    epsilon=1e-6,
                )
                if store_intermediates:
                    intermediate_latents.append(latents.clone().detach().cpu())
                if i % 10 == 0:
                    torch.cuda.synchronize()

            return {
                "latents": latents,
                "intermediate_latents": intermediate_latents,
                "predicted_noises": predicted_noises,
                "timesteps": timesteps_list,
                "live_metrics": (live_metrics.to_payload() if live_metrics is not None else None),
            }

        reversed_timesteps = list(reversed(scheduler.timesteps))

        if forward_fraction > 0 or min_forward_steps > 0:
            forward_steps = int(round(num_steps * forward_fraction))
            forward_steps = max(min_forward_steps, forward_steps)
        else:
            forward_steps = 0

        if forward_steps_override is not None:
            forward_steps = max(0, int(forward_steps_override))

        forward_steps = min(forward_steps, len(reversed_timesteps))

        for i, t in enumerate(tqdm(reversed_timesteps, desc="Inversion")):
            next_t = t
            current_t = max(
                t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps,
                0,
            )

            if store_timesteps:
                timesteps_list.append(float(current_t))

            if store_predicted_noise:
                noise_pred = self._predict_noise_for_capture(
                    latents,
                    current_t,
                    denoiser,
                    scheduler,
                    encoder_hidden_states,
                    True,
                    False,
                )
                predicted_noises.append(noise_pred.clone().detach().cpu())

            if "collect_live_binned" in locals() and collect_live_binned:
                try:
                    B = latents.shape[0]
                    Tcfg = int(scheduler.config.num_train_timesteps)
                    t_tensor = torch.tensor(current_t, device=latents.device, dtype=torch.long).repeat(B)
                    raw_pred = predict_denoiser(denoiser, latents, t_tensor, cond=encoder_hidden_states)
                    a_buf = scheduler.alphas_cumprod
                    t_idx = t_tensor.to(device=a_buf.device)
                    a = a_buf[t_idx].to(device=latents.device, dtype=latents.dtype)
                    a = a.view(B, *([1] * (latents.ndim - 1))).clamp_min(1e-8)
                    alpha_t = a.sqrt()
                    sigma_t = (1 - a).sqrt()
                    x0_ref = reference_image
                    eps_pred, _ = _get_eps_from_prediction(
                        raw_pred,
                        "epsilon",
                        latents,
                        x0_ref,
                        noise=latents.new_zeros(latents.shape),
                        alpha_t=alpha_t,
                        sigma_t=sigma_t,
                        is_ddpm_like=True,
                    )
                    eps2_vals = (eps_pred**2).flatten(1).sum(1)
                    live_metrics.eps2_mean.append(float(eps2_vals.mean().item()))
                    x0_hat = (latents - sigma_t * eps_pred) / alpha_t
                    mse_x0 = torch.mean((x0_hat - x0_ref) ** 2, dim=tuple(range(1, x0_hat.ndim))).mean()
                    live_metrics.mse_x0.append(float(mse_x0.item()))
                    ci_val = float("nan")
                    if encoder_hidden_states is not None and torch.is_floating_point(encoder_hidden_states):
                        raw_zero = denoiser(
                            latents,
                            t_tensor,
                            encoder_hidden_states=torch.zeros_like(encoder_hidden_states),
                        ).sample
                        eps_zero, _ = _get_eps_from_prediction(
                            raw_zero,
                            "epsilon",
                            latents,
                            x0_ref,
                            noise=None,
                            alpha_t=alpha_t,
                            sigma_t=sigma_t,
                            is_ddpm_like=True,
                        )
                        num = (eps_pred - eps_zero).flatten(1).norm(2, dim=1).mean()
                        den = eps_pred.flatten(1).norm(2, dim=1).mean().clamp_min(1e-12)
                        ci_val = float((num / den).item())
                    live_metrics.cond_influence.append(ci_val)
                    eps2_sq_mean = (eps2_vals**2).mean().item()
                    var_eps2 = float(eps2_sq_mean - eps2_vals.mean().item() ** 2)
                    live_metrics.eps2_var.append(var_eps2)
                    if live_metrics.eps2_hist_counts is None:
                        edges = torch.logspace(
                            -2.0,
                            2.0,
                            steps=33,
                            device=latents.device,
                            dtype=latents.dtype,
                        )
                        e2 = eps2_vals.clamp_min(edges[0]).clamp_max(edges[-1] - 1e-12)
                        bins = torch.bucketize(e2, edges) - 1
                        counts = torch.stack([(bins == bi).sum() for bi in range(edges.numel() - 1)])
                        live_metrics.eps2_hist_edges = edges.detach().cpu().tolist()
                        live_metrics.eps2_hist_counts = counts.detach().cpu().tolist()
                    live_metrics.t_norm.append(float((float(current_t) + 0.5) / max(1, Tcfg)))
                except Exception:
                    live_metrics.eps2_mean.append(float("nan"))
                    live_metrics.mse_x0.append(float("nan"))
                    live_metrics.cond_influence.append(float("nan"))
                    live_metrics.eps2_var.append(float("nan"))
                    live_metrics.t_norm.append(float("nan"))

            if i < forward_steps:
                coeff = scheduler.alphas_cumprod[next_t] if next_t >= 0 else scheduler.final_alpha_cumprod
                coeff = torch.as_tensor(coeff, device=latents.device, dtype=latents.dtype)
                if noise_generator is None:
                    noise = torch.randn_like(latents)
                else:
                    noise = torch.randn(
                        latents.shape,
                        generator=noise_generator,
                        device=latents.device,
                        dtype=latents.dtype,
                    )
                latents = coeff.sqrt() * reference_image + (1 - coeff).sqrt() * noise
            else:
                latents = self.invert_step(
                    x_t=latents,
                    t_next=next_t,
                    t_curr=current_t,
                    denoiser=denoiser,
                    scheduler=scheduler,
                    encoder_hidden_states=encoder_hidden_states,
                    use_cgd=self.use_cgd,
                    cgd_eta=self.cgd_eta,
                    use_scp=self.use_scp,
                    scp_eta=self.scp_eta,
                    x0=reference_image,
                    clip_frac=1.0,
                    epsilon=1e-6,
                )

            if store_intermediates:
                intermediate_latents.append(latents.clone().detach().cpu())

            if i % 10 == 0:
                torch.cuda.synchronize()

        return {
            "latents": latents,
            "intermediate_latents": intermediate_latents,
            "predicted_noises": predicted_noises,
            "timesteps": timesteps_list,
            "live_metrics": (live_metrics.to_payload() if live_metrics is not None else None),
        }
