import torch

from ovg.utils.predict import predict_with_cfg
from .base import BaseTrainingStrategy


class EDMTrainingStrategy(BaseTrainingStrategy):
    def compute_loss(self, batch) -> torch.Tensor:
        trainer = self.trainer
        cfg = self.cfg

        if cfg.model.space == "latent":
            trainer.vae.eval()
            with torch.no_grad():
                imgs = batch["image"].to(dtype=trainer.weight_dtype)
                x_0 = (
                    trainer.vae.encode(imgs).latent_dist.sample()
                    * trainer.vae.config.scaling_factor
                )
        else:
            x_0 = batch["image"].to(dtype=trainer.weight_dtype)

        bsz = x_0.shape[0]

        P_mean = getattr(cfg.training, "edm_p_mean", -1.2)
        P_std = getattr(cfg.training, "edm_p_std", 1.2)

        sigma_data = getattr(cfg.training, "edm_sigma_data", 0.5)

        rnd_normal = torch.randn((bsz,), device=x_0.device, dtype=x_0.dtype)
        sigma = (rnd_normal * P_std + P_mean).exp()
        sigma_view = sigma.view(-1, *([1] * (x_0.ndim - 1)))

        noise = torch.randn_like(x_0)
        noisy_latents = x_0 + noise * sigma_view

        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
        c_in = 1 / (sigma**2 + sigma_data**2).sqrt()
        c_weight = (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2

        unet_input = c_in.view(-1, *([1] * (x_0.ndim - 1))) * noisy_latents

        class_labels = self._extract_class_labels(batch)
        conditioning = self._prepare_condition_embeddings(class_labels)

        model_pred = predict_with_cfg(
            trainer.denoiser,
            unet_input.detach(),
            sigma,
            conditioning.class_embeddings,
            cfg_scale=0.0,
        )

        x_0_pred = (
            c_skip.view(-1, *([1] * (x_0.ndim - 1))) * noisy_latents
            + c_out.view(-1, *([1] * (x_0.ndim - 1))) * model_pred
        )

        loss_weights_view = c_weight.view(-1, *([1] * (x_0.ndim - 1)))
        loss = (
            loss_weights_view * (x_0_pred - x_0.to(dtype=x_0_pred.dtype)) ** 2
        ).mean()

        loss_scale = getattr(cfg.training, "loss_scale_factor", 1.0)
        return loss * loss_scale
