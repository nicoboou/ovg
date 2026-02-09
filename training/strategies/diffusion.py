import torch
import torch.nn.functional as F

from ovg.utils.predict import predict_with_cfg

from .base import BaseTrainingStrategy


class DiffusionTrainingStrategy(BaseTrainingStrategy):
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

        noise = torch.randn_like(x_0, dtype=x_0.dtype)
        if getattr(cfg.training, "noise_offset", 0.0):
            noise += cfg.training.noise_offset * torch.randn(
                (x_0.shape[0], x_0.shape[1], 1, 1),
                device=x_0.device,
                dtype=x_0.dtype,
            )

        bsz = x_0.shape[0]
        timesteps = torch.randint(
            0,
            trainer.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=x_0.device,
        ).long()
        noisy_latents = trainer.noise_scheduler.add_noise(x_0, noise, timesteps)

        class_labels = self._extract_class_labels(batch)
        conditioning = self._prepare_condition_embeddings(class_labels)

        model_pred = predict_with_cfg(
            trainer.denoiser,
            noisy_latents.detach(),
            timesteps,
            conditioning.class_embeddings,
            cfg_scale=0.0,
        )

        alphas_cumprod = trainer.noise_scheduler.alphas_cumprod.to(model_pred.device)
        alpha_t = alphas_cumprod[timesteps].sqrt().view(-1, 1, 1, 1)
        sigma_t = (1 - alphas_cumprod[timesteps]).sqrt().view(-1, 1, 1, 1)

        if trainer.noise_scheduler.config.prediction_type == "v_prediction":
            v_target = alpha_t * noise - sigma_t * x_0
            loss = F.mse_loss(model_pred, v_target.to(dtype=model_pred.dtype))
        elif trainer.noise_scheduler.config.prediction_type == "sample":
            loss = F.mse_loss(model_pred, x_0.to(dtype=model_pred.dtype))
        else:
            mse = (model_pred - noise.to(dtype=model_pred.dtype)) ** 2
            gamma = getattr(cfg.training, "min_snr_gamma", None)

            if gamma is not None:
                snr = (alpha_t**2) / (sigma_t**2 + 1e-8)
                weights = (snr.clamp_max(gamma) / (snr + 1e-8)).detach()
                loss = (weights * mse.flatten(1).mean(dim=1)).mean()
            else:
                loss = mse.mean()

        loss_scale = getattr(cfg.training, "loss_scale_factor", 1.0)
        return loss * loss_scale
