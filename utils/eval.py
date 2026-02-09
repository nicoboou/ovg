import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import warnings
import torch
import torch.nn.functional as F
from typing import Any, Optional

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import NoTrainInceptionV3

import wandb
from tqdm import tqdm


from ovg.utils.helpers import get_model_dtype
from ovg.utils.predict import predict_with_cfg

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def resize_if_needed(pil_img, max_dim=299):
    w, h = pil_img.size
    if w > max_dim or h > max_dim:
        return pil_img.resize((max_dim, max_dim), Image.BICUBIC)
    return pil_img


def update_metric_in_chunks(metric, data, chunk_size, device, **kwargs):
    for i in range(0, data.shape[0], chunk_size):
        chunk = data[i : i + chunk_size].to(device)
        metric.update(chunk, **kwargs)


def update_metric_pair_in_chunks(metric, preds, targets, chunk_size, device):
    for i in range(0, preds.shape[0], chunk_size):
        p_chunk = preds[i : i + chunk_size].to(device)
        t_chunk = targets[i : i + chunk_size].to(device)
        metric.update(p_chunk, t_chunk)


def compute_gradient_norms(unet, class_embedding):
    unet_grad_norm = 0.0
    class_embedding_grad_norm = 0.0

    if hasattr(unet, "parameters"):
        unet_grads = [p.grad for p in unet.parameters() if p.grad is not None]
        if unet_grads:
            unet_grad_norm = torch.norm(
                torch.stack([torch.norm(g.detach()) for g in unet_grads])
            ).item()

    if class_embedding is not None and hasattr(class_embedding, "parameters"):
        class_embedding_grads = [
            p.grad for p in class_embedding.parameters() if p.grad is not None
        ]
        if class_embedding_grads:
            class_embedding_grad_norm = torch.norm(
                torch.stack([torch.norm(g.detach()) for g in class_embedding_grads])
            ).item()

    return unet_grad_norm, class_embedding_grad_norm


def _sample_with_guidance(
    cfg,
    x,
    scheduler,
    denoiser,
    vae,
    class_embeddings: torch.Tensor,
    uncond_embeddings: torch.Tensor,
    guidance_scale: float,
    device: torch.device,
):
    dtype = get_model_dtype(denoiser)
    training_mode = getattr(cfg.training, "training_mode", "diffusion").lower()

    p_uncond = float(
        getattr(getattr(cfg, "training", object()), "p_uncond", 0.0) or 0.0
    )
    effective_guidance_scale = guidance_scale if p_uncond > 0.0 else 0.0
    is_latent = cfg.model.space == "latent"

    if is_latent:
        latent_height = denoiser.config.sample_size
        latent_width = denoiser.config.sample_size
        sample_shape = (1, denoiser.config.in_channels, latent_height, latent_width)
    else:
        sample_shape = (
            1,
            denoiser.config.in_channels,
            denoiser.config.sample_size,
            denoiser.config.sample_size,
        )

    x = (
        torch.randn(sample_shape, device=device, dtype=dtype)
        if x is None
        else x.to(device=device, dtype=dtype)
    )

    if scheduler is not None and hasattr(scheduler, "init_noise_sigma"):
        init_sigma = scheduler.init_noise_sigma
        if isinstance(init_sigma, torch.Tensor):
            init_sigma = init_sigma.to(device=device, dtype=dtype)
        x = x * init_sigma

    if training_mode == "edm":
        num_inf = int(getattr(cfg.training, "num_inference_steps", 50))

        sigma_min = getattr(cfg.training, "edm_sigma_min", 0.002)
        sigma_max = getattr(cfg.training, "edm_sigma_max", 80.0)

        sigmas = torch.linspace(
            math.log(sigma_max), math.log(sigma_min), num_inf + 1, device=device
        ).exp()

        x = x * sigmas[0]
        sigma_data = getattr(cfg.training, "edm_sigma_data", 0.5)

        for i in range(num_inf):
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_data_t = torch.as_tensor(
                sigma_data, device=sigma_cur.device, dtype=sigma_cur.dtype
            )
            c_skip = sigma_data_t**2 / (sigma_cur**2 + sigma_data_t**2)
            c_out = sigma_cur * sigma_data_t / (sigma_cur**2 + sigma_data_t**2).sqrt()
            c_in = 1 / (sigma_cur**2 + sigma_data_t**2).sqrt()

            unet_input = c_in * x

            F_theta = predict_with_cfg(
                denoiser,
                unet_input,
                sigma_cur.view(1),
                class_embeddings,
                effective_guidance_scale,
                null_cond=uncond_embeddings,
            )

            x_0_pred = c_skip * x + c_out * F_theta

            d = (x - x_0_pred) / sigma_cur

            dt = sigma_next - sigma_cur
            x = x + d * dt

    elif training_mode == "diffusion":
        num_inf = int(
            getattr(cfg.training, "num_inference_steps", len(scheduler.timesteps))
        )
        scheduler.set_timesteps(num_inf)

        for t in scheduler.timesteps:
            model_input = x
            if hasattr(scheduler, "scale_model_input"):
                model_input = scheduler.scale_model_input(x, t)

            noise_pred = predict_with_cfg(
                denoiser,
                model_input,
                t,
                class_embeddings,
                effective_guidance_scale,
                null_cond=uncond_embeddings,
            )
            x = scheduler.step(noise_pred, t, x).prev_sample

    if is_latent:
        x = x.to(get_model_dtype(vae))
        return vae.decode(x / vae.config.scaling_factor).sample

    return x


def compute_inception_features_in_chunks(
    images, feature_extractor, device, chunk_size=64
):
    features = []
    N = images.shape[0]
    for i in range(0, N, chunk_size):
        batch = images[i : i + chunk_size].to(device)
        with torch.no_grad():
            feat = feature_extractor(batch)
        features.append(feat.cpu())
    return torch.cat(features, dim=0)


def compute_pr_features(real_features, gen_features, k=3, quantile=0.5):
    with torch.no_grad():
        real_dists = torch.cdist(real_features, real_features)
        real_dists.fill_diagonal_(float("inf"))
        kth_vals, _ = real_dists.topk(k, largest=False)
        real_radii = kth_vals[:, -1]
        threshold = real_radii.quantile(quantile)

        gen_min_dists = torch.cdist(gen_features, real_features).min(dim=1)[0]
        precision = (gen_min_dists <= threshold).float().mean().item()

        real_min_dists = torch.cdist(real_features, gen_features).min(dim=1)[0]
        recall = (real_min_dists <= threshold).float().mean().item()

    return precision, recall


def im_set_corr(set1, set2, remove_mean=True):
    remove_im_mean = lambda data: data - data.mean(dim=(1, 2, 3), keepdims=True)

    if len(set1.shape) != 4 or len(set2.shape) != 4:
        raise ValueError("Input shape error")
    if remove_mean:
        set1 = remove_im_mean(set1)
        set2 = remove_im_mean(set2)

    norms1 = set1.norm(dim=(2, 3), keepdim=True).norm(dim=1, keepdim=True)
    norms1[norms1 == 0] = 0.001
    norms2 = set2.norm(dim=(2, 3), keepdim=True).norm(dim=1, keepdim=True)
    norms2[norms2 == 0] = 0.001

    return torch.matmul(
        ((set1 / norms1).flatten(start_dim=1)), (set2 / norms2).flatten(start_dim=1).T
    )


def create_label_image(label, size=(512, 512), device="cuda"):
    if isinstance(label, torch.Tensor):
        if label.dim() > 1:
            label_value = label.argmax().item() if label.numel() > 1 else label.item()
        else:
            label_value = label.item()
    else:
        label_value = int(label)

    pil_img = Image.new("RGB", size, color=(0, 0, 0))
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("Arial", 120)
    except IOError:
        font = ImageFont.load_default()

    text = f"Class: {label_value}"

    try:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        try:
            text_width, text_height = draw.textsize(text, font=font)
        except AttributeError:
            try:
                text_width, text_height = font.getsize(text)
            except AttributeError:
                text_width, text_height = size[0] // 4, size[1] // 8

    position = ((size[0] - text_width) // 2, (size[1] - text_height) // 2)
    draw.text(position, text, fill=(255, 255, 255), font=font)

    img_np = np.array(pil_img).transpose(2, 0, 1) / 255.0
    img_tensor = torch.from_numpy(img_np).float().to(device)

    return img_tensor


def evaluate_quality(
    vae,
    unet,
    gene_encoder,
    scheduler,
    dataloader,
    device,
    accelerator,
    num_samples=10000,
    guidance_scale=7.5,
    chunk_size=64,
    feature_extractor=None,
    eval_img_dir="eval_images",
    image_resolution=512,
    condition_type="class",
    condition_in_dim=None,
    null_class_label=None,
):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    is_metric = InceptionScore().to(device)
    kid_metric = KernelInceptionDistance(feature=2048, subset_size=8).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if feature_extractor is None:
        feature_extractor = NoTrainInceptionV3(
            name="inception-v3-compat", features_list=["2048"]
        ).to(device)
        feature_extractor.eval()

    real_features_list = []
    gen_features_list = []

    num_samples_per_process = num_samples // accelerator.num_processes
    total_processed = 0

    vae.eval()
    unet.eval()
    unet_model = unet.module if hasattr(unet, "module") else unet
    unet_dtype = get_model_dtype(unet)

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            desc="Evaluating (FID/IS/KID...)",
            leave=False,
            colour="#74D673",
            disable=not accelerator.is_main_process,
        ):
            if total_processed >= num_samples_per_process:
                break

            real_batch = batch["pixel_values"].to(device)
            gene_batch = batch["gene_labels"].to(device)

            current_batch_size = real_batch.size(0)
            if total_processed + current_batch_size > num_samples_per_process:
                current_batch_size = num_samples_per_process - total_processed
                real_batch = real_batch[:current_batch_size]
                gene_batch = gene_batch[:current_batch_size]

            if gene_encoder is not None:
                gene_embedding = gene_encoder(gene_batch)

                if null_class_label is not None:
                    special_class_idx = null_class_label
                else:
                    special_class_idx = (
                        next(iter(gene_encoder.parameters())).shape[0] - 1
                    )

                uncond_input = torch.full_like(gene_batch, fill_value=special_class_idx)
                uncond_gene = gene_encoder(uncond_input)
            else:
                gene_embedding = gene_batch
                if null_class_label is not None:
                    uncond_gene = torch.full_like(
                        gene_batch, fill_value=null_class_label
                    )
                else:
                    uncond_gene = gene_batch

            latent_height = image_resolution // 8
            latent_width = image_resolution // 8

            latents = torch.randn(
                (
                    current_batch_size,
                    unet_model.config.in_channels,
                    latent_height,
                    latent_width,
                ),
                device=device,
                dtype=unet_dtype,
            )
            if hasattr(scheduler, "init_noise_sigma"):
                init_sigma = scheduler.init_noise_sigma
                if isinstance(init_sigma, torch.Tensor):
                    init_sigma = init_sigma.to(device=device, dtype=unet_dtype)
                latents = latents * init_sigma

            for t in scheduler.timesteps:
                model_input = latents
                if hasattr(scheduler, "scale_model_input"):
                    model_input = scheduler.scale_model_input(latents, t)

                noise_pred_cond = unet(
                    model_input, t, gene_embedding[:current_batch_size]
                ).sample
                noise_pred_uncond = unet(
                    model_input, t, uncond_gene[:current_batch_size]
                ).sample
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            vae_dtype = next(vae.parameters()).dtype
            latents = latents.to(vae_dtype)
            decoded = vae.decode(latents / vae.config.scaling_factor).sample

            gen_float = torch.clamp((decoded + 1) / 2, 0, 1)
            real_float = torch.clamp((real_batch + 1) / 2, 0, 1)

            gen_uint8 = (gen_float * 255).to(torch.uint8)
            real_uint8 = (real_float * 255).to(torch.uint8)

            fid_metric.update(real_uint8, real=True)
            fid_metric.update(gen_uint8, real=False)

            is_metric.update(gen_uint8)

            kid_metric.update(real_uint8, real=True)
            kid_metric.update(gen_uint8, real=False)

            psnr_metric.update(gen_float, real_float)
            ssim_metric.update(gen_float, real_float)

            gen_resized = F.interpolate(
                gen_float, size=(299, 299), mode="bicubic", align_corners=False
            )
            real_resized = F.interpolate(
                real_float, size=(299, 299), mode="bicubic", align_corners=False
            )

            gen_feats = feature_extractor(gen_resized)
            real_feats = feature_extractor(real_resized)

            gen_features_list.append(gen_feats.cpu())
            real_features_list.append(real_feats.cpu())

            total_processed += current_batch_size

    fid_value = fid_metric.compute().item()

    is_scores = is_metric.compute()
    is_mean = is_scores[0].item()
    is_std = is_scores[1].item()

    kid_result = kid_metric.compute()
    if isinstance(kid_result, dict):
        kid_mean = kid_result["kid_mean"].item()
        kid_std = kid_result["kid_std"].item()
    else:
        kid_mean, kid_std = kid_result[0].item(), kid_result[1].item()

    psnr_value = psnr_metric.compute().item()
    ssim_value = ssim_metric.compute().item()

    if gen_features_list:
        all_gen_features = torch.cat(gen_features_list, dim=0).to(device)
        all_real_features = torch.cat(real_features_list, dim=0).to(device)

        all_gen_features = accelerator.gather(all_gen_features)
        all_real_features = accelerator.gather(all_real_features)

        precision, recall = compute_pr_features(
            all_real_features.cpu(), all_gen_features.cpu(), k=3
        )
    else:
        precision, recall = 0.0, 0.0

    fid_metric.reset()
    is_metric.reset()
    kid_metric.reset()
    psnr_metric.reset()
    ssim_metric.reset()

    if accelerator.is_main_process:
        return (
            fid_value,
            is_mean,
            is_std,
            kid_mean,
            kid_std,
            ssim_value,
            psnr_value,
            precision,
            recall,
        )
    else:
        return None


def generate_and_log_grids(
    vae,
    denoiser,
    class_embedding,
    scheduler,
    dataloader,
    cfg,
    accelerator,
    global_step,
    n_samples=10,
    n_variations=1,
    guidance_scale=7.5,
    fixed_seed=42,
    null_class_label: int | None = None,
    epoch: int | None = None,
    run_logger: Optional[Any] = None,
):
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(fixed_seed)

    dataset = dataloader.dataset
    device = next(denoiser.parameters()).device

    if hasattr(dataset, "available_classes") and dataset.available_classes:
        class_ids = sorted(list(dataset.available_classes))
    else:
        class_ids = []
        for i in range(len(dataset)):
            item = dataset[i]
            lbl = item.get("class_idx") if isinstance(item, dict) else None
            if lbl is not None:
                val = int(lbl if not torch.is_tensor(lbl) else lbl.item())
                if val not in class_ids:
                    class_ids.append(val)

    class_ids = class_ids[:n_samples]

    if not class_ids:
        print("Warning: No classes found in dataset")

        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        return

    prev_denoiser_training = denoiser.training
    denoiser.eval()
    denoiser_unwrapped = denoiser.module if hasattr(denoiser, "module") else denoiser
    if cfg.model.space == "latent" and vae is not None:
        vae.eval()

        latent_height = denoiser_unwrapped.config.sample_size
        latent_width = denoiser_unwrapped.config.sample_size
        sample_shape = (
            1,
            denoiser_unwrapped.config.in_channels,
            latent_height,
            latent_width,
        )
    else:
        height, width = (
            denoiser_unwrapped.config.sample_size,
            denoiser_unwrapped.config.sample_size,
        )
        sample_shape = (1, denoiser_unwrapped.config.in_channels, height, width)

    accelerator.wait_for_everyone()

    def _class_name(cid: int) -> str:
        name = None
        if (
            hasattr(dataset, "class_names")
            and dataset.class_names
            and cid < len(dataset.class_names)
        ):
            name = dataset.class_names[cid]
        elif hasattr(dataset, "label_to_name") and isinstance(
            dataset.label_to_name, dict
        ):
            name = dataset.label_to_name.get(cid)
        return str(name) if name is not None else f"class_{cid}"

    panels: dict[str, list] = {}
    panels_np: dict[str, list] = {}

    with torch.no_grad():
        condition_dtype = (
            torch.long if class_embedding is None else get_model_dtype(class_embedding)
        )
        dit_uncond_label = None
        if class_embedding is None:
            if null_class_label is None:
                null_class_label = int(
                    getattr(getattr(cfg, "data", object()), "num_classes")
                )
            dit_uncond_label = int(null_class_label)

        for cid in class_ids:
            cname = _class_name(cid)
            if cname not in panels:
                panels[cname] = []
            if cname not in panels_np:
                panels_np[cname] = []

        for seed_idx in range(n_samples):
            current_seed = fixed_seed * 1000 + seed_idx
            torch.manual_seed(current_seed)
            shared_noise = torch.randn(
                size=sample_shape,
                device=device,
                dtype=get_model_dtype(denoiser_unwrapped),
            )

            for cid in class_ids:
                if class_embedding is None:
                    cond_tensor = torch.tensor([cid], device=device, dtype=torch.long)
                else:
                    label = torch.tensor([cid], device=device, dtype=condition_dtype)
                    cond_tensor = class_embedding(label)

                uncond_tensor = None
                if guidance_scale and guidance_scale > 0:
                    if class_embedding is None:
                        uncond_tensor = torch.tensor(
                            [dit_uncond_label], device=device, dtype=torch.long
                        )
                    else:
                        special_idx = (
                            next(iter(class_embedding.parameters())).shape[0] - 1
                        )
                        unlabel = torch.tensor(
                            [special_idx], device=device, dtype=condition_dtype
                        )
                        uncond_tensor = class_embedding(unlabel)

                gen = _sample_with_guidance(
                    cfg,
                    shared_noise,
                    scheduler,
                    denoiser_unwrapped,
                    vae,
                    cond_tensor,
                    uncond_tensor,
                    guidance_scale,
                    device,
                )
                gen = torch.clamp((gen + 1) / 2, 0, 1).squeeze(0)

                if accelerator.is_main_process:
                    img_np = (gen * 255).byte().permute(1, 2, 0).cpu().numpy()
                    panels[_class_name(cid)].append(
                        wandb.Image(img_np, caption=f"class={cid}, seed={seed_idx}")
                    )
                    panels_np[_class_name(cid)].append(img_np)

    accelerator.wait_for_everyone()

    payload = {}
    for cname, imgs in panels.items():
        if imgs:
            payload[f"generated_samples/{cname}"] = imgs[:50]
    if epoch is not None:
        payload["epoch"] = epoch
    if payload:
        try:
            accelerator.log(payload, step=global_step)
        except Exception as e:
            print(f"Warning: Could not log images via accelerator.log: {e}")

    if run_logger is not None and accelerator.is_main_process:
        for cname, imgs_np in panels_np.items():
            if not imgs_np:
                continue

            max_imgs = min(len(imgs_np), 25)
            imgs_np = imgs_np[:max_imgs]

            cols = min(5, max_imgs)
            rows = (max_imgs + cols - 1) // cols

            h, w = imgs_np[0].shape[:2]
            pad = 2
            grid_w = cols * w + (cols - 1) * pad
            grid_h = rows * h + (rows - 1) * pad
            grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))

            for idx, arr in enumerate(imgs_np):
                r = idx // cols
                c = idx % cols
                x = c * (w + pad)
                y = r * (h + pad)
                grid.paste(Image.fromarray(arr), (x, y))

            key = f"generated_samples/{cname}_grid"
            try:
                run_logger.log_image(key=key, image=grid, step=global_step, epoch=epoch)
            except Exception as e:
                print(f"Warning: Could not log image locally for {cname}: {e}")

    torch.set_rng_state(cpu_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)

    if prev_denoiser_training:
        denoiser.train()
    else:
        denoiser.eval()

    del panels
    del panels_np
