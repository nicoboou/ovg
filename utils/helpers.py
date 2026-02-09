import re
import importlib
import torch
import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np
from scipy import stats
import warnings
from torch.nn import functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers.training_utils import cast_training_params

from peft import LoraConfig
from ovg.models.class_embedding import ResolutionEmbedding


def get_model_dtype(model):
    return next(model.parameters()).dtype


def unwrap_model(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_models(cfg):
    import hydra

    noise_scheduler = None
    if hasattr(cfg, "noise_scheduler") and cfg.noise_scheduler is not None:
        noise_scheduler = hydra.utils.instantiate(cfg.noise_scheduler)
    elif (
        hasattr(cfg.model, "noise_scheduler_path")
        and cfg.model.noise_scheduler_path is not None
    ):
        rescale_betas_zero_snr = omegaconf_select(
            cfg.model, "rescale_betas_zero_snr", default=False
        )
        noise_scheduler = DDIMScheduler.from_pretrained(
            cfg.model.noise_scheduler_path,
            subfolder="scheduler",
            rescale_betas_zero_snr=rescale_betas_zero_snr,
        )

    vae = None
    if cfg.model.space == "latent":
        vae = AutoencoderKL.from_pretrained(
            cfg.model.vae_path,
            subfolder="vae",
        )

    if cfg.model.pretrained:
        unet = UNet2DConditionModel.from_pretrained(
            cfg.model.unet_path,
            subfolder="unet",
        )

        expected_sample_size = (
            cfg.data.hr_size // 8 if cfg.model.space == "latent" else cfg.data.hr_size
        )
        if unet.config.sample_size != expected_sample_size:
            unet.config.sample_size = expected_sample_size
    else:
        if not hasattr(cfg.model, "params") or cfg.model.params is None:
            raise KeyError(
                "Expected key `params` with _target_ to instantiate the model from scratch."
            )

        model_conf = OmegaConf.to_container(cfg.model.params, resolve=True)

        if cfg.model.space == "pixel":
            model_conf["in_channels"] = 3
            model_conf["out_channels"] = 3
            model_conf["sample_size"] = cfg.data.hr_size
        elif cfg.model.space == "latent":
            model_conf["in_channels"] = 4
            model_conf["out_channels"] = 4
            model_conf["sample_size"] = cfg.data.hr_size // 8
        else:
            raise ValueError(
                f"Unsupported model space: {cfg.model.space}. Supported spaces are: 'pixel', 'latent'."
            )

        unet = hydra.utils.instantiate(model_conf)

    if cfg.model.use_lora:
        unet.requires_grad_(False)
        unet_lora_config = LoraConfig(
            r=cfg.model.lora_rank,
            lora_alpha=cfg.model.lora_rank * 2,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)

    model_class_name = unet.__class__.__name__
    is_dit = "DiT" in model_class_name or "Transformer" in model_class_name

    has_internal_class_embedding = (
        hasattr(unet, "config")
        and getattr(unet.config, "num_class_embeds", None) is not None
    )

    if is_dit or has_internal_class_embedding:
        class_embedding = None
    else:
        if hasattr(unet, "config") and hasattr(unet.config, "cross_attention_dim"):
            cross_attention_dim = unet.config.cross_attention_dim
        else:
            cross_attention_dim = model_conf.get("cross_attention_dim", 768)

        class_embedding = ResolutionEmbedding(
            num_classes=cfg.data.num_classes,
            cross_attn_dim=cross_attention_dim,
            target_dim=cross_attention_dim,
            sequence_length=1,
        )

    return noise_scheduler, vae, unet, class_embedding


def setup_weight_dtype(accelerator):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    return weight_dtype


def setup_optimizer_params(unet, class_embedding, cfg):
    base_lr = cfg.optimizer.learning_rate
    wd = cfg.optimizer.adam_weight_decay
    cond_lr_mult = float(getattr(cfg.optimizer, "cond_lr_mult", 1.0))
    name_patterns = list(getattr(cfg.optimizer, "cond_param_regex", []))
    cond_name_re = re.compile("|".join(name_patterns)) if name_patterns else None

    def no_wd(name, p):
        return (p.ndim == 1) or name.endswith(".bias")

    param_groups = []
    clip_params = []

    cond_wd, cond_nowd = [], []
    base_wd, base_nowd = [], []

    if class_embedding is not None:
        for n, p in class_embedding.named_parameters():
            if not p.requires_grad:
                continue
            clip_params.append(p)
            (cond_nowd if no_wd(n, p) else cond_wd).append(p)

    if class_embedding is None and cond_name_re is not None:
        for n, p in unet.named_parameters():
            if not p.requires_grad:
                continue
            if cond_name_re.search(n):
                clip_params.append(p)
                (cond_nowd if no_wd(n, p) else cond_wd).append(p)

    cond_ids = {id(p) for p in cond_wd + cond_nowd}

    for n, p in unet.named_parameters():
        if not p.requires_grad or id(p) in cond_ids:
            continue
        clip_params.append(p)
        (base_nowd if no_wd(n, p) else base_wd).append(p)

    if base_wd:
        param_groups.append({"params": base_wd, "lr": base_lr, "weight_decay": wd})
    if base_nowd:
        param_groups.append({"params": base_nowd, "lr": base_lr, "weight_decay": 0.0})
    if cond_wd:
        param_groups.append(
            {"params": cond_wd, "lr": base_lr * cond_lr_mult, "weight_decay": wd}
        )
    if cond_nowd:
        param_groups.append(
            {"params": cond_nowd, "lr": base_lr * cond_lr_mult, "weight_decay": 0.0}
        )

    return param_groups, clip_params


def setup_mixed_precision(models, accelerator):
    if accelerator.mixed_precision == "fp16":
        if accelerator.is_main_process:
            print("Casting trainable parameters to float32 for numerical stability.")
        for model in models:
            if model is not None:
                cast_training_params(model, dtype=torch.float32)


def move_models_to_device(models, accelerator, weight_dtype):
    for model in models:
        if model is not None:
            model.to(accelerator.device, dtype=weight_dtype)


def omegaconf_select(cfg, key, default=None):
    if isinstance(cfg, DictConfig):
        OmegaConf.set_struct(cfg, False)

        if OmegaConf.select(cfg, key, default=None) is None:
            cfg[key] = default

        OmegaConf.set_struct(cfg, True)

    value = OmegaConf.select(cfg, key, default=default)
    return None if value == "None" else value


def loss_fn(loss_type: str):
    loss_functions = {
        "l1": F.l1_loss,
        "l2": F.mse_loss,
        "huber": F.smooth_l1_loss,
    }
    if loss_type not in loss_functions:
        raise ValueError(
            f"Unsupported loss type: {loss_type}. Supported types are: {list(loss_functions.keys())}"
        )
    return loss_functions[loss_type]


def check_Gaussianity(gauss: Tensor) -> None:
    print(f"🔬 Test complet de gaussianité pour tenseur de forme {tuple(gauss.shape)}")
    print("=" * 80)

    alpha = 0.05
    results = {}

    fig, axes = plt.subplots(2, len(gauss) + 1, figsize=(4 * (len(gauss) + 1), 8))
    if len(gauss) == 1:
        axes = axes.reshape(2, 2)

    if len(gauss) == 1:
        img_data = gauss[0].cpu().numpy()
        axes[0, 0].imshow(img_data, cmap="gray")
        axes[0, 0].set_title("Tenseur (Grayscale)")
        axes[0, 0].axis("off")
        axes[1, 0].axis("off")
    else:
        if len(gauss) == 3:
            img_data = gauss.permute(1, 2, 0).cpu().numpy()

            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            axes[0, 0].imshow(img_data)
            axes[0, 0].set_title("Tenseur (RGB)")
        else:
            img_data = gauss[0].cpu().numpy()
            axes[0, 0].imshow(img_data, cmap="gray")
            axes[0, 0].set_title(f"Tenseur (Canal 0/{len(gauss)})")

        axes[0, 0].axis("off")
        axes[1, 0].axis("off")

    for idx, channel in enumerate(gauss):
        data = channel.cpu().numpy().flatten()
        n_samples = len(data)

        print(f"\n📊 CANAL {idx} ({n_samples:,} échantillons)")
        print("-" * 50)

        mean, std = np.mean(data), np.std(data)
        skewness = stats.skew(data)
        kurt = stats.kurtosis(data)

        print(f"Moyenne: {mean:.4f} | Écart-type: {std:.4f}")
        print(f"Skewness: {skewness:.4f} | Kurtosis: {kurt:.4f}")

        channel_results = {}

        if n_samples <= 5000:
            stat, p = stats.shapiro(data)
            sample_info = f"(tous les {n_samples} échantillons)"
        else:
            sample_data = np.random.choice(data, 5000, replace=False)
            stat, p = stats.shapiro(sample_data)
            sample_info = "(échantillon de 5000)"

        channel_results["shapiro"] = (stat, p)
        print(f"Shapiro-Wilk {sample_info}: W={stat:.4f}, p={p:.2e} ", end="")
        print("✅" if p > alpha else "❌")

        stat, p = stats.normaltest(data)
        channel_results["dagostino"] = (stat, p)
        print(f"D'Agostino-Pearson: χ²={stat:.4f}, p={p:.2e} ", end="")
        print("✅" if p > alpha else "❌")

        stat, p = stats.kstest(data, lambda x: stats.norm.cdf(x, mean, std))
        channel_results["ks"] = (stat, p)
        print(f"Kolmogorov-Smirnov: D={stat:.4f}, p={p:.2e} ", end="")
        print("✅" if p > alpha else "❌")

        if n_samples >= 2000:
            stat, p = stats.jarque_bera(data)
            channel_results["jarque_bera"] = (stat, p)
            print(f"Jarque-Bera: JB={stat:.4f}, p={p:.2e} ", end="")
            print("✅" if p > alpha else "❌")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.anderson(data, dist="norm")

            critical_5pct = result.critical_values[2]
            channel_results["anderson"] = (result.statistic, critical_5pct)
            print(
                f"Anderson-Darling: A²={result.statistic:.4f} vs critique={critical_5pct:.4f} ",
                end="",
            )
            print("✅" if result.statistic < critical_5pct else "❌")

        plot_idx = idx + 1

        axes[0, plot_idx].hist(
            data,
            bins=100,
            density=True,
            alpha=0.7,
            color="skyblue",
            range=(mean - 4 * std, mean + 4 * std),
        )
        x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
        axes[0, plot_idx].plot(
            x,
            stats.norm.pdf(x, mean, std),
            "r-",
            lw=2,
            label=f"N({mean:.2f}, {std:.2f}²)",
        )
        axes[0, plot_idx].set_title(f"Canal {idx} - Distribution")
        axes[0, plot_idx].legend()
        axes[0, plot_idx].grid(True, alpha=0.3)

        stats.probplot(data, dist="norm", plot=axes[1, plot_idx])
        axes[1, plot_idx].set_title(f"Canal {idx} - Q-Q Plot")
        axes[1, plot_idx].grid(True, alpha=0.3)

        results[idx] = channel_results

    print("\n🎯 SYNTHÈSE GLOBALE")
    print("=" * 50)

    tests = ["shapiro", "dagostino", "ks", "jarque_bera", "anderson"]
    test_names = [
        "Shapiro-Wilk",
        "D'Agostino-Pearson",
        "Kolmogorov-Smirnov",
        "Jarque-Bera",
        "Anderson-Darling",
    ]

    for test, name in zip(tests, test_names):
        if test == "anderson":
            passed = sum(
                1
                for ch_results in results.values()
                if test in ch_results and ch_results[test][0] < ch_results[test][1]
            )
        else:
            passed = sum(
                1
                for ch_results in results.values()
                if test in ch_results and ch_results[test][1] > alpha
            )

        total = sum(1 for ch_results in results.values() if test in ch_results)
        if total > 0:
            print(f"{name}: {passed}/{total} canaux gaussiens ", end="")
            if passed == total:
                print("🎯")
            elif passed > total // 2:
                print("⚠️")
            else:
                print("❌")

    print("\n📈 CONCLUSION")
    print("-" * 30)

    consensus_scores = []
    for idx, ch_results in results.items():
        passed = 0
        total = 0
        for test in tests:
            if test in ch_results:
                total += 1
                if test == "anderson":
                    if ch_results[test][0] < ch_results[test][1]:
                        passed += 1
                else:
                    if ch_results[test][1] > alpha:
                        passed += 1
        if total > 0:
            score = passed / total
            consensus_scores.append(score)
            print(f"Canal {idx}: {passed}/{total} tests passés ({score:.1%})")

    if consensus_scores:
        avg_score = np.mean(consensus_scores)
        print(f"\nScore moyen de gaussianité: {avg_score:.1%}")

        if avg_score >= 0.8:
            print("🎯 Forte évidence de gaussianité")
        elif avg_score >= 0.6:
            print("✅ Modérée évidence de gaussianité")
        elif avg_score >= 0.4:
            print("⚠️ Faible évidence de gaussianité")
        else:
            print("❌ Évidence contre la gaussianité")

    plt.tight_layout()
    plt.show()

    return 0


def tensor_to_image(tensor):
    tensor = tensor.cpu()
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    mn, mx = tensor.min().item(), tensor.max().item()
    choices = (
        (mn < -0.1 and mx > 0.1, (tensor + 1) / 2),
        (mx <= 1.1, tensor.clamp(0, 1)),
    )
    normalized = next(
        (result for cond, result in choices if cond), (tensor - mn) / (mx - mn + 1e-8)
    )
    return normalized.to(dtype=torch.float32)


def resize_image_tensor(image, target_size):
    if len(image.shape) == 3:
        return torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return torch.nn.functional.interpolate(
        image, size=target_size, mode="bilinear", align_corners=False
    )


def create_text_images(texts, size=(256, 256)):
    images = []
    for text in texts:
        img = Image.new("RGB", size, color=(0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        pos = ((size[0] - w) // 2, (size[1] - h) // 2)
        draw.text(pos, text, fill=(255, 255, 255), font=font)
        arr = np.array(img).transpose(2, 0, 1) / 127.5 - 1

        images.append(torch.from_numpy(arr).float())
    return images


def format_metric_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.item()
        else:
            return "tensor"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (np.floating, float)):
        return f"{float(value):.4f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)
