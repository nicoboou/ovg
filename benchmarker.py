from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from datetime import datetime
from diffusers import StableDiffusionPipeline
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import hydra
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Subset

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from ovg.utils.helpers import (
    get_model_dtype,
    normalize_tensor,
    resize_image_tensor,
    create_text_images,
    format_metric_value,
    unwrap_model,
)
from ovg.utils.logger import RunLogger
from ovg.utils.metrics_binned import (
    eval_binned_metrics,
    metrics_to_dict,
    quickplot_binned,
    render_binned_plots,
)
from ovg.data.datasets import BBBC021Dataset, Edges2ShoesDataset
from ovg.utils.helpers import get_models
from ovg.utils.checkpoint import CheckpointManager
from ovg.utils.metrics import MetricsCalculator, compute_knn_score
from ovg.schedulers.edm_scheduler import EDMSchedulerAdapter
from ovg.inversion_methods import (
    StandardInversion,
    ControlledGaussianizationInversion,
)


class SpectralAnalyzer:
    def __init__(self, device):
        self.device = device

    def compute_fft_magnitude(self, tensor: torch.Tensor):
        if tensor.shape[1] > 1:
            tensor = tensor.mean(dim=1, keepdim=True)
        fft = torch.fft.fft2(tensor.float())
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        return torch.abs(fft_shifted)

    def compute_radial_psd(self, tensor: torch.Tensor):
        magnitude = self.compute_fft_magnitude(tensor)

        b, c, h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).to(self.device)

        r_flat = r.reshape(-1).cpu().numpy().astype(int)
        mag_flat = magnitude.mean(dim=(0, 1)).reshape(-1).cpu().numpy()

        max_r = min(cx, cy)
        tbin = np.bincount(r_flat, weights=mag_flat)
        nr = np.bincount(r_flat)
        radial_profile = tbin[:max_r] / np.maximum(nr[:max_r], 1)

        return np.arange(max_r), np.log10(radial_profile + 1e-12)

    def plot_psd_comparison(self, data_dict: Dict[str, tuple], title: str):
        fig, ax = plt.subplots(figsize=(8, 6))
        for name, (radii, psd) in data_dict.items():
            ax.plot(radii, psd, label=name, linewidth=2, alpha=0.8)

        ax.set_xlabel("Spatial Frequency (Radius)")
        ax.set_ylabel("Log Magnitude (Energy)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig

    def plot_2d_spectrum_visual(self, tensors_dict: Dict[str, torch.Tensor], title: str):
        num_plots = len(tensors_dict)
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]

        for ax, (name, tensor) in zip(axes, tensors_dict.items()):
            mag = self.compute_fft_magnitude(tensor).mean(dim=(0, 1)).cpu()
            log_mag = torch.log(mag + 1e-8).numpy()

            im = ax.imshow(log_mag, cmap="inferno")
            ax.set_title(name)
            ax.axis("off")

        plt.suptitle(title)
        return fig

    def create_checkerboard_noise(self, shape, square_size=1):
        noise = torch.zeros(shape, device=self.device)
        for i in range(shape[2]):
            for j in range(shape[3]):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    noise[:, :, i, j] = 1.0
                else:
                    noise[:, :, i, j] = -1.0
        return noise


class HDF5Exporter:
    def __init__(
        self,
        output_dir: Path,
        config_name: str,
        method_name: str,
        seed: int,
        num_steps: int,
        save_intermediate_latents: bool = True,
        save_predicted_noises: bool = True,
    ):
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 export. Install with: pip install h5py")

        self.output_dir = output_dir
        self.config_name = config_name
        self.method_name = method_name
        self.seed = seed
        self.num_steps = num_steps
        self.save_intermediate_latents = save_intermediate_latents
        self.save_predicted_noises = save_predicted_noises

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.h5_path = self.output_dir / f"{config_name}_seed{seed}_{method_name}_steps{num_steps}.h5"
        self.h5_file: Optional[h5py.File] = None
        self.sample_counter = 0

    def open(
        self,
        config: Dict[str, Any],
        method_config: Optional[Dict[str, Any]] = None,
    ):
        self.h5_file = h5py.File(self.h5_path, "w")

        config_grp = self.h5_file.create_group("config")
        for key, value in config.items():
            try:
                if isinstance(value, (str, int, float, bool)):
                    config_grp.attrs[key] = value
                else:
                    config_grp.attrs[key] = str(value)
            except Exception:
                config_grp.attrs[key] = str(value)

        if method_config is not None:
            method_grp = config_grp.create_group("method_params")
            for key, value in method_config.items():
                try:
                    if isinstance(value, (str, int, float, bool)):
                        method_grp.attrs[key] = value
                    else:
                        method_grp.attrs[key] = str(value)
                except Exception:
                    method_grp.attrs[key] = str(value)

    def save_metrics(self, metrics: Dict[str, Any]):
        if self.h5_file is None:
            raise RuntimeError("HDF5 file not opened. Call open() first.")

        import json

        metrics_grp = self.h5_file.create_group("metrics")

        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool, str)):
                serializable_metrics[key] = value
            elif isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            elif hasattr(value, "item"):
                serializable_metrics[key] = value.item()
            else:
                serializable_metrics[key] = str(value)

        metrics_json = json.dumps(serializable_metrics, indent=2)
        metrics_grp.attrs["json"] = metrics_json

        for key, value in serializable_metrics.items():
            try:
                if isinstance(value, (int, float, bool, str)):
                    metrics_grp.attrs[key] = value
                elif isinstance(value, list):
                    metrics_grp.create_dataset(key, data=np.array(value))
                else:
                    metrics_grp.attrs[key] = str(value)
            except Exception:
                metrics_grp.attrs[key] = str(value)

    def save_timing(
        self,
        inversion_time_ns: int,
        sampling_time_ns: int,
        num_samples: int,
    ):
        if self.h5_file is None:
            raise RuntimeError("HDF5 file not opened. Call open() first.")

        timing_grp = self.h5_file.create_group("timing")

        inversion_time_s = inversion_time_ns / 1e9
        sampling_time_s = sampling_time_ns / 1e9
        total_time_s = inversion_time_s + sampling_time_s

        timing_grp.attrs["avg_inversion_time_s"] = inversion_time_s / max(num_samples, 1)
        timing_grp.attrs["avg_sampling_time_s"] = sampling_time_s / max(num_samples, 1)
        timing_grp.attrs["avg_total_time_s"] = total_time_s / max(num_samples, 1)

        timing_grp.attrs["num_samples"] = num_samples

    def close(self):
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None

    def save_batch(
        self,
        intermediate_latents: List[torch.Tensor],
        predicted_noises: List[torch.Tensor],
        timesteps: List[float],
        generated_images: torch.Tensor,
        original_images: torch.Tensor,
        input_images: torch.Tensor,
        sample_indices: List[int],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if self.h5_file is None:
            raise RuntimeError("HDF5 file not opened. Call open() first.")

        batch_size = generated_images.shape[0]

        for b in range(batch_size):
            sample_idx = sample_indices[b] if sample_indices else self.sample_counter
            sample_grp = self.h5_file.create_group(f"sample_{sample_idx:04d}")

            if intermediate_latents:
                if self.save_intermediate_latents:
                    latents_stacked = torch.stack(
                        [lat[b].detach().cpu() for lat in intermediate_latents], dim=0
                    ).numpy()
                else:
                    latents_stacked = intermediate_latents[-1][b].detach().cpu().unsqueeze(0).numpy()
                sample_grp.create_dataset(
                    "latents",
                    data=latents_stacked,
                    compression="gzip",
                    compression_opts=4,
                )

            if self.save_predicted_noises and predicted_noises:
                noise_stacked = torch.stack([noise[b].detach().cpu() for noise in predicted_noises], dim=0).numpy()
                sample_grp.create_dataset(
                    "predicted_noise",
                    data=noise_stacked,
                    compression="gzip",
                    compression_opts=4,
                )

            if timesteps:
                sample_grp.create_dataset("timesteps", data=np.array(timesteps))

            sample_grp.create_dataset(
                "generated_hr",
                data=generated_images[b].detach().cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )

            sample_grp.create_dataset(
                "original_hr",
                data=original_images[b].detach().cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )

            sample_grp.create_dataset(
                "input_lr",
                data=input_images[b].detach().cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )

            if metadata:
                meta_grp = sample_grp.create_group("metadata")
                for key, value in metadata.items():
                    if isinstance(value, (list, torch.Tensor)):
                        val = value[b] if hasattr(value, "__getitem__") and len(value) > b else value
                        if isinstance(val, torch.Tensor):
                            val = val.item() if val.numel() == 1 else val.tolist()
                    else:
                        val = value
                    try:
                        meta_grp.attrs[key] = val if isinstance(val, (str, int, float, bool)) else str(val)
                    except Exception:
                        meta_grp.attrs[key] = str(val)

            self.sample_counter += 1


class BenchmarkRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seed = cfg.get("seed", 42)
        self.accelerator = None
        self.device = None
        self.denoiser = None
        self.vae = None
        self.scheduler = None
        self.tokenizer = None
        self.text_encoder = None
        self.class_embedding = None
        self.dataset = None
        self.methods: List = []
        self.method_params = {}
        self.metrics = {}

        self.num_samples = cfg.benchmark.get("num_samples", 6)
        self.vis_samples = min(cfg.benchmark.get("vis_samples", 6), self.num_samples)
        self.num_inference_steps = cfg.inference.get("num_inference_steps", 50)
        self.batch_size = cfg.benchmark.get("batch_size", 2)
        self.task_mode = cfg.data.get("mode", "super_resolution")

        self.compute_distribution_metrics = self.task_mode in (
            "super_resolution",
            "translation",
            "segmentation",
        )
        self.image_prompts = None
        self._inception = None

        self.num_workers = cfg.get("num_workers", 4)
        self.use_amp = cfg.benchmark.get("use_amp", True)
        self.is_text_to_image = cfg.model_checkpoint.get("is_text_to_image", False)
        self.aggressive_cleanup = cfg.benchmark.get("aggressive_cleanup", False)

        self.export_intermediates = cfg.benchmark.get("export_intermediates", False)
        self.export_output_dir = Path(cfg.benchmark.get("export_output_dir", "results/benchmark_export"))
        self.export_save_intermediate_latents = cfg.benchmark.get("export_save_intermediate_latents", True)
        self.export_save_predicted_noises = cfg.benchmark.get("export_save_predicted_noises", True)

        self.edm_checkpoint_path = cfg.model_checkpoint.get("edm_checkpoint_path", None)

        self._setup_accelerator()
        self.logger = get_logger(__name__, log_level="INFO")
        self.device = self.accelerator.device

    def setup(self):
        self._load_models()
        self._setup_dataset()
        self.methods = self._setup_inversion_methods()
        self._setup_metrics()
        self._setup_method_params()

    def _setup_accelerator(self):
        logging_dir = Path(self.cfg.output_dir) / "logs"
        project_config = ProjectConfiguration(project_dir=self.cfg.output_dir, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            log_with=self.cfg.logging.report_to,
            project_config=project_config,
            mixed_precision=self.cfg.benchmark.get("mixed_precision", "fp16"),
        )

        if self.accelerator.is_main_process:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            config_name = hydra_cfg.job.config_name or "default"
            config_name = config_name[:-5] if config_name.endswith(".yaml") else config_name
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{config_name}_{timestamp}"

            self.cfg.logging.run_name = run_name

            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            self.accelerator.init_trackers(
                self.cfg.logging.project_name,
                config=cfg_dict,
                init_kwargs={"wandb": {"name": run_name}},
            )

        output_dir = Path(self.cfg.output_dir)
        self.run_logger = RunLogger(
            self.cfg,
            checkpoint_dir=output_dir,
            accelerator=self.accelerator,
            resume_step=0,
            wandb_root_dir=output_dir,
        )

    def _setup_dataset(self):
        name = self.cfg.data.dataset
        val_tf = hydra.utils.instantiate(self.cfg.transforms.validation)

        if name == "bbbc021":
            self.dataset = BBBC021Dataset(
                csv_path=self.cfg.data.csv_path,
                transforms=val_tf,
                hr_size=self.cfg.data.hr_size,
                scale_factor=self.cfg.data.scale_factor,
                augmentation=False,
                paired_mode=self.cfg.data.paired_mode,
                deterministic_downsample=self.cfg.data.deterministic_downsample,
                added_noise_post_upsampling=self.cfg.data.added_noise_post_upsampling,
                mode=self.cfg.data.mode,
                class_indices=self.cfg.data.class_indices,
                blur_kernel_size=self.cfg.data.blur_kernel_size,
                blur_sigma=self.cfg.data.blur_sigma,
                noise_std=self.cfg.data.noise_std,
                interpolation_mode=self.cfg.data.interpolation_mode,
                use_blur=self.cfg.data.use_blur,
                use_downsample=self.cfg.data.use_downsample,
            )
        elif name == "edges2shoes":
            self.dataset = Edges2ShoesDataset(
                csv_path=self.cfg.data.csv_path,
                root_dir=getattr(self.cfg.data, "root_dir", None),
                mode=self.cfg.data.mode,
                img_size=getattr(self.cfg.data, "img_size", getattr(self.cfg.data, "hr_size", 256)),
                transforms=True,
                augmentation=False,
                paired_mode=self.cfg.data.paired_mode,
                custom_transform=val_tf,
                class_indices=self.cfg.data.class_indices,
                normalize=getattr(self.cfg.data, "normalize", True),
                split=getattr(self.cfg.data, "split", None),
            )
        else:
            raise ValueError(f"Unsupported dataset '{name}'. Only 'bbbc021' and 'edges2shoes' are supported.")

    def _setup_metrics(self):
        self.metric_fns = []
        fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
        self.inceptionv3 = fid_metric.inception.eval()

        enable_clip = True
        enable_lpips = True
        enable_structure_distance = True
        self.metrics_calc = MetricsCalculator(
            device=self.device,
            enable_clip=enable_clip,
            enable_lpips=enable_lpips,
            enable_structure_distance=enable_structure_distance,
        )

        self.metrics = {
            "psnr (↑)": self.metrics_calc.psnr_metric,
            "haarpsi (↑)": self.metrics_calc.haarpsi_metric,
            "msssim (↑)": self.metrics_calc.msssim_metric,
        }

        if self.metrics_calc.lpips_metric is not None:
            self.metrics["lpips (↓)"] = self.metrics_calc.lpips_metric

        if self.metrics_calc.structure_distance_metric is not None:
            self.metrics["structure_distance (↓)"] = self.metrics_calc.structure_distance_metric

        if self.compute_distribution_metrics:
            self.metrics.update(
                {
                    "knn_recovery (↑)": 0.0,
                    "knn_sr": 0.0,
                    "knn_hr": 0.0,
                    "knn_lr": 0.0,
                    "fid (↓)": fid_metric,
                }
            )

        self.logger.info(f"Configured metrics: {list(self.metrics.keys())}")

    def _setup_method_params(self):
        self.method_params = {}
        if hasattr(self.cfg.benchmark, "methods"):
            for i, m in enumerate(self.cfg.benchmark.methods):
                if m.get("enabled", True):
                    cfg_id = f"config_{i}"
                    name = m.get("name")
                    if name:
                        self.method_params[f"{name}_{cfg_id}"] = {
                            "params": m.get("params", {}),
                            "description": m.get("description", ""),
                        }

    def _setup_method_params(self):
        self.method_params = {}
        if hasattr(self.cfg.benchmark, "methods"):
            for i, m in enumerate(self.cfg.benchmark.methods):
                if m.get("enabled", True):
                    cfg_id = f"config_{i}"
                    name = m.get("name")
                    if name:
                        self.method_params[f"{name}_{cfg_id}"] = {
                            "params": m.get("params", {}),
                            "description": m.get("description", ""),
                        }

    def _setup_inversion_methods(self):
        methods = []
        for i, m in enumerate(self.cfg.benchmark.methods):
            if not m.get("enabled", True):
                continue
            name = m.get("name")
            cfg_id = f"config_{i}"
            if name == "standard_inversion":
                meth = StandardInversion()
                meth.name = f"{meth.name}_{cfg_id}"
                methods.append(meth)
            elif name == "ours":
                params = m.get("params", {})
                meth = ControlledGaussianizationInversion(
                    use_cgd=params.get("use_cgd", True),
                    use_scp=params.get("use_scp", True),
                    cgd_eta=params.get("cgd_eta", 200.0),
                    scp_eta=params.get("scp_eta", 0.0002),
                    forward_fraction=params.get("forward_fraction", 0.00),
                    min_forward_steps=params.get("min_forward_steps", 0),
                )
                meth.name = f"{meth.name}_{cfg_id}"
                methods.append(meth)
        return methods

    def _load_models(self):
        path = self.cfg.model_checkpoint.checkpoint_path
        rescale = self.cfg.model_checkpoint.get("rescale_betas_zero_snr", False)
        is_t2i = self.cfg.model_checkpoint.get("is_text_to_image", False)

        model_loader = {
            True: self._load_diffusers_models,
            False: self._load_custom_models,
        }
        model_loader[is_t2i](path, rescale)

        self.dtype = get_model_dtype(self.denoiser)
        if self.denoiser is not None:
            self.denoiser = self.denoiser.to(device=self.device, dtype=self.dtype)

            unwrapped = unwrap_model(self.denoiser)
            model_class_name = unwrapped.__class__.__name__
            self.is_dit = "DiT" in model_class_name or "Transformer" in model_class_name
        else:
            self.is_dit = False
        if self.vae is not None:
            self.vae = self.vae.to(device=self.device, dtype=self.dtype)
        if self.class_embedding is not None:
            self.class_embedding = self.class_embedding.to(device=self.device, dtype=self.dtype)

    def _load_diffusers_models(self, path, rescale):
        precision = self.cfg.benchmark.get("mixed_precision", "fp16")
        torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        pipeline = StableDiffusionPipeline.from_pretrained(path, torch_dtype=torch_dtype, safety_checker=None)
        pipeline = pipeline.to(self.device)
        self.vae = pipeline.vae
        self.denoiser = pipeline.unet
        self.scheduler = pipeline.scheduler
        self.class_embedding = None
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder.to(self.device, dtype=torch.float32)

    def _load_custom_models(self, path, rescale):
        raw_checkpoint = torch.load(path, map_location="cpu")
        if "config" not in raw_checkpoint:
            self.logger.warning("Ancient checkpoint format detected. Loading using legacy loader.")
            model_cfg = self.cfg
        else:
            model_cfg = raw_checkpoint["config"]

        model_cfg.model.use_lora = self.cfg.model_checkpoint.get("use_lora", model_cfg.model.use_lora)

        self.noise_scheduler, self.vae, self.denoiser, self.class_embedding = get_models(model_cfg)

        manager = CheckpointManager()
        self.denoiser, self.vae, self.class_embedding, self.scheduler, _ = manager.load_model_weights_from_checkpoint(
            checkpoint_path=path,
            cfg=model_cfg,
            denoiser=self.denoiser,
            vae=self.vae,
            class_embedding=self.class_embedding,
            scheduler=self.noise_scheduler,
        )

        self.model_training_mode = model_cfg.training.get("training_mode", "diffusion").lower()
        if self.model_training_mode == "edm":
            self.scheduler = EDMSchedulerAdapter(model_cfg, device=self.device)
        else:
            if self.scheduler is None:
                raise ValueError("Échec du chargement d'un scheduler DDIM pour un modèle de diffusion.")
            self.logger.info(f"Utilisation du scheduler {self.scheduler.__class__.__name__} du checkpoint.")

        self.tokenizer = None
        self.text_encoder = None

    def _unload_models(self):
        if self.denoiser is not None:
            del self.denoiser
        if self.vae is not None:
            del self.vae
        if self.class_embedding is not None:
            del self.class_embedding
        if self.scheduler is not None:
            del self.scheduler
        self.denoiser = None
        self.vae = None
        self.class_embedding = None
        self.scheduler = None
        torch.cuda.empty_cache()

    def _prepare_editing_samples(self):
        image_paths = list(self.cfg.data.get("image_paths", []))
        image_prompts = self.cfg.data.get("image_prompts", [])

        if len(image_paths) != len(image_prompts):
            raise ValueError(f"Mismatch: {len(image_paths)} images but {len(image_prompts)} prompt pairs")

        transform = hydra.utils.instantiate(self.cfg.transforms.validation)

        images = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            tensor_img = transform(img)
            images.append(tensor_img)
        original = torch.stack(images, dim=0)

        source_prompts = []
        target_prompts = []
        for prompt_pair in image_prompts:
            source_prompts.append(prompt_pair["source"])
            target_prompts.append(prompt_pair["target"])

        self.image_prompts = list(zip(source_prompts, target_prompts))
        labels = torch.arange(len(image_paths), dtype=torch.long)

        return {
            "original_images": original,
            "downsampled": original.clone(),
            "latents": None,
            "knn_labels": labels,
            "source_prompts": source_prompts,
            "target_prompts": target_prompts,
        }

    def _prepare_samples(self):
        if self.task_mode == "t2i_editing":
            return self._prepare_editing_samples()

        available = len(self.dataset)
        if self.num_samples > available:
            self.num_samples = available

        all_hr = []
        all_lr = []
        all_classes = []

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        indices = torch.randperm(available, generator=gen)[: self.num_samples].tolist()
        subset = Subset(self.dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        for batch in loader:
            all_hr.append(batch["hr"])
            all_lr.append(batch["lr"])
            all_classes.append(batch["class_idx"])

        images = torch.cat(all_hr, dim=0)
        downsampled = torch.cat(all_lr, dim=0)
        knn_labels = torch.cat(all_classes, dim=0).cpu()

        self.logger.info(f"knn_labels unique values: {knn_labels.unique().tolist()}")
        self.image_prompts = None

        source_class = torch.zeros_like(knn_labels)
        target_class = torch.ones_like(knn_labels)

        return {
            "original_images": images,
            "downsampled": downsampled,
            "latents": None,
            "knn_labels": knn_labels,
            "source_class": source_class,
            "target_class": target_class,
        }

    @torch.no_grad()
    def _extract_inception_features(self, images: torch.Tensor) -> np.ndarray:
        if images.dim() == 4 and images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        imgs = (images.clamp(0, 1) * 255).to(torch.uint8)
        batch_size = 32
        feats = []
        for i in range(0, imgs.shape[0], batch_size):
            batch = imgs[i : i + batch_size].to(self.device, non_blocking=True)
            f = self.inceptionv3(batch)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _get_embeddings(self, editing_mode: bool, source_data, target_data):
        if editing_mode:
            cond_emb_source, _ = self.encode_embeddings(source_data)
            target_condition_emb, target_uncond = self.encode_embeddings(target_data)
            return cond_emb_source, target_condition_emb, target_uncond
        else:
            source_data_device = source_data.to(self.device, non_blocking=True)
            target_data_device = target_data.to(self.device, non_blocking=True)

            cond_emb_source, _ = self.encode_embeddings(source_data_device)
            target_condition_emb, _ = self.encode_embeddings(target_data_device)
            return cond_emb_source, target_condition_emb, None

    @torch.no_grad()
    def encode_embeddings(self, condition):
        if self.is_text_to_image:
            if isinstance(condition, str):
                condition = [condition]

            if self.tokenizer is None or self.text_encoder is None:
                raise ValueError("Tokenizer and text_encoder must be provided for text-to-image models")

            text_inputs = self.tokenizer(
                condition,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            denoiser_device = next(self.denoiser.parameters()).device
            text_input_ids = text_inputs.input_ids.to(denoiser_device)
            cond_emb = self.text_encoder(text_input_ids)[0]

            uncond_text = [""] * len(condition)
            uncond_inputs = self.tokenizer(
                uncond_text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_input_ids = uncond_inputs.input_ids.to(denoiser_device)
            uncond_emb = self.text_encoder(uncond_input_ids)[0]

            target_dtype = get_model_dtype(self.denoiser)
            cond_emb = cond_emb.to(device=denoiser_device, dtype=target_dtype)
            uncond_emb = uncond_emb.to(device=denoiser_device, dtype=target_dtype)

            return cond_emb, uncond_emb

        else:
            if getattr(self, "is_dit", False):
                denoiser_device = next(self.denoiser.parameters()).device

                if torch.is_tensor(condition):
                    cond_labels = condition.to(device=denoiser_device, dtype=torch.long)
                else:
                    cond_labels = torch.tensor(condition, device=denoiser_device, dtype=torch.long)

                unwrapped = unwrap_model(self.denoiser)
                config = getattr(unwrapped, "config", None)
                num_classes = getattr(config, "num_classes", None)
                if num_classes is None and self.class_embedding is not None:
                    num_classes = getattr(self.class_embedding, "num_classes", 1000)
                if num_classes is None:
                    num_classes = 1000
                uncond_labels = torch.full_like(cond_labels, fill_value=num_classes)
                return cond_labels, uncond_labels

            if self.class_embedding is None:
                return None, None

            cond_emb = self.class_embedding(condition)
            uncond_labels = torch.full_like(condition, self.class_embedding.num_classes)
            uncond_emb = self.class_embedding(uncond_labels)

            target_dtype = get_model_dtype(self.denoiser)
            denoiser_device = next(self.denoiser.parameters()).device

            if cond_emb.dim() == 2:
                cond_emb = cond_emb.unsqueeze(1)
            if uncond_emb.dim() == 2:
                uncond_emb = uncond_emb.unsqueeze(1)
            cond_emb = cond_emb.to(device=denoiser_device, dtype=target_dtype)
            uncond_emb = uncond_emb.to(device=denoiser_device, dtype=target_dtype)
            return cond_emb, uncond_emb

    def _encode_latents(self, batch_input: torch.Tensor) -> torch.Tensor:
        has_vae = self.vae is not None
        encoder = {
            True: lambda: self.vae.encode(
                batch_input.to(device=self.device, dtype=self.dtype, non_blocking=True)
            ).latent_dist.sample()
            * self.vae.config.scaling_factor,
            False: lambda: batch_input.to(device=self.device, dtype=self.dtype, non_blocking=True),
        }
        return encoder[has_vae]()

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        has_vae = self.vae is not None
        decoder = {
            True: lambda: self.vae.decode(latents / self.vae.config.scaling_factor).sample,
            False: lambda: latents,
        }
        return decoder[has_vae]()

    def _compute_knn_recovery(
        self,
        feats_sr_full: np.ndarray,
        knn_labels: torch.Tensor,
        knn_lr: float | None,
        knn_hr: float | None,
    ) -> Dict[str, float | bool]:
        if self.compute_distribution_metrics:
            return self.metrics_calc.compute_knn_recovery(
                feats_sr=feats_sr_full,
                labels=knn_labels,
                knn_lr=knn_lr,
                knn_hr=knn_hr,
            )
        return {}

    def log_results_wandb(self, results: Dict):
        report_to = getattr(self.cfg, "logging", {}).get("report_to", None)
        if isinstance(report_to, str):
            use_wandb_direct = report_to.lower() == "wandb"
        else:
            use_wandb_direct = bool(report_to) and "wandb" in report_to

        if use_wandb_direct:
            import wandb

        method_results = results.get("method_results", {})
        available_methods = [m for m in self.methods if m.name in method_results]

        num = results["original_images"].shape[0]
        vis = min(self.vis_samples, num)
        cols = 2 + len(available_methods)
        headers = ["Original HR", "Downsampled LR"] + [m.name for m in available_methods]

        header_imgs = create_text_images(headers)
        th, tw = results["original_images"][0].shape[1:]
        resized_headers = [resize_image_tensor(h, (th, tw)) for h in header_imgs]

        resized_headers = [normalize_tensor(img.cpu()) for img in resized_headers]

        all_imgs = []

        for img in resized_headers:
            all_imgs.append(img)

        for i in range(vis):
            orig_img = normalize_tensor(results["original_images"][i].cpu())
            down_img = normalize_tensor(results["downsampled"][i].cpu())

            all_imgs.append(orig_img)
            all_imgs.append(down_img)

            for m in available_methods:
                gen = method_results[m.name]["generated_images"][i].cpu()
                all_imgs.append(gen.clamp(0, 1))

        grid = torchvision.utils.make_grid(torch.stack(all_imgs), nrow=cols, padding=2)

        grid_uint8 = (grid.clamp(0, 1) * 255).to(torch.uint8)
        if grid_uint8.dim() == 3 and grid_uint8.size(0) in (1, 3):
            grid_uint8 = grid_uint8.permute(1, 2, 0).contiguous()

        self.run_logger.log_image(
            key="inversion_benchmark/results_grid",
            image=grid_uint8.cpu().numpy(),
            step=0,
        )

        try:

            def _to_hwc(img: torch.Tensor | np.ndarray) -> np.ndarray:
                if torch.is_tensor(img):
                    img = img.detach().cpu().numpy()
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = np.transpose(img, (1, 2, 0))
                return img

            output_dir = Path(self.cfg.output_dir)
            if not output_dir.is_absolute():
                output_dir = Path(hydra.utils.get_original_cwd()) / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            run_name = getattr(getattr(self.cfg, "logging", None), "run_name", None)
            if not run_name:
                run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            svg_path = output_dir / f"results_grid_{run_name}.svg"

            rows = vis + 1
            dpi = 300

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5), dpi=dpi)

            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)

            for j, header_img in enumerate(resized_headers):
                axes[0, j].imshow(_to_hwc(header_img))
                axes[0, j].axis("off")

            for i in range(vis):
                axes[i + 1, 0].imshow(_to_hwc(normalize_tensor(results["original_images"][i].cpu())))
                axes[i + 1, 0].axis("off")

                axes[i + 1, 1].imshow(_to_hwc(normalize_tensor(results["downsampled"][i].cpu())))
                axes[i + 1, 1].axis("off")

                for j, m in enumerate(available_methods):
                    gen = method_results[m.name]["generated_images"][i].cpu().clamp(0, 1)
                    axes[i + 1, 2 + j].imshow(_to_hwc(gen))
                    axes[i + 1, 2 + j].axis("off")

            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            fig.savefig(svg_path, format="svg", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            self.logger.info(f"Saved results grid SVG to: {svg_path}")

        except Exception as e:
            self.logger.warning(f"Failed to export SVG: {e}")

        if use_wandb_direct:
            wandb.log({"inversion_benchmark/results_grid": wandb.Image(grid_uint8.cpu().numpy())})

        metrics_data = {"Method": [m.name for m in available_methods]}

        ordered_metric_keys = list(self.metrics.keys())
        extra_metric_keys: set[str] = set()
        for method in available_methods:
            extra_metric_keys.update(method_results[method.name]["metrics"].keys())
        for mk in ordered_metric_keys:
            extra_metric_keys.discard(mk)
        ordered_metric_keys += sorted(extra_metric_keys)

        for key in ordered_metric_keys:
            metrics_data[key] = []
            for method in available_methods:
                raw_val = method_results[method.name]["metrics"].get(key)
                metrics_data[key].append(format_metric_value(raw_val))

        for i, method in enumerate(available_methods):
            metric_dict = {f"benchmark/{key}": metrics_data[key][i] for key in ordered_metric_keys}
            metric_dict["method"] = method.name
            self.run_logger.log(metric_dict, step=i)

        try:
            import matplotlib.pyplot as _plt

            live_base_step = max(1, len(available_methods))

            for m in available_methods:
                m_idx = available_methods.index(m)
                step_for_method = live_base_step + m_idx
                for phase, key, prefix in (
                    ("inversion", "live_binned_inversion", "live/inversion"),
                    ("sampling", "live_binned_sampling", "live/sampling"),
                ):
                    live = method_results.get(m.name, {}).get(key)
                    if not live:
                        continue

                    errs = live.get("errors", [])
                    if errs:
                        try:
                            self.logger.warning(
                                f"[live-metrics] {phase} errors for {m.name}: count={len(errs)}; first='"
                                + errs[0]
                                + "'"
                            )
                        except Exception:
                            pass
                    t = live.get("t_norm", [])

                    fig1 = _plt.figure()
                    ax1 = fig1.add_subplot(1, 1, 1)
                    ax1.plot(t, live.get("mse_x0", []), label="MSE(x0)")
                    ax1.set_title(f"Live {phase.capitalize()}: MSE(x0) — {m.name}")
                    ax1.set_xlabel("t/T")
                    ax1.set_ylabel("MSE")
                    ax1.grid(True, alpha=0.3)
                    self.run_logger.log_image(f"{prefix}/{m.name}/mse_x0", fig1, step=step_for_method, epoch=0)
                    _plt.close(fig1)

                    fig2 = _plt.figure()
                    ax2 = fig2.add_subplot(1, 1, 1)
                    ax2.plot(t, live.get("eps2_mean", []), label="mean ||eps||^2")
                    ax2.set_title(f"Live {phase.capitalize()}: mean ||eps||^2 — {m.name}")
                    ax2.set_xlabel("t/T")
                    ax2.set_ylabel("mean")
                    ax2.grid(True, alpha=0.3)
                    self.run_logger.log_image(
                        f"{prefix}/{m.name}/eps2_mean",
                        fig2,
                        step=step_for_method,
                        epoch=0,
                    )
                    _plt.close(fig2)

                    em = live.get("eps_mse", [])
                    if em:
                        fig2b = _plt.figure()
                        ax2b = fig2b.add_subplot(1, 1, 1)
                        ax2b.plot(t, em, label="eps MSE (whiteness)")
                        ax2b.set_title(f"Live {phase.capitalize()}: eps MSE — {m.name}")
                        ax2b.set_xlabel("t/T")
                        ax2b.set_ylabel("mse")
                        ax2b.grid(True, alpha=0.3)
                        self.run_logger.log_image(
                            f"{prefix}/{m.name}/eps_mse",
                            fig2b,
                            step=step_for_method,
                            epoch=0,
                        )
                        _plt.close(fig2b)

                    ci = live.get("cond_influence", [])
                    if ci:
                        fig3 = _plt.figure()
                        ax3 = fig3.add_subplot(1, 1, 1)
                        ax3.plot(t, ci, label="cond influence")
                        ax3.set_title(f"Live {phase.capitalize()}: cond influence — {m.name}")
                        ax3.set_xlabel("t/T")
                        ax3.set_ylabel("ratio")
                        ax3.grid(True, alpha=0.3)
                        self.run_logger.log_image(
                            f"{prefix}/{m.name}/cond_influence",
                            fig3,
                            step=step_for_method,
                            epoch=0,
                        )
                        _plt.close(fig3)

                    cf = live.get("class_flip", [])
                    if cf:
                        figc = _plt.figure()
                        axc = figc.add_subplot(1, 1, 1)
                        axc.plot(t, cf, label="class flip delta")
                        axc.set_title(f"Live {phase.capitalize()}: class flip — {m.name}")
                        axc.set_xlabel("t/T")
                        axc.set_ylabel("delta")
                        axc.grid(True, alpha=0.3)
                        self.run_logger.log_image(
                            f"{prefix}/{m.name}/class_flip",
                            figc,
                            step=step_for_method,
                            epoch=0,
                        )
                        _plt.close(figc)

                    varn = live.get("eps2_var", [])
                    if varn:
                        fig4 = _plt.figure()
                        ax4 = fig4.add_subplot(1, 1, 1)
                        ax4.plot(t, varn, label="Var(||eps||^2)")
                        ax4.set_title(f"Live {phase.capitalize()}: Var(||eps||^2) — {m.name}")
                        ax4.set_xlabel("t/T")
                        ax4.set_ylabel("variance")
                        ax4.grid(True, alpha=0.3)
                        self.run_logger.log_image(
                            f"{prefix}/{m.name}/eps2_var",
                            fig4,
                            step=step_for_method,
                            epoch=0,
                        )
                        _plt.close(fig4)

                    edges = live.get("eps2_hist_edges", None)
                    counts = live.get("eps2_hist_counts", None)
                    if edges and counts:
                        import numpy as _np

                        edges_np = _np.array(edges)
                        counts_np = _np.array(counts)
                        centers = 0.5 * (edges_np[:-1] + edges_np[1:])
                        fig5 = _plt.figure()
                        ax5 = fig5.add_subplot(1, 1, 1)
                        ax5.bar(
                            centers,
                            counts_np,
                            width=(edges_np[1:] - edges_np[:-1]),
                            align="center",
                        )
                        ax5.set_title(f"Live {phase.capitalize()}: hist ||eps||^2 @ top step — {m.name}")
                        ax5.set_xlabel("||eps||^2")
                        ax5.set_ylabel("count")
                        self.run_logger.log_image(
                            f"{prefix}/{m.name}/eps2_hist_top",
                            fig5,
                            step=step_for_method,
                            epoch=0,
                        )
                        _plt.close(fig5)
        except Exception:
            pass

        if use_wandb_direct:
            table = wandb.Table(
                data=[
                    [metrics_data["Method"][i]] + [metrics_data[k][i] for k in ordered_metric_keys]
                    for i in range(len(metrics_data["Method"]))
                ],
                columns=["Method"] + ordered_metric_keys,
            )

            pareto = {"Method": [], "MS-SSIM (fidelity)": [], "Laplacian (realism)": []}
            for m in available_methods:
                res = method_results[m.name]["metrics"]
                if "msssim (↑)" in res and "laplacian (↑)" in res:
                    pareto["Method"].append(m.name)
                    pareto["MS-SSIM (fidelity)"].append(res["msssim (↑)"])
                    pareto["Laplacian (realism)"].append(res["laplacian (↑)"])

            wandb.log(
                {
                    "inversion_benchmark/metrics_table": table,
                    "inversion_benchmark/pareto_front": wandb.plot.scatter(
                        wandb.Table(
                            data=[
                                [
                                    pareto["Method"][i],
                                    pareto["MS-SSIM (fidelity)"][i],
                                    pareto["Laplacian (realism)"][i],
                                ]
                                for i in range(len(pareto["Method"]))
                            ],
                            columns=[
                                "Method",
                                "MS-SSIM (fidelity)",
                                "Laplacian (realism)",
                            ],
                        ),
                        x="MS-SSIM (fidelity)",
                        y="Laplacian (realism)",
                        title=f"Pareto Front: Fidelity vs Realism ({num} samples)",
                    ),
                }
            )

        features = results["features"]
        feats_hr = features["hr"]
        feats_lr = features["lr"]
        feats_sr_dict = features.get("sr", {})

        knn_labels = results["knn_labels"]

        assert feats_hr.shape[0] == knn_labels.shape[0], (
            f"HR features/labels mismatch: {feats_hr.shape[0]} vs {knn_labels.shape[0]}"
        )
        assert feats_lr.shape[0] == knn_labels.shape[0], (
            f"LR features/labels mismatch: {feats_lr.shape[0]} vs {knn_labels.shape[0]}"
        )
        for mname, feats_sr in feats_sr_dict.items():
            assert feats_sr.shape[0] == knn_labels.shape[0], (
                f"SR features/labels mismatch for {mname}: {feats_sr.shape[0]} vs {knn_labels.shape[0]}"
            )

        if len(available_methods) > 0:
            ordered = [method.name for method in available_methods]
            fig = self.metrics_calc.create_pca_plot(
                feats_hr=feats_hr,
                feats_lr=feats_lr,
                feats_sr_dict=feats_sr_dict,
                labels=knn_labels,
                seed=self.seed,
                method_order=ordered,
            )
            if fig is not None:
                self.run_logger.log_image(
                    key="feature_clustering_pca",
                    image=fig,
                    step=live_base_step + len(available_methods) + 1,
                )

                if use_wandb_direct:
                    wandb.log({"feature_clustering_pca": wandb.Image(fig)})
                plt.close(fig)

            method_metrics = {method.name: method_results[method.name]["metrics"] for method in available_methods}
            metric_order = [
                "structure_distance (↓)",
                "psnr (↑)",
                "lpips (↓)",
                "haarpsi (↑)",
                "msssim (↑)",
            ]
            metric_display_names = {
                "structure_distance (↓)": "Structure Distance",
                "psnr (↑)": "PSNR",
                "lpips (↓)": "LPIPS",
                "haarpsi (↑)": "HaarPSI",
                "msssim (↑)": "MS-SSIM",
            }

            has_knn_metric = any(
                "knn_recovery (↑)" in method_results[method.name]["metrics"] for method in available_methods
            )

            fig_metrics = None
            if has_knn_metric:
                fig_metrics = self.metrics_calc.plot_metrics_vs_knn(
                    method_metrics=method_metrics,
                    metric_order=metric_order,
                    metric_display_names=metric_display_names,
                    knn_key="knn_recovery (↑)",
                    title="Metrics vs KNN recovery",
                )

            if fig_metrics is not None:
                self.run_logger.log_image(
                    key="metrics_vs_knn",
                    image=fig_metrics,
                    step=live_base_step + len(available_methods) + 2,
                )

                if use_wandb_direct:
                    wandb.log({"metrics_vs_knn": wandb.Image(fig_metrics)})
                plt.close(fig_metrics)

    def compute_binned_metrics(self, label_override: int | None = None, log_tag: str | None = None):
        def _sched_kind(scheduler):
            ddpm_like = hasattr(scheduler, "alphas_cumprod")

            edm_like = (not ddpm_like) and hasattr(scheduler, "inversion_timesteps")
            return ddpm_like, edm_like

        ddpm_like, edm_like = _sched_kind(self.scheduler)
        pred_type_override = None
        if hasattr(self, "cfg") and hasattr(self.cfg, "metrics"):
            pred_type_override = self.cfg.metrics.get("prediction_type", None)
        if not ddpm_like and not edm_like and pred_type_override not in {"epsilon", "sample", "v_prediction"}:
            self.logger.warning(
                "Binned metrics require DDPM/DDIM, EDM-like scheduler, or prediction_type override; skipping."
            )
            return None

        if self.dataset is None:
            self.logger.warning("No dataset available for binned metrics; skipping.")
            return None

        subset_size = min(self.num_samples, len(self.dataset))
        base_subset = Subset(self.dataset, list(range(subset_size)))

        if label_override is not None:
            import torch as _torch

            class _LabelOverrideDataset(torch.utils.data.Dataset):
                def __init__(self, base, value: int):
                    self.base = base
                    self.value = int(value)

                def __len__(self):
                    return len(self.base)

                def __getitem__(self, idx):
                    item = self.base[idx]
                    if isinstance(item, dict):
                        out = dict(item)
                        for k in (
                            "class_idx",
                            "res_class",
                            "cond_class",
                            "label",
                            "labels",
                        ):
                            if k in out:
                                v = out[k]
                                if hasattr(v, "shape"):
                                    out[k] = _torch.full_like(v.long(), self.value)
                                else:
                                    out[k] = self.value
                        return out
                    return item

            subset = _LabelOverrideDataset(base_subset, label_override)
        else:
            subset = base_subset
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        pred_type = getattr(
            self.cfg.metrics,
            "prediction_type",
            getattr(self.scheduler.config, "prediction_type", "epsilon"),
        )
        bm = eval_binned_metrics(
            vae=self.vae,
            denoiser=self.denoiser,
            class_embedding=self.class_embedding,
            noise_scheduler=self.scheduler,
            dataloader=loader,
            device=self.device,
            n_bins=int(self.cfg.metrics.get("n_bins", 20)) if hasattr(self.cfg, "metrics") else 20,
            max_batches=int(self.cfg.metrics.get("max_batches", 10)) if hasattr(self.cfg, "metrics") else 10,
            prediction_type=pred_type,
            compute_x0_mse=True,
            compute_class_flip=True,
            compute_cond_influence=(self.class_embedding is not None),
            compute_ema_delta=False,
        )

        payload = metrics_to_dict(bm)
        if self.run_logger:
            figs = render_binned_plots(bm, title_prefix=f"Benchmarker/{log_tag}" if log_tag else "Benchmarker")
            for m in self.methods:
                ns = f"eval/binned_metrics/{m.name}" if not log_tag else f"eval/binned_metrics_{log_tag}/{m.name}"
                self.run_logger.log({ns: payload}, step=0, epoch=0)
                for k_fig, fig in figs.items():
                    new_key = k_fig
                    if log_tag:
                        new_key = new_key.replace("eval/plots/", f"eval/plots/{log_tag}/{m.name}/")
                    else:
                        new_key = new_key.replace("eval/plots/", f"eval/plots/{m.name}/")
                    self.run_logger.log_image(new_key, fig, step=0, epoch=0)
            try:
                import matplotlib.pyplot as _plt

                for fig in figs.values():
                    _plt.close(fig)
            except Exception:
                pass
        try:
            quickplot_binned(bm, title="Benchmarker")
        except Exception:
            pass
        return bm

    @torch.no_grad()
    def run_spectral_analysis(self, results: Dict):
        self.logger.info("[Paper Exp] Running Spectral Analysis...")
        analyzer = SpectralAnalyzer(self.device)
        psd_results = {}

        first_method_data = list(results["method_results"].values())[0]
        ref_latent = first_method_data["inverted_latents"]
        gaussian_noise = torch.randn_like(ref_latent)
        psd_results["Gaussian Prior (Theory)"] = analyzer.compute_radial_psd(gaussian_noise)
        for method_name, data in results["method_results"].items():
            psd_results[method_name] = analyzer.compute_radial_psd(data["inverted_latents"])
        fig_psd = analyzer.plot_psd_comparison(psd_results, "Latent Space Spectral Density")
        self.run_logger.log_image("paper_experiments/spectral_density_latents_line", fig_psd, step=0)
        plt.close(fig_psd)

        psd_img_results = {}
        psd_img_results["HR Reference"] = analyzer.compute_radial_psd(results["original_images"])
        psd_img_results["LR Input"] = analyzer.compute_radial_psd(results["downsampled"])
        for method_name, data in results["method_results"].items():
            imgs = data["generated_images"]
            psd_img_results[method_name] = analyzer.compute_radial_psd(imgs)
        fig_psd_img = analyzer.plot_psd_comparison(psd_img_results, title="Image Space Spectral Density")
        self.run_logger.log_image("paper_experiments/spectral_density_images", fig_psd_img, step=0)
        plt.close(fig_psd_img)

        heatmap_tensors = {
            "Gaussian (Theory)": gaussian_noise,
        }
        for k in results["method_results"].keys():
            if "standard" in k.lower():
                heatmap_tensors["Standard"] = results["method_results"][k]["inverted_latents"]
            if "ours" in k.lower():
                heatmap_tensors["Ours (Proposed)"] = results["method_results"][k]["inverted_latents"]
        if len(heatmap_tensors) == 1:
            keys = list(results["method_results"].keys())
            heatmap_tensors[keys[0]] = results["method_results"][keys[0]]["inverted_latents"]
            heatmap_tensors[keys[-1]] = results["method_results"][keys[-1]]["inverted_latents"]
        fig_heatmap = analyzer.plot_2d_spectrum_visual(heatmap_tensors, "Latent Space 2D Spectrum (Log Mag)")
        self.run_logger.log_image("paper_experiments/spectral_density_latents_heatmap", fig_heatmap, step=0)
        plt.close(fig_heatmap)

    @torch.no_grad()
    def run_differential_spectral_analysis(self):
        self.logger.info("[Paper Exp] Running Differential Spectral Analysis...")
        analyzer = SpectralAnalyzer(self.device)

        samples = self._prepare_samples()
        image = samples["downsampled"][:1].to(self.device)

        if self.vae is not None:
            image_latent = self._encode_latents(image)
        else:
            image_latent = image.to(self.device)

        if self.task_mode == "t2i_editing":
            cond, _ = self.encode_embeddings(samples["source_prompts"][:1])
        else:
            source_class = samples["source_class"][:1].to(self.device)
            cond, _ = self.encode_embeddings(source_class)

        methods_to_test = {
            "Standard": StandardInversion(),
            "Ours": ControlledGaussianizationInversion(
                use_cgd=True,
                cgd_eta=200.0,
                use_scp=True,
                scp_eta=0.002,
            ),
        }

        figs = {}

        for name, method in methods_to_test.items():
            invert_output = method.invert(
                image=image_latent,
                denoiser=self.denoiser,
                scheduler=self.scheduler,
                encoder_hidden_states=cond,
                x0=image_latent,
                num_steps=self.num_inference_steps,
                store_intermediates=True,
            )

            if isinstance(invert_output, dict):
                intermediates = invert_output.get("intermediate_latents", [])
            elif len(invert_output) == 3:
                _, intermediates, _ = invert_output
            else:
                _, intermediates = invert_output

            spectrogram_cols = []
            for t in range(len(intermediates) - 1):
                delta = intermediates[t + 1] - intermediates[t]
                radii, psd = analyzer.compute_radial_psd(delta)
                spectrogram_cols.append(psd)

            if not spectrogram_cols:
                self.logger.warning(f"No spectrogram data for {name}, skipping")
                continue

            spectrogram = np.stack(spectrogram_cols, axis=1)
            print(f"Method: {name}, vmin: {np.percentile(spectrogram, 1)}, vmax: {np.percentile(spectrogram, 99)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(
                spectrogram,
                aspect="auto",
                origin="lower",
                cmap="magma",
                interpolation="nearest",
                vmin=-1.5,
                vmax=0.5,
            )

            ax.set_title(f"Spectral Velocity: {name} \n $\\Delta x = x_{{t+1}} - x_{{t}}$")
            ax.set_xlabel("Inversion Step ($t \\to T$)")
            ax.set_ylabel("Spatial Frequency (Low $\\to$ High)")
            fig.colorbar(im, ax=ax, label="Log Energy of Update $\\Delta x$")

            figs[name] = fig
            self.run_logger.log_image(f"paper_experiments/velocity_spectrum_{name}", fig, step=0)
            plt.close(fig)

        self.logger.info("Differential Spectral Analysis Complete.")

    @torch.no_grad()
    def run_null_space_sensitivity_test(self):
        self.logger.info("[Paper Exp] Running Null Space Sensitivity Test...")
        analyzer = SpectralAnalyzer(self.device)

        samples = self._prepare_samples()
        images_lr = samples["downsampled"][: self.batch_size].to(self.device)

        if self.task_mode == "t2i_editing":
            cond_lr, _ = self.encode_embeddings(samples["source_prompts"][: self.batch_size])
        else:
            source_labels = samples["source_class"][: self.batch_size].to(self.device)
            cond_lr, _ = self.encode_embeddings(source_labels)

        latents = self._encode_latents(images_lr)

        perturbation = analyzer.create_checkerboard_noise(latents.shape)
        perturbation = perturbation * 1.0

        latents_perturbed = latents + perturbation

        t_idx = torch.tensor([0] * latents.shape[0], device=self.device).long()

        noise_pred_clean = self.denoiser(latents, t_idx, encoder_hidden_states=cond_lr).sample
        noise_pred_perturbed = self.denoiser(latents_perturbed, t_idx, encoder_hidden_states=cond_lr).sample

        diff = noise_pred_clean - noise_pred_perturbed
        l2_diff = diff.pow(2).sum().item()

        self.logger.info(f"Null Space Sensitivity (L2 Diff): {l2_diff:.6f}")

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(latents[0, 0].cpu(), cmap="gray")
        ax[0].set_title("Original Latent ($z_{LR}$)")

        ax[1].imshow((latents_perturbed[0, 0]).cpu(), cmap="gray")
        ax[1].set_title("Perturbed Latent ($z_{LR} + \delta_{HF}$)")

        im = ax[2].imshow(
            (noise_pred_clean[0, 0] - noise_pred_perturbed[0, 0]).abs().cpu(),
            cmap="inferno",
        )
        ax[2].set_title(f"Prediction Diff\nTotal L2: {l2_diff:.4f}")
        fig.colorbar(im, ax=ax[2])

        self.run_logger.log_image("paper_experiments/null_space_sensitivity", fig, step=0)
        self.run_logger.log({"paper_experiments/null_space_l2_score": l2_diff}, step=0)
        plt.close(fig)

    @torch.no_grad()
    def run_blur_variance_analysis(self, sigma_range: List[float] = None):
        self.logger.info("[Paper Exp] Running Blur vs Latent Variance Analysis...")

        if sigma_range is None:
            sigma_range = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

        samples = self._prepare_samples()
        image_hr = samples["original_images"][:1].to(self.device)

        if self.task_mode == "t2i_editing":
            cond, _ = self.encode_embeddings(samples["source_prompts"][:1])
        else:
            source_class = samples["source_class"][:1].to(self.device)
            cond, _ = self.encode_embeddings(source_class)

        methods_to_test = {
            "Standard": StandardInversion(),
            "Ours": ControlledGaussianizationInversion(
                use_cgd=True,
                cgd_eta=200.0,
                use_scp=True,
                scp_eta=0.002,
            ),
        }

        variance_results = {name: [] for name in methods_to_test.keys()}
        variance_per_channel = {name: [] for name in methods_to_test.keys()}

        gaussian_variances = []

        for sigma in sigma_range:
            self.logger.info(f"  Processing sigma = {sigma:.1f}")

            if sigma > 0:
                kernel_size = int(6 * sigma + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                kernel_size = max(3, kernel_size)

                blurred_image = torchvision.transforms.functional.gaussian_blur(
                    image_hr,
                    kernel_size=[kernel_size, kernel_size],
                    sigma=[sigma, sigma],
                )
            else:
                blurred_image = image_hr.clone()

            latent_input = self._encode_latents(blurred_image)

            gaussian_noise = torch.randn_like(latent_input)
            gaussian_variances.append(gaussian_noise.var().item())

            for method_name, method in methods_to_test.items():
                invert_output = method.invert(
                    image=latent_input,
                    denoiser=self.denoiser,
                    scheduler=self.scheduler,
                    encoder_hidden_states=cond,
                    x0=latent_input,
                    num_steps=self.num_inference_steps,
                    store_intermediates=False,
                )

                if isinstance(invert_output, dict):
                    inverted_latent = invert_output["latents"]
                elif isinstance(invert_output, tuple):
                    inverted_latent = invert_output[0]
                else:
                    inverted_latent = invert_output

                total_var = inverted_latent.var().item()
                variance_results[method_name].append(total_var)

                per_channel_var = inverted_latent.var(dim=(0, 2, 3)).cpu().numpy()
                variance_per_channel[method_name].append(per_channel_var)

        fig1, ax1 = plt.subplots(figsize=(10, 6))

        mean_gaussian_var = np.mean(gaussian_variances)
        ax1.axhline(
            y=mean_gaussian_var,
            color="gray",
            linestyle="--",
            linewidth=2,
            label=f"Gaussian Prior (Var ≈ {mean_gaussian_var:.3f})",
            alpha=0.7,
        )

        colors = {"Standard": "tab:blue", "Ours": "tab:orange"}
        markers = {"Standard": "o", "Ours": "s"}

        for method_name, variances in variance_results.items():
            ax1.plot(
                sigma_range,
                variances,
                color=colors.get(method_name, "tab:green"),
                marker=markers.get(method_name, "^"),
                linewidth=2,
                markersize=8,
                label=method_name,
            )

        ax1.set_xlabel(r"Blur $\sigma$", fontsize=12)
        ax1.set_ylabel("Latent Variance", fontsize=12)
        ax1.set_title(r"Latent Variance vs Gaussian Blur $\sigma$", fontsize=14)
        ax1.legend(loc="best", fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-0.5, max(sigma_range) + 0.5)

        self.run_logger.log_image("paper_experiments/blur_variance_analysis", fig1, step=0)
        plt.close(fig1)

        fig2, axes = plt.subplots(1, len(methods_to_test), figsize=(6 * len(methods_to_test), 5))
        if len(methods_to_test) == 1:
            axes = [axes]

        for ax, (method_name, per_channel_data) in zip(axes, variance_per_channel.items()):
            heatmap_data = np.stack(per_channel_data, axis=0)

            im = ax.imshow(
                heatmap_data.T,
                aspect="auto",
                cmap="viridis",
                origin="lower",
                extent=[
                    sigma_range[0],
                    sigma_range[-1],
                    -0.5,
                    heatmap_data.shape[1] - 0.5,
                ],
            )
            ax.set_xlabel(r"Blur $\sigma$", fontsize=12)
            ax.set_ylabel("Latent Channel", fontsize=12)
            ax.set_title(f"{method_name}: Per-Channel Variance", fontsize=12)
            fig2.colorbar(im, ax=ax, label="Variance")

        plt.suptitle(r"Per-Channel Latent Variance vs Blur $\sigma$", fontsize=14)
        plt.tight_layout()
        self.run_logger.log_image("paper_experiments/blur_variance_per_channel", fig2, step=0)
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 6))

        ax3.axhline(
            y=1.0,
            color="gray",
            linestyle="--",
            linewidth=2,
            label="Gaussian Prior",
            alpha=0.7,
        )

        for method_name, variances in variance_results.items():
            normalized_var = np.array(variances) / mean_gaussian_var
            ax3.plot(
                sigma_range,
                normalized_var,
                color=colors.get(method_name, "tab:green"),
                marker=markers.get(method_name, "^"),
                linewidth=2,
                markersize=8,
                label=method_name,
            )

        ax3.set_xlabel(r"Blur $\sigma$", fontsize=12)
        ax3.set_ylabel("Normalized Variance (ratio to Gaussian)", fontsize=12)
        ax3.set_title(r"Normalized Latent Variance vs Blur $\sigma$", fontsize=14)
        ax3.legend(loc="best", fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-0.5, max(sigma_range) + 0.5)

        self.run_logger.log_image("paper_experiments/blur_variance_normalized", fig3, step=0)
        plt.close(fig3)

        summary_data = {
            "paper_experiments/blur_variance_summary": {
                "sigma_range": sigma_range,
                "gaussian_reference_variance": mean_gaussian_var,
            }
        }
        for method_name, variances in variance_results.items():
            summary_data["paper_experiments/blur_variance_summary"][f"{method_name}_variances"] = variances

        self.run_logger.log(summary_data, step=0)

        self.logger.info("[Paper Exp] Blur vs Latent Variance Analysis Complete.")

        return {
            "sigma_range": sigma_range,
            "variance_results": variance_results,
            "gaussian_reference": mean_gaussian_var,
        }

    @torch.no_grad()
    def benchmark(self) -> Dict:
        samples = self._prepare_samples()
        editing_mode = self.task_mode == "t2i_editing"

        vis_idx = slice(self.vis_samples)
        results = {
            "original_images": samples["original_images"][vis_idx].clone(),
            "downsampled": samples["downsampled"][vis_idx].clone(),
            "knn_labels": samples["knn_labels"],
        }

        if editing_mode:
            results.update(
                {
                    "source_prompts": samples["source_prompts"],
                    "target_prompts": samples["target_prompts"],
                }
            )

        original_hr_norm = normalize_tensor(samples["original_images"])
        original_lr_norm = normalize_tensor(samples["downsampled"])

        feats_hr = self._extract_inception_features(original_hr_norm)
        feats_lr = self._extract_inception_features(original_lr_norm)

        if self.compute_distribution_metrics:
            knn_lr_value = compute_knn_score(feats_lr, samples["knn_labels"])
            knn_hr_value = compute_knn_score(feats_hr, samples["knn_labels"])
            knn_lr = knn_hr_value if editing_mode else knn_lr_value
            knn_hr = knn_hr_value
        else:
            knn_lr = knn_hr = None

        results["features"] = {"hr": feats_hr, "lr": feats_lr, "sr": {}}
        method_results = {}
        num_samples = samples["original_images"].shape[0]

        for method in self.methods:
            self.logger.info(f"Processing method: {method.name}")

            if not (self.model_training_mode == "diffusion") and not (
                method.name.startswith("standard_inversion") or method.name.startswith("ours")
            ):
                self.logger.warning(
                    f"Skipping incompatible method {method.name} for model type {self.model_training_mode}"
                )
                continue

            h5_exporter = None
            if self.export_intermediates:
                dataset_name = self.cfg.data.get("dataset", "unknown")

                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                config_name = hydra_cfg.job.config_name or "config"
                config_name = config_name[:-5] if config_name.endswith(".yaml") else config_name

                if "_config_" in method.name:
                    parts = method.name.split("_config_")
                    base_method_name = parts[0]
                    config_idx = parts[1] if len(parts) > 1 else "0"

                    try:
                        config_idx_int = int(config_idx) + 29
                        config_idx = str(config_idx_int)
                    except ValueError:
                        pass
                else:
                    base_method_name = method.name
                    config_idx = "0"

                h5_method_name = f"{base_method_name}_config{config_idx}"

                h5_exporter = HDF5Exporter(
                    output_dir=self.export_output_dir,
                    config_name=config_name,
                    method_name=h5_method_name,
                    seed=self.seed,
                    num_steps=self.num_inference_steps,
                    save_intermediate_latents=self.export_save_intermediate_latents,
                    save_predicted_noises=self.export_save_predicted_noises,
                )
                export_config = {
                    "dataset": dataset_name,
                    "method": method.name,
                    "training_mode": self.model_training_mode,
                    "num_steps": self.num_inference_steps,
                    "num_samples": num_samples,
                    "batch_size": self.batch_size,
                    "checkpoint": str(self.cfg.model_checkpoint.get("checkpoint_path", "")),
                    "seed": self.seed,
                }
                method_config = method.get_config()
                h5_exporter.open(export_config, method_config=method_config)
                self.logger.info(f"HDF5 export enabled: {h5_exporter.h5_path}")

            for metric_fn in self.metrics.values():
                if hasattr(metric_fn, "reset"):
                    metric_fn.reset()

            metric_values = {k: 0.0 for k in self.metrics}
            sr_feature_chunks: List[np.ndarray] = []
            vis_generated_images = []
            vis_inverted_latents = []

            total_inversion_time_ns = 0
            total_sampling_time_ns = 0

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)

                batch_original = samples["original_images"][start_idx:end_idx]
                batch_down = samples["downsampled"][start_idx:end_idx]
                batch_input = batch_original if editing_mode else batch_down

                if editing_mode:
                    source_data = samples["source_prompts"][start_idx:end_idx]
                    target_data = samples["target_prompts"][start_idx:end_idx]
                else:
                    source_data = samples["source_class"][start_idx:end_idx]
                    target_data = samples["target_class"][start_idx:end_idx]

                batch_latents = self._encode_latents(batch_input)
                cond_emb_source, target_condition_emb, target_uncond = self._get_embeddings(
                    editing_mode, source_data, target_data
                )

                run_kwargs = {
                    "image": batch_latents,
                    "denoiser": self.denoiser,
                    "scheduler": self.scheduler,
                    "encoder_hidden_states": cond_emb_source,
                    "guidance_scale": 1.0,
                    "num_steps": self.num_inference_steps,
                    "batch_size": self.batch_size,
                }

                if editing_mode:
                    run_kwargs.update(
                        {
                            "sample_encoder_hidden_states": target_condition_emb,
                            "sample_uncond_embeddings": target_uncond,
                        }
                    )
                else:
                    run_kwargs.update(
                        {
                            "sample_encoder_hidden_states": target_condition_emb,
                        }
                    )

                method_params = self.method_params.get(method.name, {}).get("params", {})
                run_kwargs.update(method_params)

                if "x0" not in run_kwargs:
                    run_kwargs["x0"] = batch_latents

                live_binned_flag = False
                if hasattr(self.cfg, "metrics"):
                    live_binned_flag = bool(self.cfg.metrics.get("live_binned", False))
                if live_binned_flag:
                    run_kwargs["collect_live_binned"] = True

                    run_kwargs["inversion_live_flip_embeddings"] = target_condition_emb

                    run_kwargs["sample_live_flip_embeddings"] = cond_emb_source

                if h5_exporter is not None:
                    run_kwargs["store_intermediates"] = True
                    run_kwargs["store_predicted_noise"] = True
                    run_kwargs["store_timesteps"] = True

                run_result = method.run(**run_kwargs)

                intermediate_latents = []
                predicted_noises = []
                timesteps_list = []
                batch_inversion_time_ns = 0
                batch_sampling_time_ns = 0
                if isinstance(run_result, dict):
                    generated = run_result["generated"]
                    inverted_latents = run_result["inverted_latents"]
                    intermediate_latents = run_result.get("intermediate_latents", [])
                    predicted_noises = run_result.get("predicted_noises", [])
                    timesteps_list = run_result.get("timesteps", [])
                    batch_inversion_time_ns = run_result.get("inversion_time_ns", 0)
                    batch_sampling_time_ns = run_result.get("sampling_time_ns", 0)
                else:
                    generated, inverted_latents = run_result

                total_inversion_time_ns += batch_inversion_time_ns
                total_sampling_time_ns += batch_sampling_time_ns

                generated = self._decode_latents(generated)

                if h5_exporter is not None:
                    sample_indices = list(range(start_idx, end_idx))
                    h5_exporter.save_batch(
                        intermediate_latents=intermediate_latents,
                        predicted_noises=predicted_noises,
                        timesteps=timesteps_list,
                        generated_images=generated,
                        original_images=batch_original,
                        input_images=batch_down,
                        sample_indices=sample_indices,
                        metadata={"knn_labels": samples["knn_labels"][start_idx:end_idx]},
                    )

                gen_norm = normalize_tensor(generated)
                orig_norm = normalize_tensor(batch_original)

                if gen_norm.shape != orig_norm.shape:
                    gen_norm = resize_image_tensor(gen_norm, (orig_norm.shape[2], orig_norm.shape[3]))

                gen_norm_cpu = gen_norm.detach().cpu()

                for metric_name, metric_fn in self.metrics.items():
                    if hasattr(metric_fn, "update"):
                        if metric_name == "fid (↓)":
                            gen_uint8 = (gen_norm.clamp(0, 1) * 255).to(torch.uint8).to(self.device)
                            orig_uint8 = (orig_norm.clamp(0, 1) * 255).to(torch.uint8).to(self.device)
                            metric_fn.update(gen_uint8, real=False)
                            metric_fn.update(orig_uint8, real=True)
                        elif metric_name == "lpips (↓)":
                            gen_metric = gen_norm.clamp(0.0, 1.0).mul(2.0).sub(1.0)
                            orig_metric = orig_norm.clamp(0.0, 1.0).mul(2.0).sub(1.0)
                            gen_metric = gen_metric.to(device=self.device, dtype=torch.float32)
                            orig_metric = orig_metric.to(device=self.device, dtype=torch.float32)
                            metric_fn.update(gen_metric, orig_metric)
                        elif metric_name == "haarpsi (↑)":
                            gen_metric = gen_norm.to(device=self.device, dtype=torch.float32)
                            orig_metric = orig_norm.to(device=self.device, dtype=torch.float32)
                            metric_fn.update(gen_metric, orig_metric)
                        else:
                            metric_device = getattr(metric_fn, "device", self.device)
                            if hasattr(metric_fn, "parameters") and len(list(metric_fn.parameters())) > 0:
                                first_param = next(metric_fn.parameters())
                                target_device = first_param.device
                                target_dtype = first_param.dtype
                            else:
                                target_device = metric_device
                                target_dtype = self.dtype

                            gen_metric = gen_norm.to(device=target_device, dtype=target_dtype)
                            orig_metric = orig_norm.to(device=target_device, dtype=target_dtype)

                            metric_fn.update(gen_metric, orig_metric)

                batch_feats = self._extract_inception_features(gen_norm_cpu)
                sr_feature_chunks.append(batch_feats)

                vis_count = max(0, min(end_idx, self.vis_samples) - start_idx)
                if vis_count > 0:
                    vis_generated_images.append(gen_norm_cpu[:vis_count].clone())
                    vis_inverted_latents.append(inverted_latents[:vis_count].detach().cpu().clone())

                del (
                    batch_latents,
                    cond_emb_source,
                    target_condition_emb,
                    generated,
                    inverted_latents,
                    gen_norm,
                    gen_norm_cpu,
                    batch_feats,
                    orig_norm,
                )
                if self.aggressive_cleanup:
                    torch.cuda.empty_cache()

            for metric_name, metric_fn in self.metrics.items():
                if hasattr(metric_fn, "compute"):
                    computed_value = metric_fn.compute()
                    if torch.is_tensor(computed_value):
                        metric_values[metric_name] = float(computed_value.item())
                    else:
                        metric_values[metric_name] = float(computed_value)
                    self.logger.info(f"Metric {metric_name}: {metric_values[metric_name]}")

            if sr_feature_chunks:
                feats_sr_full = np.concatenate(sr_feature_chunks, axis=0)
            else:
                feats_sr_full = np.zeros((0, feats_hr.shape[1]), dtype=np.float32)
            metric_values.update(self._compute_knn_recovery(feats_sr_full, samples["knn_labels"], knn_lr, knn_hr))

            vis_gen = (
                torch.cat(vis_generated_images, dim=0)
                if vis_generated_images
                else torch.zeros_like(results["original_images"])
            )
            vis_lat = (
                torch.cat(vis_inverted_latents, dim=0)
                if vis_inverted_latents
                else torch.zeros_like(results["downsampled"])
            )

            results["features"]["sr"][method.name] = feats_sr_full
            method_results[method.name] = {
                "generated_images": vis_gen,
                "inverted_latents": vis_lat,
                "metrics": metric_values,
            }

            live_inv = getattr(method, "live_binned", None)
            if live_inv is not None:
                method_results[method.name]["live_binned_inversion"] = live_inv
            live_samp = getattr(method, "live_binned_sampling", None)
            if live_samp is not None:
                method_results[method.name]["live_binned_sampling"] = live_samp

            if h5_exporter is not None:
                h5_exporter.save_metrics(metric_values)
                h5_exporter.save_timing(
                    inversion_time_ns=total_inversion_time_ns,
                    sampling_time_ns=total_sampling_time_ns,
                    num_samples=num_samples,
                )
                h5_exporter.close()
                self.logger.info(f"HDF5 export complete: {h5_exporter.h5_path}")
                self.logger.info(
                    f"Timing - Inversion: {total_inversion_time_ns / 1e9:.4f}s, "
                    f"Sampling: {total_sampling_time_ns / 1e9:.4f}s, "
                    f"Total: {(total_inversion_time_ns + total_sampling_time_ns) / 1e9:.4f}s "
                    f"({num_samples} samples)"
                )

            self.logger.info(f"Final metrics for {method.name}: {list(metric_values.keys())}")

            del (
                sr_feature_chunks,
                vis_generated_images,
                vis_inverted_latents,
                feats_sr_full,
            )
            if self.aggressive_cleanup:
                torch.cuda.empty_cache()

        if self.edm_checkpoint_path is not None:
            self.logger.info("=" * 60)
            self.logger.info("Running EDM checkpoint evaluation...")
            self.logger.info("=" * 60)

            self._unload_models()
            self._load_custom_models(self.edm_checkpoint_path, rescale=False)

            self.dtype = get_model_dtype(self.denoiser)
            if self.denoiser is not None:
                self.denoiser = self.denoiser.to(device=self.device, dtype=self.dtype).eval()
            if self.vae is not None:
                self.vae = self.vae.to(device=self.device, dtype=self.dtype).eval()
            if self.class_embedding is not None:
                self.class_embedding = self.class_embedding.to(device=self.device, dtype=self.dtype).eval()

            edm_method = StandardInversion()
            edm_method_name = "edm_standard_inversion"

            for metric_fn in self.metrics.values():
                if hasattr(metric_fn, "reset"):
                    metric_fn.reset()

            metric_values = {k: 0.0 for k in self.metrics}
            sr_feature_chunks: List[np.ndarray] = []
            vis_generated_images = []
            vis_inverted_latents = []

            total_inversion_time_ns = 0
            total_sampling_time_ns = 0

            h5_exporter = None
            if self.export_intermediates:
                dataset_name = self.cfg.data.get("dataset", "unknown")

                hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
                config_name = hydra_cfg.job.config_name or "config"
                config_name = config_name[:-5] if config_name.endswith(".yaml") else config_name

                h5_exporter = HDF5Exporter(
                    output_dir=self.export_output_dir,
                    config_name=config_name,
                    method_name=edm_method_name,
                    seed=self.seed,
                    num_steps=self.num_inference_steps,
                    save_intermediate_latents=self.export_save_intermediate_latents,
                    save_predicted_noises=self.export_save_predicted_noises,
                )
                export_config = {
                    "dataset": dataset_name,
                    "method": edm_method_name,
                    "num_steps": self.num_inference_steps,
                    "num_samples": num_samples,
                    "batch_size": self.batch_size,
                    "checkpoint": str(self.edm_checkpoint_path),
                    "training_mode": "edm",
                    "seed": self.seed,
                }

                edm_method_config = {
                    "name": edm_method_name,
                    "description": "Standard EDM Inversion",
                    "training_mode": "edm",
                }
                h5_exporter.open(export_config, method_config=edm_method_config)
                self.logger.info(f"HDF5 export enabled for EDM: {h5_exporter.h5_path}")

            for start_idx in range(0, num_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, num_samples)

                batch_original = samples["original_images"][start_idx:end_idx]
                batch_down = samples["downsampled"][start_idx:end_idx]
                batch_input = batch_original if editing_mode else batch_down

                if editing_mode:
                    source_data = samples["source_prompts"][start_idx:end_idx]
                    target_data = samples["target_prompts"][start_idx:end_idx]
                else:
                    source_data = samples["source_class"][start_idx:end_idx]
                    target_data = samples["target_class"][start_idx:end_idx]

                batch_latents = self._encode_latents(batch_input)
                cond_emb_source, target_condition_emb, target_uncond = self._get_embeddings(
                    editing_mode, source_data, target_data
                )

                run_kwargs = {
                    "image": batch_latents,
                    "denoiser": self.denoiser,
                    "scheduler": self.scheduler,
                    "encoder_hidden_states": cond_emb_source,
                    "guidance_scale": 1.0,
                    "num_steps": self.num_inference_steps,
                    "batch_size": self.batch_size,
                    "sample_encoder_hidden_states": target_condition_emb,
                    "x0": batch_latents,
                }

                if h5_exporter is not None:
                    run_kwargs["store_intermediates"] = True
                    run_kwargs["store_predicted_noise"] = True
                    run_kwargs["store_timesteps"] = True

                run_result = edm_method.run(**run_kwargs)

                intermediate_latents = []
                predicted_noises = []
                timesteps_list = []
                batch_inversion_time_ns = 0
                batch_sampling_time_ns = 0
                if isinstance(run_result, dict):
                    generated = run_result["generated"]
                    inverted_latents = run_result["inverted_latents"]
                    intermediate_latents = run_result.get("intermediate_latents", [])
                    predicted_noises = run_result.get("predicted_noises", [])
                    timesteps_list = run_result.get("timesteps", [])
                    batch_inversion_time_ns = run_result.get("inversion_time_ns", 0)
                    batch_sampling_time_ns = run_result.get("sampling_time_ns", 0)
                else:
                    generated, inverted_latents = run_result

                total_inversion_time_ns += batch_inversion_time_ns
                total_sampling_time_ns += batch_sampling_time_ns

                generated = self._decode_latents(generated)

                if h5_exporter is not None:
                    sample_indices = list(range(start_idx, end_idx))
                    h5_exporter.save_batch(
                        intermediate_latents=intermediate_latents,
                        predicted_noises=predicted_noises,
                        timesteps=timesteps_list,
                        generated_images=generated,
                        original_images=batch_original,
                        input_images=batch_down,
                        sample_indices=sample_indices,
                        metadata={"knn_labels": samples["knn_labels"][start_idx:end_idx]},
                    )

                gen_norm = normalize_tensor(generated)
                orig_norm = normalize_tensor(batch_original)

                if gen_norm.shape != orig_norm.shape:
                    gen_norm = resize_image_tensor(gen_norm, (orig_norm.shape[2], orig_norm.shape[3]))

                gen_norm_cpu = gen_norm.detach().cpu()

                for metric_name, metric_fn in self.metrics.items():
                    if hasattr(metric_fn, "update"):
                        if metric_name == "fid (↓)":
                            gen_uint8 = (gen_norm.clamp(0, 1) * 255).to(torch.uint8).to(self.device)
                            orig_uint8 = (orig_norm.clamp(0, 1) * 255).to(torch.uint8).to(self.device)
                            metric_fn.update(gen_uint8, real=False)
                            metric_fn.update(orig_uint8, real=True)
                        elif metric_name == "lpips (↓)":
                            gen_metric = gen_norm.clamp(0.0, 1.0).mul(2.0).sub(1.0)
                            orig_metric = orig_norm.clamp(0.0, 1.0).mul(2.0).sub(1.0)
                            gen_metric = gen_metric.to(device=self.device, dtype=torch.float32)
                            orig_metric = orig_metric.to(device=self.device, dtype=torch.float32)
                            metric_fn.update(gen_metric, orig_metric)
                        elif metric_name == "haarpsi (↑)":
                            gen_metric = gen_norm.to(device=self.device, dtype=torch.float32)
                            orig_metric = orig_norm.to(device=self.device, dtype=torch.float32)
                            metric_fn.update(gen_metric, orig_metric)
                        else:
                            metric_device = getattr(metric_fn, "device", self.device)
                            if hasattr(metric_fn, "parameters") and len(list(metric_fn.parameters())) > 0:
                                first_param = next(metric_fn.parameters())
                                target_device = first_param.device
                                target_dtype = first_param.dtype
                            else:
                                target_device = metric_device
                                target_dtype = self.dtype

                            gen_metric = gen_norm.to(device=target_device, dtype=target_dtype)
                            orig_metric = orig_norm.to(device=target_device, dtype=target_dtype)

                            metric_fn.update(gen_metric, orig_metric)

                batch_feats = self._extract_inception_features(gen_norm_cpu)
                sr_feature_chunks.append(batch_feats)

                vis_count = max(0, min(end_idx, self.vis_samples) - start_idx)
                if vis_count > 0:
                    vis_generated_images.append(gen_norm_cpu[:vis_count].clone())
                    vis_inverted_latents.append(inverted_latents[:vis_count].detach().cpu().clone())

            for metric_name, metric_fn in self.metrics.items():
                if hasattr(metric_fn, "compute"):
                    computed_value = metric_fn.compute()
                    if torch.is_tensor(computed_value):
                        metric_values[metric_name] = float(computed_value.item())
                    else:
                        metric_values[metric_name] = float(computed_value)
                    self.logger.info(f"EDM Metric {metric_name}: {metric_values[metric_name]}")

            if sr_feature_chunks:
                feats_sr_full = np.concatenate(sr_feature_chunks, axis=0)
            else:
                feats_sr_full = np.zeros((0, feats_hr.shape[1]), dtype=np.float32)
            metric_values.update(self._compute_knn_recovery(feats_sr_full, samples["knn_labels"], knn_lr, knn_hr))

            vis_gen = (
                torch.cat(vis_generated_images, dim=0)
                if vis_generated_images
                else torch.zeros_like(results["original_images"])
            )
            vis_lat = (
                torch.cat(vis_inverted_latents, dim=0)
                if vis_inverted_latents
                else torch.zeros_like(results["downsampled"])
            )

            results["features"]["sr"][edm_method_name] = feats_sr_full
            method_results[edm_method_name] = {
                "generated_images": vis_gen,
                "inverted_latents": vis_lat,
                "metrics": metric_values,
            }

            if h5_exporter is not None:
                h5_exporter.save_metrics(metric_values)
                h5_exporter.save_timing(
                    inversion_time_ns=total_inversion_time_ns,
                    sampling_time_ns=total_sampling_time_ns,
                    num_samples=num_samples,
                )
                h5_exporter.close()
                self.logger.info(f"HDF5 export complete for EDM: {h5_exporter.h5_path}")
                total_inv_s = total_inversion_time_ns / 1e9
                total_samp_s = total_sampling_time_ns / 1e9
                self.logger.info(
                    f"EDM Timing - Inversion: {total_inv_s:.4f}s ({total_inversion_time_ns} ns), "
                    f"Sampling: {total_samp_s:.4f}s ({total_sampling_time_ns} ns), "
                    f"Total: {total_inv_s + total_samp_s:.4f}s "
                    f"({num_samples} samples)"
                )

            self.logger.info(f"EDM evaluation complete. Metrics: {list(metric_values.keys())}")

        results["method_results"] = method_results

        try:
            run_offline = bool(getattr(getattr(self.cfg, "metrics", {}), "run_offline_in_benchmark", False))
            if (
                run_offline
                and self.compute_distribution_metrics
                and self.class_embedding is not None
                and hasattr(self.cfg, "metrics")
            ):
                num_classes = int(getattr(self.cfg.data, "num_classes", 0))
                if num_classes >= 2:
                    self.compute_binned_metrics(label_override=0, log_tag="inversion")

                    self.compute_binned_metrics(label_override=1, log_tag="sampling")
                else:
                    self.compute_binned_metrics()
        except Exception as e:
            self.logger.warning(f"[metrics] binned computation skipped: {e}")

        if len(method_results) > 0:
            try:
                self.run_spectral_analysis(results)
            except Exception as e:
                self.logger.warning(f"Spectral Analysis failed: {e}")

        try:
            self.run_null_space_sensitivity_test()
        except Exception as e:
            self.logger.warning(f"Null Space Test failed: {e}")

        try:
            self.run_differential_spectral_analysis()
        except Exception as e:
            self.logger.warning(f"Differential Spectral Analysis failed: {e}")

        try:
            self.run_blur_variance_analysis()
        except Exception as e:
            self.logger.warning(f"Blur Variance Analysis failed: {e}")

        return results
