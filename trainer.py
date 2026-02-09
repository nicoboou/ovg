#!/usr/bin/env python
# coding=utf-8

import math
import os
import hydra
from omegaconf import OmegaConf
import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import numpy as np
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler

from ovg.utils.eval import (
    generate_and_log_grids,
    compute_gradient_norms,
    evaluate_quality,
)
from ovg.utils.helpers import get_model_dtype
from ovg.utils.metrics_binned import eval_binned_metrics, render_binned_plots
from ovg.data.datasets import BBBC021Dataset, Edges2ShoesDataset
from ovg.utils.ema import ExponentialMovingAverage, EMAModelWrapper
from ovg.utils.logger import RunLogger
from ovg.utils.checkpoint import CheckpointManager
from ovg.utils.helpers import (
    get_models,
    setup_weight_dtype,
    setup_optimizer_params,
    unwrap_model,
)
from ovg.utils.checkpoint import load_training_checkpoint, get_last_step
from ovg.training.strategies import build_training_strategy


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_step = 0
        self.current_epoch = 0

        self.logger = get_logger(__name__, log_level="INFO")
        self.training_strategy = None

        self.setup_environment()
        self.setup_data()
        self.setup_models()
        self.setup_training()
        self.training_strategy = build_training_strategy(self)

        resume_metrics_path = self.get_checkpoint_dir() / "metrics.jsonl"
        resume_step = get_last_step(resume_metrics_path) if resume_metrics_path.exists() else 0

        self.run_logger = RunLogger(
            self.cfg,
            checkpoint_dir=self.get_checkpoint_dir(),
            accelerator=self.accelerator,
            resume_step=resume_step,
            wandb_root_dir=Path(self.cfg.output_dir),
        )
        self.setup_logging()

        self.checkpoint_manager = CheckpointManager(self)

    def setup_environment(self):
        logging_dir = Path(self.cfg.output_dir, "logs")

        accelerator_project_config = ProjectConfiguration(project_dir=self.cfg.output_dir, logging_dir=logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            log_with=self.cfg.logging.report_to,
            project_config=accelerator_project_config,
        )

        if self.accelerator.state.device.type == "cuda":
            torch.cuda.set_device(self.accelerator.state.device)

        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        ovg_run_name = os.environ.get("ovg_RUN_NAME")

        if slurm_job_id:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            config_name = hydra_cfg.job.config_name or "default"
            if config_name.endswith(".yaml"):
                config_name = config_name[:-5]
            run_name = f"{config_name}_{slurm_job_id}"
            if self.accelerator.is_main_process:
                self.logger.info(f"Using SLURM_JOB_ID for run_name: {run_name}")
        elif ovg_run_name:
            run_name = ovg_run_name
            if self.accelerator.is_main_process:
                self.logger.info(f"Using ovg_RUN_NAME from environment: {run_name}")
        else:
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            config_name = hydra_cfg.job.config_name or "default"
            if config_name.endswith(".yaml"):
                config_name = config_name[:-5]

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{config_name}_{timestamp}"

            if self.accelerator.is_main_process:
                self.logger.info(f"Generated local run_name: {run_name}")

        self.cfg.logging.run_name = run_name
        if self.accelerator.is_main_process:
            ckpt_dir = Path(self.cfg.output_dir) / "checkpoints" / run_name
            self.logger.info(f"Run name = {run_name}")
            self.logger.info(f"Checkpoint dir = {ckpt_dir}")

        if self.accelerator.is_main_process:
            config_dict = OmegaConf.to_container(self.cfg, resolve=True)
            self.accelerator.init_trackers(
                self.cfg.logging.project_name,
                config=config_dict,
                init_kwargs={"wandb": {"name": run_name}},
            )

        self.accelerator.wait_for_everyone()

        if self.cfg.seed is not None:
            set_seed(self.cfg.seed)
            if self.accelerator.num_processes > 1:
                self.logger.info(
                    f"Process {self.accelerator.process_index}/{self.accelerator.num_processes} initialized"
                )

    def setup_data(self):
        self.val_dataset = None
        self.val_dataloader = None
        if self.cfg.data.dataset == "bbbc021":
            self.train_dataset = BBBC021Dataset(
                csv_path=self.cfg.data.csv_path,
                hr_size=self.cfg.data.hr_size,
                scale_factor=self.cfg.data.scale_factor,
                augmentation=self.cfg.data.augmentation,
                paired_mode=self.cfg.data.paired_mode,
                custom_transform=hydra.utils.instantiate(self.cfg.transforms.train),
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
        elif self.cfg.data.dataset == "edges2shoes":
            self.train_dataset = Edges2ShoesDataset(
                csv_path=self.cfg.data.csv_path,
                root_dir=getattr(self.cfg.data, "root_dir", None),
                img_size=getattr(self.cfg.data, "img_size", getattr(self.cfg.data, "hr_size", 256)),
                augmentation=self.cfg.data.augmentation,
                paired_mode=self.cfg.data.paired_mode,
                custom_transform=hydra.utils.instantiate(self.cfg.transforms.train),
                mode=self.cfg.data.mode,
                class_indices=self.cfg.data.class_indices,
                normalize=getattr(self.cfg.data, "normalize", True),
                split=getattr(self.cfg.data, "split", None),
            )

        def _seed_worker(worker_id: int):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.data.num_workers,
            worker_init_fn=_seed_worker,
            pin_memory=True,
        )
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                shuffle=False,
                batch_size=self.cfg.training.batch_size,
                num_workers=self.cfg.data.num_workers,
                worker_init_fn=_seed_worker,
            )

    def setup_models(self):
        self.weight_dtype = setup_weight_dtype(self.accelerator)
        self.noise_scheduler, self.vae, self.denoiser, self.class_embedding = get_models(self.cfg)
        self.unet = self.denoiser

        if self.cfg.training.gradient_checkpointing:
            self.denoiser.enable_gradient_checkpointing()

        if getattr(self.cfg.model, "enable_xformers_memory_efficient_attention", False):
            if hasattr(self.denoiser, "enable_xformers_memory_efficient_attention"):
                self.denoiser.enable_xformers_memory_efficient_attention()
                self.logger.info("Enabled xformers memory efficient attention for denoiser")
            else:
                self.logger.warning(
                    "xformers is not available or model does not support it. "
                    "Proceeding without memory efficient attention."
                )

        if self.noise_scheduler is not None:
            self.noise_scheduler.set_timesteps(self.cfg.training.num_inference_steps)

        self.ema_denoiser = None
        self.ema_class_embedding = None
        if getattr(self.cfg.training, "use_ema", False):
            self.ema_denoiser = ExponentialMovingAverage(
                self.denoiser,
                decay=getattr(self.cfg.training, "ema_decay", 0.9999),
                min_decay=getattr(self.cfg.training, "ema_min_decay", 0.0),
            )
            if self.class_embedding is not None:
                self.ema_class_embedding = ExponentialMovingAverage(
                    self.class_embedding,
                    decay=getattr(self.cfg.training, "ema_decay", 0.9999),
                    min_decay=getattr(self.cfg.training, "ema_min_decay", 0.0),
                )
        if self.class_embedding is not None:
            self.condition_dtype = get_model_dtype(self.class_embedding)
            base_embedding = getattr(self.class_embedding, "module", self.class_embedding)
            self.null_class_label = getattr(base_embedding, "num_classes", None)
        else:
            self.condition_dtype = torch.long

            base_denoiser = unwrap_model(self.denoiser)
            num_class_embeds = getattr(getattr(base_denoiser, "config", object()), "num_class_embeds", None)
            if num_class_embeds is not None:
                num_classes = int(getattr(getattr(self.cfg, "data", object()), "num_classes", -1))
                if num_class_embeds > num_classes:
                    self.null_class_label = num_classes
                else:
                    self.null_class_label = -1
            else:
                num_classes = int(getattr(getattr(self.cfg, "data", object()), "num_classes", -1))
                if num_classes < 0:
                    raise ValueError("cfg.data.num_classes must be defined and non-negative for DiT models.")
                self.null_class_label = num_classes

        if self.null_class_label is None:
            self.null_class_label = int(getattr(getattr(self.cfg, "data", object()), "num_classes", -1))

        base_denoiser = unwrap_model(self.denoiser)
        if self.null_class_label is not None:
            setattr(base_denoiser, "_ovg_null_class_label", int(self.null_class_label))

    def setup_training(self):
        optimizer_param_groups, clip_params = setup_optimizer_params(self.denoiser, self.class_embedding, self.cfg)
        self.optimizer = torch.optim.AdamW(
            optimizer_param_groups,
            lr=self.cfg.optimizer.learning_rate,
            betas=(self.cfg.optimizer.adam_beta1, self.cfg.optimizer.adam_beta2),
            weight_decay=self.cfg.optimizer.adam_weight_decay,
            eps=self.cfg.optimizer.adam_epsilon,
        )

        total_batch_size = (
            self.cfg.training.batch_size
            * self.accelerator.num_processes
            * self.cfg.training.gradient_accumulation_steps
        )
        num_update_steps_per_epoch = math.ceil(len(self.train_dataset) / total_batch_size)

        self.num_update_steps_per_epoch = num_update_steps_per_epoch

        if self.cfg.training.max_num_steps is None:
            self.cfg.training.max_num_steps = self.cfg.training.num_epochs * num_update_steps_per_epoch
        else:
            self.cfg.training.num_epochs = math.ceil(self.cfg.training.max_num_steps / num_update_steps_per_epoch)
            self.logger.info(
                f"Adjusted num_epochs to {self.cfg.training.num_epochs} based on max_num_steps={self.cfg.training.max_num_steps}"
            )

        extra_sched_kwargs = {}
        if hasattr(self.cfg.optimizer, "num_cycles") and self.cfg.optimizer.num_cycles is not None:
            extra_sched_kwargs["num_cycles"] = int(self.cfg.optimizer.num_cycles)

        self.lr_scheduler = get_scheduler(
            self.cfg.optimizer.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.optimizer.lr_warmup_steps,
            num_training_steps=self.cfg.training.max_num_steps,
            **extra_sched_kwargs,
        )

        models_to_prepare = [
            self.denoiser,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ]

        if self.class_embedding is not None:
            models_to_prepare.insert(1, self.class_embedding)
        if self.val_dataloader is not None:
            models_to_prepare.append(self.val_dataloader)

        if self.vae is not None:
            self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
            self.vae.eval()
            self.vae.requires_grad_(False)

        self.denoiser.requires_grad_(True)

        prepared = self.accelerator.prepare(*models_to_prepare)

        if self.class_embedding is not None:
            if self.val_dataloader is not None:
                (
                    self.denoiser,
                    self.class_embedding,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                    self.val_dataloader,
                ) = prepared
            else:
                (
                    self.denoiser,
                    self.class_embedding,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                ) = prepared
        else:
            if self.val_dataloader is not None:
                (
                    self.denoiser,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                    self.val_dataloader,
                ) = prepared
            else:
                (
                    self.denoiser,
                    self.optimizer,
                    self.train_dataloader,
                    self.lr_scheduler,
                ) = prepared

        self.logger.info("Models prepared with success through Accelerator")

        if getattr(self.cfg.training, "use_ema", False):
            if self.ema_denoiser is not None:
                self.ema_denoiser.to(self.accelerator.device)
            if self.ema_class_embedding is not None:
                self.ema_class_embedding.to(self.accelerator.device)

        self.optimizer_params = clip_params

        checkpoint_dir = self.get_checkpoint_dir()
        latest_path = checkpoint_dir / "latest.pt"
        chosen_path = None
        if latest_path.exists():
            chosen_path = latest_path
        else:
            candidates = list(checkpoint_dir.glob("checkpoint-*.pt"))

            def _extract_step(p: Path) -> int:
                try:
                    return int(p.stem.split("-")[-1])
                except Exception:
                    return -1

            if candidates:
                candidates.sort(key=_extract_step)
                chosen_path = candidates[-1]

        def _try_load(path: Path) -> bool:
            if path is None or not path.exists():
                return False
            try:
                if self.accelerator.is_main_process:
                    self.logger.info(f"Loading checkpoint from {path}")
                load_training_checkpoint(path, self)
                if self.accelerator.is_main_process:
                    self.logger.info(f"Resumed from step {self.global_step}, epoch {self.current_epoch}")
                return True
            except Exception as e:
                if self.accelerator.is_main_process:
                    self.logger.warning(f"Failed to load checkpoint {path}: {e}")
                return False

        loaded = _try_load(chosen_path)
        if not loaded:
            candidates = list(checkpoint_dir.glob("checkpoint-*.pt"))

            def _extract_step2(p: Path) -> int:
                try:
                    return int(p.stem.split("-")[-1])
                except Exception:
                    return -1

            candidates.sort(key=_extract_step2, reverse=True)
            for cand in candidates:
                if _try_load(cand):
                    loaded = True
                    break

        if not loaded:
            if self.accelerator.is_main_process:
                self.logger.info(f"No valid checkpoints under {checkpoint_dir}. Starting from scratch.")

    def setup_logging(self):
        if self.accelerator.is_main_process:
            total_batch_size = (
                self.cfg.training.batch_size
                * self.accelerator.num_processes
                * self.cfg.training.gradient_accumulation_steps
            )

            self.logger.info("***** Running training *****")
            self.logger.info(f"------ Num GPUs detected by torch = {torch.cuda.device_count()}")
            self.logger.info(f"------ Num processes (Accelerate) = {self.accelerator.num_processes}")
            self.logger.info(f"------ Distributed type = {self.accelerator.distributed_type}")
            self.logger.info(f"------ Mixed precision = {self.accelerator.mixed_precision}")
            self.logger.info(f"------ Num examples = {len(self.train_dataset)}")
            self.logger.info(f"------ Num Epochs = {self.cfg.training.num_epochs}")
            self.logger.info(f"------ Batch size per device = {self.cfg.training.batch_size}")
            self.logger.info(f"------ Efficient batch size (w. distributed & accumulation) = {total_batch_size}")
            self.logger.info(f"------ Gradient Accumulation steps = {self.cfg.training.gradient_accumulation_steps}")
            self.logger.info(f"------ Total optimization steps = {self.cfg.training.max_num_steps}")
            self.logger.info(f"------ Learning rate = {self.cfg.optimizer.learning_rate}")
            self.logger.info(f"------ Noise offset = {self.cfg.training.noise_offset}")
            self.logger.info(f"------ Guidance scale = {self.cfg.training.guidance_scale}")
            self.logger.info(f"------ Seed = {self.cfg.seed}")
            self.logger.info(f"------ LoRA = {self.cfg.model.use_lora}")
            self.logger.info(f"------ LoRA rank = {self.cfg.model.lora_rank}")
            self.logger.info(f"------ Use EMA = {getattr(self.cfg.training, 'use_ema', False)}")
            if getattr(self.cfg.training, "use_ema", False):
                self.logger.info(f"------ EMA decay = {getattr(self.cfg.training, 'ema_decay', 0.9999)}")
                self.logger.info(f"------ EMA update after = {getattr(self.cfg.training, 'ema_update_after', 100)}")
            self.logger.info(f"------ Logging to: {self.cfg.output_dir}")

    def _apply_lr_floor(self):
        min_lr = getattr(self.cfg.optimizer, "min_lr", None)
        if not min_lr:
            return
        for g in self.optimizer.param_groups:
            if "lr" in g and g["lr"] < min_lr:
                g["lr"] = min_lr

    def get_checkpoint_dir(self):
        base = Path(self.cfg.output_dir)
        if not base.is_absolute():
            base = base.resolve()
        run_name = getattr(self.cfg.logging, "run_name", None)
        return (base / "checkpoints" / run_name) if run_name else (base / "checkpoints")

    def save_checkpoint_and_sync(self, reason: str = "periodic", force_latest: bool = True) -> Path:
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return None

        checkpoint_dir = self.get_checkpoint_dir()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        step_path = checkpoint_dir / f"checkpoint-{self.global_step}.pt"
        tmp_path = checkpoint_dir / f".checkpoint-{self.global_step}.pt.tmp"

        self.checkpoint_manager.save_training_checkpoint(str(tmp_path))
        tmp_path.rename(step_path)

        if force_latest:
            latest_tmp = checkpoint_dir / ".latest.pt.tmp"
            latest_path = checkpoint_dir / "latest.pt"
            self.checkpoint_manager.save_training_checkpoint(str(latest_tmp))
            latest_tmp.rename(latest_path)

        self.logger.info(
            f"Checkpoint saved ({reason}): {step_path.name} [step={self.global_step}, epoch={self.current_epoch}]"
        )

        self.accelerator.wait_for_everyone()
        return step_path

    def compute_loss(self, batch):
        if self.training_strategy is None:
            raise RuntimeError("Training strategy has not been initialized.")
        return self.training_strategy.compute_loss(batch)

    def train_step(self, batch):
        models_to_accumulate = [self.denoiser]
        if self.class_embedding is not None:
            models_to_accumulate.append(self.class_embedding)

        with self.accelerator.accumulate(models_to_accumulate):
            loss = self.compute_loss(batch)
            self.accelerator.backward(loss)
            step_incremented = False

            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(self.optimizer_params, self.cfg.training.max_grad_norm)
                self.global_step += 1
                step_incremented = True

                self.log_training_metrics(loss, grad_norm)

                if getattr(self.cfg.training, "use_ema", False):
                    if self.ema_denoiser is not None:
                        self.ema_denoiser.update(
                            self.accelerator.unwrap_model(self.denoiser),
                            self.global_step,
                        )
                    if self.ema_class_embedding is not None and self.class_embedding is not None:
                        self.ema_class_embedding.update(
                            self.accelerator.unwrap_model(self.class_embedding),
                            self.global_step,
                        )

            self.optimizer.step()
            self.lr_scheduler.step()
            self._apply_lr_floor()
            self.optimizer.zero_grad()

        return step_incremented

    def log_training_metrics(self, loss, grad_norm=None):
        unet_grad_norm, class_embedding_grad_norm = compute_gradient_norms(self.denoiser, self.class_embedding)

        current_lr = None
        try:
            if self.optimizer.param_groups:
                current_lr = float(self.optimizer.param_groups[0].get("lr", None))
        except Exception:
            current_lr = None

        log_dict = {
            "train/loss": loss.item(),
            "train/unet_grad_norm": unet_grad_norm,
            "train/class_embedding_grad_norm": class_embedding_grad_norm,
            "train/lr": current_lr if current_lr is not None else self.lr_scheduler.get_last_lr()[0],
        }

        if grad_norm is not None:
            log_dict["train/total_grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        if self.class_embedding is not None:
            device = next(self.class_embedding.parameters()).device
            try:
                class_embedding_0 = self.class_embedding(torch.tensor([0], device=device))
                class_embedding_1 = self.class_embedding(torch.tensor([1], device=device))
                cosine_sim = F.cosine_similarity(class_embedding_0, class_embedding_1, dim=-1).mean()
                log_dict["train/cosine_sim_class_embed"] = cosine_sim.item()
            except Exception as e:
                self.logger.warning(f"Error computing cosine similarity: {e}")

        if self.run_logger:
            self.run_logger.log(log_dict, step=self.global_step, epoch=self.current_epoch)

        log_dict["epoch"] = self.current_epoch
        self.accelerator.log(log_dict, step=self.global_step)

    def _check_and_handle_requeue(self) -> bool:
        checkpoint_dir = self.get_checkpoint_dir()
        request_file = checkpoint_dir / ".requeue_request"

        if self.accelerator.is_main_process:
            self.logger.debug(f"[REQUEUE CHECK] checkpoint_dir = {checkpoint_dir}")
            self.logger.debug(f"[REQUEUE CHECK] request_file = {request_file}")
            self.logger.debug(f"[REQUEUE CHECK] request_file.exists() = {request_file.exists()}")

        if not request_file.exists():
            return False

        self.logger.warning(f"Submitit requeue detected - saving checkpoint at step {self.global_step}")
        self.save_checkpoint_and_sync(reason="submitit_requeue")

        if self.accelerator.is_main_process:
            done_file = checkpoint_dir / ".requeue_done"
            try:
                done_file.write_text(str(self.global_step))
                request_file.unlink()
                self.logger.info(f"Requeue handshake completed at step {self.global_step}")
            except Exception as e:
                self.logger.warning(f"Failed to complete requeue handshake: {e}")

        return True

    def train_epoch(self, epoch):
        self.current_epoch = epoch
        self.denoiser.train()
        if self.class_embedding is not None:
            self.class_embedding.train()

        if self.cfg.seed is not None:
            set_seed(int(self.cfg.seed) + int(epoch))

        batches_to_skip = 0
        if getattr(self, "num_update_steps_per_epoch", None):
            if self.global_step > 0:
                steps_done_in_epoch = self.global_step % self.num_update_steps_per_epoch
                if steps_done_in_epoch > 0:
                    batches_to_skip = steps_done_in_epoch * self.cfg.training.gradient_accumulation_steps
                    if self.accelerator.is_main_process:
                        self.logger.info(
                            f"Resuming epoch {epoch}: skipping {batches_to_skip} batches "
                            f"({steps_done_in_epoch} update steps already completed in this epoch)"
                        )

        skipped_batches = 0

        for _, batch in tqdm(
            enumerate(self.train_dataloader),
            desc="Steps",
            leave=False,
            total=len(self.train_dataloader),
            initial=min(batches_to_skip, len(self.train_dataloader)),
            colour="#83ABF7",
        ):
            self._check_and_handle_requeue()
            if skipped_batches < batches_to_skip:
                skipped_batches += 1
                continue
            step_incremented = self.train_step(batch)
            self._check_and_handle_requeue()

            if step_incremented and self.global_step % self.cfg.logging.checkpointing_steps == 0:
                self.save_checkpoint_and_sync(reason="periodic_steps")

            if self.global_step % 1000 == 0:
                try:
                    use_ema = getattr(self.cfg.training, "use_ema", False)
                    is_dit = self.class_embedding is None

                    if use_ema and self.ema_denoiser is not None:
                        if is_dit:
                            with EMAModelWrapper(self.denoiser, self.ema_denoiser) as ema_unet:
                                generate_and_log_grids(
                                    self.vae,
                                    ema_unet,
                                    None,
                                    self.noise_scheduler,
                                    self.train_dataloader,
                                    self.cfg,
                                    self.accelerator,
                                    self.global_step,
                                    n_samples=8,
                                    n_variations=1,
                                    null_class_label=self.null_class_label,
                                    run_logger=self.run_logger,
                                )
                        elif self.ema_class_embedding is not None:
                            with (
                                EMAModelWrapper(self.denoiser, self.ema_denoiser) as ema_unet,
                                EMAModelWrapper(self.class_embedding, self.ema_class_embedding) as ema_class_embedding,
                            ):
                                generate_and_log_grids(
                                    self.vae,
                                    ema_unet,
                                    ema_class_embedding,
                                    self.noise_scheduler,
                                    self.train_dataloader,
                                    self.cfg,
                                    self.accelerator,
                                    self.global_step,
                                    n_samples=8,
                                    n_variations=1,
                                    null_class_label=self.null_class_label,
                                    run_logger=self.run_logger,
                                )
                        else:
                            generate_and_log_grids(
                                self.vae,
                                self.denoiser,
                                self.class_embedding,
                                self.noise_scheduler,
                                self.train_dataloader,
                                self.cfg,
                                self.accelerator,
                                self.global_step,
                                n_samples=8,
                                n_variations=1,
                                null_class_label=self.null_class_label,
                                run_logger=self.run_logger,
                            )
                    else:
                        generate_and_log_grids(
                            self.vae,
                            self.denoiser,
                            self.class_embedding,
                            self.noise_scheduler,
                            self.train_dataloader,
                            self.cfg,
                            self.accelerator,
                            self.global_step,
                            n_samples=8,
                            n_variations=1,
                            null_class_label=self.null_class_label,
                            run_logger=self.run_logger,
                        )

                    self.accelerator.wait_for_everyone()

                except Exception as e:
                    self.logger.error("=" * 80)
                    self.logger.error(f"❌ ERROR during image generation at step {self.global_step}: {e}")
                    self.logger.error("=" * 80)
                    import traceback

                    self.logger.error(traceback.format_exc())
                    self.logger.warning("Continuing training despite generation error...")

            if self.global_step >= self.cfg.training.max_num_steps:
                break

    def validate(self, epoch):
        if hasattr(self.cfg.evaluation, "val_interval") and epoch % self.cfg.evaluation.val_interval == 0:
            try:
                use_ema = getattr(self.cfg.training, "use_ema", False)
                if use_ema and self.ema_denoiser and self.ema_class_embedding:
                    with (
                        EMAModelWrapper(self.denoiser, self.ema_denoiser) as ema_unet,
                        EMAModelWrapper(self.class_embedding, self.ema_class_embedding) as ema_class_embedding,
                    ):
                        generate_and_log_grids(
                            self.vae,
                            ema_unet,
                            ema_class_embedding,
                            self.noise_scheduler,
                            self.train_dataloader,
                            self.cfg,
                            self.accelerator,
                            self.global_step,
                            null_class_label=self.null_class_label,
                            run_logger=self.run_logger,
                        )
                else:
                    generate_and_log_grids(
                        self.vae,
                        self.denoiser,
                        self.class_embedding,
                        self.noise_scheduler,
                        self.train_dataloader,
                        self.cfg,
                        self.accelerator,
                        self.global_step,
                        null_class_label=self.null_class_label,
                        run_logger=self.run_logger,
                    )

                mcfg = getattr(self.cfg, "metrics", None)
                if mcfg and getattr(mcfg, "enabled", True):
                    try:
                        bm = eval_binned_metrics(
                            vae=self.vae,
                            denoiser=self.denoiser,
                            class_embedding=self.class_embedding,
                            noise_scheduler=self.noise_scheduler,
                            dataloader=self.train_dataloader,
                            device=self.accelerator.device,
                            n_bins=int(getattr(mcfg, "n_bins", 20)),
                            max_batches=int(getattr(mcfg, "max_batches", 10)),
                            prediction_type=getattr(
                                mcfg,
                                "prediction_type",
                                getattr(
                                    self.noise_scheduler.config,
                                    "prediction_type",
                                    "epsilon",
                                ),
                            ),
                            compute_x0_mse=bool(getattr(mcfg, "compute_x0_mse", True)),
                            compute_class_flip=bool(getattr(mcfg, "compute_class_flip", True)),
                            compute_cond_influence=bool(getattr(mcfg, "compute_cond_influence", True)),
                            compute_ema_delta=False,
                        )

                        metric_key = "eval/mse_eps"

                        payload = {
                            "eval/bins/mid_t": bm.bins_mid_t.tolist(),
                            f"{metric_key}/all": bm.mse_eps_all.tolist(),
                            f"{metric_key}/class_0": bm.mse_eps_class_0.tolist(),
                            f"{metric_key}/class_1": bm.mse_eps_class_1.tolist(),
                            "eval/eps2/mean_all": bm.eps2_mean_all.tolist(),
                            "eval/eps2/var_all": bm.eps2_var_all.tolist(),
                            "eval/eps2/hist_edges": bm.eps2_hist_edges.tolist(),
                            "eval/eps2/hist_counts": bm.eps2_hist_counts.tolist(),
                        }

                        if bm.mse_x0_all is not None:
                            payload["eval/mse_x0/all"] = bm.mse_x0_all.tolist()
                        if bm.class_flip_delta is not None:
                            payload["eval/delta/class_flip"] = bm.class_flip_delta.tolist()
                        if bm.cond_influence_ratio is not None:
                            payload["eval/cond_influence/ratio"] = bm.cond_influence_ratio.tolist()
                        if bm.failure_rate_lastbin is not None:
                            payload["eval/failure_rate_lastbin"] = bm.failure_rate_lastbin

                        if self.run_logger:
                            self.run_logger.log(
                                payload,
                                step=self.global_step,
                                epoch=epoch,
                            )

                            figs = render_binned_plots(bm, title_prefix="Eval")
                            for fig_key, fig in figs.items():
                                self.run_logger.log_image(
                                    fig_key,
                                    fig,
                                    step=self.global_step,
                                    epoch=epoch,
                                )
                                try:
                                    import matplotlib.pyplot as plt

                                    plt.close(fig)
                                except Exception:
                                    pass

                        self.accelerator.log(payload, step=self.global_step)

                    except Exception as e:
                        self.logger.warning(f"[metrics] binned eval skipped: {e}")

            except Exception as e:
                self.logger.error("=" * 80)
                self.logger.error(f"❌ ERROR during validation at epoch {epoch}: {e}")
                self.logger.error("=" * 80)
                import traceback

                self.logger.error(traceback.format_exc())
                self.logger.warning("Continuing training despite validation error...")

        if (
            self.val_dataloader is not None
            and hasattr(self.cfg.evaluation, "val_interval")
            and self.cfg.evaluation.val_interval > 0
            and epoch % self.cfg.evaluation.val_interval == 0
        ):
            self.logger.info(f"Starting FID evaluation at epoch {epoch}...")

            unet_to_eval = self.ema_denoiser.ema_model if self.ema_denoiser else self.denoiser
            gene_encoder_to_eval = (
                self.ema_class_embedding.ema_model if self.ema_class_embedding else self.class_embedding
            )

            try:
                metrics = evaluate_quality(
                    vae=self.vae,
                    unet=unet_to_eval,
                    gene_encoder=gene_encoder_to_eval,
                    scheduler=self.noise_scheduler,
                    dataloader=self.val_dataloader,
                    device=self.accelerator.device,
                    accelerator=self.accelerator,
                    num_samples=getattr(self.cfg.evaluation, "num_samples", 10000),
                    guidance_scale=getattr(self.cfg.training, "guidance_scale", 7.5),
                    image_resolution=getattr(self.cfg.data, "hr_size", 256),
                    null_class_label=self.null_class_label,
                )

                if self.accelerator.is_main_process and metrics:
                    (
                        fid,
                        is_mean,
                        is_std,
                        kid_mean,
                        kid_std,
                        ssim,
                        psnr,
                        precision,
                        recall,
                    ) = metrics

                    self.logger.info(f"FID: {fid:.4f}, IS: {is_mean:.4f}, KID: {kid_mean:.4f}")

                    log_dict = {
                        "eval/fid": fid,
                        "eval/is_mean": is_mean,
                        "eval/is_std": is_std,
                        "eval/kid_mean": kid_mean,
                        "eval/kid_std": kid_std,
                        "eval/ssim": ssim,
                        "eval/psnr": psnr,
                        "eval/precision": precision,
                        "eval/recall": recall,
                        "epoch": epoch,
                    }

                    if self.run_logger:
                        self.run_logger.log(
                            log_dict,
                            step=self.global_step,
                            epoch=epoch,
                        )

                    self.accelerator.log(log_dict, step=self.global_step)

            except Exception as e:
                self.logger.error(f"Error during FID evaluation: {e}")
                import traceback

                traceback.print_exc()

    def train(self):
        start_epoch = int(getattr(self, "current_epoch", 0))

        if getattr(self, "num_update_steps_per_epoch", None):
            if self.global_step > 0 and (self.global_step % self.num_update_steps_per_epoch == 0):
                start_epoch = min(start_epoch + 1, self.cfg.training.num_epochs - 1)

        for epoch in tqdm(
            range(start_epoch, self.cfg.training.num_epochs),
            desc="Epoch",
            leave=True,
            total=self.cfg.training.num_epochs,
            initial=start_epoch,
            colour="#EE7A3E",
        ):
            self.train_epoch(epoch)
            self.validate(epoch)
            if self.global_step >= self.cfg.training.max_num_steps:
                break

        if self.accelerator.is_main_process:
            self.logger.info("Saving final model...")
            checkpoint_path = self.get_checkpoint_dir() / "checkpoint-final.pt"
            self.checkpoint_manager.save_training_checkpoint(checkpoint_path)
            self.logger.info(f"Final checkpoint saved: {checkpoint_path}")

        if self.run_logger:
            self.run_logger.close()
        self.accelerator.end_training()
