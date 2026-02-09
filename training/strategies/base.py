from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ovg.trainer import Trainer


@dataclass
class ConditioningBatch:
    class_embeddings: torch.Tensor
    unconditional_embeddings: torch.Tensor
    original_labels: torch.Tensor
    dropped_labels: torch.Tensor


class BaseTrainingStrategy:
    def __init__(self, trainer: "Trainer") -> None:
        self.trainer = trainer

    @property
    def cfg(self):
        return self.trainer.cfg

    @property
    def accelerator(self):
        return self.trainer.accelerator

    def _extract_class_labels(self, batch) -> torch.Tensor:
        cfg = self.cfg
        condition_dtype = self.trainer.condition_dtype

        device = self.device

        if cfg.data.mode == "super_resolution" and "res_class" in batch:
            class_labels = batch["res_class"].to(device=device, dtype=condition_dtype)
        elif (
            cfg.data.mode in ["conditional", "segmentation", "translation"]
            and "cond_class" in batch
        ):
            class_labels = batch["cond_class"].to(device=device, dtype=condition_dtype)
        elif "class_idx" in batch:
            class_labels = batch["class_idx"].to(device=device, dtype=condition_dtype)
        else:
            raise KeyError(
                f"Appropriate class labels not found in batch for mode '{cfg.data.mode}'"
            )

        return class_labels

    def _prepare_condition_embeddings(
        self, class_labels: torch.Tensor
    ) -> ConditioningBatch:
        class_embedding_module = self.trainer.class_embedding

        if class_embedding_module is None:
            drop_prob = float(getattr(self.cfg.training, "p_uncond", 0.0))
            drop_mask = (
                torch.rand(class_labels.shape[0], device=class_labels.device)
                < drop_prob
            )

            class_labels_dropped = class_labels.clone()
            uncond_label = getattr(self.trainer, "null_class_label", None)

            cfg_supported = uncond_label is not None and int(uncond_label) >= 0

            if drop_prob > 0.0 and not cfg_supported:
                raise RuntimeError(
                    "Classifier-free guidance (p_uncond > 0) requires an unconditional class label. "
                    "For UNet2DModel with num_class_embeds, ensure num_class_embeds > num_classes "
                    "to reserve one embedding for the unconditional case."
                )

            if drop_mask.any() and cfg_supported:
                class_labels_dropped[drop_mask] = uncond_label

            fill_value = int(uncond_label) if cfg_supported else -1
            unconditional_embeddings = torch.full_like(
                class_labels, fill_value=fill_value, dtype=torch.long
            )

            return ConditioningBatch(
                class_embeddings=class_labels_dropped.long(),
                unconditional_embeddings=unconditional_embeddings,
                original_labels=class_labels,
                dropped_labels=class_labels_dropped,
            )

        num_classes = getattr(
            class_embedding_module, "module", class_embedding_module
        ).num_classes
        max_class_idx = num_classes - 1

        if class_labels.max() >= num_classes:
            raise ValueError(
                f"Class label {class_labels.max().item()} >= num_classes {num_classes}. "
                f"This would collide with the unconditional embedding at index {num_classes}!"
            )

        class_labels = torch.clamp(class_labels, 0, max_class_idx)

        drop_prob = float(getattr(self.cfg.training, "p_uncond", 0.0))
        drop_mask = (
            torch.rand(class_labels.shape[0], device=class_labels.device) < drop_prob
        )

        class_labels_dropped = class_labels.clone()
        if drop_mask.any():
            class_labels_dropped[drop_mask] = num_classes

        class_embeddings = class_embedding_module(
            class_labels_dropped.to(device=self.device)
        )

        special_class_idx = num_classes
        uncond_labels = torch.full_like(class_labels, fill_value=special_class_idx)
        unconditional_embeddings = class_embedding_module(
            uncond_labels.to(device=self.device)
        )

        return ConditioningBatch(
            class_embeddings=class_embeddings,
            unconditional_embeddings=unconditional_embeddings,
            original_labels=class_labels,
            dropped_labels=class_labels_dropped,
        )

    @property
    def device(self):
        return self.trainer.accelerator.device
