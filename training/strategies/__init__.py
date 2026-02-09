from typing import TYPE_CHECKING

from ovg.training.strategies.diffusion import DiffusionTrainingStrategy
from ovg.training.strategies.edm import EDMTrainingStrategy

if TYPE_CHECKING:
    from ovg.trainer import Trainer


def build_training_strategy(trainer: "Trainer"):
    mode = getattr(trainer.cfg.training, "training_mode", "diffusion").lower()

    if mode == "diffusion":
        return DiffusionTrainingStrategy(trainer)

    elif mode == "edm":
        return EDMTrainingStrategy(trainer)
    else:
        raise ValueError(
            f"Unsupported training mode: {mode}. Expected 'diffusion' or 'edm'."
        )
