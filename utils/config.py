from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import logging
from hydra.core.hydra_config import HydraConfig

from ovg.utils.helpers import omegaconf_select

log = logging.getLogger(__name__)


def setup_config(cfg: DictConfig = None):
    if cfg is None:
        from hydra.core.global_hydra import GlobalHydra

        cfg = GlobalHydra.instance().cfg

    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name or ""
    if "/" in config_name:
        parts = [p for p in config_name.split("/") if p]

        hierarchy = parts[:-1]
        candidate = cfg
        for key in hierarchy:
            if isinstance(candidate, DictConfig) and key in candidate:
                candidate = candidate[key]
            else:
                candidate = None
                break
        if isinstance(candidate, DictConfig):
            cfg = candidate

            try:
                cli_overrides = HydraConfig.get().overrides.task
                if cli_overrides:
                    OmegaConf.set_struct(cfg, False)
                    for override in cli_overrides:
                        if "=" in override:
                            key_part, value_part = override.split("=", 1)

                            key = key_part.lstrip("+")

                            try:
                                import yaml

                                parsed_value = yaml.safe_load(value_part)
                            except Exception:
                                parsed_value = value_part

                            OmegaConf.update(cfg, key, parsed_value, merge=False)
                    OmegaConf.set_struct(cfg, True)
            except Exception as e:
                log.warning(
                    "Failed to re-apply CLI overrides to flattened config: %s. Using flattened config as-is.",
                    e,
                )
        else:
            log.warning(
                "Unable to flatten config hierarchy for '%s' (keys missing). Continuing with original config.",
                config_name,
            )

    run_dir = HydraConfig.get().run.dir
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    cfg.run_dir = omegaconf_select(cfg, "run_dir", default=run_dir)

    cfg.model.conditioning_mode = omegaconf_select(
        cfg.model, "conditioning_mode", default="cross_attention"
    )
    cfg.training.training_mode = omegaconf_select(
        cfg.training, "training_mode", default="diffusion"
    )

    if hasattr(cfg, "inference"):
        cfg.inference.output_dir = Path(run_dir) / "results"
        cfg.inference.output_dir.mkdir(parents=True, exist_ok=True)

        cfg.inference.input_dir = Path(run_dir) / "test_images/lr"
        cfg.inference.input_dir.mkdir(parents=True, exist_ok=True)

        cfg.inference.input_path = Path(run_dir) / "test_images/lr/test_image.png"

    try:
        if hasattr(cfg, "data") and hasattr(cfg.data, "csv_path"):
            log.info(f"Using data.csv_path: {cfg.data.csv_path}")
    except Exception:
        pass

    return cfg
