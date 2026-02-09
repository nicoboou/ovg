import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import dataclasses
import datetime
import torch
from peft import get_peft_model_state_dict, PeftModel
from omegaconf import DictConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    version: str = "3.0"
    created_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )
    global_step: int = 0
    epoch: int = 0
    use_lora: Optional[bool] = None
    use_ema: Optional[bool] = None
    model_space: Optional[str] = None
    model_class_name: Optional[str] = None


@dataclass
class CheckpointSchema:
    metadata: CheckpointMetadata
    config: Optional[Any] = None

    denoiser_state_dict: Optional[Dict[str, torch.Tensor]] = None
    class_embedding_state_dict: Optional[Dict[str, torch.Tensor]] = None
    vae_state_dict: Optional[Dict[str, torch.Tensor]] = None

    ema_denoiser_state_dict: Optional[Dict[str, torch.Tensor]] = None
    ema_class_embedding_state_dict: Optional[Dict[str, torch.Tensor]] = None

    optimizer_state_dict: Optional[Dict[str, Any]] = None
    scheduler_state_dict: Optional[Dict[str, Any]] = None


class CheckpointManager:
    def __init__(self, trainer=None):
        self.trainer = trainer

    def _safe_load_metadata(self, meta: Any) -> CheckpointMetadata:
        if isinstance(meta, CheckpointMetadata):
            return meta
        if isinstance(meta, dict):
            known_keys = {f.name for f in dataclasses.fields(CheckpointMetadata)}
            filtered = {k: v for k, v in meta.items() if k in known_keys}
            return CheckpointMetadata(**filtered)

        try:
            meta_as_dict = dataclasses.asdict(meta)
            known_keys = {f.name for f in dataclasses.fields(CheckpointMetadata)}
            filtered = {k: v for k, v in meta_as_dict.items() if k in known_keys}
            return CheckpointMetadata(**filtered)
        except Exception:
            raise ValueError(
                "Unsupported metadata format in checkpoint: expected dict or CheckpointMetadata"
            )

    def _get_model_state_dict(
        self, model: torch.nn.Module, use_lora: bool
    ) -> Dict[str, torch.Tensor]:
        if use_lora:
            if isinstance(model, PeftModel):
                return get_peft_model_state_dict(model)
            else:
                logger.warning(
                    "use_lora=True mais le modèle n'est pas un PeftModel. Sauvegarde du state_dict complet."
                )
                return model.state_dict()
        else:
            return model.state_dict()

    def save_training_checkpoint(self, output_path: str):
        if not self.trainer:
            raise ValueError(
                "Un 'trainer' doit être fourni au CheckpointManager pour la sauvegarde."
            )

        trainer = self.trainer
        cfg = trainer.cfg
        use_lora = getattr(cfg.model, "use_lora", False)
        use_ema = getattr(cfg.training, "use_ema", False)

        unwrapped_denoiser = trainer.accelerator.unwrap_model(trainer.denoiser)
        denoiser_sd = self._get_model_state_dict(unwrapped_denoiser, use_lora)

        ema_denoiser_sd = None
        if use_ema and trainer.ema_denoiser:
            ema_denoiser_sd = self._get_model_state_dict(
                trainer.ema_denoiser.ema_model, use_lora
            )

        class_embedding_sd = None
        ema_class_embedding_sd = None
        if trainer.class_embedding:
            unwrapped_embed = trainer.accelerator.unwrap_model(trainer.class_embedding)
            class_embedding_sd = self._get_model_state_dict(unwrapped_embed, False)
            if use_ema and trainer.ema_class_embedding:
                ema_class_embedding_sd = self._get_model_state_dict(
                    trainer.ema_class_embedding.ema_model, False
                )

        metadata = CheckpointMetadata(
            global_step=trainer.global_step,
            epoch=trainer.current_epoch,
            use_lora=use_lora,
            use_ema=use_ema,
            model_space=getattr(cfg.model, "space", None),
            model_class_name=unwrapped_denoiser.__class__.__name__,
        )

        schema = CheckpointSchema(
            metadata=metadata,
            config=cfg,
            denoiser_state_dict=denoiser_sd,
            class_embedding_state_dict=class_embedding_sd,
            ema_denoiser_state_dict=ema_denoiser_sd,
            ema_class_embedding_state_dict=ema_class_embedding_sd,
            optimizer_state_dict=trainer.optimizer.state_dict(),
            scheduler_state_dict=trainer.lr_scheduler.state_dict(),
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {k: v for k, v in schema.__dict__.items() if v is not None}
        if isinstance(payload.get("metadata"), CheckpointMetadata):
            payload["metadata"] = dataclasses.asdict(payload["metadata"])
        tmp_path = f"{output_path}.tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, output_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
        logger.info(f"Checkpoint de training sauvegardé : {output_path}")

    def load_training_checkpoint(self, checkpoint_path: str) -> int:
        if not self.trainer:
            raise ValueError(
                "Un 'trainer' doit être fourni au CheckpointManager pour le chargement."
            )

        logger.info(f"Chargement du checkpoint de training: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer = self.trainer

        if "metadata" not in checkpoint:
            logger.warning(
                "Ancien format de checkpoint (v1) détecté. Migration en cours..."
            )
            schema = self._migrate_legacy_checkpoint(checkpoint, trainer.cfg)
        else:
            metadata = self._safe_load_metadata(checkpoint["metadata"])

            if "denoiser_state_dict" in checkpoint:
                logger.info("Format de checkpoint v3 détecté.")
                denoiser_sd = checkpoint.get("denoiser_state_dict")
                class_embed_sd = checkpoint.get("class_embedding_state_dict")
                vae_sd = checkpoint.get("vae_state_dict")
                ema_denoiser_sd = checkpoint.get("ema_denoiser_state_dict")
                ema_class_embed_sd = checkpoint.get("ema_class_embedding_state_dict")

            elif "model_state_dict" in checkpoint:
                logger.info("Format de checkpoint v2 détecté. Migration...")
                model_sd = checkpoint.get("model_state_dict", {})
                vae_sd, denoiser_sd, class_embed_sd, _ = self._legacy_parse_state_dicts(
                    model_sd
                )

                ema_denoiser_sd = None
                ema_class_embed_sd = None
                if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
                    ema_sd = checkpoint.get("ema_state_dict", {})
                    _, ema_denoiser_sd, ema_class_embed_sd, _ = (
                        self._legacy_parse_state_dicts(ema_sd)
                    )

            else:
                raise ValueError(
                    "Format de checkpoint inconnu. Ni 'denoiser_state_dict' ni 'model_state_dict' n'ont été trouvés."
                )

        schema = CheckpointSchema(
            metadata=metadata,
            config=checkpoint.get("config"),
            denoiser_state_dict=denoiser_sd,
            class_embedding_state_dict=class_embed_sd,
            vae_state_dict=vae_sd,
            ema_denoiser_state_dict=ema_denoiser_sd,
            ema_class_embedding_state_dict=ema_class_embed_sd,
            optimizer_state_dict=checkpoint.get("optimizer_state_dict"),
            scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
        )

        if schema.denoiser_state_dict:
            trainer.accelerator.unwrap_model(trainer.denoiser).load_state_dict(
                schema.denoiser_state_dict,
                strict=False,
            )

        if trainer.class_embedding and schema.class_embedding_state_dict:
            trainer.accelerator.unwrap_model(trainer.class_embedding).load_state_dict(
                schema.class_embedding_state_dict
            )

        if schema.ema_denoiser_state_dict and trainer.ema_denoiser:
            trainer.ema_denoiser.ema_model.load_state_dict(
                schema.ema_denoiser_state_dict, strict=False
            )
        if schema.ema_class_embedding_state_dict and trainer.ema_class_embedding:
            trainer.ema_class_embedding.ema_model.load_state_dict(
                schema.ema_class_embedding_state_dict
            )

        if schema.optimizer_state_dict:
            trainer.optimizer.load_state_dict(schema.optimizer_state_dict)
        if schema.scheduler_state_dict:
            trainer.lr_scheduler.load_state_dict(schema.scheduler_state_dict)

        trainer.global_step = schema.metadata.global_step
        trainer.current_epoch = schema.metadata.epoch

        logger.info(
            f"Reprise depuis step {trainer.global_step}, epoch {trainer.current_epoch}"
        )
        return trainer.global_step


def get_last_step(metrics_file: Path) -> int:
    if not metrics_file.exists():
        return 0
    last_step = 0
    with open(metrics_file) as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                step = data.get("step", 0)
                last_step = max(last_step, int(step))
            except (json.JSONDecodeError, ValueError):
                continue
    return last_step


def load_training_checkpoint(checkpoint_path: Path, trainer):
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    manager = CheckpointManager(trainer=trainer)
    global_step = manager.load_training_checkpoint(str(checkpoint_path))
    logger.info(
        f"Resumed from step {trainer.global_step}, epoch {trainer.current_epoch}"
    )
    return global_step

    def load_model_weights_from_checkpoint(
        self,
        checkpoint_path: str,
        cfg: DictConfig,
        denoiser: torch.nn.Module,
        vae: Optional[torch.nn.Module],
        class_embedding: Optional[torch.nn.Module],
        scheduler: Any,
    ):
        logger.info(f"Chargement des poids du modèle depuis: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "metadata" not in checkpoint:
            logger.warning(
                "Ancien format de checkpoint (v1) détecté. Migration en cours..."
            )
            schema = self._migrate_legacy_checkpoint(checkpoint, cfg)
        else:
            metadata = self._safe_load_metadata(checkpoint["metadata"])

            if "denoiser_state_dict" in checkpoint:
                logger.info("Format de checkpoint v3 détecté.")
                denoiser_sd = checkpoint.get("denoiser_state_dict")
                class_embed_sd = checkpoint.get("class_embedding_state_dict")
                vae_sd = checkpoint.get("vae_state_dict")
                ema_denoiser_sd = checkpoint.get("ema_denoiser_state_dict")
                ema_class_embed_sd = checkpoint.get("ema_class_embedding_state_dict")

            elif "model_state_dict" in checkpoint:
                logger.info("Format de checkpoint v2 détecté. Migration...")
                model_sd = checkpoint.get("model_state_dict", {})
                vae_sd, denoiser_sd, class_embed_sd, _ = self._legacy_parse_state_dicts(
                    model_sd
                )

                ema_denoiser_sd = None
                ema_class_embed_sd = None
                if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"]:
                    ema_sd = checkpoint.get("ema_state_dict", {})
                    _, ema_denoiser_sd, ema_class_embed_sd, _ = (
                        self._legacy_parse_state_dicts(ema_sd)
                    )

            else:
                raise ValueError(
                    "Format de checkpoint inconnu. Ni 'denoiser_state_dict' ni 'model_state_dict' n'ont été trouvés."
                )

            schema = CheckpointSchema(
                metadata=metadata,
                config=checkpoint.get("config"),
                denoiser_state_dict=denoiser_sd,
                class_embedding_state_dict=class_embed_sd,
                vae_state_dict=vae_sd,
                ema_denoiser_state_dict=ema_denoiser_sd,
                ema_class_embedding_state_dict=ema_class_embed_sd,
            )

        use_ema = schema.metadata.use_ema

        denoiser_weights = (
            schema.ema_denoiser_state_dict
            if use_ema and schema.ema_denoiser_state_dict
            else schema.denoiser_state_dict
        )
        class_embed_weights = (
            schema.ema_class_embedding_state_dict
            if use_ema and schema.ema_class_embedding_state_dict
            else schema.class_embedding_state_dict
        )
        vae_weights = schema.vae_state_dict

        if use_ema and schema.ema_denoiser_state_dict:
            logger.info("Utilisation des poids EMA pour le chargement.")
        else:
            logger.info("Utilisation des poids de modèle standard pour le chargement.")

        if denoiser_weights:
            denoiser.load_state_dict(denoiser_weights, strict=False)
            logger.info(f"Poids du Denoiser chargés ({len(denoiser_weights)} clés).")
        else:
            logger.warning("Aucun poids de denoiser trouvé dans le checkpoint.")

        if class_embedding and class_embed_weights:
            class_embedding.load_state_dict(class_embed_weights)
            logger.info(
                f"Poids de Class Embedding chargés ({len(class_embed_weights)} clés)."
            )

        if vae and vae_weights:
            vae.load_state_dict(vae_weights)
            logger.info(f"Poids du VAE chargés ({len(vae_weights)} clés).")

        loaded_cfg = schema.config if schema.config else cfg

        return denoiser, vae, class_embedding, scheduler, loaded_cfg

    def _migrate_legacy_checkpoint(
        self, checkpoint: Dict, cfg: Any
    ) -> CheckpointSchema:
        logger.info("Migration d'un ancien format de checkpoint...")
        use_ema = getattr(cfg.training, "use_ema", False)

        ema_weights = None
        if use_ema:
            if "ema_state_dict" in checkpoint:
                logger.info("Migration: Trouvé 'ema_state_dict' (format final v2).")
                ema_weights = checkpoint["ema_state_dict"]
            elif "ema_unet" in checkpoint:
                logger.info("Migration: Trouvé 'ema_unet' (format requeue v1).")

                ema_weights = {
                    "unet." + k: v for k, v in checkpoint["ema_unet"].items()
                }
                if "ema_class_embedding" in checkpoint:
                    for k, v in checkpoint["ema_class_embedding"].items():
                        ema_weights["class_embed." + k] = v
            elif "ema_unet_state" in checkpoint:
                logger.info("Migration: Trouvé 'ema_unet_state' (format requeue v2).")

                try:
                    ema_state = checkpoint["ema_unet_state"]

                    unet_param_names = [
                        k for k, v in self.trainer.denoiser.named_parameters()
                    ]
                    shadow_params = ema_state["shadow_params"]
                    if len(unet_param_names) == len(shadow_params):
                        ema_weights = {
                            "unet." + name: param
                            for name, param in zip(unet_param_names, shadow_params)
                        }
                        logger.info(
                            f"Migration: 'ema_unet_state' mappé avec succès ({len(ema_weights)} clés)."
                        )
                    else:
                        logger.warning(
                            "Migration: 'ema_unet_state' shadow_params ne correspond pas aux noms de paramètres. Ignoré."
                        )
                except Exception as e:
                    logger.warning(
                        f"Migration: Échec du mappage de 'ema_unet_state': {e}. Ignoré."
                    )

        non_ema_weights = None
        if "model_state_dict" in checkpoint:
            logger.info("Migration: Trouvé 'model_state_dict' (format v2).")
            non_ema_weights = checkpoint["model_state_dict"]
        elif "unet_state_dict" in checkpoint:
            logger.info("Migration: Trouvé 'unet_state_dict' (format requeue v2).")
            non_ema_weights = {
                f"unet.{k}": v for k, v in checkpoint["unet_state_dict"].items()
            }
            if "class_embedding_state_dict" in checkpoint:
                for k, v in checkpoint["class_embedding_state_dict"].items():
                    non_ema_weights[f"class_embed.{k}"] = v

        weights_to_parse = (
            ema_weights if use_ema and ema_weights is not None else non_ema_weights
        )

        if weights_to_parse is None:
            raise ValueError(
                "Migration a échoué: Impossible de trouver des poids de modèle valides (ni EMA, ni standard)."
            )

        if use_ema and ema_weights is not None:
            logger.info("Utilisation des poids EMA pour la migration.")
        else:
            logger.info(
                "Utilisation des poids non-EMA pour la migration (EMA non trouvé ou non activé)."
            )

        vae_sd, unet_sd, class_sd, _ = self._legacy_parse_state_dicts(weights_to_parse)

        ema_denoiser_sd, ema_class_embed_sd = None, None
        if use_ema and ema_weights:
            _, ema_denoiser_sd, ema_class_embed_sd, _ = self._legacy_parse_state_dicts(
                ema_weights
            )

        elif use_ema and ema_weights is None and "ema_state_dict" in checkpoint:
            _, ema_denoiser_sd, ema_class_embed_sd, _ = self._legacy_parse_state_dicts(
                checkpoint["ema_state_dict"]
            )
            if ema_denoiser_sd:
                logger.info(
                    "Utilisation de 'ema_state_dict' comme source EMA de secours."
                )

        metadata = self._safe_load_metadata(checkpoint.get("metadata", {}))
        metadata.global_step = checkpoint.get("global_step", checkpoint.get("step", 0))
        metadata.epoch = checkpoint.get("epoch", checkpoint.get("current_epoch", 0))
        metadata.use_ema = use_ema

        return CheckpointSchema(
            metadata=metadata,
            config=checkpoint.get("config", cfg),
            denoiser_state_dict=unet_sd,
            class_embedding_state_dict=class_sd,
            vae_state_dict=vae_sd,
            ema_denoiser_state_dict=ema_denoiser_sd,
            ema_class_embedding_state_dict=ema_class_embed_sd,
            optimizer_state_dict=checkpoint.get(
                "optimizer_state_dict", checkpoint.get("optimizer")
            ),
            scheduler_state_dict=checkpoint.get(
                "scheduler_state_dict", checkpoint.get("lr_scheduler")
            ),
        )

    def _legacy_extract_model_weights(
        self, checkpoint: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        if "ema_state_dict" in checkpoint:
            logger.info("Migration: Utilisation de 'ema_state_dict'.")
            ema_state_dict = checkpoint["ema_state_dict"]
            if "shadow_params" in ema_state_dict and "param_names" in ema_state_dict:
                return dict(
                    zip(ema_state_dict["param_names"], ema_state_dict["shadow_params"])
                )

            return ema_state_dict
        if "model_state_dict" in checkpoint:
            logger.info("Migration: Utilisation de 'model_state_dict'.")
            return checkpoint["model_state_dict"]

        logger.info(
            "Migration: Utilisation des clés racines (denoiser, class_embedding, ...)."
        )
        model_weights = {}
        keys_to_check = [
            "unet",
            "class_embedding",
            "unet_state_dict",
            "class_embedding_state_dict",
            "unet_lora_state_dict",
        ]
        for key in keys_to_check:
            if key in checkpoint and checkpoint[key] is not None:
                prefix = "unet" if key.startswith("unet") else "class_embed"
                for k, v in checkpoint[key].items():
                    model_weights[f"{prefix}.{k}"] = v
        return model_weights

    def _legacy_parse_state_dicts(
        self, model_weights: Dict[str, torch.Tensor]
    ) -> Tuple[Dict, Dict, Dict, Dict]:
        vae_state_dict = {}
        unet_state_dict = {}
        class_embed_state_dict = {}
        scheduler_buffers = {}
        for key, value in model_weights.items():
            if key.startswith("vae."):
                vae_state_dict[key[4:]] = value
            elif key.startswith("unet."):
                unet_state_dict[key[5:]] = value
            elif key.startswith("class_embed."):
                class_embed_state_dict[key[12:]] = value
            elif key.endswith("_buf"):
                scheduler_buffers[key] = value
        return (
            vae_state_dict,
            unet_state_dict,
            class_embed_state_dict,
            scheduler_buffers,
        )
