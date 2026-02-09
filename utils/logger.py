#!/usr/bin/env python
# coding=utf-8

import json
import os
import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import wandb
from omegaconf import OmegaConf


class RunLogger:
    def __init__(
        self,
        cfg,
        checkpoint_dir: Path,
        accelerator=None,
        resume_step: int | None = 0,
        wandb_root_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.checkpoint_dir = Path(checkpoint_dir)
        self.wandb_root_dir = (
            Path(wandb_root_dir) if wandb_root_dir else self.checkpoint_dir
        )
        self.accelerator = accelerator
        self.is_main = accelerator.is_main_process if accelerator else True

        self.metrics_file = self.checkpoint_dir / "metrics.jsonl"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

        self.images_dir = self.checkpoint_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.images_manifest = self.images_dir / "manifest.jsonl"

        self._metrics_fh = open(self.metrics_file, "a") if self.is_main else None
        self._images_fh = open(self.images_manifest, "a") if self.is_main else None

        self.wandb_run = None
        self.is_wandb_owner = False

        self._wandb_id_file = self.checkpoint_dir / ".wandb_id"

        if self.is_main and accelerator:
            try:
                for tracker in accelerator.trackers:
                    if tracker.name == "wandb":
                        self.wandb_run = tracker.tracker
                        break
            except Exception:
                pass

        if self.is_main and self.wandb_run is None:
            base_url = getattr(getattr(cfg, "logging", {}), "base_url", None)
            if base_url and not os.environ.get("WANDB_BASE_URL"):
                os.environ["WANDB_BASE_URL"] = str(base_url)

            wandb_id = os.environ.get("WANDB_RUN_ID")
            if not wandb_id:
                try:
                    if self._wandb_id_file.exists():
                        wandb_id = self._wandb_id_file.read_text().strip() or None
                        if wandb_id:
                            print(f"RunLogger: Loaded stored wandb ID: {wandb_id}")
                except Exception:
                    wandb_id = None
            if wandb_id is None:
                wandb_id = self.find_wandb_id()

            run_name = getattr(getattr(cfg, "logging", {}), "run_name", None)
            if not run_name:
                run_name = (
                    f"offline_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )

            try:
                init_args = {
                    "project": getattr(
                        getattr(cfg, "logging", {}), "project_name", None
                    ),
                    "config": OmegaConf.to_container(self.cfg, resolve=True),
                    "name": run_name,
                    "dir": str(self.wandb_root_dir.resolve()),
                    "resume": "allow",
                }
                entity = getattr(getattr(cfg, "logging", {}), "entity", None)
                if entity:
                    init_args["entity"] = entity
                if wandb_id:
                    init_args["id"] = wandb_id

                self.wandb_run = wandb.init(**init_args)
                self.is_wandb_owner = True

                try:
                    if self.wandb_run and getattr(self.wandb_run, "id", None):
                        self._wandb_id_file.write_text(str(self.wandb_run.id))
                        os.environ["WANDB_RUN_ID"] = str(self.wandb_run.id)
                except Exception:
                    pass

                try:
                    if resume_step and int(resume_step) > 0:
                        self._backfill_metrics(upto_step=int(resume_step))
                        if (self.images_manifest).exists():
                            self._backfill_images(upto_step=int(resume_step))
                except Exception as e:
                    print(f"WARN: Backfill failed: {e}")

                try:
                    print(
                        "RunLogger/W&B:",
                        {
                            "mode": os.environ.get("WANDB_MODE"),
                            "base_url": os.environ.get("WANDB_BASE_URL"),
                            "entity": init_args.get("entity"),
                            "project": init_args.get("project"),
                            "id": getattr(self.wandb_run, "id", None),
                            "name": run_name,
                            "dir": init_args.get("dir"),
                        },
                    )
                except Exception:
                    pass

            except Exception as e:
                print(f"WARN: Failed to init offline wandb: {e}")

    def _backfill_metrics(self, upto_step: int, max_lines: int | None = None) -> None:
        if not self.wandb_run:
            return

        marker = self.checkpoint_dir / f".wandb_backfill_upto_{upto_step}.done"
        if marker.exists():
            print(f"RunLogger: Backfill already done up to step {upto_step}, skipping.")
            return

        if not self.metrics_file.exists():
            print("RunLogger: No metrics.jsonl to backfill.")
            return

        count = 0
        with open(self.metrics_file, "r") as fh:
            for line in fh:
                if max_lines is not None and count >= int(max_lines):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                step = data.pop("step", None)
                if step is None:
                    continue
                try:
                    step = int(step)
                except Exception:
                    continue
                if step > upto_step:
                    continue

                self.wandb_run.log(data, step=step)
                count += 1

        try:
            marker.write_text(
                json.dumps(
                    {
                        "upto_step": upto_step,
                        "count": count,
                        "ts": datetime.datetime.now().isoformat(),
                    }
                )
            )
        except Exception:
            pass

    def find_wandb_id(self):
        wandb_dir = self.wandb_root_dir / "wandb"
        if not wandb_dir.exists():
            return None

        latest_run_symlink = wandb_dir / "latest-run"
        run_dir = None

        if latest_run_symlink.is_symlink():
            try:
                run_dir = latest_run_symlink.resolve()
            except Exception:
                run_dir = None

        if not run_dir or not run_dir.exists():
            candidates = []
            candidates.extend(d for d in wandb_dir.glob("run-*-*-*") if d.is_dir())
            candidates.extend(
                d for d in wandb_dir.glob("offline-run-*-*") if d.is_dir()
            )
            candidates.extend(d for d in wandb_dir.glob("*-run-*-*") if d.is_dir())
            if candidates:
                candidates.sort()
                run_dir = candidates[-1]

        if run_dir:
            rid = run_dir.name.split("-")[-1]
            if rid:
                print(f"RunLogger: Found existing wandb run ID for resume: {rid}")
                return rid

        print("RunLogger: No existing wandb run ID found.")
        return None

    def log(self, metrics: dict[str, Any], step: int, epoch: int | None = None):
        if not self.is_main:
            return

        row = {"step": step}
        if epoch is not None:
            row["epoch"] = epoch
        row.update(metrics)
        self._metrics_fh.write(json.dumps(row) + "\n")
        self._metrics_fh.flush()

        if self.wandb_run:
            log_dict = dict(metrics)
            if epoch is not None:
                log_dict["epoch"] = epoch
            self.wandb_run.log(log_dict, step=step)

    def log_image(self, key: str, image: Any, step: int, epoch: int | None = None):
        if not self.is_main:
            return

        pil_img = self._to_pil(image)
        img_path = self._save_image(pil_img, key, step, epoch)

        row = {
            "step": step,
            "epoch": epoch,
            "key": key,
            "path": str(img_path),
        }
        self._images_fh.write(json.dumps(row) + "\n")
        self._images_fh.flush()

        if self.wandb_run:
            try:
                self.wandb_run.log({key: wandb.Image(pil_img)}, step=step)
            except Exception as e:
                print(f"WARN: Failed to log image to wandb: {e}")

    def _to_pil(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image

        try:
            from matplotlib.figure import Figure as _MplFigure
            from matplotlib.axes import Axes as _MplAxes
        except Exception:
            _MplFigure = None
            _MplAxes = None

        if _MplFigure is not None and isinstance(image, _MplFigure):
            from io import BytesIO

            buf = BytesIO()
            try:
                image.savefig(buf, format="png", bbox_inches="tight")
            except Exception:
                image.savefig(buf, format="png")
            buf.seek(0)
            return Image.open(buf).convert("RGB")

        if _MplAxes is not None and isinstance(image, _MplAxes):
            return self._to_pil(image.figure)

        if hasattr(image, "detach"):
            try:
                image = image.detach().cpu().numpy()
            except Exception:
                pass

        try:
            arr = np.array(image)
        except Exception:
            txt = str(image)
            canvas = Image.new(
                "RGB", (max(64, 8 * len(txt)), 24), color=(255, 255, 255)
            )
            return canvas

        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.dtype.kind == "f":
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr.squeeze(axis=2)

        return Image.fromarray(arr)

    def _save_image(
        self, img: Image.Image, key: str, step: int, epoch: int | None
    ) -> Path:
        epoch_str = f"e{epoch}" if epoch is not None else "eNA"
        safe_key = key.replace("/", "_")
        filename = f"{epoch_str}_s{step}_{safe_key}.png"
        path = self.images_dir / filename
        img.save(path)
        return path

    def _backfill_images(self, upto_step: int, max_lines: int | None = None) -> None:
        if not self.wandb_run:
            return
        manifest = self.images_manifest
        if not manifest.exists():
            return
        marker = self.checkpoint_dir / f".wandb_backfill_images_upto_{upto_step}.done"
        if marker.exists():
            return
        import json as _json
        from PIL import Image as _Image

        count = 0
        with open(manifest, "r") as fh:
            for line in fh:
                if max_lines is not None and count >= int(max_lines):
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = _json.loads(line)
                except Exception:
                    continue
                step = row.get("step")
                try:
                    step = int(step) if step is not None else None
                except Exception:
                    step = None
                if step is None or step > upto_step:
                    continue
                key = row.get("key", "image")
                path = row.get("path")
                if not path:
                    continue
                try:
                    img = _Image.open(path)
                    self.wandb_run.log({key: wandb.Image(img)}, step=step)
                    count += 1
                except Exception:
                    continue
        try:
            marker.write_text(
                _json.dumps(
                    {
                        "upto_step": upto_step,
                        "count": count,
                        "ts": datetime.datetime.now().isoformat(),
                    }
                )
            )
        except Exception:
            pass

    def close(self):
        if self.is_main:
            if self._metrics_fh:
                self._metrics_fh.close()
                self._metrics_fh = None
            if self._images_fh:
                self._images_fh.close()
                self._images_fh = None
            if self.wandb_run and self.is_wandb_owner:
                self.wandb_run.finish()
                self.wandb_run = None
