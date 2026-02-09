import pandas as pd
from typing import Dict, Optional, Callable, List, Tuple
import logging
from pathlib import Path
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torchvision.transforms as T

from ovg.data.transforms import (
    get_transforms,
    DegradationTransform,
    UpscaleTransform,
)

log = logging.getLogger(__name__)


class BBBC021Dataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        mode: str = "classification",
        hr_size: int = 256,
        scale_factor: int = 4,
        transforms: bool = True,
        augmentation: bool = True,
        paired_mode: Optional[bool] = None,
        custom_transform: Optional[Callable] = None,
        deterministic_downsample: bool = True,
        added_noise_post_upsampling: bool = 0.0,
        class_indices: Optional[List[int]] = None,
        normalize: bool = True,
        blur_kernel_size: int = 1,
        blur_sigma: float = 0.0,
        noise_std: float = 0.01,
        interpolation_mode: str = "nearest",
        use_blur: bool = False,
        use_downsample: bool = True,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self._custom_transform = custom_transform
        self.mode = mode
        self.hr_size = hr_size
        self.scale_factor = scale_factor
        self.transforms = transforms
        self.augmentation = augmentation
        self.custom_transform = custom_transform
        self.deterministic_downsample = deterministic_downsample
        self.added_noise_post_upsampling = added_noise_post_upsampling
        self.class_indices = class_indices
        self.normalize = normalize
        self.loader = default_loader

        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.interpolation_mode = interpolation_mode
        self.use_blur = use_blur
        self.use_downsample = use_downsample

        if mode == "super_resolution":
            if hr_size % scale_factor != 0:
                raise ValueError(f"hr_size ({hr_size}) must be divisible by scale_factor ({scale_factor})")
            if paired_mode is None:
                self.paired_mode = "unpaired" not in self.csv_path.name.lower()
            else:
                self.paired_mode = paired_mode

        self.data = pd.read_csv(csv_path)
        self._create_class_index()
        self._filter_classes()
        self._setup_transforms()

        log.info(f"Loaded {len(self.data)} samples from {csv_path}")
        if mode == "super_resolution":
            log.info(f"Super-resolution mode: {'paired' if self.paired_mode else 'unpaired'}")

    def _filter_classes(self):
        if self.mode == "classification" and self.class_indices is not None:
            original_len = len(self.data)
            self.data = self.data[self.data["class_idx"].isin(self.class_indices)].copy()
            self.class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(self.class_indices))}
            self.data["mapped_class_idx"] = self.data["class_idx"].map(self.class_mapping)
            self._create_class_index()
            log.info(f"Filtered from {original_len} to {len(self.data)} samples")
            log.info(f"Class mapping: {self.class_mapping}")

    def _setup_transforms(self):
        if self.mode == "super_resolution":
            self.hr_transform = get_transforms(
                size=self.hr_size,
                train=self.augmentation,
                normalize=True,
                custom_transform=self.custom_transform,
            )
            self.degradation = DegradationTransform(
                scale_factor=self.scale_factor,
                hr_size=self.hr_size,
                kernel_size=self.blur_kernel_size,
                sigma=self.blur_sigma,
                noise_std=self.noise_std,
                interpolation_mode=self.interpolation_mode,
                use_blur=self.use_blur,
                use_downsample=self.use_downsample,
            )
            self.upscale = UpscaleTransform(target_size=self.hr_size, interpolation_mode=self.interpolation_mode)

        elif self.mode == "classification":
            if self.transforms:
                self.transform = get_transforms(
                    size=self.hr_size,
                    train=self.augmentation,
                    normalize=self.normalize,
                    custom_transform=self.custom_transform,
                )
            else:
                self.transform = T.ToTensor()

    def get_num_classes(self) -> int:
        if self.mode != "classification":
            raise ValueError("get_num_classes() is only valid for classification mode")
        if self.class_indices is not None:
            return len(self.class_indices)
        return len(self.available_classes)

    def get_class_mapping(self) -> Dict[int, int]:
        if self.mode != "classification":
            raise ValueError("get_class_mapping() is only valid for classification mode")
        if hasattr(self, "class_mapping"):
            return self.class_mapping.copy()
        return {idx: idx for idx in self.available_classes}

    def _create_class_index(self):
        self.class_to_indices = {}

        if self.mode == "super_resolution":
            for idx, row in self.data.iterrows():
                res_class = row["res_class"] if "res_class" in row else row["class_idx"]
                if res_class not in self.class_to_indices:
                    self.class_to_indices[res_class] = []
                self.class_to_indices[res_class].append(idx)
        else:
            for idx, row in self.data.iterrows():
                class_idx = row["mapped_class_idx"] if "mapped_class_idx" in row else row["class_idx"]
                if class_idx not in self.class_to_indices:
                    self.class_to_indices[class_idx] = []
                self.class_to_indices[class_idx].append(idx)

        self.available_classes = list(self.class_to_indices.keys())
        log.info(
            f"Created class index for mode '{self.mode}': {len(self.available_classes)} classes, {dict(zip(self.available_classes, [len(indices) for indices in self.class_to_indices.values()]))}"
        )

    @property
    def task_type(self) -> str:
        return self.mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data.iloc[idx]
        img_path = sample["path"]

        try:
            img = Image.open(img_path)
            img.load()
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            log.warning(f"Cannot load image at index {idx}: {img_path}. Error: {e}")

            if idx < len(self) - 1:
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)

        if self.mode == "classification":
            return self._get_classification_sample(img, sample, img_path)
        elif self.mode == "super_resolution":
            return self._get_super_resolution_sample(img, sample, img_path, idx)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _get_classification_sample(
        self, img: Image.Image, sample: pd.Series, img_path: str
    ) -> Dict[str, torch.Tensor]:
        img_tensor = self.transform(img)

        result = {
            "image": img_tensor,
            "path": img_path,
        }

        if self.class_indices is not None and hasattr(self, "class_mapping"):
            result["class_idx"] = torch.tensor(sample["mapped_class_idx"], dtype=torch.long)
            result["original_class_idx"] = torch.tensor(sample["class_idx"], dtype=torch.long)
        else:
            result["class_idx"] = torch.tensor(sample["class_idx"], dtype=torch.long)

        if "class_name" in sample:
            result["class_name"] = sample["class_name"]

        return result

    def _get_super_resolution_sample(
        self, img: Image.Image, sample: pd.Series, img_path: str, idx: int
    ) -> Dict[str, torch.Tensor]:
        is_high_res = sample["res_class"] == 1

        if self.deterministic_downsample:
            rng_state = torch.get_rng_state()
            torch.manual_seed(idx)

        try:
            if self.paired_mode:
                return self._get_paired_sample(img, img_path, sample)
            else:
                return self._get_unpaired_sample(img, sample, img_path, is_high_res)
        finally:
            if self.deterministic_downsample:
                torch.set_rng_state(rng_state)

    def _get_paired_sample(self, img: Image.Image, img_path: str, sample: pd.Series) -> Dict[str, torch.Tensor]:
        hr_tensor = self.hr_transform(img)
        lr_tensor = self.upscale(self.degradation(hr_tensor))

        result = {
            "hr": hr_tensor,
            "lr": lr_tensor,
            "path": img_path,
            "sr_factor": torch.tensor(self.scale_factor, dtype=torch.float),
            "class_idx": torch.tensor(sample["class_idx"], dtype=torch.long),
        }

        if "class_name" in sample:
            result["class_name"] = sample["class_name"]

        return result

    def _get_unpaired_sample(
        self, img: Image.Image, sample: pd.Series, img_path: str, is_high_res: bool
    ) -> Dict[str, torch.Tensor]:
        if is_high_res:
            img_tensor = self.hr_transform(img)
            sr_factor = 1.0
        else:
            hr_tensor = self.hr_transform(img)
            lr_tensor = self.degradation(hr_tensor)
            img_tensor = self.upscale(lr_tensor)
            sr_factor = float(self.scale_factor)

        result = {
            "image": img_tensor,
            "path": img_path,
            "class_idx": torch.tensor(sample["class_idx"], dtype=torch.long),
            "res_class": torch.tensor(sample["res_class"], dtype=torch.long),
            "sr_factor": torch.tensor(sr_factor, dtype=torch.float),
        }

        if "class_name" in sample:
            result["class_name"] = sample["class_name"]

        return result

    def get_samples_by_class(
        self, class_idx: int, n_samples: int = 1, seed: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        if class_idx not in self.class_to_indices:
            raise ValueError(f"Class {class_idx} not found in dataset")

        available_indices = self.class_to_indices[class_idx]
        if n_samples > len(available_indices):
            log.warning(
                f"Requested {n_samples} samples from class {class_idx}, but only {len(available_indices)} available"
            )
            n_samples = len(available_indices)

        if seed is not None:
            random.seed(seed)

        selected_indices = random.sample(available_indices, n_samples)
        return [self[idx] for idx in selected_indices]

    def get_balanced_samples(
        self,
        n_samples_per_class: int = 1,
        classes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        if classes is None:
            classes = self.available_classes

        samples = []
        for i, class_idx in enumerate(classes):
            try:
                class_seed = seed + i if seed is not None else None
                class_samples = self.get_samples_by_class(class_idx, n_samples_per_class, class_seed)
                samples.extend(class_samples)
            except ValueError as e:
                log.warning(f"Skipping class {class_idx}: {e}")

        return samples

    def get_random_samples(self, n_samples: int, seed: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        if seed is not None:
            random.seed(seed)

        if n_samples > len(self):
            log.warning(f"Requested {n_samples} samples, but dataset only has {len(self)}")
            n_samples = len(self)

        indices = random.sample(range(len(self)), n_samples)
        return [self[idx] for idx in indices]

    def get_deterministic_samples_by_class(
        self, class_idx: int, n_samples: int = 1, start_idx: int = 0
    ) -> List[Dict[str, torch.Tensor]]:
        if class_idx not in self.class_to_indices:
            raise ValueError(f"Class {class_idx} not found in dataset")

        available_indices = self.class_to_indices[class_idx]
        if start_idx >= len(available_indices):
            log.warning(f"start_idx {start_idx} >= available samples {len(available_indices)} for class {class_idx}")
            start_idx = 0

        if n_samples > len(available_indices):
            log.warning(
                f"Requested {n_samples} samples from class {class_idx}, but only {len(available_indices)} available"
            )
            n_samples = len(available_indices)

        end_idx = min(start_idx + n_samples, len(available_indices))
        selected_indices = available_indices[start_idx:end_idx]

        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            selected_indices.extend(available_indices[:remaining])

        return [self[idx] for idx in selected_indices]


class Edges2ShoesDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        root_dir: Optional[Path] = None,
        mode: str = "translation",
        img_size: int = 256,
        transforms: bool = True,
        augmentation: bool = True,
        paired_mode: Optional[bool] = None,
        custom_transform: Optional[Callable] = None,
        class_indices: Optional[List[int]] = None,
        normalize: bool = True,
        split: Optional[str] = None,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)
        self.root_dir = Path(root_dir) if root_dir else self.csv_path.parent
        self.mode = mode
        self.img_size = img_size
        self.transforms = transforms
        self.augmentation = augmentation
        self.custom_transform = custom_transform
        self.class_indices = class_indices
        self.normalize = normalize
        self.loader = default_loader

        if paired_mode is None:
            self.paired_mode = mode == "translation"
        else:
            self.paired_mode = paired_mode

        self.data = pd.read_csv(csv_path)
        if split is not None:
            self.data = self.data[self.data["split"] == split].reset_index(drop=True)

        self._create_class_index()
        self._filter_classes()
        self._setup_transforms()

        log.info(
            f"Loaded {len(self.data)} Edges2Shoes samples from {csv_path} "
            f"(mode={self.mode}, paired={self.paired_mode})"
        )

    def _filter_classes(self):
        if self.class_indices is not None:
            original_len = len(self.data)
            self.data = self.data[self.data["class_idx"].isin(self.class_indices)].reset_index(drop=True)
            self.class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(self.class_indices))}
            self.data["mapped_class_idx"] = self.data["class_idx"].map(self.class_mapping)
            self._create_class_index()
            log.info(f"Filtered from {original_len} to {len(self.data)} samples")

    def _setup_transforms(self):
        if self.transforms:
            self.transform = get_transforms(
                size=self.img_size,
                train=self.augmentation,
                normalize=self.normalize,
                custom_transform=self.custom_transform,
            )
        else:
            self.transform = T.ToTensor()

    def _create_class_index(self):
        self.class_to_indices = {}
        for idx, row in self.data.iterrows():
            class_idx = row.get("mapped_class_idx", row.get("class_idx", 0))
            self.class_to_indices.setdefault(class_idx, []).append(idx)
        self.available_classes = list(self.class_to_indices.keys())

    @property
    def task_type(self) -> str:
        return self.mode

    def __len__(self) -> int:
        return len(self.data)

    def _split_image(self, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        width, height = img.size
        half_width = width // 2
        edge_img = img.crop((0, 0, half_width, height))
        shoe_img = img.crop((half_width, 0, width, height))
        return edge_img, shoe_img

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data.iloc[idx]
        img_path = self.root_dir / sample["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
            img.load()
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            log.warning(f"Cannot load image at index {idx}: {img_path}. Error: {e}")
            return self.__getitem__((idx + 1) % len(self))

        edge_img, shoe_img = self._split_image(img)

        if self.paired_mode:
            return self._get_paired_sample(edge_img, shoe_img, sample, str(img_path))
        else:
            return self._get_unpaired_sample(edge_img, shoe_img, sample, str(img_path), idx)

    def _get_paired_sample(
        self,
        edge_img: Image.Image,
        shoe_img: Image.Image,
        sample: pd.Series,
        img_path: str,
    ) -> Dict[str, torch.Tensor]:
        edge_tensor = self.transform(edge_img)
        shoe_tensor = self.transform(shoe_img)

        class_idx = sample.get("mapped_class_idx", sample.get("class_idx", 0))
        result = {
            "hr": shoe_tensor,
            "lr": edge_tensor,
            "path": img_path,
            "image_id": sample["image_id"],
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
        }

        return result

    def _get_unpaired_sample(
        self,
        edge_img: Image.Image,
        shoe_img: Image.Image,
        sample: pd.Series,
        img_path: str,
        idx: int,
    ) -> Dict[str, torch.Tensor]:
        if "cond_class" in sample:
            cond_class = int(sample["cond_class"])
        else:
            cond_class = idx % 2

        if cond_class == 1:
            img_tensor = self.transform(shoe_img)
        else:
            img_tensor = self.transform(edge_img)

        class_idx = sample.get("mapped_class_idx", sample.get("class_idx", 0))
        result = {
            "image": img_tensor,
            "path": img_path,
            "image_id": sample["image_id"],
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
            "cond_class": torch.tensor(cond_class, dtype=torch.long),
        }

        return result

    def get_num_classes(self) -> int:
        if self.class_indices is not None:
            return len(self.class_indices)
        return len(self.available_classes)

    def get_samples_by_class(
        self, class_idx: int, n_samples: int = 1, seed: Optional[int] = None
    ) -> List[Dict[str, torch.Tensor]]:
        if class_idx not in self.class_to_indices:
            raise ValueError(f"Class {class_idx} not found in dataset")

        available_indices = self.class_to_indices[class_idx]
        if n_samples > len(available_indices):
            log.warning(
                f"Requested {n_samples} samples from class {class_idx}, but only {len(available_indices)} available"
            )
            n_samples = len(available_indices)

        if seed is not None:
            random.seed(seed)

        selected_indices = random.sample(available_indices, n_samples)
        return [self[idx] for idx in selected_indices]

    def get_random_samples(self, n_samples: int, seed: Optional[int] = None) -> List[Dict[str, torch.Tensor]]:
        if seed is not None:
            random.seed(seed)

        if n_samples > len(self):
            log.warning(f"Requested {n_samples} samples, but dataset only has {len(self)}")
            n_samples = len(self)

        indices = random.sample(range(len(self)), n_samples)
        return [self[idx] for idx in indices]

    def get_deterministic_samples_by_class(
        self, class_idx: int, n_samples: int = 1, start_idx: int = 0
    ) -> List[Dict[str, torch.Tensor]]:
        if class_idx not in self.class_to_indices:
            raise ValueError(f"Class {class_idx} not found in dataset")

        available_indices = self.class_to_indices[class_idx]
        if start_idx >= len(available_indices):
            log.warning(f"start_idx {start_idx} >= available samples {len(available_indices)} for class {class_idx}")
            start_idx = 0

        if n_samples > len(available_indices):
            log.warning(
                f"Requested {n_samples} samples from class {class_idx}, but only {len(available_indices)} available"
            )
            n_samples = len(available_indices)

        end_idx = min(start_idx + n_samples, len(available_indices))
        selected_indices = available_indices[start_idx:end_idx]

        if len(selected_indices) < n_samples:
            remaining = n_samples - len(selected_indices)
            selected_indices.extend(available_indices[:remaining])

        return [self[idx] for idx in selected_indices]
