import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import mode
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError
from torch.utils.hooks import RemovableHandle
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def compute_knn_score(
    feats: np.ndarray, labels: torch.Tensor, k: int = 5, per_class: bool = True
) -> float:
    labs = labels.detach().cpu().numpy().astype(int)
    num = feats.shape[0]
    if num <= 1:
        return 0.0
    effective_k = min(k, num - 1)
    nn = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    nn.fit(feats)
    _, idx = nn.kneighbors(feats)
    neighbor_idx = idx[:, 1:]
    neighbor_labels = labs[neighbor_idx]
    pred, _ = mode(neighbor_labels, axis=1, keepdims=False)
    pred = pred.flatten()

    if per_class:
        unique_labels = np.unique(labs)
        class_scores = []
        for lab in unique_labels:
            mask = labs == lab
            if not np.any(mask):
                continue
            class_scores.append(float((pred[mask] == lab).mean()))
        if class_scores:
            return float(np.mean(class_scores))

    return float((pred == labs).mean())


class HaarPSI:
    def __init__(
        self,
        C: float = 30.0,
        alpha: float = 4.2,
        preprocess_with_subsampling: bool = True,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.C = C
        self.alpha = alpha
        self.preprocess_with_subsampling = preprocess_with_subsampling
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype
        self._range_warning_emitted = False
        self.reset()

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = (
                device if isinstance(device, torch.device) else torch.device(device)
            )
        if dtype is not None:
            self.dtype = dtype
        return self

    def reset(self):
        self._values = []

    def _prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(tensor):
            raise TypeError("Expected a torch.Tensor for HaarPSI computation")

        tensor = tensor.to(self.device, dtype=torch.float32)
        min_val = tensor.amin().item()
        max_val = tensor.amax().item()

        needs_rescale = min_val < 0.0 or max_val > 1.0
        if needs_rescale:
            if min_val >= -1.05 and max_val <= 1.05:
                tensor = (tensor + 1.0) / 2.0
            else:
                range_val = max_val - min_val
                if range_val > 1e-6:
                    tensor = (tensor - min_val) / range_val
                else:
                    tensor = torch.zeros_like(tensor)

            if not self._range_warning_emitted:
                warnings.warn(
                    "HaarPSIMetric: inputs outside [0,1] detected; automatically rescaling.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                self._range_warning_emitted = True

        return tensor.clamp(0.0, 1.0).to(dtype=self.dtype)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = self._prepare_tensor(target)
        preds = self._prepare_tensor(preds)
        similarity, _, _ = haarpsi(
            target, preds, self.C, self.alpha, self.preprocess_with_subsampling
        )
        self._values.append(similarity.detach().to("cpu", copy=True).view(-1))

    def compute(self) -> torch.Tensor:
        if not self._values:
            return torch.zeros(1, device=self.device, dtype=self.dtype).squeeze(0)
        stacked = torch.cat(self._values, dim=0)
        return stacked.to(self.device, dtype=self.dtype).mean()


def haarpsi(ref, deg, C, α, preprocess_with_subsampling=True):
    assert torch.is_tensor(ref) and torch.is_tensor(deg)
    assert ref.shape == deg.shape
    assert len(ref.shape) in {2, 3, 4}
    assert ref.dtype == torch.float32
    assert ref.dtype == deg.dtype
    assert torch.all(ref >= 0) and torch.all(ref <= 1)
    assert torch.all(deg >= 0) and torch.all(deg <= 1)

    ref = 255 * ref
    deg = 255 * deg

    if len(ref.shape) == 2:
        ref = ref.unsqueeze(0).unsqueeze(0)
        deg = deg.unsqueeze(0).unsqueeze(0)
    elif len(ref.shape) == 3:
        ref = ref.unsqueeze(0)
        deg = deg.unsqueeze(0)

    assert deg.shape[1] in {1, 3}
    assert ref.shape[1] == deg.shape[1]

    is_color_image = deg.shape[1] == 3

    if is_color_image:
        ref_y = (
            0.299 * ref[:, 0, :, :] + 0.587 * ref[:, 1, :, :] + 0.114 * ref[:, 2, :, :]
        )
        deg_y = (
            0.299 * deg[:, 0, :, :] + 0.587 * deg[:, 1, :, :] + 0.114 * deg[:, 2, :, :]
        )
        ref_i = (
            0.596 * ref[:, 0, :, :] - 0.274 * ref[:, 1, :, :] - 0.322 * ref[:, 2, :, :]
        )
        deg_i = (
            0.596 * deg[:, 0, :, :] - 0.274 * deg[:, 1, :, :] - 0.322 * deg[:, 2, :, :]
        )
        ref_q = (
            0.211 * ref[:, 0, :, :] - 0.523 * ref[:, 1, :, :] + 0.312 * ref[:, 2, :, :]
        )
        deg_q = (
            0.211 * deg[:, 0, :, :] - 0.523 * deg[:, 1, :, :] + 0.312 * deg[:, 2, :, :]
        )
        ref_y = ref_y.unsqueeze(1)
        deg_y = deg_y.unsqueeze(1)
        ref_i = ref_i.unsqueeze(1)
        deg_i = deg_i.unsqueeze(1)
        ref_q = ref_q.unsqueeze(1)
        deg_q = deg_q.unsqueeze(1)
    else:
        ref_y = ref
        deg_y = deg

    if preprocess_with_subsampling:
        ref_y = _subsample(ref_y)
        deg_y = _subsample(deg_y)
        if is_color_image:
            ref_i = _subsample(ref_i)
            deg_i = _subsample(deg_i)
            ref_q = _subsample(ref_q)
            deg_q = _subsample(deg_q)

    n_scales = 3
    coeffs_ref_y = _haar_wavelet_decompose(ref_y, n_scales)
    coeffs_deg_y = _haar_wavelet_decompose(deg_y, n_scales)

    if is_color_image:
        coefficients_ref_i = torch.abs(_convolve2d(ref_i, torch.ones((2, 2)) / 4.0))
        coefficients_deg_i = torch.abs(_convolve2d(deg_i, torch.ones((2, 2)) / 4.0))
        coefficients_ref_q = torch.abs(_convolve2d(ref_q, torch.ones((2, 2)) / 4.0))
        coefficients_deg_q = torch.abs(_convolve2d(deg_q, torch.ones((2, 2)) / 4.0))

    n_samples, _, height, width = ref_y.shape
    n_channels = 3 if is_color_image else 2

    local_similarities = torch.zeros(
        n_channels, n_samples, 1, height, width, device=ref_y.device
    )
    weights = torch.zeros(n_channels, n_samples, 1, height, width, device=ref_y.device)

    for orientation in [0, 1]:
        weights[orientation] = _get_weights_for_orientation(
            coeffs_deg_y,
            coeffs_ref_y,
            n_scales,
            orientation,
        )
        local_similarities[orientation] = _get_local_similarity_for_orientation(
            C, coeffs_deg_y, coeffs_ref_y, n_scales, orientation
        )

    if is_color_image:
        similarity_i = (2 * coefficients_ref_i * coefficients_deg_i + C) / (
            coefficients_ref_i**2 + coefficients_deg_i**2 + C
        )
        similarity_q = (2 * coefficients_ref_q * coefficients_deg_q + C) / (
            coefficients_ref_q**2 + coefficients_deg_q**2 + C
        )
        local_similarities[2, :, :, :, :] = (similarity_i + similarity_q) / 2
        weights[2, :, :, :, :] = (weights[0, :, :, :, :] + weights[1, :, :, :, :]) / 2

    a = torch.sigmoid(α * local_similarities) * weights
    b = weights

    dims = (0, 3, 4)
    pre_logit = torch.sum(a, dim=dims) / torch.sum(b, dim=dims)

    logit = lambda value, α: torch.log(value / (1 - value)) / α
    similarity = logit(pre_logit, α) ** 2

    local_similarities = local_similarities[:, :, 0].permute(1, 0, 2, 3)
    weights = weights[:, :, 0].permute(1, 0, 2, 3)
    return similarity, local_similarities, weights


def _get_local_similarity_for_orientation(
    C, coeffs_deg_y, coeffs_ref_y, n_scales, orientation
):
    coeffs_ref_y_magnitude = coeffs_ref_y.abs()[
        (orientation * n_scales, 1 + orientation * n_scales), :, :
    ]
    coeffs_deg_y_magnitude = coeffs_deg_y.abs()[
        (orientation * n_scales, 1 + orientation * n_scales), :, :
    ]
    a = 2 * coeffs_ref_y_magnitude * coeffs_deg_y_magnitude + C
    b = coeffs_ref_y_magnitude**2 + coeffs_deg_y_magnitude**2 + C
    frac = a / b
    local_similarity = (frac[0] + frac[1]) / 2
    return local_similarity


def _get_weights_for_orientation(coeffs_deg_y, coeffs_ref_y, n_scales, orientation):
    maxs = torch.maximum(
        coeffs_ref_y[2 + orientation * n_scales].abs(),
        coeffs_deg_y[2 + orientation * n_scales].abs(),
    )
    return maxs


def _subsample(image):
    kernel = torch.ones(2, 2, device=image.device) / 4.0
    subsampled_image = _convolve2d(image, kernel)
    return subsampled_image[:, :, ::2, ::2]


def _convolve2d(data, kernel):
    width, height = data.shape[-2:]
    kernel = kernel.to(device=data.device, dtype=data.dtype).unsqueeze(0).unsqueeze(0)

    rotated_data = torch.rot90(data, 2, [2, 3])
    padding = (kernel.shape[2] // 2, kernel.shape[3] // 2)
    res = torch.nn.functional.conv2d(rotated_data, kernel, padding=padding)
    res = torch.nn.functional.interpolate(
        res, (width, height), mode="nearest", align_corners=None
    )
    return torch.rot90(res, 2, [2, 3])


def _get_haar_filter(scale, device):
    haar_filter = 2**-scale * torch.ones(2**scale, 2**scale, device=device)
    haar_filter[: haar_filter.shape[0] // 2, :] = -haar_filter[
        : haar_filter.shape[0] // 2, :
    ]
    return haar_filter


def _haar_wavelet_decompose(image, number_of_scales):
    coefficients = torch.zeros(2 * number_of_scales, *image.shape, device=image.device)
    for scale in range(1, number_of_scales + 1):
        haar_filter = _get_haar_filter(scale, image.device)
        coefficients[scale - 1] = _convolve2d(image, haar_filter)
        coefficients[scale + number_of_scales - 1] = _convolve2d(image, haar_filter.t())
    return coefficients


class DinoV2StructureExtractor:
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        device: torch.device | str = "cpu",
        image_size: int = 224,
    ) -> None:
        self.device = torch.device(device)
        self.model_name = model_name
        self.image_size = image_size
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(
            self.device
        )
        self.model.eval()

        first_block = self.model.blocks[0]
        self.num_heads = first_block.attn.num_heads
        self.embed_dim = first_block.attn.qkv.in_features
        self.head_dim = self.embed_dim // self.num_heads

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

        self._handles: List[RemovableHandle] = []
        self._qkv_output: Optional[torch.Tensor] = None

    def _clear_handles(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _register_hooks(self, layer_idx: int) -> None:
        block = self.model.blocks[layer_idx]

        def save_qkv(_module, _inp, output):
            self._qkv_output = output

        handle = block.attn.qkv.register_forward_hook(save_qkv)
        self._handles.append(handle)

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 3:
            raise ValueError("Expected image tensor with shape (C, H, W)")
        image_cpu = image.detach().cpu().clamp(0.0, 1.0)
        processed = self.transform(image_cpu)
        return processed.unsqueeze(0).to(self.device)

    def _extract_qkv(self, image: torch.Tensor, layer_idx: int) -> torch.Tensor:
        self._qkv_output = None
        self._register_hooks(layer_idx)
        try:
            with torch.no_grad():
                _ = self.model(image)
        finally:
            self._clear_handles()

        if self._qkv_output is None:
            raise RuntimeError("Failed to capture qkv output from DINOv2 block")
        return self._qkv_output

    def keys_self_similarity(self, image: torch.Tensor, layer_idx: int) -> torch.Tensor:
        prep = self._preprocess(image)
        qkv = self._extract_qkv(prep, layer_idx)

        if qkv.dim() != 3:
            raise ValueError("Unexpected qkv tensor dimensionality")

        batch, tokens, three_dim = qkv.shape
        if batch != 1:
            raise ValueError("Structure distance expects batch size of 1")

        if three_dim % 3 != 0:
            raise ValueError("qkv features size must be divisible by 3")

        hidden_dim = three_dim // 3
        if hidden_dim % self.num_heads != 0:
            raise ValueError("Hidden dimension must be divisible by number of heads")

        qkv = qkv.reshape(1, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        keys = qkv[1]
        keys = keys.reshape(1, self.num_heads, tokens, self.head_dim)
        keys = keys.permute(0, 2, 1, 3).reshape(
            1, tokens, self.num_heads * self.head_dim
        )
        keys = F.normalize(keys, dim=-1)

        similarity = torch.matmul(keys, keys.transpose(1, 2))
        return similarity.squeeze(0)


class StructureDistanceCore:
    def __init__(
        self,
        device: torch.device | str,
        model_name: str = "dinov2_vitb14",
        image_size: int = 224,
        layer_idx: int = -1,
    ) -> None:
        self.device = torch.device(device)
        self.layer_idx = layer_idx
        self.extractor = DinoV2StructureExtractor(
            model_name=model_name,
            device=self.device,
            image_size=image_size,
        )

    def distance_from_tensors(
        self, target: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        target = target.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        prediction = prediction.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)

        with torch.no_grad():
            target_sim = self.extractor.keys_self_similarity(target, self.layer_idx)
            pred_sim = self.extractor.keys_self_similarity(prediction, self.layer_idx)
            loss = F.mse_loss(pred_sim, target_sim, reduction="mean")
        return loss


class StructureDistanceMetric:
    def __init__(self, core: StructureDistanceCore) -> None:
        self.core = core
        self.reset()

    def reset(self) -> None:
        self._values: List[torch.Tensor] = []

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.to(self.core.device, dtype=torch.float32)
        target = target.to(self.core.device, dtype=torch.float32)
        for idx in range(preds.shape[0]):
            value = self.core.distance_from_tensors(target[idx], preds[idx])
            self._values.append(value.detach())

    def compute(self) -> torch.Tensor:
        if not self._values:
            return torch.tensor(0.0, device=self.core.device, dtype=torch.float32)
        stacked = torch.stack(self._values)
        return stacked.mean()


class MetricsCalculator:
    def __init__(
        self,
        device,
        enable_clip: bool = True,
        enable_lpips: bool = True,
        enable_structure_distance: bool = True,
    ):
        self.device = device

        self.clip_metric = None
        if enable_clip:
            try:
                self.clip_metric = CLIPScore(
                    model_name_or_path="openai/clip-vit-large-patch14"
                ).to(device)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load CLIP model (offline?): {e}. CLIP similarity will be unavailable."
                )
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

        self.lpips_metric = None
        if enable_lpips:
            try:
                self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
                    net_type="alex"
                ).to(device)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load LPIPS model (offline?): {e}. LPIPS will be unavailable."
                )
        self.mse_metric = MeanSquaredError().to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.msssim_metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)
        self.haarpsi_metric = HaarPSI().to(device=device)

        self.structure_distance_core = None
        self.structure_distance_metric = None
        if enable_structure_distance:
            try:
                self.structure_distance_core = StructureDistanceCore(device=self.device)
                self.structure_distance_metric = StructureDistanceMetric(
                    self.structure_distance_core
                )
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Failed to load DINOv2 model (offline?): {e}. Structure distance will be unavailable."
                )

    def calculate_clip_similarity(self, img, txt, mask=None):
        if self.clip_metric is None:
            import warnings

            warnings.warn("CLIP metric not available (offline mode or load failure).")
            return 0.0

        img = np.array(img)

        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)

        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        score = self.clip_metric(img_tensor, txt)
        return score.cpu().item()

    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = (
            torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        img_gt_tensor = (
            torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        score = self.psnr_metric(img_pred_tensor, img_gt_tensor)
        return score.cpu().item()

    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        if self.lpips_metric is None:
            import warnings

            warnings.warn("LPIPS metric not available (offline mode or load failure).")
            return 0.0

        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = (
            torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        img_gt_tensor = (
            torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        score = self.lpips_metric(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        return score.cpu().item()

    def calculate_haarpsi(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = (
                img_pred * mask_pred[..., None]
                if mask_pred.ndim == 2
                else img_pred * mask_pred
            )
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = (
                img_gt * mask_gt[..., None] if mask_gt.ndim == 2 else img_gt * mask_gt
            )

        pred_tensor = (
            torch.tensor(img_pred)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        gt_tensor = (
            torch.tensor(img_gt)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )

        metric = HaarPSI().to(device=self.device)
        metric.update(pred_tensor, gt_tensor)
        score = metric.compute()
        return float(score.item())

    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)

        score = self.mse_metric(
            img_pred_tensor.contiguous(), img_gt_tensor.contiguous()
        )
        return score.cpu().item()

    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = (
            torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        img_gt_tensor = (
            torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        score = self.ssim_metric(img_pred_tensor, img_gt_tensor)
        return score.cpu().item()

    def calculate_structure_distance(
        self, img_pred, img_gt, mask_pred=None, mask_gt=None
    ):
        if self.structure_distance_core is None:
            import warnings

            warnings.warn(
                "Structure distance metric not available (offline mode or load failure)."
            )
            return 0.0

        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = (
                img_pred * mask_pred[..., None]
                if mask_pred.ndim == 2
                else img_pred * mask_pred
            )
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = (
                img_gt * mask_gt[..., None] if mask_gt.ndim == 2 else img_gt * mask_gt
            )

        img_pred_tensor = (
            torch.tensor(img_pred).permute(2, 0, 1).to(self.device, dtype=torch.float32)
        )
        img_gt_tensor = (
            torch.tensor(img_gt).permute(2, 0, 1).to(self.device, dtype=torch.float32)
        )

        score = self.structure_distance_core.distance_from_tensors(
            img_gt_tensor, img_pred_tensor
        )
        return float(score.item())

    def compute_knn_recovery(
        self,
        feats_sr: np.ndarray,
        labels: torch.Tensor,
        knn_lr: float = None,
        knn_hr: float = None,
        k: int = 5,
    ) -> dict:
        knn_sr = compute_knn_score(feats_sr, labels, k=k)
        tau = (knn_hr or 0.0) - (knn_lr or 0.0)
        valid = knn_lr is not None and knn_hr is not None and abs(tau) >= 1e-6
        recovery = (knn_sr - (knn_lr or 0.0)) / tau if valid else 0.0

        return {
            "knn_sr": knn_sr,
            "knn_hr": knn_hr,
            "knn_lr": knn_lr,
            "knn_recovery (↑)": float(recovery),
            "knn_recovery_conf": float(abs(tau) if valid else 0.0),
            "knn_recovery_valid": bool(valid),
        }

    def create_pca_plot(
        self,
        feats_hr: np.ndarray,
        feats_lr: np.ndarray,
        feats_sr_dict: dict,
        labels: torch.Tensor,
        output_path: Optional[str] = None,
        seed: int = 42,
        max_cols: int = 3,
        method_order: Optional[List[str]] = None,
    ) -> Optional[plt.Figure]:
        if not feats_sr_dict:
            return None

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        labels_np = labels.detach().cpu().numpy().astype(int)
        num_samples = labels_np.shape[0]

        uniq = np.unique(labels_np)
        tab = cm.get_cmap("tab10", max(len(uniq), 2))
        lut = {lab: mcolors.to_hex(tab(i % tab.N)) for i, lab in enumerate(uniq)}
        colors = np.array([lut[label] for label in labels_np])

        ordered_methods = method_order or list(feats_sr_dict.keys())
        ordered_methods = [m for m in ordered_methods if m in feats_sr_dict]
        if not ordered_methods:
            return None

        num_methods = len(ordered_methods)
        ncols = min(max_cols, num_methods)
        nrows = int(np.ceil(num_methods / ncols))

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows)
        )
        axes = np.array(axes).reshape(-1)

        for ax_idx, method_name in enumerate(ordered_methods):
            feats_sr = feats_sr_dict[method_name]
            ax = axes[ax_idx]

            combined = np.concatenate([feats_hr, feats_lr, feats_sr], axis=0)
            pca = PCA(n_components=2, random_state=seed)
            projected = pca.fit_transform(combined)
            var1, var2 = pca.explained_variance_ratio_[:2] * 100

            hr_2d = projected[:num_samples]
            lr_2d = projected[num_samples : 2 * num_samples]
            sr_2d = projected[2 * num_samples : 3 * num_samples]

            ax.scatter(
                hr_2d[:, 0],
                hr_2d[:, 1],
                c=colors,
                marker="o",
                s=50,
                label="HR",
                alpha=0.7,
            )
            ax.scatter(
                lr_2d[:, 0],
                lr_2d[:, 1],
                c=colors,
                marker="s",
                s=50,
                label="LR",
                alpha=0.7,
            )
            ax.scatter(
                sr_2d[:, 0],
                sr_2d[:, 1],
                c=colors,
                marker="^",
                s=50,
                label=f"{method_name} SR",
                alpha=0.7,
            )

            ax.set_title(method_name)
            ax.set_xlabel(f"PC1 ({var1:.1f}%)")
            ax.set_ylabel(f"PC2 ({var2:.1f}%)")
            ax.legend()

        for idx in range(num_methods, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return None
        return fig

    def plot_metrics_vs_knn(
        self,
        method_metrics: dict,
        metric_order: list[str],
        metric_display_names: dict[str, str] = None,
        knn_key: str = "knn_recovery (↑)",
        title: str = "Metrics vs KNN recovery",
    ) -> plt.Figure:
        if not method_metrics:
            return None

        if metric_display_names is None:
            metric_display_names = {m: m for m in metric_order}

        metrics_to_plot = metric_order

        methods = list(method_metrics.keys())

        knn_percentages = {
            m: float(method_metrics[m][knn_key]) * 100.0 for m in methods
        }

        sorted_methods = sorted(methods, key=lambda m: knn_percentages[m])

        sorted_knn = np.array([knn_percentages[m] for m in sorted_methods])

        metric_values = {
            metric: np.array([float(method_metrics[m][metric]) for m in sorted_methods])
            for metric in metrics_to_plot
        }

        knn_min = sorted_knn.min()
        knn_max = sorted_knn.max()
        if np.isclose(knn_min, knn_max):
            delta = max(abs(knn_min) * 0.05, 1.0)
            knn_min -= delta
            knn_max += delta
        knn_range = knn_max - knn_min

        all_z = np.concatenate(list(metric_values.values()))
        z_min = all_z.min()
        z_max = all_z.max()
        z_range = z_max - z_min if z_max > z_min else 1.0

        z_base = z_min - 0.12 * z_range
        z_top = z_max + 0.18 * z_range

        y_spacing = 12.0
        y_positions = {
            metric: idx * y_spacing for idx, metric in enumerate(metrics_to_plot)
        }

        left_method = sorted_methods[0]
        right_method = sorted_methods[-1]
        mid_method = sorted_methods[len(sorted_methods) // 2]

        representative_methods = {
            "left": left_method,
            "mid": mid_method,
            "right": right_method,
        }

        start_color = "#FF8C00"
        mid_color = "#DC143C"
        end_color = "#1E90FF"
        representative_colors = {
            "left": start_color,
            "mid": mid_color,
            "right": end_color,
        }
        representative_markers = {"left": "^", "mid": "D", "right": "o"}

        baseline_color = "#222222"
        arrow_length_ratio = 0.18
        upward_arrow_ratio = 0.025

        fig = plt.figure(figsize=(12, 9.5))
        ax = fig.add_subplot(111, projection="3d")

        line_cmap = plt.get_cmap("tab20")
        method_palette = plt.get_cmap("tab10")
        method_colors = {
            name: method_palette(i % method_palette.N)
            for i, name in enumerate(sorted_methods)
        }

        connector_points = {"left": [], "mid": [], "right": []}

        for idx, metric in enumerate(metrics_to_plot):
            y = y_positions[metric]
            x = sorted_knn
            z = metric_values[metric]

            line_color = line_cmap((idx * 1.7) % 1.0)

            ax.plot(
                x, np.full_like(x, y), z, color=line_color, linewidth=2.6, alpha=0.92
            )

            x_base = np.array([knn_min, knn_max])
            ax.plot(
                x_base,
                np.full_like(x_base, y),
                np.full_like(x_base, z_base),
                color=baseline_color,
                linewidth=1.2,
            )

            label_offset = 0.018 * z_range
            ax.text(
                knn_min,
                y,
                z_base - 1.4 * label_offset,
                f"{knn_min:.2f}",
                color="black",
                fontsize=9,
                ha="center",
            )
            ax.text(
                knn_max,
                y,
                z_base - 1.4 * label_offset,
                f"{knn_max:.2f}",
                color="black",
                fontsize=9,
                ha="center",
            )
            ax.text(
                (knn_min + knn_max) / 2,
                y,
                z_base + label_offset,
                metric_display_names[metric],
                color=line_color,
                fontsize=12,
                ha="center",
                weight="bold",
            )

            for i, method in enumerate(sorted_methods):
                if method in representative_methods.values():
                    continue
                x_val = x[i]
                z_val = z[i]
                color = method_colors[method]
                ax.scatter(
                    x_val,
                    y,
                    z_val,
                    marker="o",
                    s=70,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.85,
                    zorder=9,
                )

            for key, method in representative_methods.items():
                i = sorted_methods.index(method)
                x_val = x[i]
                z_val = z[i]
                color = representative_colors[key]
                marker = representative_markers[key]
                ax.scatter(
                    x_val,
                    y,
                    z_val,
                    marker=marker,
                    s=130,
                    color=color,
                    edgecolor="black",
                    linewidth=0.9,
                    zorder=12,
                )
                ax.text(
                    x_val,
                    y,
                    z_val + label_offset,
                    f"{z_val:.2f}",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

                connector_points[key].append((x_val, y, z_val))

            start_point = (x[0], y, z[0])
            ax.plot(
                [start_point[0], start_point[0]],
                [start_point[1], start_point[1]],
                [z_base, start_point[2]],
                color="gray",
                linestyle="-",
                linewidth=1.0,
                alpha=0.7,
            )

            arrow_dx = max(knn_range * 0.08, 0.5)
            ax.quiver(
                knn_max - arrow_dx,
                y,
                z_base,
                arrow_dx,
                0,
                0,
                color="black",
                lw=1.2,
                arrow_length_ratio=arrow_length_ratio,
            )
            ax.quiver(
                knn_min,
                y,
                z_base,
                0,
                0,
                z_top - z_base,
                color="black",
                lw=1.2,
                arrow_length_ratio=upward_arrow_ratio,
            )

        for key, points in connector_points.items():
            if len(points) < 2:
                continue
            pts = np.array(points)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="gray",
                linestyle="--",
                linewidth=1.3,
                alpha=0.7,
            )

        ax.view_init(elev=20, azim=-65)
        ax.set_box_aspect((2.0, 2.2, 1.5))

        ax.xaxis.set_pane_color((1, 1, 1, 0.0))
        ax.yaxis.set_pane_color((1, 1, 1, 0.0))
        ax.zaxis.set_pane_color((1, 1, 1, 0.0))
        ax.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.25)

        x_pad = 0.05 * knn_range
        ax.set_xlim(knn_min - x_pad, knn_max + x_pad)

        y_values = list(y_positions.values())
        y_min = min(y_values) - 0.6 * y_spacing if y_values else -y_spacing
        y_max = max(y_values) + 0.6 * y_spacing if y_values else y_spacing
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_base - 0.05 * z_range, z_top)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel(knn_key)
        ax.set_ylabel("")
        ax.set_zlabel("Metric value")

        ax.set_title(title)
        plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

        from matplotlib.lines import Line2D

        legend_handles = []
        for key in ["left", "mid", "right"]:
            method = representative_methods[key]
            knn_value = knn_percentages[method]
            handle = Line2D(
                [0],
                [0],
                marker=representative_markers[key],
                color="none",
                markerfacecolor=representative_colors[key],
                markeredgecolor="black",
                markeredgewidth=0.9,
                linestyle="",
                markersize=8,
                label=f"{method} — KNN {knn_value:.1f}%",
            )
            legend_handles.append(handle)

        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=False,
            )

        return fig
