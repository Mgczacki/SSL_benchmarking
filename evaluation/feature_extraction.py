"""Feature extraction pipeline with disk caching for SSL benchmarking.

This module provides efficient feature extraction from self-supervised models
with automatic disk caching, mixed-precision inference, and multi-GPU support.
It is designed for large-scale evaluation of JEPA and other SSL models across
multiple downstream datasets.

Typical usage::

    extractor = FeatureExtractor(model, dataloader)
    features, labels = extractor.extract()

    # Or with disk caching (recommended for repeated evaluations):
    features, labels = extract_and_cache(
        model, dataloader,
        cache_dir="./cache",
        dataset_name="cifar100",
        model_name="ijepa_vith14",
    )
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known feature dimensions for common SSL architectures.  Used by
# ``get_feature_dim`` so callers can pre-allocate buffers or configure
# downstream heads without instantiating the full model.
# ---------------------------------------------------------------------------
_FEATURE_DIMS: dict[str, int] = {
    "vit_tiny": 192,
    "vit_small": 384,
    "vit_base": 768,
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant": 1408,
    "resnet50": 2048,
    "resnet101": 2048,
    "resnet152": 2048,
}


# ---------------------------------------------------------------------------
# Protocol describing the minimal model interface we expect.
# ---------------------------------------------------------------------------
class FeatureModel(Protocol):
    """Protocol for models that expose an ``extract_features`` method."""

    def extract_features(self, images: torch.Tensor) -> torch.Tensor: ...


class MultiLayerFeatureModel(Protocol):
    """Protocol for models that can return features from specific layers.

    ``extract_features`` must accept an optional ``layer`` keyword argument
    that selects which transformer block (0-indexed from the input) to read
    features from.  When *layer* is ``None`` the model should return the
    default (usually last-layer) representation.
    """

    def extract_features(
        self, images: torch.Tensor, *, layer: Optional[int] = None
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Auto-detect the best available accelerator.

    Preference order: CUDA -> MPS (Apple Silicon) -> CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when wrapped in DataParallel."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Feature processing utilities
# ---------------------------------------------------------------------------

def normalize_features(features: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize feature vectors along *dim*.

    Args:
        features: Tensor of shape ``(N, D)`` (or any shape with a feature
            dimension at position *dim*).
        dim: Dimension along which to normalize. Defaults to ``1``.
        eps: Small constant to avoid division by zero.

    Returns:
        Tensor of the same shape as *features* with unit L2-norm along *dim*.
    """
    return torch.nn.functional.normalize(features, p=2, dim=dim, eps=eps)


def get_feature_dim(model_name: str, arch: str, num_layers: int = 1) -> int:
    """Return the expected feature dimension for a given architecture.

    When *num_layers* > 1 the returned value accounts for the concatenation
    of features from multiple transformer blocks (i.e. ``base_dim * num_layers``).

    Args:
        model_name: Logical model name (e.g. ``"ijepa"``).  Currently unused
            but reserved so that future model-specific overrides can be added
            without breaking the public API.
        arch: Architecture identifier, e.g. ``"vit_huge"`` or ``"resnet50"``.
            Must be a key in the internal dimension table.
        num_layers: Number of layers whose features will be concatenated.

    Returns:
        Integer feature dimension.

    Raises:
        ValueError: If *arch* is not recognised.
    """
    key = arch.lower().replace("-", "_")
    if key not in _FEATURE_DIMS:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Known architectures: {sorted(_FEATURE_DIMS.keys())}"
        )
    return _FEATURE_DIMS[key] * num_layers


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _cache_paths(
    cache_dir: Union[str, Path],
    model_name: str,
    dataset_name: str,
    split: str = "train",
) -> Tuple[Path, Path]:
    """Return ``(features_path, labels_path)`` for a given cache slot."""
    base = Path(cache_dir) / model_name
    features_path = base / f"{dataset_name}_{split}_features.pt"
    labels_path = base / f"{dataset_name}_{split}_labels.pt"
    return features_path, labels_path


def load_cached_features(
    cache_dir: Union[str, Path],
    model_name: str,
    dataset_name: str,
    split: str = "train",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load pre-extracted features and labels from disk.

    Args:
        cache_dir: Root directory used by the caching system.
        model_name: Sub-directory name identifying the model.
        dataset_name: Dataset identifier used when the cache was written.
        split: Dataset split (e.g. ``"train"``, ``"val"``, ``"test"``).

    Returns:
        ``(features, labels)`` tensors on CPU.

    Raises:
        FileNotFoundError: If cache files do not exist at the expected path.
        RuntimeError: If the cached files are corrupted and cannot be loaded.
    """
    feat_path, label_path = _cache_paths(cache_dir, model_name, dataset_name, split)

    if not feat_path.exists():
        raise FileNotFoundError(f"Cached features not found at {feat_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Cached labels not found at {label_path}")

    try:
        features = torch.load(feat_path, map_location="cpu", weights_only=True)
        labels = torch.load(label_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load cached features from {feat_path} or {label_path}. "
            f"The files may be corrupted. Delete them and re-extract. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "Loaded cached features from %s  [%s, %s]",
        feat_path.parent,
        tuple(features.shape),
        tuple(labels.shape),
    )
    return features, labels


def _save_cache(
    features: torch.Tensor,
    labels: torch.Tensor,
    cache_dir: Union[str, Path],
    model_name: str,
    dataset_name: str,
    split: str = "train",
) -> None:
    """Persist features and labels to disk."""
    feat_path, label_path = _cache_paths(cache_dir, model_name, dataset_name, split)
    feat_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(features, feat_path)
    torch.save(labels, label_path)
    logger.info("Saved cached features to %s", feat_path.parent)


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract features from an SSL model over an entire :class:`DataLoader`.

    The extractor runs inference with ``torch.no_grad`` and mixed-precision
    ``autocast`` (on CUDA), accumulates results on CPU to avoid GPU OOM, and
    returns L2-normalised feature / label tensors.

    Args:
        model: Any module that implements ``extract_features(images) -> Tensor``.
            May be wrapped in :class:`~torch.nn.DataParallel`.
        dataloader: A :class:`~torch.utils.data.DataLoader` that yields
            ``(images, labels)`` tuples.
        device: Device to run inference on.  ``None`` means auto-detect.
        normalize: Whether to L2-normalize features before returning.
            Defaults to ``True``.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: Optional[torch.device] = None,
        normalize: bool = True,
    ) -> None:
        self.model = model
        self.dataloader = dataloader
        self.device = device or _get_device()
        self.normalize = normalize

    # ---- public API -------------------------------------------------------

    def extract(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run feature extraction over the full dataloader.

        Returns:
            ``(features, labels)`` where *features* has shape ``(N, D)`` and
            *labels* has shape ``(N,)``.  Both reside on CPU.
        """
        self.model.eval()
        self.model.to(self.device)

        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        use_amp = self.device.type == "cuda"
        raw_model = _unwrap_model(self.model)

        with torch.no_grad():
            for images, labels in tqdm(
                self.dataloader,
                desc="Extracting features",
                leave=False,
            ):
                images = images.to(self.device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        feats = raw_model.extract_features(images)
                else:
                    feats = raw_model.extract_features(images)

                # Move to CPU immediately to keep GPU memory bounded.
                all_features.append(feats.cpu())
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        if self.normalize:
            features = normalize_features(features)

        return features, labels

    def extract_layer(self, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from a *specific* transformer layer.

        This requires the underlying model to accept a ``layer`` keyword
        argument in its ``extract_features`` method (see
        :class:`MultiLayerFeatureModel`).

        Args:
            layer: 0-indexed transformer block to read from.

        Returns:
            ``(features, labels)`` on CPU.
        """
        self.model.eval()
        self.model.to(self.device)

        all_features: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        use_amp = self.device.type == "cuda"
        raw_model = _unwrap_model(self.model)

        with torch.no_grad():
            for images, labels in tqdm(
                self.dataloader,
                desc=f"Extracting features (layer {layer})",
                leave=False,
            ):
                images = images.to(self.device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        feats = raw_model.extract_features(images, layer=layer)
                else:
                    feats = raw_model.extract_features(images, layer=layer)

                all_features.append(feats.cpu())
                all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        if self.normalize:
            features = normalize_features(features)

        return features, labels


# ---------------------------------------------------------------------------
# Multi-layer feature concatenation
# ---------------------------------------------------------------------------

def concat_multi_layer_features(
    model: nn.Module,
    dataloader: DataLoader,
    num_layers: int = 4,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract and concatenate features from the last *num_layers* layers.

    Models like I-JEPA benefit from using a richer representation formed by
    concatenating the [CLS] token (or average-pooled patch tokens) from
    several of the final transformer blocks.

    The underlying model must support a ``layer`` keyword argument in its
    ``extract_features`` method so that individual blocks can be queried.
    See :class:`MultiLayerFeatureModel`.

    Args:
        model: SSL model with per-layer extraction support.
        dataloader: DataLoader yielding ``(images, labels)`` tuples.
        num_layers: Number of *final* layers to concatenate.  For example,
            ``num_layers=4`` on a 24-block ViT reads layers 20, 21, 22, 23.
        device: Inference device.  ``None`` for auto-detect.
        normalize: L2-normalize the concatenated features.

    Returns:
        ``(features, labels)`` where *features* has shape
        ``(N, D * num_layers)`` and resides on CPU.
    """
    device = device or _get_device()
    model.eval()
    model.to(device)

    raw_model = _unwrap_model(model)

    # Determine total number of layers so we can index from the end.
    # We try common attribute names used by timm / HuggingFace ViTs.
    total_layers: Optional[int] = None
    for attr in ("num_layers", "depth", "n_layers", "num_blocks"):
        total_layers = getattr(raw_model, attr, None)
        if total_layers is not None:
            break

    # Fallback: look for a ``blocks`` or ``layers`` sequence.
    if total_layers is None:
        for attr in ("blocks", "layers", "encoder"):
            container = getattr(raw_model, attr, None)
            if container is not None and hasattr(container, "__len__"):
                total_layers = len(container)
                break

    if total_layers is None:
        raise RuntimeError(
            "Cannot determine the number of layers in the model. "
            "Ensure the model exposes one of: num_layers, depth, n_layers, "
            "num_blocks, or has a 'blocks' / 'layers' attribute with a length."
        )

    if num_layers > total_layers:
        raise ValueError(
            f"Requested {num_layers} layers but the model only has {total_layers}."
        )

    target_layers = list(range(total_layers - num_layers, total_layers))
    logger.info(
        "Concatenating features from layers %s (total model layers: %d)",
        target_layers,
        total_layers,
    )

    # We build a *non-normalizing* extractor for each layer and normalize
    # only the final concatenated result (if requested).
    extractor = FeatureExtractor(
        model, dataloader, device=device, normalize=False,
    )

    layer_features: list[torch.Tensor] = []
    labels: Optional[torch.Tensor] = None

    for layer_idx in target_layers:
        feats, layer_labels = extractor.extract_layer(layer_idx)
        layer_features.append(feats)
        # Labels are identical across layers; keep only one copy.
        if labels is None:
            labels = layer_labels

    # (N, D*num_layers)
    features = torch.cat(layer_features, dim=1)

    if normalize:
        features = normalize_features(features)

    assert labels is not None  # guaranteed by at least one iteration
    return features, labels


# ---------------------------------------------------------------------------
# High-level caching entry point
# ---------------------------------------------------------------------------

def extract_and_cache(
    model: nn.Module,
    dataloader: DataLoader,
    cache_dir: Union[str, Path],
    dataset_name: str,
    model_name: str,
    split: str = "train",
    force_recompute: bool = False,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract features with transparent disk caching.

    On the first call the features are extracted from *model* using *dataloader*
    and persisted under *cache_dir*.  Subsequent calls with the same arguments
    load directly from disk, skipping inference entirely.

    Args:
        model: SSL model implementing ``extract_features``.
        dataloader: DataLoader that yields ``(images, labels)``.
        cache_dir: Root directory for the on-disk cache.
        dataset_name: Human-readable dataset identifier (e.g. ``"cifar100"``).
        model_name: Human-readable model identifier used as a sub-directory.
        split: Dataset split name (``"train"``, ``"val"``, ``"test"``).
        force_recompute: If ``True``, ignore existing cache and re-extract.
        device: Inference device.  ``None`` for auto-detect.
        normalize: L2-normalize the extracted features.

    Returns:
        ``(features, labels)`` tensors on CPU.
    """
    feat_path, label_path = _cache_paths(cache_dir, model_name, dataset_name, split)

    # ----- try loading from cache ------------------------------------------
    if not force_recompute and feat_path.exists() and label_path.exists():
        try:
            return load_cached_features(cache_dir, model_name, dataset_name, split)
        except RuntimeError:
            logger.warning(
                "Cached features at %s appear corrupted; re-extracting.",
                feat_path.parent,
            )

    # ----- extract ---------------------------------------------------------
    logger.info(
        "Extracting features for %s / %s / %s ...",
        model_name,
        dataset_name,
        split,
    )
    extractor = FeatureExtractor(
        model, dataloader, device=device, normalize=normalize,
    )
    features, labels = extractor.extract()

    # ----- persist ---------------------------------------------------------
    _save_cache(features, labels, cache_dir, model_name, dataset_name, split)

    return features, labels
