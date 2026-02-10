"""
Fine-tuning evaluation module for SSL benchmarking.

Provides end-to-end fine-tuning of pre-trained self-supervised backbones with
best-practice recipes drawn from the MAE, DINOv2, and I-JEPA literature:

- Layer-wise learning-rate decay (LLRD)
- Cosine annealing with linear warmup
- Mixup / CutMix data augmentation (via ``timm``, when available)
- Label-smoothing cross-entropy
- Mixed-precision training with ``torch.amp``
- Gradient checkpointing for memory-constrained settings
- Low-shot (e.g. ImageNet-1 %) evaluation with stratified sub-sampling

Typical usage::

    from evaluation.finetune import FineTuner

    tuner = FineTuner(backbone, num_classes=1000, epochs=100)
    results = tuner.train(train_loader, val_loader)
    print(results["best_val_top1"])
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalisation (consistent with utils/datasets.py)
# ---------------------------------------------------------------------------
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ===================================================================
# Device helpers
# ===================================================================

def _resolve_device(device: str) -> torch.device:
    """Resolve the ``"auto"`` device string to an actual device.

    Selection order: CUDA -> MPS -> CPU.

    Parameters
    ----------
    device:
        ``"auto"`` for automatic selection, or any string accepted by
        ``torch.device``.

    Returns
    -------
    torch.device
    """
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===================================================================
# Accuracy
# ===================================================================

def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> List[float]:
    """Compute top-k accuracy for the given predictions and ground truth.

    Parameters
    ----------
    output:
        Logits tensor of shape ``(batch_size, num_classes)``.
    target:
        Ground-truth label tensor of shape ``(batch_size,)``.
    topk:
        Tuple of *k* values for which accuracy is computed.

    Returns
    -------
    list[float]
        Accuracy values (in **percent**) for each requested *k*, in the same
        order as *topk*.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        if batch_size == 0:
            return [0.0] * len(topk)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results: List[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


# ===================================================================
# Classification head
# ===================================================================

class _ClassificationWrapper(nn.Module):
    """Thin wrapper that pairs an SSL backbone with a linear head.

    The wrapper calls ``backbone.forward`` to get feature embeddings, then
    passes them through a linear classifier.  If the backbone returns a
    dictionary (common in many JEPA / DINO implementations), the wrapper
    attempts to extract the CLS token embedding automatically.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.zeros_(self.head.bias)
        self._drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        # Handle various backbone output formats.
        if isinstance(features, dict):
            # Try common keys used by JEPA / DINO-style encoders.
            for key in ("x_norm_clstoken", "cls_token", "x", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
            else:
                # Fall back to the first tensor value in the dict.
                features = next(iter(features.values()))
        if isinstance(features, (tuple, list)):
            features = features[0]
        # If we still have a 3-D tensor (batch, tokens, dim), take mean pool.
        if features.dim() == 3:
            features = features.mean(dim=1)
        return self.head(features)


def attach_classification_head(
    model: nn.Module,
    feature_dim: int,
    num_classes: int,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """Wrap an SSL backbone with a linear classification head.

    Parameters
    ----------
    model:
        Pre-trained backbone (frozen or unfrozen -- caller decides).
    feature_dim:
        Dimensionality of the backbone's output feature vector.
    num_classes:
        Number of target classes.
    drop_path_rate:
        Stochastic depth rate applied inside the backbone, if supported.

    Returns
    -------
    nn.Module
        A ``_ClassificationWrapper`` instance.
    """
    return _ClassificationWrapper(
        backbone=model,
        feature_dim=feature_dim,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )


# ===================================================================
# Layer-wise learning-rate decay
# ===================================================================

def _get_num_layers(model: nn.Module) -> int:
    """Heuristically determine the number of "layers" in a ViT-like model.

    The function inspects named parameters for patterns such as
    ``blocks.<idx>``, ``layers.<idx>``, ``layer.<idx>``, or
    ``encoder.layer.<idx>`` and returns ``max(idx) + 1``.  If no such
    pattern is found, it falls back to a flat count of top-level children.
    """
    block_pattern = re.compile(
        r"(?:blocks|layers|layer|encoder\.layer)\.(\d+)\."
    )
    max_idx = -1
    for name, _ in model.named_parameters():
        m = block_pattern.search(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx >= 0:
        return max_idx + 1
    # Fallback: count immediate children that have parameters.
    children_with_params = [
        c for c in model.children()
        if any(True for _ in c.parameters())
    ]
    return max(len(children_with_params), 1)


def _get_layer_id(name: str, num_layers: int) -> int:
    """Return the layer index (0 = shallowest) for a parameter name.

    Embedding / patch-embed parameters -> layer 0.
    Transformer block *i* -> layer *i + 1*.
    Head / norm parameters after the last block -> ``num_layers``.
    """
    if any(
        tok in name
        for tok in (
            "cls_token", "pos_embed", "patch_embed",
            "embed", "mask_token",
        )
    ):
        return 0

    block_pattern = re.compile(
        r"(?:blocks|layers|layer|encoder\.layer)\.(\d+)\."
    )
    m = block_pattern.search(name)
    if m:
        return int(m.group(1)) + 1

    # Everything else is assumed to sit on top of the last block.
    return num_layers


def get_layer_wise_lr_groups(
    model: nn.Module,
    base_lr: float,
    layer_decay: float,
) -> List[Dict[str, Any]]:
    """Build parameter groups with layer-wise learning-rate decay for AdamW.

    Deeper (later) layers receive the full ``base_lr``; earlier layers are
    scaled down by ``layer_decay`` raised to the distance from the top:

        LR(layer) = base_lr * layer_decay ** (num_layers - layer_idx)

    Parameters
    ----------
    model:
        The full model (backbone + head).
    base_lr:
        Learning rate assigned to the deepest layer (and the head).
    layer_decay:
        Multiplicative decay factor per layer.

    Returns
    -------
    list[dict]
        A list of dicts suitable for ``torch.optim.AdamW(params=...)``.
        Each dict has keys ``"params"``, ``"lr"``, and ``"layer_id"``.
    """
    num_layers = _get_num_layers(model)
    # Total depth includes embedding (layer 0) + transformer blocks + head.
    total_depth = num_layers + 1  # +1 for head / final norm

    layer_params: Dict[int, List[torch.nn.Parameter]] = defaultdict(list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = _get_layer_id(name, num_layers)
        layer_params[layer_id].append(param)

    param_groups: List[Dict[str, Any]] = []
    for layer_id in sorted(layer_params.keys()):
        scale = layer_decay ** (total_depth - layer_id)
        param_groups.append(
            {
                "params": layer_params[layer_id],
                "lr": base_lr * scale,
                "layer_id": layer_id,
            }
        )
    return param_groups


# ===================================================================
# Learning-rate schedule: cosine annealing with linear warmup
# ===================================================================

def _build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a cosine-decay LR scheduler with a linear warmup phase.

    During warmup (epoch < warmup_epochs) the LR ramps linearly from
    ``min_lr`` to the group's base LR.  After warmup, the LR follows a
    cosine decay from the base LR to ``min_lr``.

    Parameters
    ----------
    optimizer:
        Optimiser whose parameter groups already contain the target LR.
    warmup_epochs:
        Number of warmup epochs.
    total_epochs:
        Total training epochs (including warmup).
    min_lr:
        Minimum learning rate at the end of training.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(current_epoch: int, group_idx: int) -> float:
        base = base_lrs[group_idx]
        if base == 0:
            return 0.0
        if current_epoch < warmup_epochs:
            # Linear warmup from min_lr to base_lr.
            alpha = current_epoch / max(warmup_epochs, 1)
            return (min_lr + alpha * (base - min_lr)) / base
        # Cosine decay from base_lr to min_lr.
        progress = (current_epoch - warmup_epochs) / max(
            total_epochs - warmup_epochs, 1
        )
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (min_lr + cosine_value * (base - min_lr)) / base

    # ``LambdaLR`` calls the lambda with ``(epoch,)`` but we need the
    # group index too.  We create one lambda per group.
    lambdas = [
        (lambda idx: (lambda epoch: lr_lambda(epoch, idx)))(i)
        for i in range(len(base_lrs))
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


# ===================================================================
# Data augmentation transforms
# ===================================================================

def get_finetune_train_transform(image_size: int = 224) -> transforms.Compose:
    """Build a training transform with RandAugment and RandomErasing.

    Pipeline::

        RandomResizedCrop(image_size)
        RandomHorizontalFlip
        RandAugment(num_ops=2, magnitude=9)
        ToTensor
        Normalize(IMAGENET_MEAN, IMAGENET_STD)
        RandomErasing(p=0.25)

    Parameters
    ----------
    image_size:
        Target spatial resolution.

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.08, 1.0), interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def get_finetune_val_transform(image_size: int = 224) -> transforms.Compose:
    """Standard evaluation transform (resize + center crop + normalise).

    Pipeline::

        Resize(round(image_size * 256/224))
        CenterCrop(image_size)
        ToTensor
        Normalize(IMAGENET_MEAN, IMAGENET_STD)

    Parameters
    ----------
    image_size:
        Target spatial resolution.

    Returns
    -------
    transforms.Compose
    """
    resize_size = int(math.ceil(image_size * 256 / 224))
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ===================================================================
# Mixup / CutMix helper
# ===================================================================

def _build_mixup_fn(
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    num_classes: int = 1000,
) -> Optional[Any]:
    """Try to instantiate ``timm.data.Mixup`` if *timm* is installed.

    If *timm* is not available, return ``None`` so that callers can
    gracefully skip Mixup/CutMix augmentation.

    Parameters
    ----------
    mixup_alpha:
        Alpha for Mixup Beta distribution.  Set to 0 to disable Mixup.
    cutmix_alpha:
        Alpha for CutMix Beta distribution.  Set to 0 to disable CutMix.
    num_classes:
        Number of target classes (needed for one-hot label generation).

    Returns
    -------
    timm.data.Mixup | None
    """
    if mixup_alpha <= 0 and cutmix_alpha <= 0:
        return None
    try:
        from timm.data import Mixup  # type: ignore[import-untyped]

        return Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.0,  # we apply smoothing in the loss instead
            num_classes=num_classes,
        )
    except ImportError:
        logger.warning(
            "timm is not installed -- Mixup / CutMix augmentation will be "
            "skipped.  Install timm (pip install timm) for full recipe."
        )
        return None


# ===================================================================
# Label-smoothing cross-entropy loss
# ===================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Supports both hard (integer) and soft (one-hot / mixed) targets so that
    it works seamlessly with Mixup / CutMix.

    Parameters
    ----------
    smoothing:
        Label-smoothing factor in ``[0, 1)``.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Parameters
        ----------
        logits:
            Raw predictions of shape ``(N, C)``.
        target:
            Either integer labels ``(N,)`` or soft targets ``(N, C)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        if target.dim() == 1:
            # Hard targets -- apply smoothing manually.
            num_classes = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)
            with torch.no_grad():
                smooth_target = torch.full_like(
                    log_probs, self.smoothing / (num_classes - 1)
                )
                smooth_target.scatter_(
                    1, target.unsqueeze(1), 1.0 - self.smoothing
                )
            return (-smooth_target * log_probs).sum(dim=-1).mean()
        else:
            # Soft targets (from Mixup/CutMix) -- treat as distribution.
            log_probs = F.log_softmax(logits, dim=-1)
            return (-target * log_probs).sum(dim=-1).mean()


# ===================================================================
# FineTuner
# ===================================================================

class FineTuner:
    """End-to-end fine-tuning evaluator for SSL backbones.

    Implements the standard fine-tuning recipe used by MAE, I-JEPA, and
    DINOv2: full backbone unfreezing, layer-wise LR decay, cosine schedule
    with warmup, Mixup/CutMix, label smoothing, and mixed-precision
    training.

    Parameters
    ----------
    model:
        Pre-trained backbone module.  All parameters will be unfrozen
        during fine-tuning.
    num_classes:
        Number of target classes for the classification head.
    lr:
        Peak learning rate (assigned to the deepest layer / head).
    weight_decay:
        Weight decay for AdamW.
    epochs:
        Total number of training epochs.
    batch_size:
        Mini-batch size (used only by ``low_shot_finetune`` when creating
        data loaders internally; ``train`` expects pre-built loaders).
    layer_decay:
        Layer-wise LR decay factor.  Each layer closer to the input is
        scaled by an additional factor of ``layer_decay``.
    warmup_epochs:
        Number of linear-warmup epochs at the start of training.
    mixup_alpha:
        Alpha for Mixup Beta distribution (0 disables Mixup).
    cutmix_alpha:
        Alpha for CutMix Beta distribution (0 disables CutMix).
    label_smoothing:
        Label-smoothing coefficient for the loss function.
    drop_path_rate:
        Stochastic depth rate injected into the backbone (if the wrapper
        supports it).
    device:
        Target device.  ``"auto"`` selects CUDA > MPS > CPU.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        epochs: int = 100,
        batch_size: int = 64,
        layer_decay: float = 0.65,
        warmup_epochs: int = 5,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        label_smoothing: float = 0.1,
        drop_path_rate: float = 0.1,
        device: str = "auto",
    ) -> None:
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.layer_decay = layer_decay
        self.warmup_epochs = warmup_epochs
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.drop_path_rate = drop_path_rate
        self.device = _resolve_device(device)

        # Deep-copy so that the original pre-trained weights are not mutated.
        self._backbone = copy.deepcopy(model)

        # Detect feature dimension by running a dummy forward pass.
        self._feature_dim = self._infer_feature_dim(self._backbone)

        # Build classification wrapper (backbone + linear head).
        self._model = attach_classification_head(
            self._backbone,
            self._feature_dim,
            self.num_classes,
            drop_path_rate=self.drop_path_rate,
        )

        # State populated during training.
        self._best_state_dict: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Feature-dimension inference
    # ------------------------------------------------------------------

    def _infer_feature_dim(self, backbone: nn.Module) -> int:
        """Run a single dummy forward pass to discover the output dimension.

        Inspects the backbone for common attributes first (``embed_dim``,
        ``num_features``, ``feature_dim``).  Falls back to a forward pass
        with a 1x3x224x224 tensor if no attribute is found.
        """
        for attr in ("embed_dim", "num_features", "feature_dim", "hidden_size"):
            if hasattr(backbone, attr):
                dim = getattr(backbone, attr)
                if isinstance(dim, int) and dim > 0:
                    return dim

        # Forward-pass fallback.
        backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = backbone(dummy)
        if isinstance(out, dict):
            for key in ("x_norm_clstoken", "cls_token", "x", "last_hidden_state"):
                if key in out:
                    out = out[key]
                    break
            else:
                out = next(iter(out.values()))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 3:
            out = out.mean(dim=1)
        return out.shape[-1]

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    @staticmethod
    def _enable_gradient_checkpointing(model: nn.Module) -> None:
        """Enable gradient checkpointing on the backbone if supported.

        Works with ``timm`` ViT models (``set_grad_checkpointing``) as
        well as HuggingFace models (``gradient_checkpointing_enable``).
        """
        backbone = model.backbone if hasattr(model, "backbone") else model
        if hasattr(backbone, "set_grad_checkpointing"):
            backbone.set_grad_checkpointing(enable=True)
            logger.info("Gradient checkpointing enabled (timm-style).")
        elif hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled (HF-style).")
        else:
            logger.debug(
                "Backbone does not expose a gradient-checkpointing API -- "
                "skipping."
            )

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scaler: Optional[torch.amp.GradScaler],
        mixup_fn: Optional[Any],
        epoch: int,
    ) -> float:
        """Run one training epoch and return the mean loss."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        use_amp = scaler is not None

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            if mixup_fn is not None:
                images, targets = mixup_fn(images, targets)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type, enabled=use_amp,
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if use_amp:
                scaler.scale(loss).backward()  # type: ignore[union-attr]
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
    ) -> Tuple[float, float]:
        """Evaluate the model and return ``(top1, top5)`` accuracy (%)."""
        model.eval()
        top1_sum = 0.0
        top5_sum = 0.0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=(self.device.type == "cuda"),
            ):
                logits = model(images)

            batch_size = targets.size(0)
            # Clamp topk to number of classes to avoid runtime errors on
            # datasets with fewer than 5 classes.
            maxk = min(5, logits.size(-1))
            topk_vals = (1,) if maxk < 5 else (1, 5)
            acc = accuracy(logits, targets, topk=topk_vals)
            top1_sum += acc[0] * batch_size
            top5_sum += (acc[1] if len(acc) > 1 else acc[0]) * batch_size
            total_samples += batch_size

        if total_samples == 0:
            return 0.0, 0.0
        return top1_sum / total_samples, top5_sum / total_samples

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        gradient_checkpointing: bool = False,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Fine-tune the backbone end-to-end and evaluate on the validation set.

        The full backbone is unfrozen and trained jointly with the linear
        classification head.

        Parameters
        ----------
        train_loader:
            ``DataLoader`` for the training set.
        val_loader:
            ``DataLoader`` for the validation set.
        gradient_checkpointing:
            If ``True``, enable gradient checkpointing to save memory at
            the cost of extra computation.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``best_val_top1`` (float): Best validation top-1 accuracy (%).
            - ``best_val_top5`` (float): Best validation top-5 accuracy (%).
            - ``training_history`` (list[dict]): Per-epoch metrics with keys
              ``epoch``, ``train_loss``, ``val_top1``, ``val_top5``, ``lr``.
        """
        _set_seed(seed)

        model = self._model
        model.to(self.device)

        # Unfreeze every parameter in the backbone + head.
        for param in model.parameters():
            param.requires_grad = True

        if gradient_checkpointing:
            self._enable_gradient_checkpointing(model)

        # --- Optimizer with layer-wise LR decay ---------------------------
        param_groups = get_layer_wise_lr_groups(
            model, base_lr=self.lr, layer_decay=self.layer_decay,
        )
        # Apply weight_decay to all groups.
        no_decay_keywords = ("bias", "norm", "ln", "bn", "layernorm", "batchnorm")
        refined_groups: List[Dict[str, Any]] = []
        for group in param_groups:
            decay_params: List[nn.Parameter] = []
            no_decay_params: List[nn.Parameter] = []
            for p in group["params"]:
                # Determine the name (best-effort) to decide on weight decay.
                name = ""
                for n, pp in model.named_parameters():
                    if pp is p:
                        name = n.lower()
                        break
                if any(kw in name for kw in no_decay_keywords) or p.dim() == 1:
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
            if decay_params:
                refined_groups.append(
                    {
                        "params": decay_params,
                        "lr": group["lr"],
                        "weight_decay": self.weight_decay,
                    }
                )
            if no_decay_params:
                refined_groups.append(
                    {
                        "params": no_decay_params,
                        "lr": group["lr"],
                        "weight_decay": 0.0,
                    }
                )

        optimizer = torch.optim.AdamW(refined_groups, lr=self.lr, weight_decay=self.weight_decay)

        # --- LR scheduler -------------------------------------------------
        scheduler = _build_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=self.warmup_epochs,
            total_epochs=self.epochs,
        )

        # --- Loss ----------------------------------------------------------
        criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)

        # --- Mixup / CutMix -----------------------------------------------
        mixup_fn = _build_mixup_fn(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            num_classes=self.num_classes,
        )

        # --- Mixed precision -----------------------------------------------
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        # --- Training loop -------------------------------------------------
        best_top1 = 0.0
        best_top5 = 0.0
        history: List[Dict[str, Any]] = []

        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, mixup_fn,
                epoch,
            )
            scheduler.step()

            val_top1, val_top5 = self._validate(model, val_loader)
            current_lr = optimizer.param_groups[0]["lr"]

            logger.info(
                "Epoch %3d/%d  |  train_loss=%.4f  |  val_top1=%.2f  |  "
                "val_top5=%.2f  |  lr=%.2e",
                epoch + 1, self.epochs, train_loss, val_top1, val_top5,
                current_lr,
            )

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_top1": val_top1,
                    "val_top5": val_top5,
                    "lr": current_lr,
                }
            )

            if val_top1 > best_top1:
                best_top1 = val_top1
                best_top5 = val_top5
                self._best_state_dict = copy.deepcopy(model.state_dict())

        return {
            "best_val_top1": best_top1,
            "best_val_top5": best_top5,
            "training_history": history,
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save the best model checkpoint to disk.

        Parameters
        ----------
        path:
            File path for the checkpoint ``.pt`` file.

        Raises
        ------
        RuntimeError
            If ``train`` has not been called yet (no best checkpoint exists).
        """
        if self._best_state_dict is None:
            raise RuntimeError(
                "No best checkpoint available -- call train() first."
            )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._best_state_dict, path)
        logger.info("Saved best checkpoint to %s", path)

    def load_checkpoint(self, path: str) -> None:
        """Load a checkpoint into the fine-tuned model.

        Parameters
        ----------
        path:
            File path for the checkpoint ``.pt`` file.
        """
        state = torch.load(path, map_location=self.device, weights_only=True)
        self._model.load_state_dict(state)
        logger.info("Loaded checkpoint from %s", path)


# ===================================================================
# Low-shot fine-tuning
# ===================================================================

def _stratified_subsample(
    dataset: Dataset,
    percent: float,
    num_classes: int,
    seed: int = 42,
) -> Subset:
    """Create a stratified subset containing ``percent`` of the data.

    Ensures every class is represented with at least one sample (if the
    original dataset has at least one sample per class).

    Parameters
    ----------
    dataset:
        Full training dataset.  Must support ``len`` and integer indexing
        that returns ``(image, label)`` tuples.  The ``targets`` attribute
        is used when available (e.g. ``torchvision`` datasets), falling
        back to iterating over the dataset otherwise.
    percent:
        Fraction to keep, e.g. ``0.01`` for 1 %.
    num_classes:
        Total number of classes in the dataset.
    seed:
        Random seed for deterministic sampling.

    Returns
    -------
    Subset
    """
    # Collect per-class indices.
    class_indices: Dict[int, List[int]] = defaultdict(list)

    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            targets = targets.tolist()
        for idx, label in enumerate(targets):
            class_indices[label].append(idx)
    elif hasattr(dataset, "samples"):
        # ImageFolder-style dataset.
        for idx, (_, label) in enumerate(dataset.samples):
            class_indices[label].append(idx)
    else:
        # Slow fallback: iterate the dataset.
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_indices[label].append(idx)

    rng = random.Random(seed)
    selected: List[int] = []

    for cls in sorted(class_indices.keys()):
        indices = class_indices[cls]
        n_keep = max(1, round(len(indices) * percent))
        n_keep = min(n_keep, len(indices))
        sampled = rng.sample(indices, n_keep)
        selected.extend(sampled)

    selected.sort()
    return Subset(dataset, selected)


def low_shot_finetune(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    num_classes: int,
    percent: float = 0.01,
    seed: int = 42,
    epochs: int = 100,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Fine-tune a backbone on a small fraction of the training data.

    This implements the standard *low-shot* evaluation protocol (e.g.
    ImageNet-1 %) used in the SSL literature.  Training data is
    sub-sampled in a stratified manner to maintain class balance, and
    augmentation intensity is reduced to prevent overfitting on the tiny
    training set.

    Parameters
    ----------
    model:
        Pre-trained backbone.
    train_dataset:
        Full training dataset (will be sub-sampled).
    val_dataset:
        Validation dataset (used in its entirety).
    num_classes:
        Number of target classes.
    percent:
        Fraction of training data to keep (e.g. ``0.01`` for 1 %).
    seed:
        Random seed for reproducibility (both sub-sampling and training).
    epochs:
        Number of fine-tuning epochs.
    **kwargs:
        Additional keyword arguments forwarded to ``FineTuner.__init__``
        (e.g. ``lr``, ``weight_decay``, ``device``).

    Returns
    -------
    dict
        Result dictionary identical to ``FineTuner.train``, with an
        additional key ``num_train_samples`` indicating the effective
        training-set size.
    """
    _set_seed(seed)

    # --- Stratified sub-sampling ------------------------------------------
    train_subset = _stratified_subsample(
        train_dataset, percent=percent, num_classes=num_classes, seed=seed,
    )
    logger.info(
        "Low-shot fine-tuning: using %d / %d training samples (%.1f%%).",
        len(train_subset), len(train_dataset), percent * 100,
    )

    # --- Reduced augmentation for low-shot --------------------------------
    # Scale down Mixup/CutMix to avoid over-regularising a tiny dataset.
    low_shot_defaults: Dict[str, Any] = {
        "mixup_alpha": 0.2,
        "cutmix_alpha": 0.2,
        "label_smoothing": 0.05,
        "warmup_epochs": min(5, epochs // 10),
        "epochs": epochs,
    }
    # User-provided kwargs override the defaults.
    merged_kwargs = {**low_shot_defaults, **kwargs}

    batch_size = merged_kwargs.pop("batch_size", 64)

    # --- Build data loaders -----------------------------------------------
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )

    # --- Fine-tune --------------------------------------------------------
    tuner = FineTuner(
        model=model,
        num_classes=num_classes,
        batch_size=batch_size,
        **merged_kwargs,
    )
    results = tuner.train(train_loader, val_loader, seed=seed)
    results["num_train_samples"] = len(train_subset)
    return results
