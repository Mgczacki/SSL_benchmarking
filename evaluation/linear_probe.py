"""Linear probe evaluation for self-supervised learning benchmarks.

This module provides three approaches for linear probe evaluation:
1. **LinearProbeTrainer** -- Full SGD-based training of a linear classifier on
   frozen features. This is the standard protocol used in most SSL papers.
2. **lr_sweep** -- Convenience wrapper that trains probes across a grid of
   learning rates and reports the best result.
3. **quick_linear_probe** -- A fast alternative that uses scikit-learn's
   LogisticRegression (L-BFGS solver) for rapid validation.  Much faster
   than SGD training (minutes vs. hours) at the cost of slightly lower
   accuracy.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    """Return an explicit ``torch.device`` from the *device* string.

    When *device* is ``"auto"`` the function picks CUDA if available, then
    MPS (Apple Silicon), and finally falls back to CPU.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    """Compute top-k accuracy for the given *output* logits and *target* labels.

    Parameters
    ----------
    output : torch.Tensor
        Logit tensor of shape ``(N, C)``.
    target : torch.Tensor
        Ground-truth label tensor of shape ``(N,)``.
    topk : tuple[int, ...]
        Tuple of *k* values for which to compute accuracy.

    Returns
    -------
    list[float]
        Accuracy values (in percent, 0--100) for each requested *k*.
    """
    with torch.no_grad():
        maxk = min(max(topk), output.size(1))
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results: list[float] = []
        for k in topk:
            k_clamped = min(k, output.size(1))
            correct_k = correct[:k_clamped].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size).item())
        return results


def _normalize(features: torch.Tensor) -> torch.Tensor:
    """L2-normalise *features* along the last dimension."""
    return F.normalize(features, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Evaluate helper (module-level)
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 512,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Evaluate a linear classifier and return top-1 and top-5 accuracy.

    Parameters
    ----------
    model : nn.Module
        The linear classifier (typically ``nn.Linear``).
    features : torch.Tensor
        Feature tensor of shape ``(N, D)``.
    labels : torch.Tensor
        Label tensor of shape ``(N,)``.
    batch_size : int
        Batch size used during evaluation to avoid OOM on large sets.
    device : torch.device | None
        Device on which to evaluate.  If *None* the device of the model
        parameters is used.

    Returns
    -------
    tuple[float, float]
        ``(top1_accuracy, top5_accuracy)`` in percent (0--100).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_top1_correct = 0
    all_top5_correct = 0
    total = 0

    with torch.no_grad():
        for feat_batch, label_batch in loader:
            feat_batch = feat_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            logits = model(feat_batch)

            top1, top5 = _accuracy(logits, label_batch, topk=(1, 5))
            n = label_batch.size(0)
            all_top1_correct += top1 * n / 100.0
            all_top5_correct += top5 * n / 100.0
            total += n

    top1_acc = all_top1_correct / total * 100.0
    top5_acc = all_top5_correct / total * 100.0
    return top1_acc, top5_acc


# ---------------------------------------------------------------------------
# LinearProbeTrainer
# ---------------------------------------------------------------------------

class LinearProbeTrainer:
    """Train a linear classifier on top of frozen SSL features.

    This is the standard linear evaluation protocol described in most
    self-supervised learning papers.  A single ``nn.Linear`` layer is
    trained using SGD with momentum on pre-extracted features while the
    backbone is kept frozen.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the input feature vectors.
    num_classes : int
        Number of target classes.
    lr : float
        Initial learning rate for SGD.
    momentum : float
        SGD momentum factor.
    weight_decay : float
        L2 weight-decay coefficient.
    epochs : int
        Total number of training epochs.
    batch_size : int
        Mini-batch size used for training and evaluation.
    lr_schedule : str
        Learning-rate schedule: ``"cosine"`` for cosine annealing or
        ``"step"`` for step decay (factor 0.1 at epochs 60 and 80).
    warmup_epochs : int
        Number of linear warmup epochs.  Set to 0 to disable warmup.
    normalize_features : bool
        If *True*, L2-normalise feature vectors before training.
    device : str
        Device string.  ``"auto"`` selects CUDA > MPS > CPU automatically.

    Example
    -------
    >>> trainer = LinearProbeTrainer(feature_dim=768, num_classes=1000)
    >>> results = trainer.train(train_feats, train_lbls, val_feats, val_lbls)
    >>> print(f"Best val top-1: {results['best_val_top1']:.2f}%")
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        epochs: int = 100,
        batch_size: int = 256,
        lr_schedule: str = "cosine",
        warmup_epochs: int = 0,
        normalize_features: bool = True,
        device: str = "auto",
    ) -> None:
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr_schedule = lr_schedule
        self.warmup_epochs = warmup_epochs
        self.normalize_features = normalize_features
        self.device = _resolve_device(device)

        if lr_schedule not in ("cosine", "step"):
            raise ValueError(
                f"Unknown lr_schedule '{lr_schedule}'. Expected 'cosine' or 'step'."
            )

    # ---- internal helpers ------------------------------------------------

    def _build_model(self) -> nn.Linear:
        """Create a fresh linear head and move it to *self.device*."""
        model = nn.Linear(self.feature_dim, self.num_classes)
        nn.init.trunc_normal_(model.weight, std=0.01)
        nn.init.zeros_(model.bias)
        return model.to(self.device)

    def _get_lr(self, epoch: int, base_lr: float) -> float:
        """Compute the learning rate for *epoch* respecting warmup + schedule.

        During warmup the LR increases linearly from 0 to *base_lr*.
        After warmup:
        - ``cosine``: cosine annealing to 0.
        - ``step``: decay by 0.1 at epochs 60 and 80.
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            return base_lr * (epoch + 1) / self.warmup_epochs

        if self.lr_schedule == "cosine":
            effective_epoch = epoch - self.warmup_epochs
            effective_total = self.epochs - self.warmup_epochs
            return base_lr * 0.5 * (1.0 + math.cos(math.pi * effective_epoch / effective_total))

        # Step schedule: decay by 0.1 at milestones 60 and 80
        factor = 1.0
        for milestone in (60, 80):
            if epoch >= milestone:
                factor *= 0.1
        return base_lr * factor

    def _set_lr(self, optimizer: torch.optim.Optimizer, lr: float) -> None:
        """Set the learning rate on all parameter groups of *optimizer*."""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        """Optionally L2-normalise and move features to float32."""
        features = features.float()
        if self.normalize_features:
            features = _normalize(features)
        return features

    # ---- public API ------------------------------------------------------

    def train(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Train a linear probe and return evaluation metrics.

        Parameters
        ----------
        train_features : torch.Tensor
            Training feature tensor of shape ``(N_train, feature_dim)``.
        train_labels : torch.Tensor
            Training labels of shape ``(N_train,)``.
        val_features : torch.Tensor
            Validation feature tensor of shape ``(N_val, feature_dim)``.
        val_labels : torch.Tensor
            Validation labels of shape ``(N_val,)``.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:

            - ``best_val_top1`` (float): Best validation top-1 accuracy.
            - ``best_val_top5`` (float): Best validation top-5 accuracy.
            - ``final_val_top1`` (float): Final-epoch validation top-1 accuracy.
            - ``final_val_top5`` (float): Final-epoch validation top-5 accuracy.
            - ``training_history`` (list[dict]): Per-epoch metrics.
        """
        _set_seed(seed)

        # Prepare features
        train_features = self._prepare_features(train_features)
        val_features = self._prepare_features(val_features)
        train_labels = train_labels.long()
        val_labels = val_labels.long()

        # Build model, optimizer, loss
        model = self._build_model()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Data loader
        train_dataset = TensorDataset(train_features, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        # Mixed-precision scaler (CUDA only)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Training loop
        best_val_top1 = 0.0
        best_val_top5 = 0.0
        training_history: list[dict[str, Any]] = []

        pbar = tqdm(range(self.epochs), desc="Linear probe", leave=True)
        for epoch in pbar:
            # -- adjust lr --
            current_lr = self._get_lr(epoch, self.lr)
            self._set_lr(optimizer, current_lr)

            # -- train one epoch --
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for feat_batch, label_batch in train_loader:
                feat_batch = feat_batch.to(self.device, non_blocking=True)
                label_batch = label_batch.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=self.device.type, enabled=use_amp
                ):
                    logits = model(feat_batch)
                    loss = criterion(logits, label_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_size_actual = label_batch.size(0)
                running_loss += loss.item() * batch_size_actual
                running_correct += (logits.argmax(dim=1) == label_batch).sum().item()
                running_total += batch_size_actual

            train_loss = running_loss / running_total
            train_acc = running_correct / running_total * 100.0

            # -- validate --
            val_top1, val_top5 = evaluate(
                model, val_features, val_labels,
                batch_size=self.batch_size, device=self.device,
            )

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                best_val_top5 = val_top5

            training_history.append({
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_top1": val_top1,
                "val_top5": val_top5,
            })

            pbar.set_postfix(
                lr=f"{current_lr:.1e}",
                train_loss=f"{train_loss:.4f}",
                val_acc=f"{val_top1:.2f}",
            )

        final_val_top1, final_val_top5 = evaluate(
            model, val_features, val_labels,
            batch_size=self.batch_size, device=self.device,
        )

        results: dict[str, Any] = {
            "best_val_top1": best_val_top1,
            "best_val_top5": best_val_top5,
            "final_val_top1": final_val_top1,
            "final_val_top5": final_val_top5,
            "training_history": training_history,
        }

        logger.info(
            "Linear probe complete -- best val top-1: %.2f%%, top-5: %.2f%%",
            best_val_top1, best_val_top5,
        )
        return results


# ---------------------------------------------------------------------------
# LR Sweep
# ---------------------------------------------------------------------------

def lr_sweep(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    feature_dim: int,
    lr_values: list[float] | None = None,
    epochs: int = 100,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train linear probes across a grid of learning rates and report the best.

    This is useful when the optimal learning rate is unknown.  Each LR
    candidate trains an independent probe from scratch.

    Parameters
    ----------
    train_features : torch.Tensor
        Training features of shape ``(N, D)``.
    train_labels : torch.Tensor
        Training labels of shape ``(N,)``.
    val_features : torch.Tensor
        Validation features of shape ``(N, D)``.
    val_labels : torch.Tensor
        Validation labels of shape ``(N,)``.
    num_classes : int
        Number of target classes.
    feature_dim : int
        Dimensionality of input features.
    lr_values : list[float] | None
        Learning rate grid to sweep.  If *None* the default grid
        ``[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`` is used.
    epochs : int
        Number of training epochs per run.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`LinearProbeTrainer`.

    Returns
    -------
    dict[str, Any]
        Dictionary with:

        - ``best_lr`` (float): Learning rate that achieved the best top-1.
        - ``best_val_top1`` (float): Best top-1 accuracy across all LRs.
        - ``best_val_top5`` (float): Corresponding top-5 accuracy.
        - ``all_results`` (dict[float, dict]): Per-LR result dictionaries.
    """
    if lr_values is None:
        lr_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    all_results: dict[float, dict[str, Any]] = {}
    best_lr = lr_values[0]
    best_top1 = 0.0
    best_top5 = 0.0

    print(f"Starting LR sweep over {len(lr_values)} values: {lr_values}")
    print(f"  feature_dim={feature_dim}, num_classes={num_classes}, epochs={epochs}")
    print("-" * 72)

    for i, lr in enumerate(lr_values):
        start_time = time.time()
        print(f"[{i + 1}/{len(lr_values)}] Training with lr={lr} ...")

        trainer = LinearProbeTrainer(
            feature_dim=feature_dim,
            num_classes=num_classes,
            lr=lr,
            epochs=epochs,
            **kwargs,
        )
        result = trainer.train(
            train_features, train_labels, val_features, val_labels,
        )
        all_results[lr] = result

        elapsed = time.time() - start_time
        print(
            f"  -> best_val_top1={result['best_val_top1']:.2f}%, "
            f"best_val_top5={result['best_val_top5']:.2f}%, "
            f"time={elapsed:.1f}s"
        )

        if result["best_val_top1"] > best_top1:
            best_top1 = result["best_val_top1"]
            best_top5 = result["best_val_top5"]
            best_lr = lr

    print("-" * 72)
    print(
        f"LR sweep complete.  Best lr={best_lr}, "
        f"top-1={best_top1:.2f}%, top-5={best_top5:.2f}%"
    )

    return {
        "best_lr": best_lr,
        "best_val_top1": best_top1,
        "best_val_top5": best_top5,
        "all_results": all_results,
    }


# ---------------------------------------------------------------------------
# Quick linear probe (sklearn)
# ---------------------------------------------------------------------------

def quick_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    C_values: list[float] | None = None,
) -> dict[str, Any]:
    """Fast linear probe using scikit-learn's LogisticRegression.

    This is significantly faster than full SGD training (minutes instead of
    hours) and is useful for quick validation during development.  Accuracy
    is typically slightly lower than the SGD protocol.

    Parameters
    ----------
    train_features : torch.Tensor
        Training features of shape ``(N, D)``.
    train_labels : torch.Tensor
        Training labels of shape ``(N,)``.
    val_features : torch.Tensor
        Validation features of shape ``(N, D)``.
    val_labels : torch.Tensor
        Validation labels of shape ``(N,)``.
    C_values : list[float] | None
        Inverse regularization strengths to sweep.  If *None* the default
        grid ``[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]`` is used.

    Returns
    -------
    dict[str, Any]
        Dictionary with:

        - ``best_C`` (float): Regularization strength with the best top-1.
        - ``best_acc`` (float): Best top-1 accuracy (percent).
        - ``all_results`` (dict[float, float]): Accuracy for each C value.
    """
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Move to numpy
    X_train = train_features.cpu().float().numpy()
    y_train = train_labels.cpu().numpy()
    X_val = val_features.cpu().float().numpy()
    y_val = val_labels.cpu().numpy()

    # L2-normalise features
    norms = np.linalg.norm(X_train, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    X_train = X_train / norms

    norms = np.linalg.norm(X_val, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    X_val = X_val / norms

    all_results: dict[float, float] = {}
    best_C = C_values[0]
    best_acc = 0.0

    print(f"Quick linear probe -- sweeping C over {C_values}")

    for C in C_values:
        start_time = time.time()

        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=1000,
            multi_class="multinomial",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val) * 100.0

        elapsed = time.time() - start_time
        all_results[C] = acc
        print(f"  C={C:<10g}  val_acc={acc:.2f}%  ({elapsed:.1f}s)")

        if acc > best_acc:
            best_acc = acc
            best_C = C

    print(f"Best: C={best_C}, val_acc={best_acc:.2f}%")

    return {
        "best_C": best_C,
        "best_acc": best_acc,
        "all_results": all_results,
    }
