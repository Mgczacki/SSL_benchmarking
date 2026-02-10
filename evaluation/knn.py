"""k-Nearest Neighbors evaluator for self-supervised learning benchmarks.

Standard evaluation protocol used to measure the quality of frozen feature
representations.  The evaluator supports both a pure-PyTorch implementation
and an optional FAISS-accelerated path that is significantly faster for
large-scale datasets such as ImageNet-1K (~1.28M training features).

Typical settings
----------------
* ImageNet-1K : k=20,  temperature=0.07
* CIFAR-100   : k=200, temperature=0.07

References
----------
* Wu et al., "Unsupervised Feature Learning via Non-Parametric Instance
  Discrimination", CVPR 2018.
* Caron et al., "Emerging Properties in Self-Supervised Vision Transformers"
  (DINO), ICCV 2021.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FAISS import
# ---------------------------------------------------------------------------
try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """L2-normalize feature vectors along the last dimension."""
    return F.normalize(x, p=2, dim=-1)


def _to_contiguous_float32(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is contiguous float32 (required by FAISS)."""
    return x.contiguous().float()


# ---------------------------------------------------------------------------
# KNNEvaluator
# ---------------------------------------------------------------------------

class KNNEvaluator:
    """Weighted k-NN classifier on frozen representations.

    For every query (validation) sample the *k* nearest neighbours in the
    training feature bank are retrieved.  Class votes are weighted by
    ``exp(similarity / temperature)`` where similarity is the cosine
    similarity (i.e. the inner product of L2-normalised features).

    Parameters
    ----------
    k : int
        Number of nearest neighbours.  Common choices: 20 (ImageNet),
        200 (CIFAR).
    temperature : float
        Softmax temperature applied to the similarity scores before
        accumulating votes.  Lower values make the weighting sharper.
    use_faiss : bool
        If *True* and the ``faiss`` package is installed, use a FAISS
        ``IndexFlatIP`` index for the nearest-neighbour search.  This
        can be orders of magnitude faster for large training sets.  If
        ``faiss`` is not available the evaluator falls back to the pure
        PyTorch path transparently.
    """

    def __init__(
        self,
        k: int = 20,
        temperature: float = 0.07,
        use_faiss: bool = False,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.k = k
        self.temperature = temperature

        if use_faiss and not _FAISS_AVAILABLE:
            logger.warning(
                "use_faiss=True but faiss is not installed. "
                "Falling back to pure PyTorch k-NN."
            )
        self.use_faiss = use_faiss and _FAISS_AVAILABLE

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        num_classes: int,
        batch_size: int = 256,
    ) -> dict[str, object]:
        """Run the k-NN evaluation.

        All features are L2-normalised internally so the caller does **not**
        need to normalise them beforehand.

        Parameters
        ----------
        train_features : torch.Tensor
            Feature bank of shape ``(N_train, D)``.
        train_labels : torch.Tensor
            Integer class labels of shape ``(N_train,)``.
        val_features : torch.Tensor
            Query features of shape ``(N_val, D)``.
        val_labels : torch.Tensor
            Integer class labels of shape ``(N_val,)``.
        num_classes : int
            Total number of classes (used for the vote accumulator).
        batch_size : int
            Number of query samples processed at once.  Lower values use
            less memory at the cost of slightly slower execution.

        Returns
        -------
        dict
            ``top1_accuracy``    -- float, top-1 classification accuracy.
            ``top5_accuracy``    -- float, top-5 classification accuracy.
            ``per_class_accuracy`` -- torch.Tensor of shape ``(num_classes,)``
                with the per-class accuracy (NaN for classes with no val
                samples).
        """
        # Validate inputs -------------------------------------------------- #
        if train_features.ndim != 2 or val_features.ndim != 2:
            raise ValueError(
                "Features must be 2-D tensors of shape (N, D). "
                f"Got train_features.shape={train_features.shape}, "
                f"val_features.shape={val_features.shape}."
            )
        if train_features.shape[1] != val_features.shape[1]:
            raise ValueError(
                "Feature dimensions must match. "
                f"train={train_features.shape[1]}, val={val_features.shape[1]}."
            )

        k = min(self.k, train_features.shape[0])

        # L2 normalise ------------------------------------------------------ #
        train_features = _l2_normalize(train_features.float())
        val_features = _l2_normalize(val_features.float())
        train_labels = train_labels.long()
        val_labels = val_labels.long()

        # Dispatch ---------------------------------------------------------- #
        if self.use_faiss:
            predictions, top5_predictions = self._evaluate_faiss(
                train_features,
                train_labels,
                val_features,
                num_classes,
                k,
                batch_size,
            )
        else:
            predictions, top5_predictions = self._evaluate_pytorch(
                train_features,
                train_labels,
                val_features,
                num_classes,
                k,
                batch_size,
            )

        # Metrics ----------------------------------------------------------- #
        val_labels_device = val_labels.to(predictions.device)

        top1_correct = predictions.eq(val_labels_device).sum().item()
        top1_accuracy = top1_correct / val_labels_device.shape[0]

        top5_correct = top5_predictions.eq(
            val_labels_device.unsqueeze(1)
        ).any(dim=1).sum().item()
        top5_accuracy = top5_correct / val_labels_device.shape[0]

        # Per-class accuracy ------------------------------------------------ #
        per_class_accuracy = torch.full(
            (num_classes,), float("nan"), dtype=torch.float32
        )
        for c in range(num_classes):
            mask = val_labels_device == c
            if mask.any():
                per_class_accuracy[c] = (
                    predictions[mask].eq(c).float().mean().item()
                )

        return {
            "top1_accuracy": top1_accuracy,
            "top5_accuracy": top5_accuracy,
            "per_class_accuracy": per_class_accuracy,
        }

    # --------------------------------------------------------------------- #
    # Pure PyTorch path
    # --------------------------------------------------------------------- #

    def _evaluate_pytorch(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        num_classes: int,
        k: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batched k-NN using PyTorch matrix multiplications.

        Returns the top-1 predicted labels and the top-5 predicted labels for
        every validation sample.
        """
        device = val_features.device
        n_val = val_features.shape[0]

        # Move training data to the same device
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)

        all_top1: list[torch.Tensor] = []
        all_top5: list[torch.Tensor] = []

        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            query = val_features[start:end]  # (B, D)

            # Cosine similarity (features are already L2-normalised)
            similarity = query @ train_features.T  # (B, N_train)

            # Top-k neighbours
            topk_sim, topk_idx = similarity.topk(k, dim=1)  # (B, k)

            # Retrieve neighbour labels
            topk_labels = train_labels[topk_idx]  # (B, k)

            # Temperature-weighted voting
            weights = torch.exp(topk_sim / self.temperature)  # (B, k)

            # Accumulate votes per class
            # votes shape: (B, num_classes)
            votes = torch.zeros(
                query.shape[0], num_classes, device=device, dtype=weights.dtype
            )
            votes.scatter_add_(
                dim=1,
                index=topk_labels,
                src=weights,
            )

            # Top-1 and top-5 predictions
            top5_preds = votes.topk(min(5, num_classes), dim=1).indices  # (B, 5)
            top1_preds = top5_preds[:, 0]  # (B,)

            all_top1.append(top1_preds)
            all_top5.append(top5_preds)

        return torch.cat(all_top1, dim=0), torch.cat(all_top5, dim=0)

    # --------------------------------------------------------------------- #
    # FAISS-accelerated path
    # --------------------------------------------------------------------- #

    def _evaluate_faiss(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        num_classes: int,
        k: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """k-NN using a FAISS inner-product index.

        Inner product on L2-normalised vectors is equivalent to cosine
        similarity, so we use ``faiss.IndexFlatIP``.

        Returns the top-1 predicted labels and the top-5 predicted labels.
        """
        device = val_features.device
        n_val = val_features.shape[0]
        dim = train_features.shape[1]

        # Build the FAISS index on CPU (numpy arrays, float32, contiguous)
        train_np = _to_contiguous_float32(train_features).cpu().numpy()
        index = faiss.IndexFlatIP(dim)
        index.add(train_np)

        train_labels_cpu = train_labels.long().cpu()

        all_top1: list[torch.Tensor] = []
        all_top5: list[torch.Tensor] = []

        for start in range(0, n_val, batch_size):
            end = min(start + batch_size, n_val)
            query_np = (
                _to_contiguous_float32(val_features[start:end]).cpu().numpy()
            )

            # FAISS search returns (distances, indices), both (B, k)
            distances, indices = index.search(query_np, k)

            # Back to torch tensors
            sim = torch.from_numpy(distances)   # (B, k) -- cosine similarities
            idx = torch.from_numpy(indices).long()  # (B, k)

            topk_labels = train_labels_cpu[idx]  # (B, k)

            weights = torch.exp(sim / self.temperature)  # (B, k)

            votes = torch.zeros(
                sim.shape[0], num_classes, dtype=weights.dtype
            )
            votes.scatter_add_(
                dim=1,
                index=topk_labels,
                src=weights,
            )

            top5_preds = votes.topk(min(5, num_classes), dim=1).indices
            top1_preds = top5_preds[:, 0]

            all_top1.append(top1_preds)
            all_top5.append(top5_preds)

        predictions = torch.cat(all_top1, dim=0).to(device)
        top5_predictions = torch.cat(all_top5, dim=0).to(device)
        return predictions, top5_predictions


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_knn_evaluation(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    k_values: Optional[list[int]] = None,
    temperature: float = 0.07,
    use_faiss: bool = False,
    batch_size: int = 256,
) -> dict[str, object]:
    """Run k-NN evaluation for multiple values of *k*.

    This is a thin convenience wrapper around :class:`KNNEvaluator` that
    iterates over several *k* values and collects all results in a single
    dictionary.

    Parameters
    ----------
    train_features : torch.Tensor
        Training feature bank of shape ``(N_train, D)``.
    train_labels : torch.Tensor
        Training labels of shape ``(N_train,)``.
    val_features : torch.Tensor
        Validation features of shape ``(N_val, D)``.
    val_labels : torch.Tensor
        Validation labels of shape ``(N_val,)``.
    num_classes : int
        Total number of classes.
    k_values : list[int], optional
        List of *k* values to evaluate.  Defaults to ``[10, 20, 200]``.

        * ``k=20`` -- standard for ImageNet-1K.
        * ``k=200`` -- standard for CIFAR-100.
        * ``k=10`` -- a commonly reported additional data-point.
    temperature : float
        Temperature for the soft voting.  0.07 is standard.
    use_faiss : bool
        Whether to attempt FAISS acceleration (see :class:`KNNEvaluator`).
    batch_size : int
        Batch size for the similarity computation.

    Returns
    -------
    dict
        Nested results keyed by ``"k={value}"``.  Each entry is a dict with
        ``top1_accuracy``, ``top5_accuracy``, and ``per_class_accuracy``.

        Example::

            {
                "k=10":  {"top1_accuracy": 0.72, "top5_accuracy": 0.91, ...},
                "k=20":  {"top1_accuracy": 0.73, "top5_accuracy": 0.92, ...},
                "k=200": {"top1_accuracy": 0.70, "top5_accuracy": 0.89, ...},
            }
    """
    if k_values is None:
        k_values = [10, 20, 200]

    results: dict[str, object] = {}

    for k in sorted(k_values):
        logger.info("Running k-NN evaluation with k=%d, T=%.4f", k, temperature)
        evaluator = KNNEvaluator(
            k=k,
            temperature=temperature,
            use_faiss=use_faiss,
        )
        metrics = evaluator.evaluate(
            train_features=train_features,
            train_labels=train_labels,
            val_features=val_features,
            val_labels=val_labels,
            num_classes=num_classes,
            batch_size=batch_size,
        )
        results[f"k={k}"] = metrics
        logger.info(
            "  k=%d  top-1=%.2f%%  top-5=%.2f%%",
            k,
            metrics["top1_accuracy"] * 100,
            metrics["top5_accuracy"] * 100,
        )

    return results
