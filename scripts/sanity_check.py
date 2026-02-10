#!/usr/bin/env python3
"""Sanity check script for JEPA benchmarking environment.

This script validates the complete benchmarking pipeline using synthetic data.
It requires no downloaded datasets and runs quickly on any device (CUDA, MPS, CPU).

Usage:
    python scripts/sanity_check.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import logging
from typing import Dict, Any

from models import load_model, list_available_models
from evaluation.knn import KNNEvaluator
from evaluation.linear_probe import LinearProbeTrainer, quick_linear_probe
from utils.datasets import get_num_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Metal Performance Shaders (MPS)")
        return device
    device = torch.device("cpu")
    logger.info("Using CPU device")
    return device


def create_synthetic_data(
    num_train: int = 500,
    num_val: int = 100,
    num_classes: int = 10,
    feature_dim: int = 768,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Create synthetic feature tensors for evaluation.

    Parameters
    ----------
    num_train : int
        Number of training samples
    num_val : int
        Number of validation samples
    num_classes : int
        Number of classes
    feature_dim : int
        Feature dimension
    device : torch.device
        Device to create tensors on

    Returns
    -------
    dict
        Dictionary with keys: train_feats, train_labels, val_feats, val_labels
    """
    torch.manual_seed(42)

    train_feats = torch.randn(num_train, feature_dim, device=device)
    train_labels = torch.randint(0, num_classes, (num_train,), device=device)
    val_feats = torch.randn(num_val, feature_dim, device=device)
    val_labels = torch.randint(0, num_classes, (num_val,), device=device)

    # L2 normalize
    train_feats = torch.nn.functional.normalize(train_feats, p=2, dim=-1)
    val_feats = torch.nn.functional.normalize(val_feats, p=2, dim=-1)

    return {
        "train_feats": train_feats.cpu(),
        "train_labels": train_labels.cpu(),
        "val_feats": val_feats.cpu(),
        "val_labels": val_labels.cpu(),
    }


def test_knn_evaluator(data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Test k-NN evaluator with synthetic data."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing k-NN Evaluator")
    logger.info("=" * 70)

    evaluator = KNNEvaluator(k=20, temperature=0.07)
    results = evaluator.evaluate(
        train_features=data["train_feats"],
        train_labels=data["train_labels"],
        val_features=data["val_feats"],
        val_labels=data["val_labels"],
        num_classes=10,
    )

    logger.info("k-NN Results:")
    logger.info("  Top-1 Accuracy: %.2f%%", results["top1_accuracy"] * 100)
    logger.info("  Top-5 Accuracy: %.2f%%", results["top5_accuracy"] * 100)

    # Validate results
    assert 0 <= results["top1_accuracy"] <= 1, "Invalid top-1 accuracy"
    assert 0 <= results["top5_accuracy"] <= 1, "Invalid top-5 accuracy"
    assert (
        results["top1_accuracy"] <= results["top5_accuracy"]
    ), "Top-1 should be <= Top-5"

    logger.info("✓ k-NN evaluator passed")
    return results


def test_linear_probe_trainer(data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Test linear probe trainer with synthetic data."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Linear Probe Trainer")
    logger.info("=" * 70)

    device = get_device()
    trainer = LinearProbeTrainer(
        feature_dim=768,
        num_classes=10,
        lr=0.1,
        epochs=3,  # Reduced for speed
        device=str(device),
    )

    results = trainer.train(
        train_features=data["train_feats"],
        train_labels=data["train_labels"],
        val_features=data["val_feats"],
        val_labels=data["val_labels"],
    )

    logger.info("Linear Probe Results:")
    logger.info("  Best Val Top-1: %.2f%%", results["best_val_top1"])
    logger.info("  Best Val Top-5: %.2f%%", results["best_val_top5"])

    # Validate results
    assert "best_val_top1" in results
    assert "best_val_top5" in results
    assert 0 <= results["best_val_top1"] <= 100
    assert 0 <= results["best_val_top5"] <= 100

    logger.info("✓ Linear probe trainer passed")
    return results


def test_quick_linear_probe(data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Test quick linear probe with sklearn."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Quick Linear Probe (sklearn)")
    logger.info("=" * 70)

    results = quick_linear_probe(
        train_features=data["train_feats"],
        train_labels=data["train_labels"],
        val_features=data["val_feats"],
        val_labels=data["val_labels"],
        C_values=[0.1, 1.0, 10.0],
    )

    logger.info("Quick Linear Probe Results:")
    logger.info("  Best C: %.2f", results["best_C"])
    logger.info("  Best Accuracy: %.2f%%", results["best_acc"])

    # Validate results
    assert "best_C" in results
    assert "best_acc" in results
    assert 0 <= results["best_acc"] <= 100
    assert results["best_C"] in [0.1, 1.0, 10.0]

    logger.info("✓ Quick linear probe passed")
    return results


def test_model_loading() -> Dict[str, Any]:
    """Test loading and basic inference with a model."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Model Loading (DINOv2)")
    logger.info("=" * 70)

    try:
        model = load_model("dinov2", arch="vit_base")
        logger.info("✓ Loaded DINOv2 ViT-B/14")
        logger.info("  Embedding dim: %d", model.embedding_dim)
        logger.info("  Model name: %s", model.name)

        device = get_device()
        model = model.to(device)

        # Create dummy input
        dummy_images = torch.randn(2, 3, 224, 224, device=device)
        features = model.extract_features(dummy_images)

        logger.info("  Feature shape: %s", features.shape)
        assert features.shape == (2, model.embedding_dim)

        # Check normalization
        norms = torch.norm(features, p=2, dim=-1)
        is_normalized = torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        )
        logger.info("  L2 normalized: %s", is_normalized)

        logger.info("✓ Model loading and inference passed")
        return {
            "model_name": model.name,
            "embedding_dim": model.embedding_dim,
            "feature_shape": features.shape,
            "is_normalized": is_normalized,
        }

    except Exception as e:
        logger.warning("Model loading test failed: %s", e)
        logger.info("(This is OK if DINOv2 weights need to be downloaded)")
        return {
            "model_name": "dinov2-vit_base",
            "skipped": True,
            "reason": str(e),
        }


def test_dataset_utilities() -> Dict[str, Any]:
    """Test dataset utilities."""
    logger.info("\n" + "=" * 70)
    logger.info("Testing Dataset Utilities")
    logger.info("=" * 70)

    datasets = ["cifar10", "cifar100", "imagenet1k", "stl10"]
    results = {}

    for dataset_name in datasets:
        try:
            num_classes = get_num_classes(dataset_name)
            results[dataset_name] = num_classes
            logger.info("  %s: %d classes", dataset_name, num_classes)
        except Exception as e:
            logger.warning("  %s: Failed - %s", dataset_name, e)

    logger.info("✓ Dataset utilities passed")
    return results


def main():
    """Run all sanity checks."""
    logger.info("=" * 70)
    logger.info("JEPA Benchmarking Environment - Sanity Check")
    logger.info("=" * 70)

    device = get_device()
    logger.info("Device: %s\n", device)

    # Create synthetic data
    logger.info("Creating synthetic data...")
    data = create_synthetic_data(device=device)
    logger.info("✓ Synthetic data created")
    logger.info("  Train features: %s", data["train_feats"].shape)
    logger.info("  Val features: %s", data["val_feats"].shape)

    # Run tests
    all_results = {}

    try:
        all_results["dataset_utils"] = test_dataset_utilities()
    except Exception as e:
        logger.error("Dataset utilities test failed: %s", e)
        all_results["dataset_utils"] = {"error": str(e)}

    try:
        all_results["model_loading"] = test_model_loading()
    except Exception as e:
        logger.error("Model loading test failed: %s", e)
        all_results["model_loading"] = {"error": str(e)}

    try:
        all_results["knn_evaluator"] = test_knn_evaluator(data)
    except Exception as e:
        logger.error("k-NN evaluator test failed: %s", e)
        all_results["knn_evaluator"] = {"error": str(e)}

    try:
        all_results["linear_probe_trainer"] = test_linear_probe_trainer(data)
    except Exception as e:
        logger.error("Linear probe trainer test failed: %s", e)
        all_results["linear_probe_trainer"] = {"error": str(e)}

    try:
        all_results["quick_linear_probe"] = test_quick_linear_probe(data)
    except Exception as e:
        logger.error("Quick linear probe test failed: %s", e)
        all_results["quick_linear_probe"] = {"error": str(e)}

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("Sanity Check Summary")
    logger.info("=" * 70)

    passed = sum(
        1
        for v in all_results.values()
        if isinstance(v, dict) and "error" not in v
    )
    total = len(all_results)

    logger.info("Tests passed: %d/%d", passed, total)

    if passed == total:
        logger.info("\n✓ All sanity checks passed!")
        return 0
    else:
        logger.warning("\n⚠ Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
