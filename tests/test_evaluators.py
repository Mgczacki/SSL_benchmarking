"""Unit tests for evaluation modules."""

import pytest
import torch

from evaluation.knn import KNNEvaluator
from evaluation.linear_probe import LinearProbeTrainer, quick_linear_probe


class TestKNNEvaluator:
    """Test k-NN evaluator."""

    def test_knn_basic(self, synthetic_features):
        """Test basic k-NN evaluation."""
        evaluator = KNNEvaluator(k=20, temperature=0.07)
        results = evaluator.evaluate(
            train_features=synthetic_features["train_feats"],
            train_labels=synthetic_features["train_labels"],
            val_features=synthetic_features["val_feats"],
            val_labels=synthetic_features["val_labels"],
            num_classes=synthetic_features["num_classes"],
        )

        assert "top1_accuracy" in results
        assert "top5_accuracy" in results
        assert "per_class_accuracy" in results
        assert 0 <= results["top1_accuracy"] <= 1
        assert 0 <= results["top5_accuracy"] <= 1
        assert results["top1_accuracy"] <= results["top5_accuracy"]  # top5 >= top1

    def test_knn_small_k(self, synthetic_features):
        """Test k-NN with small k value."""
        evaluator = KNNEvaluator(k=1)
        results = evaluator.evaluate(
            train_features=synthetic_features["train_feats"],
            train_labels=synthetic_features["train_labels"],
            val_features=synthetic_features["val_feats"],
            val_labels=synthetic_features["val_labels"],
            num_classes=synthetic_features["num_classes"],
        )
        assert results["top1_accuracy"] >= 0

    def test_knn_large_k(self, synthetic_features):
        """Test k-NN with k larger than training set."""
        evaluator = KNNEvaluator(k=1000)  # Larger than training set
        results = evaluator.evaluate(
            train_features=synthetic_features["train_feats"],
            train_labels=synthetic_features["train_labels"],
            val_features=synthetic_features["val_feats"],
            val_labels=synthetic_features["val_labels"],
            num_classes=synthetic_features["num_classes"],
        )
        assert results["top1_accuracy"] >= 0


class TestLinearProbeTrainer:
    """Test linear probe trainer."""

    def test_linear_probe_training(self, synthetic_features, device):
        """Test linear probe training convergence."""
        trainer = LinearProbeTrainer(
            feature_dim=synthetic_features["feature_dim"],
            num_classes=synthetic_features["num_classes"],
            lr=0.1,
            epochs=5,
            device=str(device),
        )

        results = trainer.train(
            train_features=synthetic_features["train_feats"],
            train_labels=synthetic_features["train_labels"],
            val_features=synthetic_features["val_feats"],
            val_labels=synthetic_features["val_labels"],
        )

        assert "best_val_top1" in results
        assert "best_val_top5" in results
        assert "training_history" in results
        assert len(results["training_history"]) == 5
        assert 0 <= results["best_val_top1"] <= 100
        assert 0 <= results["best_val_top5"] <= 100

    def test_linear_probe_normalization(self, synthetic_features, device):
        """Test linear probe with feature normalization."""
        # Create unnormalized features
        torch.manual_seed(42)
        train_feats = torch.randn(100, 256)
        val_feats = torch.randn(50, 256)

        trainer = LinearProbeTrainer(
            feature_dim=256,
            num_classes=10,
            epochs=3,
            normalize_features=True,
            device=str(device),
        )

        results = trainer.train(
            train_features=train_feats,
            train_labels=torch.randint(0, 10, (100,)),
            val_features=val_feats,
            val_labels=torch.randint(0, 10, (50,)),
        )

        assert results["best_val_top1"] >= 0


class TestQuickLinearProbe:
    """Test sklearn-based quick linear probe."""

    def test_quick_linear_probe(self, synthetic_features):
        """Test quick linear probe."""
        results = quick_linear_probe(
            train_features=synthetic_features["train_feats"],
            train_labels=synthetic_features["train_labels"],
            val_features=synthetic_features["val_feats"],
            val_labels=synthetic_features["val_labels"],
            C_values=[0.1, 1.0, 10.0],
        )

        assert "best_C" in results
        assert "best_acc" in results
        assert "all_results" in results
        assert 0 <= results["best_acc"] <= 100
        assert results["best_C"] in [0.1, 1.0, 10.0]
