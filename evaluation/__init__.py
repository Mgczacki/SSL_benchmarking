"""Evaluation protocols for SSL benchmarking."""

from evaluation.feature_extraction import (
    FeatureExtractor,
    extract_and_cache,
    load_cached_features,
    normalize_features,
)
from evaluation.knn import KNNEvaluator, run_knn_evaluation
from evaluation.linear_probe import (
    LinearProbeTrainer,
    lr_sweep,
    quick_linear_probe,
)
from evaluation.finetune import FineTuner, low_shot_finetune

__all__ = [
    "FeatureExtractor",
    "extract_and_cache",
    "load_cached_features",
    "normalize_features",
    "KNNEvaluator",
    "run_knn_evaluation",
    "LinearProbeTrainer",
    "lr_sweep",
    "quick_linear_probe",
    "FineTuner",
    "low_shot_finetune",
]
