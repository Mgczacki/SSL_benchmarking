"""Unit tests for dataset utilities."""

import pytest
import torch
from torch.utils.data import DataLoader

from utils.datasets import (
    get_eval_transform,
    get_train_transform,
    get_num_classes,
)


class TestTransforms:
    """Test image transforms."""

    def test_eval_transform(self):
        """Test evaluation transform."""
        transform = get_eval_transform(image_size=224)
        assert transform is not None

        # Create a dummy PIL image-like tensor
        dummy_img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        # Transforms expect PIL Image, so this is just a sanity check
        # In practice, this would be called on PIL images by the dataset

    def test_train_transform(self):
        """Test training transform."""
        transform = get_train_transform(image_size=224)
        assert transform is not None

    def test_eval_transform_custom_size(self):
        """Test evaluation transform with custom size."""
        transform = get_eval_transform(image_size=256)
        assert transform is not None

    def test_train_transform_custom_size(self):
        """Test training transform with custom size."""
        transform = get_train_transform(image_size=256)
        assert transform is not None


class TestNumClasses:
    """Test number of classes lookup."""

    def test_cifar10_classes(self):
        """Test CIFAR-10 class count."""
        assert get_num_classes("cifar10") == 10
        assert get_num_classes("CIFAR-10") == 10  # Case insensitive
        assert get_num_classes("cifar_10") == 10  # Underscore insensitive

    def test_cifar100_classes(self):
        """Test CIFAR-100 class count."""
        assert get_num_classes("cifar100") == 100

    def test_imagenet1k_classes(self):
        """Test ImageNet-1K class count."""
        assert get_num_classes("imagenet1k") == 1000
        assert get_num_classes("imagenet-1k") == 1000

    def test_imagenet100_classes(self):
        """Test ImageNet-100 class count."""
        assert get_num_classes("imagenet100") == 100

    def test_stl10_classes(self):
        """Test STL-10 class count."""
        assert get_num_classes("stl10") == 10

    def test_tinyimagenet_classes(self):
        """Test Tiny ImageNet class count."""
        assert get_num_classes("tinyimagenet") == 200

    def test_inaturalist_classes(self):
        """Test iNaturalist class count."""
        assert get_num_classes("inaturalist2018") == 8142

    def test_places205_classes(self):
        """Test Places205 class count."""
        assert get_num_classes("places205") == 205

    def test_unknown_dataset(self):
        """Test that unknown dataset raises ValueError."""
        with pytest.raises(ValueError):
            get_num_classes("unknown_dataset")
