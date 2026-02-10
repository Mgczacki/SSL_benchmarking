"""Pytest configuration and shared fixtures for benchmarking tests."""

import pytest
import torch


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    parser.addoption(
        "--device",
        action="store",
        default="auto",
        help="Device to use for tests: 'cuda', 'mps', 'cpu', or 'auto' (default)",
    )
    parser.addoption(
        "--skip-model-tests",
        action="store_true",
        help="Skip model loading tests (useful without GPU)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "model: marks tests for model loading")
    config.addinivalue_line("markers", "integration: marks integration tests")


@pytest.fixture(scope="session")
def test_device(request):
    """Get device for tests with optional override from command line."""
    device_arg = request.config.getoption("--device")

    if device_arg == "auto":
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"\nUsing CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("\nUsing Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            print("\nUsing CPU device")
    else:
        # Use specified device
        try:
            device = torch.device(device_arg)
            print(f"\nUsing specified device: {device}")
        except RuntimeError as e:
            raise pytest.UsageError(f"Invalid device: {device_arg}") from e

    return device


@pytest.fixture(scope="session")
def skip_model_tests(request):
    """Check if model tests should be skipped."""
    return request.config.getoption("--skip-model-tests")


@pytest.fixture
def device(test_device):
    """Per-test device fixture (uses session-scoped test_device)."""
    return test_device


@pytest.fixture
def cpu_device():
    """Force CPU device for CPU-only tests."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """CUDA device fixture (skips if CUDA not available)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def mps_device():
    """MPS device fixture (skips if MPS not available)."""
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    return torch.device("mps")


@pytest.fixture
def synthetic_features(device):
    """Create synthetic feature tensors for evaluation.

    Returns a dictionary with:
    - train_feats: (500, 768) L2-normalized
    - train_labels: (500,) class labels (0-9)
    - val_feats: (100, 768) L2-normalized
    - val_labels: (100,) class labels (0-9)
    """
    torch.manual_seed(42)

    num_classes = 10
    num_train = 500
    num_val = 100
    feature_dim = 768

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
        "num_classes": num_classes,
        "feature_dim": feature_dim,
    }


@pytest.fixture
def small_synthetic_features(device):
    """Create smaller synthetic features (for quick tests)."""
    torch.manual_seed(42)

    num_classes = 10
    num_train = 100
    num_val = 20
    feature_dim = 256

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
        "num_classes": num_classes,
        "feature_dim": feature_dim,
    }


@pytest.fixture
def dummy_images(device):
    """Create dummy image tensors (B, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 224, 224, device=device)
