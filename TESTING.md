# Testing Guide

This document provides comprehensive instructions for running tests and validating the JEPA benchmarking environment.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Types](#test-types)
3. [Device Selection](#device-selection)
4. [Running Unit Tests](#running-unit-tests)
5. [Running Sanity Check](#running-sanity-check)
6. [Running Full Benchmarks](#running-full-benchmarks)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
```

Or with test dependencies:

```bash
pip install -e ".[dev]"
```

### 2. Run Sanity Check (Recommended First Step)

```bash
python scripts/sanity_check.py
```

This validates the entire pipeline with synthetic data on your available device (CUDA, MPS, or CPU).
Expected runtime: **2-5 minutes** depending on device.

### 3. Run Unit Tests

```bash
pytest tests/ -v
```

This runs all unit tests with synthetic data and mocks, no dataset downloads required.
Expected runtime: **1-2 minutes**.

## Test Types

### Unit Tests (tests/)

**Purpose**: Validate individual components in isolation with synthetic/mock data.

**Coverage**:
- `test_datasets.py`: Transform utilities, dataset metadata
- `test_evaluators.py`: k-NN, linear probe, quick linear probe
- `test_models.py`: Model loading, initialization, feature extraction

**Data Requirements**: None (synthetic tensors only)
**Runtime**: ~1-2 minutes (CPU), ~30 seconds (CUDA/MPS)
**Device**: Any (CUDA, MPS, CPU)

### Sanity Check (scripts/sanity_check.py)

**Purpose**: End-to-end validation of the complete pipeline with synthetic data.

**Validates**:
1. Device detection and allocation
2. Model loading (DINOv2 as example)
3. Feature extraction
4. k-NN evaluation
5. Linear probe training
6. Quick linear probe (sklearn)
7. Dataset utilities

**Data Requirements**: None (synthetic tensors only)
**Runtime**: 2-5 minutes depending on device
**Device**: Any (CUDA, MPS, CPU) - auto-detected

### Integration Tests (via sanity_check.py)

**Purpose**: Validate real models and evaluations on synthetic features.

Implicit in sanity check - tests loading actual model checkpoints and running full evaluation pipelines.

### Full Benchmarks (scripts/run_benchmark.py)

**Purpose**: Evaluate SSL models on real datasets.

**Data Requirements**:
- CIFAR-100 (auto-downloaded): ~160 MB
- ImageNet-1K (manual): ~150 GB
- Others as configured

**Runtime**:
- CIFAR-100 full suite: 30-60 minutes (single GPU)
- ImageNet-1K k-NN: 2-4 hours (single GPU)
- ImageNet-1K linear probe: 6-12 hours (single GPU)

## Device Selection

All tests support automatic device detection in this priority order:

1. **CUDA** (NVIDIA GPUs) - Fastest
2. **MPS** (Apple Silicon) - Good performance on M1/M2/M3
3. **CPU** - Slowest, but always available

### Manual Device Selection

For unit tests:
```bash
pytest tests/ -v  # Auto-detects device
```

For sanity check:
```bash
python scripts/sanity_check.py  # Auto-detects device
```

For benchmarks:
```bash
python scripts/run_benchmark.py --config configs/default.yaml --device cuda
```

### Checking Available Devices

```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

print("MPS available:",
      hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
```

## Running Unit Tests

### Basic Usage

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test class
pytest tests/test_evaluators.py::TestKNNEvaluator -v

# Run specific test method
pytest tests/test_evaluators.py::TestKNNEvaluator::test_knn_basic -v
```

### Common pytest Options

```bash
# Stop at first failure
pytest tests/ -x

# Show print statements
pytest tests/ -v -s

# Run only tests matching pattern
pytest tests/ -k "knn" -v

# Run with coverage report
pytest tests/ --cov=models --cov=evaluation --cov=utils -v

# Run in parallel (install pytest-xdist first)
pip install pytest-xdist
pytest tests/ -n auto
```

### Test Output Example

```
tests/test_evaluators.py::TestKNNEvaluator::test_knn_basic PASSED     [ 10%]
tests/test_evaluators.py::TestKNNEvaluator::test_knn_small_k PASSED   [ 20%]
tests/test_evaluators.py::TestKNNEvaluator::test_knn_large_k PASSED   [ 30%]
tests/test_evaluators.py::TestLinearProbeTrainer::test_linear_probe_training PASSED [ 40%]
tests/test_evaluators.py::TestLinearProbeTrainer::test_linear_probe_normalization PASSED [ 50%]
tests/test_evaluators.py::TestQuickLinearProbe::test_quick_linear_probe PASSED [ 60%]
tests/test_datasets.py::TestTransforms::test_eval_transform PASSED   [ 70%]
tests/test_datasets.py::TestTransforms::test_train_transform PASSED  [ 80%]
tests/test_datasets.py::TestNumClasses::test_cifar10_classes PASSED   [ 90%]
tests/test_models.py::TestModelRegistry::test_list_available_models PASSED [100%]

======================== 10 passed in 45.23s ========================
```

## Running Sanity Check

### Basic Usage

```bash
python scripts/sanity_check.py
```

### Expected Output

```
2025-02-09 10:30:15,123 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,123 - __main__ - INFO - JEPA Benchmarking Environment - Sanity Check
2025-02-09 10:30:15,123 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,124 - __main__ - INFO - Using CUDA device: NVIDIA A100-SXM4-80GB
2025-02-09 10:30:15,124 - __main__ - INFO - Device: cuda

2025-02-09 10:30:15,126 - __main__ - INFO - Creating synthetic data...
2025-02-09 10:30:15,130 - __main__ - INFO - ✓ Synthetic data created
2025-02-09 10:30:15,130 - __main__ - INFO -   Train features: torch.Size([500, 768])
2025-02-09 10:30:15,130 - __main__ - INFO -   Val features: torch.Size([100, 768])

2025-02-09 10:30:15,131 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,131 - __main__ - INFO - Testing Dataset Utilities
2025-02-09 10:30:15,131 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,132 - __main__ - INFO -   cifar10: 10 classes
2025-02-09 10:30:15,132 - __main__ - INFO -   cifar100: 100 classes
2025-02-09 10:30:15,133 - __main__ - INFO -   imagenet1k: 1000 classes
2025-02-09 10:30:15,133 - __main__ - INFO -   stl10: 10 classes
2025-02-09 10:30:15,133 - __main__ - INFO - ✓ Dataset utilities passed

2025-02-09 10:30:15,134 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,134 - __main__ - INFO - Testing Model Loading (DINOv2)
2025-02-09 10:30:15,150 - __main__ - INFO - ✓ Loaded DINOv2 ViT-B/14
2025-02-09 10:30:15,150 - __main__ - INFO -   Embedding dim: 768
2025-02-09 10:30:15,150 - __main__ - INFO -   Model name: dinov2-vit_base
2025-02-09 10:30:15,250 - __main__ - INFO -   Feature shape: torch.Size([2, 768])
2025-02-09 10:30:15,250 - __main__ - INFO -   L2 normalized: True
2025-02-09 10:30:15,250 - __main__ - INFO - ✓ Model loading and inference passed

2025-02-09 10:30:15,251 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,251 - __main__ - INFO - Testing k-NN Evaluator
2025-02-09 10:30:15,260 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,290 - __main__ - INFO - k-NN Results:
2025-02-09 10:30:15,290 - __main__ - INFO -   Top-1 Accuracy: 42.00%
2025-02-09 10:30:15,290 - __main__ - INFO -   Top-5 Accuracy: 85.00%
2025-02-09 10:30:15,290 - __main__ - INFO - ✓ k-NN evaluator passed

2025-02-09 10:30:15,291 - __main__ - INFO - ======================================================================
2025-02-09 10:30:15,291 - __main__ - INFO - Testing Linear Probe Trainer
2025-02-09 10:30:15,291 - __main__ - INFO - ======================================================================
2025-02-09 10:30:20,123 - __main__ - INFO - Linear Probe Results:
2025-02-09 10:30:20,123 - __main__ - INFO -   Best Val Top-1: 65.00%
2025-02-09 10:30:20,123 - __main__ - INFO -   Best Val Top-5: 85.00%
2025-02-09 10:30:20,123 - __main__ - INFO - ✓ Linear probe trainer passed

2025-02-09 10:30:20,124 - __main__ - INFO - ======================================================================
2025-02-09 10:30:20,124 - __main__ - INFO - Testing Quick Linear Probe (sklearn)
2025-02-09 10:30:20,126 - __main__ - INFO - ======================================================================
2025-02-09 10:30:20,230 - __main__ - INFO - Quick Linear Probe Results:
2025-02-09 10:30:20,230 - __main__ - INFO -   Best C: 1.00
2025-02-09 10:30:20,230 - __main__ - INFO -   Best Accuracy: 58.00%
2025-02-09 10:30:20,230 - __main__ - INFO - ✓ Quick linear probe passed

2025-02-09 10:30:20,231 - __main__ - INFO - ======================================================================
2025-02-09 10:30:20,231 - __main__ - INFO - Sanity Check Summary
2025-02-09 10:30:20,231 - __main__ - INFO - ======================================================================
2025-02-09 10:30:20,232 - __main__ - INFO - Tests passed: 5/5
2025-02-09 10:30:20,232 - __main__ - INFO - ✓ All sanity checks passed!
```

### Exit Codes

- `0`: All checks passed
- `1`: Some checks failed (review output for details)

## Running Full Benchmarks

### CIFAR-100 Quick Benchmark

```bash
python scripts/run_benchmark.py --config configs/default.yaml
```

**Runtime**: 30-60 minutes (single GPU)
**Dataset size**: ~160 MB (auto-downloads)

### ImageNet-1K Full Benchmark

First, download ImageNet-1K from https://www.image-net.org/download.php

Then:

```bash
python scripts/run_benchmark.py --config configs/imagenet_full.yaml \
    --data_dir /path/to/imagenet1k
```

**Runtime**: 12-24 hours for full suite (single A100)
**Dataset size**: ~150 GB

### Custom Configuration

Create `configs/custom.yaml`:

```yaml
# Datasets to evaluate on
datasets:
  - name: cifar100
    evaluate: true
  - name: imagenet100
    evaluate: false

# Models to benchmark
models:
  - name: dinov2
    arch: vit_base
    evaluate: true
  - name: ijepa
    arch: vit_base
    checkpoint_path: /path/to/ijepa.pth
    evaluate: true

# Evaluation protocols
evaluation:
  knn:
    enabled: true
    k: 20
  linear_probe:
    enabled: true
    lr_sweep: [0.001, 0.01, 0.1, 1.0, 10.0]
  finetune:
    enabled: false
  low_shot:
    enabled: false
```

Then run:

```bash
python scripts/run_benchmark.py --config configs/custom.yaml
```

## Troubleshooting

### Test Fails: "Module not found"

**Problem**: `ModuleNotFoundError: No module named 'models'`

**Solution**: Install the project in development mode:

```bash
pip install -e .
```

### Test Fails: "CUDA out of memory"

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
1. Run on CPU: Tests auto-fall back if CUDA fails
2. Reduce batch size in test configuration
3. Use a smaller device: `pytest tests/ --device cpu`

### Test Fails: "Model loading failed"

**Problem**: Model checkpoint download fails

**Solution**:
1. Check internet connection
2. Check disk space (need ~20 GB for all model weights)
3. Manually download checkpoints first (see models/\_\_init\_\_.py for URLs)

### Model Loading Takes Forever

**Problem**: First run of model loading downloads weights

**Solution**: This is normal! First DINOv2 download is ~350 MB (goes to `~/.cache/huggingface`).

For subsequent runs: cached weights will be used (much faster).

### Device Not Detected

**Problem**: "Using CPU device" but you have a GPU

**Solutions**:

For CUDA:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

For MPS (Apple):
```python
import torch
print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
```

If False, reinstall PyTorch for your device:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For MPS (Apple)
pip install torch torchvision torchaudio
```

### Tests Pass Locally but Fail in CI

**Problem**: Device-dependent test failures

**Solution**: Tests auto-skip if device required is unavailable:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
```

Use `pytest -v` to see which tests were skipped.

## Performance Benchmarks

Expected runtimes on standard hardware:

### Unit Tests

| Device | Time | Notes |
|--------|------|-------|
| NVIDIA A100 | 30 sec | 80GB VRAM |
| NVIDIA RTX 4090 | 45 sec | 24GB VRAM |
| NVIDIA RTX 3080 | 60 sec | 10GB VRAM |
| Apple M2 (MPS) | 90 sec | Using unified memory |
| CPU (Intel Xeon) | 2-3 min | Single core bottleneck |

### Sanity Check

| Device | Time | Notes |
|--------|------|-------|
| NVIDIA A100 | 2 min | Includes model download (~350MB) |
| NVIDIA RTX 4090 | 3 min | First run: longer due to download |
| Apple M2 (MPS) | 3-4 min | Excellent performance on M2+ |
| CPU (Intel Xeon) | 5-8 min | CPU bottleneck for inference |

### Full Benchmarks (ImageNet-1K)

| Model | Device | k-NN | Linear Probe | Fine-tune |
|-------|--------|------|------|-----------|
| DINOv2 ViT-B | A100 | 2h | 12h | 2 days |
| DINOv2 ViT-B | RTX 4090 | 3h | 16h | 3 days |
| DINOv2 ViT-B | RTX 3080 | 4h | 20h | 4 days |
| DINOv2 ViT-B | M2 (MPS) | 4-5h | 18-24h | Not recommended |

## Continuous Integration

For GitHub Actions / GitLab CI, use:

```yaml
test:
  script:
    # Quick smoke test on CPU
    - python scripts/sanity_check.py
    - pytest tests/ -v --tb=short
```

For GPU runners:

```yaml
test_gpu:
  script:
    - pytest tests/ -v --device cuda
  tags:
    - gpu
    - cuda
```

## Next Steps

After all tests pass:

1. **Run CIFAR-100 benchmark** for quick validation:
   ```bash
   python scripts/run_benchmark.py --config configs/default.yaml
   ```

2. **Prepare ImageNet-1K** if you have the data:
   ```bash
   # Download from https://www.image-net.org/download.php
   # Then run:
   python scripts/run_benchmark.py --config configs/imagenet_full.yaml \
       --data_dir /path/to/imagenet
   ```

3. **Add your own model** to the benchmark:
   - Implement a model class inheriting from `BaseSSLModel`
   - Register it in `models/__init__.py`
   - Test with sanity check
   - Run full benchmark

4. **View results** in generated reports:
   ```bash
   # Open in browser
   open outputs/results.html

   # Or check markdown table
   cat outputs/results.md
   ```
