# Testing Quick Start Guide

Get started with testing the JEPA benchmarking environment in minutes.

## 1-Minute Setup

```bash
# Install dependencies (if not already done)
pip install -e ".[dev]"

# Run sanity check (validates entire pipeline)
python scripts/sanity_check.py
```

Expected output: "✓ All sanity checks passed!" (takes 2-5 minutes)

## Common Testing Tasks

### I want to...

#### Run all unit tests
```bash
pytest tests/ -v
```
⏱ **Time**: 1-2 minutes | 📍 **Location**: `tests/` | 🔌 **No data needed**

#### Test a specific component
```bash
# Test model loading
pytest tests/test_models.py -v

# Test evaluators
pytest tests/test_evaluators.py -v

# Test datasets
pytest tests/test_datasets.py -v

# Test reporting
pytest tests/test_reporting.py -v
```

#### Test a specific feature
```bash
# Test k-NN evaluator only
pytest tests/test_evaluators.py::TestKNNEvaluator -v

# Test DINOv2 model loading
pytest tests/test_models.py::TestDINOv2Loading -v

# Test dataset class counts
pytest tests/test_datasets.py::TestNumClasses -v
```

#### Run tests on CPU only
```bash
pytest tests/ --device cpu -v
```

#### Run quick smoke tests
```bash
pytest tests/ -k "basic" -v
```

#### Get test coverage report
```bash
pytest tests/ --cov=models --cov=evaluation --cov=utils --cov-report=html -v
```

#### Stop at first failure
```bash
pytest tests/ -x -v
```

#### Run tests in parallel (faster)
```bash
pip install pytest-xdist
pytest tests/ -n auto -v
```

#### Skip slow/GPU tests
```bash
pytest tests/ -m "not slow" -v
pytest tests/ -m "not gpu" -v
```

#### Run with detailed output
```bash
pytest tests/ -vv -s  # Extra verbose + show print statements
```

### Validate the entire pipeline
```bash
python scripts/sanity_check.py
```

This single script validates:
- ✅ Device detection (CUDA/MPS/CPU)
- ✅ Dataset utilities
- ✅ Model loading
- ✅ Feature extraction
- ✅ k-NN evaluation
- ✅ Linear probe training
- ✅ Quick linear probe

No dataset downloads required. Takes 2-5 minutes depending on device.

## What Each Test Covers

### Unit Tests (~50 tests)

| Module | Tests | What's Tested |
|--------|-------|---------------|
| `test_evaluators.py` | 6 | k-NN, linear probe, quick linear probe |
| `test_datasets.py` | 12 | Transforms, class counts, metadata |
| `test_models.py` | 23 | Model loading, feature extraction, interface |
| `test_reporting.py` | 17 | Result tracking, visualization, export |

### Sanity Check Script (~5 major tests)

1. **Dataset Utilities** - Class count lookup for CIFAR/ImageNet/STL-10
2. **Model Loading** - Load DINOv2, extract features, validate normalization
3. **k-NN Evaluation** - Run k-NN with synthetic features
4. **Linear Probe** - Train linear classifier on features
5. **Quick Linear Probe** - sklearn-based C-value sweep

## Expected Runtimes

| Task | Device | Time | Notes |
|------|--------|------|-------|
| All unit tests | CUDA A100 | 30 sec | Synthetic data only |
| All unit tests | CPU | 2-3 min | CPU bottleneck |
| Sanity check | CUDA A100 | 2 min | Includes model download |
| Sanity check | Apple M2 | 3-4 min | Great MPS performance |
| Sanity check | CPU | 5-8 min | CPU bottleneck |

## Device Selection

Tests automatically detect your device in this order:
1. **CUDA** (NVIDIA GPUs) - Fastest
2. **MPS** (Apple Silicon M1/M2/M3+) - Good performance
3. **CPU** - Always available, slower

### Override device selection
```bash
pytest tests/ --device cuda   # Force CUDA
pytest tests/ --device mps    # Force MPS
pytest tests/ --device cpu    # Force CPU
```

## Troubleshooting

### Tests fail with "ModuleNotFoundError"
```bash
# Install project in development mode
pip install -e .
```

### CUDA out of memory
```bash
# Run tests on CPU instead
pytest tests/ --device cpu -v
```

### Model loading takes forever
This is normal on first run! Models download to `~/.cache/huggingface` (~350 MB).
Second run will be much faster.

### Tests pass locally but fail on different device
Tests gracefully skip if device is unavailable. Use `-v` flag to see which tests skipped.

### Need to see print statements
```bash
pytest tests/ -s -v  # -s = show output
```

## Continuous Integration

For GitHub Actions / GitLab CI:

```yaml
# Quick smoke test (2 minutes)
python scripts/sanity_check.py

# Or full test suite on CPU
pytest tests/ -v --device cpu
```

## Next Steps After Tests Pass

1. **Run CIFAR-100 benchmark** (30-60 min):
   ```bash
   python scripts/run_benchmark.py --config configs/default.yaml
   ```

2. **Prepare ImageNet-1K** (12+ hours):
   ```bash
   # Download from https://www.image-net.org/download.php
   python scripts/run_benchmark.py --config configs/imagenet_full.yaml --data_dir /path/to/imagenet
   ```

3. **Add your model** and test it:
   - Implement `BaseSSLModel` subclass in `models/`
   - Register in `models/__init__.py`
   - Run sanity check: `python scripts/sanity_check.py`
   - Run benchmark: `python scripts/run_benchmark.py --config configs/custom.yaml`

## For More Details

- **Comprehensive guide**: See `TESTING.md`
- **Test infrastructure overview**: See `TEST_INFRASTRUCTURE.md`
- **Full benchmark setup**: See `README.md`
- **Running full benchmarks**: See `scripts/run_benchmark.py --help`

## Common Commands Cheat Sheet

```bash
# Validate everything works (START HERE!)
python scripts/sanity_check.py

# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_models.py::TestDINOv2Loading -v

# Quick smoke test
pytest tests/ -k "basic" -v

# Parallel testing (faster)
pytest tests/ -n auto -v

# With coverage
pytest tests/ --cov=models --cov-report=html -v

# Stop on first failure
pytest tests/ -x -v

# CPU only
pytest tests/ --device cpu -v

# Full benchmark (after tests pass)
python scripts/run_benchmark.py --config configs/default.yaml
```
