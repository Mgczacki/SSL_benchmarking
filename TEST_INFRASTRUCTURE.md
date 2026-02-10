# Testing Infrastructure Summary

This document summarizes the comprehensive testing framework that has been implemented for the JEPA benchmarking environment.

## Overview

The testing infrastructure consists of three complementary testing approaches:

1. **Unit Tests** - Fast, isolated component testing with synthetic data
2. **Sanity Check Script** - End-to-end validation of the entire pipeline
3. **Integration Tests** - Real model loading and feature extraction validation

All tests support automatic device detection (CUDA > MPS > CPU) and require no dataset downloads.

## Files Added

### Test Source Files

#### `/Users/mario/jepa-benchmark/tests/conftest.py` (165 lines)
Pytest configuration and shared fixtures providing:
- Device auto-detection (CUDA > MPS > CPU)
- Synthetic feature tensors (500 train, 100 val, 768-dim, L2-normalized)
- Smaller synthetic features for quick tests (100 train, 20 val, 256-dim)
- Dummy image tensors (2, 3, 224, 224)
- Command-line options (`--device`, `--skip-model-tests`)
- Custom pytest markers (slow, gpu, model, integration)

**Key Fixtures**:
```python
test_device()       # Session-scoped device detection
device()            # Per-test device (uses test_device)
synthetic_features()    # (500, 768) L2-normalized + labels
small_synthetic_features()  # (100, 256) for quick tests
dummy_images()      # (2, 3, 224, 224) image tensors
```

#### `/Users/mario/jepa-benchmark/tests/test_evaluators.py` (126 lines)
Tests for evaluation modules:

**TestKNNEvaluator** (3 tests):
- `test_knn_basic`: Tests basic k-NN evaluation with default k=20
- `test_knn_small_k`: Tests with k=1
- `test_knn_large_k`: Tests with k=1000 (larger than training set)

**TestLinearProbeTrainer** (2 tests):
- `test_linear_probe_training`: Tests 5-epoch convergence
- `test_linear_probe_normalization`: Tests feature normalization

**TestQuickLinearProbe** (1 test):
- `test_quick_linear_probe`: Tests sklearn C-value sweep [0.1, 1.0, 10.0]

#### `/Users/mario/jepa-benchmark/tests/test_datasets.py` (85 lines)
Tests for dataset utilities:

**TestTransforms** (4 tests):
- `test_eval_transform`: Tests evaluation transform
- `test_train_transform`: Tests training transform
- `test_eval_transform_custom_size`: Tests with 256x256
- `test_train_transform_custom_size`: Tests with 256x256

**TestNumClasses** (8 tests):
- Tests `get_num_classes()` for 8 datasets:
  - CIFAR-10/100, ImageNet-1K/100, STL-10, Tiny ImageNet, iNaturalist, Places205
- Tests case/underscore insensitivity
- Tests ValueError for unknown datasets

#### `/Users/mario/jepa-benchmark/tests/test_models.py` (335 lines)
Comprehensive model loading tests:

**TestModelRegistry** (4 tests):
- `test_list_available_models`: Validates registry structure
- `test_list_available_architectures`: Validates all models have archs
- `test_checkpoint_info_lookup`: Tests checkpoint metadata
- `test_checkpoint_info_missing`: Tests None for unknown models

**TestDINOv2Loading** (2 tests):
- Parametrized tests for vit_small and vit_base
- Validates model properties and embedding dimensions

**TestDINOv1Loading** (2 tests):
- Parametrized tests for vit_small and vit_base
- Tests architecture alias resolution (vit_base → dino_vitb16)

**TestMAELoading** (1 test):
- Parametrized tests for vit_base and vit_large

**TestIJEPALoading** (2 tests):
- Tests checkpoint requirement
- Tests concat_n_layers parameter acceptance

**TestiBOTLoading** (1 test):
- Tests checkpoint requirement

**TestLeJEPALoading** (1 test):
- Tests checkpoint requirement

**TestModelInterface** (3 tests):
- `test_model_has_required_attributes`: Validates interface compliance
- `test_model_eval_method`: Tests .eval() support
- `test_model_to_method`: Tests .to(device) support

**TestModelFeatureExtraction** (4 tests):
- `test_dinov2_feature_extraction`: Tests actual feature extraction
- `test_dinov1_feature_extraction`: Tests DINOv1 features
- `test_feature_output_is_normalized`: Validates L2 normalization
- `test_feature_dtype_is_float32`: Validates dtype consistency

**TestUnknownModel** (2 tests):
- Tests error handling for unknown models and architectures

#### `/Users/mario/jepa-benchmark/tests/test_reporting.py` (281 lines)
Tests for reporting and visualization:

**TestBenchmarkResults** (9 tests):
- `test_init`: Initialization
- `test_add_result`: Adding single results
- `test_add_multiple_results`: Bulk result addition
- `test_get_model_results`: Filtering by model
- `test_get_dataset_results`: Filtering by dataset
- `test_comparison_table`: Generating comparison tables
- `test_to_markdown`: Markdown export
- `test_empty_results_markdown`: Edge case handling
- `test_summary_stats`: Summary statistics
- `test_save_and_load`: JSON persistence

**TestBenchmarkResultsEdgeCases** (5 tests):
- Tests NaN, infinity, negative values
- Tests special characters and unicode in names

**TestBenchmarkComparisonLogic** (3 tests):
- Model ranking by metric
- Finding best models per dataset
- Comparison across datasets

### Configuration Files

#### `/Users/mario/jepa-benchmark/pytest.ini` (38 lines)
Pytest configuration with:
- Test discovery patterns
- Custom markers (slow, gpu, model, integration)
- Output formatting (verbose, short traceback)
- Timeout settings (300 seconds)
- Coverage configuration

**Custom Options**:
```bash
pytest tests/ --device cuda          # Use CUDA
pytest tests/ --skip-model-tests     # Skip model loading tests
```

### Standalone Scripts

#### `/Users/mario/jepa-benchmark/scripts/sanity_check.py` (348 lines)
End-to-end validation script that:

**Device Detection**:
- Auto-detects CUDA, MPS, or CPU
- Logs device information and memory

**Tests Performed**:
1. Synthetic data creation (500 train, 100 val, 10 classes, 768-dim)
2. Dataset utilities (class counts for CIFAR-10/100, ImageNet-1K, STL-10)
3. Model loading (DINOv2 ViT-B as example)
4. Feature extraction (validates L2 normalization)
5. k-NN evaluation (k=20, temperature=0.07)
6. Linear probe training (3 epochs for speed)
7. Quick linear probe (sklearn-based C-value sweep)

**Runtime**: 2-5 minutes depending on device
**Exit Code**: 0 = all passed, 1 = some failed

**Usage**:
```bash
python scripts/sanity_check.py
```

### Documentation Files

#### `/Users/mario/jepa-benchmark/TESTING.md` (350+ lines)
Comprehensive testing guide with:
- Quick start instructions
- Test type descriptions and runtimes
- Device selection guidance
- Unit test examples and options
- Sanity check usage
- Full benchmark running instructions
- Troubleshooting guide
- Performance benchmarks table
- CI/CD integration examples

## Test Statistics

| Category | Count | Type | Runtime |
|----------|-------|------|---------|
| Unit tests | 50+ | Evaluators, Datasets, Models, Reporting | ~1-2 min |
| Test classes | 16 | Organized by component | - |
| Fixtures | 6 | Device, features, images | - |
| Custom markers | 4 | slow, gpu, model, integration | - |
| Sanity checks | 5 | Dataset→Model→Eval→Report | 2-5 min |

## Test Coverage

### Evaluators Module
- ✅ k-NN evaluation (basic, small k, large k)
- ✅ Linear probe training (basic, with normalization)
- ✅ Quick linear probe (sklearn-based)

### Datasets Module
- ✅ Transform utilities (eval, train, custom sizes)
- ✅ Class count lookup (8 datasets, case-insensitive)

### Models Module
- ✅ Model registry and factory
- ✅ DINOv2 loading and inference
- ✅ DINOv1 loading with arch aliases
- ✅ MAE, I-JEPA, iBOT, LeJEPA loading
- ✅ Unified model interface
- ✅ Feature extraction and normalization

### Reporting Module
- ✅ Result aggregation
- ✅ Filtering and ranking
- ✅ Markdown/JSON export
- ✅ Edge case handling (NaN, inf, unicode)

## Device Support

All tests support automatic device detection:

```python
# CUDA (highest priority)
if torch.cuda.is_available():
    device = torch.device("cuda")

# MPS (Apple Silicon)
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")

# CPU (fallback)
else:
    device = torch.device("cpu")
```

**Manual Override**:
```bash
pytest tests/ --device cpu      # Force CPU
pytest tests/ --device cuda     # Force CUDA
```

## How to Run Tests

### All Unit Tests
```bash
pytest tests/ -v
```

### Specific Test File
```bash
pytest tests/test_models.py -v
```

### Specific Test Class
```bash
pytest tests/test_evaluators.py::TestKNNEvaluator -v
```

### With Coverage Report
```bash
pytest tests/ --cov=models --cov=evaluation --cov=utils -v
```

### Sanity Check (Full Pipeline)
```bash
python scripts/sanity_check.py
```

### Skip Model Loading Tests (for CPU-only)
```bash
pytest tests/ --skip-model-tests -v
```

## Expected Test Results

### Sample Output
```
tests/test_evaluators.py::TestKNNEvaluator::test_knn_basic PASSED     [ 10%]
tests/test_models.py::TestDINOv2Loading::test_dinov2_initialization PASSED [ 20%]
tests/test_datasets.py::TestNumClasses::test_cifar10_classes PASSED    [ 30%]
tests/test_reporting.py::TestBenchmarkResults::test_init PASSED        [ 40%]
...
======================== 50+ passed in 45s ========================
```

### Sanity Check Output
```
JEPA Benchmarking Environment - Sanity Check
======================================================================
Using CUDA device: NVIDIA A100-SXM4-80GB

✓ Dataset utilities passed
✓ Model loading and inference passed
✓ k-NN evaluator passed
✓ Linear probe trainer passed
✓ Quick linear probe passed

======================================================================
Tests passed: 5/5
✓ All sanity checks passed!
```

## Requirements

### For Running Tests
```
torch>=2.0
torchvision>=0.15
pytest>=7.0
pytest-cov  # For coverage reports
```

### For Running Benchmarks
```
timm>=0.9.0
transformers>=4.30
scikit-learn
faiss-gpu  # Optional, for faster k-NN
```

## Next Steps

After tests pass successfully:

1. **Run CIFAR-100 benchmark** (30-60 minutes):
   ```bash
   python scripts/run_benchmark.py --config configs/default.yaml
   ```

2. **Prepare for ImageNet-1K** (12+ hours):
   ```bash
   # Download ImageNet first, then:
   python scripts/run_benchmark.py --config configs/imagenet_full.yaml
   ```

3. **Add custom model** to benchmarks:
   - Implement `BaseSSLModel` subclass
   - Register in `models/__init__.py`
   - Run sanity check validation
   - Run full benchmark

## Troubleshooting

**Device not detected**:
```bash
pytest tests/ -v -s  # Show print statements
```

**Out of memory**:
```bash
pytest tests/ --device cpu  # Force CPU
```

**Model download fails**:
- Check internet connection
- Manually download checkpoint
- Provide `checkpoint_path` to `load_model()`

See `TESTING.md` for comprehensive troubleshooting guide.
