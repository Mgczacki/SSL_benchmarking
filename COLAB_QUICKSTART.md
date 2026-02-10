# Google Colab Testing - Quick Start (30 Seconds)

## 🚀 Start Here

### Option 1: Open from GitHub (Easiest)

Replace `yourusername` with your GitHub username, then open this link in your browser:

```
https://colab.research.google.com/github/yourusername/jepa-benchmark/blob/main/JEPA_Benchmarking_Tests.ipynb
```

### Option 2: Manual Upload

1. Go to https://colab.research.google.com
2. Click **File** → **Upload Notebook**
3. Select `JEPA_Benchmarking_Tests.ipynb`
4. Click **Open**

---

## ▶️ Run Tests (2 Minutes)

Once the notebook is open, simply run cells from top to bottom:

### Cell 1: Setup (1-2 min)
```python
!git clone https://github.com/yourusername/jepa-benchmark.git
%cd jepa-benchmark
!pip install -e . -q && pip install pytest -q
```

### Cell 2: Check System (10 sec)
Shows your GPU type and PyTorch version.

### Cell 3: Sanity Check (2-5 min)
```python
!python scripts/sanity_check.py
```

Expected output:
```
✓ Dataset utilities passed
✓ Model loading and inference passed
✓ k-NN evaluator passed
✓ Linear probe trainer passed
✓ Quick linear probe passed

✓ All sanity checks passed!
```

### Cell 4: Unit Tests (1-2 min)
```python
!pytest tests/ -v
```

Expected output:
```
tests/test_datasets.py::TestTransforms::test_eval_transform PASSED
tests/test_evaluators.py::TestKNNEvaluator::test_knn_basic PASSED
tests/test_models.py::TestDINOv2Loading::test_dinov2_initialization PASSED
... [50+ tests] ...

======================== 50+ passed in 45s ========================
```

---

## ✨ What Gets Tested

✅ **No downloads needed** - Everything uses synthetic data
✅ **50+ unit tests** across 4 modules
✅ **Full pipeline validation** with sanity check
✅ **Works on any GPU** (T4, A100, V100) or CPU
✅ **Fast feedback** - Complete in ~10 minutes

---

## 📊 Test Coverage

| Component | Tests | Status |
|-----------|-------|--------|
| Datasets (transforms, class counts) | 12 | ✅ |
| Evaluators (k-NN, linear probe) | 6 | ✅ |
| Models (loading, inference) | 23 | ✅ |
| Reporting (results, export) | 17 | ✅ |
| **Total** | **50+** | **✅** |

---

## ⏱️ Expected Times

| Task | Time | Notes |
|------|------|-------|
| Setup | 1-2 min | First time only |
| Sanity Check | 2-5 min | Full pipeline validation |
| Unit Tests | 1-2 min | 50+ tests |
| **Total** | **5-10 min** | Everything |

---

## 🎯 What Each Test Validates

### Sanity Check
1. ✓ Dataset utilities (CIFAR, ImageNet, STL-10 class counts)
2. ✓ Model loading (DINOv2 example)
3. ✓ Feature extraction (L2 normalization)
4. ✓ k-NN evaluation (k=20 neighbors)
5. ✓ Linear probe training (3 epochs)
6. ✓ Quick linear probe (sklearn C-sweep)

### Unit Tests
- **test_datasets.py** (12 tests)
  - Transform utilities (eval, train, custom sizes)
  - Class counts (8 different datasets)

- **test_evaluators.py** (6 tests)
  - k-NN basic, small k, large k
  - Linear probe training and normalization
  - Quick linear probe sklearn-based

- **test_models.py** (23 tests)
  - Model registry and factory
  - DINOv2, DINOv1, MAE, I-JEPA, iBOT, LeJEPA loading
  - Feature extraction and normalization
  - Interface compliance (.eval(), .to())

- **test_reporting.py** (17 tests)
  - Result aggregation and filtering
  - Markdown/JSON export
  - Edge case handling

---

## 🖥️ GPU Allocation

Colab automatically selects a GPU. To verify:

Run the **System Information** cell - it shows:
```
GPU Device: Tesla T4        ← Your GPU
GPU Memory: 16.0 GB
```

To change GPU type:
1. Click **Runtime** → **Change Runtime Type**
2. Select **GPU** (required)
3. Choose GPU type: T4, A100, V100, etc.
4. Click **Save**

---

## ❌ If Tests Fail

### Issue: "Repository not found"
- Make sure you replaced `yourusername` with your actual GitHub username
- Repository must be public

### Issue: "CUDA out of memory"
- Switch to a larger GPU (A100 instead of T4)
- Or run just `tests/test_datasets.py` (no model loading)

### Issue: "Model download is slow"
- Normal! First run downloads DINOv2 (~350 MB)
- Subsequent runs will be fast (cached)

### Issue: "Tests timeout"
- You can increase timeout with Colab Pro
- Or run smaller test batches

For more troubleshooting, see `COLAB_SETUP.md`.

---

## 📚 Full Documentation

- **This file** - Quick start (30 seconds)
- **COLAB_SETUP.md** - Detailed Colab guide (workflows, troubleshooting)
- **TESTING_QUICKSTART.md** - Testing commands cheat sheet
- **TESTING.md** - Comprehensive testing documentation
- **TEST_INFRASTRUCTURE.md** - Technical overview of testing

---

## 🎉 Success Criteria

You'll see "✅ ALL TESTS PASSED!" when:

```
✓ All sanity checks passed!     ← From sanity_check.py
======================== 50+ passed in 45s ========================  ← From pytest
```

---

## 🚀 Next Steps (After Tests Pass)

1. **Run CIFAR-100 benchmark** (30-60 min):
   ```bash
   python scripts/run_benchmark.py --config configs/default.yaml
   ```

2. **Prepare ImageNet-1K** (if you have it):
   ```bash
   python scripts/run_benchmark.py --config configs/imagenet_full.yaml
   ```

3. **Add your own model**:
   - Implement `BaseSSLModel` in `models/`
   - Register in `models/__init__.py`
   - Run sanity check
   - Benchmark against baselines

---

## ✅ Ready to Start?

1. Open notebook: https://colab.research.google.com/github/yourusername/jepa-benchmark/blob/main/JEPA_Benchmarking_Tests.ipynb
2. Run cells from top to bottom
3. See "✅ ALL TESTS PASSED!" in ~10 minutes

That's it! 🎉

---

**Questions?** Check `COLAB_SETUP.md` for detailed documentation.
