# Using the Testing Notebook in Google Colab

This guide shows you how to run the complete JEPA benchmarking test suite in Google Colab.

## Quick Start (2 Steps)

### Step 1: Open the Notebook in Colab

**Option A: Direct GitHub Link** (Recommended)

Replace `yourusername` with your GitHub username, then click this link:
```
https://colab.research.google.com/github/yourusername/jepa-benchmark/blob/main/JEPA_Benchmarking_Tests.ipynb
```

**Option B: Upload Manually**

1. Go to https://colab.research.google.com
2. Click **File** → **Upload Notebook**
3. Select `JEPA_Benchmarking_Tests.ipynb` from your computer
4. Click **Open**

**Option C: From Google Drive**

1. Upload `JEPA_Benchmarking_Tests.ipynb` to your Google Drive
2. Right-click → **Open with** → **Google Colaboratory**

### Step 2: Run All Tests

Simply run the cells in order from top to bottom:

1. **Setup & Installation** (1-2 minutes)
   - Clones repository
   - Installs dependencies

2. **System Information** (10 seconds)
   - Shows GPU/CPU info
   - Displays PyTorch version

3. **Run Sanity Check** (2-5 minutes)
   - Tests entire pipeline
   - Uses synthetic data only

4. **Run Unit Tests** (1-2 minutes)
   - 50+ comprehensive tests
   - Tests all components

5. **(Optional) Generate Coverage Report** (1-2 minutes)

6. **(Optional) Save to Google Drive**
   - Saves all results to your Drive

That's it! You'll see detailed output for each test.

## Notebook Sections

### 1️⃣ Setup & Installation
- Clones the GitHub repository
- Installs all required dependencies (torch, pytest, etc.)

**Expected time:** 1-2 minutes

### 2️⃣ System Information
- Shows GPU type and memory (if available)
- Displays Python and PyTorch versions
- Shows working directory

**Expected time:** 10 seconds

### 3️⃣ Sanity Check (Full Pipeline)
- Creates synthetic test data
- Tests dataset utilities
- Tests model loading
- Tests feature extraction
- Tests k-NN evaluation
- Tests linear probe training
- Tests quick linear probe

**Expected time:** 2-5 minutes
**Exit code:** 0 = success, 1 = failure

### 4️⃣ Unit Tests (Complete Suite)
Tests 50+ individual tests across:
- Dataset utilities (12 tests)
- Evaluators (6 tests)
- Models (23 tests)
- Reporting (17 tests)

**Expected time:** 1-2 minutes

### 5️⃣ Individual Test Suites
Run each test module separately:
- **test_datasets.py** - Transform utilities, class counts
- **test_evaluators.py** - k-NN, linear probe, quick probe
- **test_models.py** - Model loading, feature extraction
- **test_reporting.py** - Result tracking, export

Each can be run independently.

### 6️⃣ Test Specific Components
Run targeted tests:
- DINOv2 model loading only
- k-NN evaluator only
- Other individual components

### 7️⃣ Test Summary & Results
Displays a comprehensive summary of:
- Number of tests
- What was tested
- Next steps

### 8️⃣ Save Results to Google Drive
- Mounts your Google Drive
- Saves test results and summary

## Common Workflows

### Workflow 1: Quick Validation (5 minutes)
1. Run **Setup & Installation**
2. Run **System Information**
3. Run **Sanity Check**

If "✅ All sanity checks passed!" appears, everything works!

### Workflow 2: Complete Testing (10 minutes)
1. Run **Setup & Installation**
2. Run **System Information**
3. Run **Sanity Check**
4. Run **All Unit Tests**
5. Run **Test Summary**

### Workflow 3: Detailed Component Testing (15 minutes)
1. Run **Setup & Installation**
2. Run **System Information**
3. Run **Sanity Check**
4. Run each **Individual Test Suite** separately:
   - Dataset Tests
   - Evaluator Tests
   - Model Tests
   - Reporting Tests
5. Run **Coverage Report**
6. Run **Test Summary**

### Workflow 4: Save Results (20 minutes)
1. Run all test sections above
2. Run **Mount Google Drive**
3. Run **Save Test Results to Drive**

Results will be saved to: `My Drive/jepa_test_results_YYYYMMDD_HHMMSS/`

## GPU Selection in Colab

### Check GPU Allocation
Run the **System Information** cell - it shows:
```
GPU Available: True
GPU Device: Tesla T4 (or A100, V100, etc.)
GPU Memory: 16.0 GB
```

### Change GPU Type
1. Click **Runtime** → **Change Runtime Type**
2. Select **GPU** for GPU acceleration
3. Choose from available GPU types (T4, A100, etc.)
4. Click **Save**

### Run on CPU Only (if GPU unavailable)
Add this to any test cell:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
```

## Expected Output Examples

### Sanity Check Success
```
======================================================================
JEPA Benchmarking Environment - Sanity Check
======================================================================
Using CUDA device: Tesla T4

Creating synthetic data...
✓ Synthetic data created

======================================================================
Testing Dataset Utilities
======================================================================
  cifar10: 10 classes
  cifar100: 100 classes
  ✓ Dataset utilities passed

======================================================================
Testing k-NN Evaluator
======================================================================
k-NN Results:
  Top-1 Accuracy: 42.00%
  Top-5 Accuracy: 85.00%
✓ k-NN evaluator passed

======================================================================
Sanity Check Summary
======================================================================
Tests passed: 5/5
✓ All sanity checks passed!
```

### Unit Tests Success
```
tests/test_datasets.py::TestTransforms::test_eval_transform PASSED
tests/test_datasets.py::TestTransforms::test_train_transform PASSED
tests/test_evaluators.py::TestKNNEvaluator::test_knn_basic PASSED
tests/test_models.py::TestDINOv2Loading::test_dinov2_initialization PASSED
... [50+ tests] ...

======================== 50+ passed in 45s ========================
```

## Troubleshooting

### "Repository not found"
- Ensure you've replaced `yourusername` with your actual GitHub username
- Make the repository public or add authentication

### "ModuleNotFoundError: No module named 'torch'"
- This means pip install is still running
- Wait a few seconds and re-run the cell
- Or restart the runtime: **Runtime** → **Restart Runtime**

### "CUDA out of memory"
- Switch to a larger GPU: **Runtime** → **Change Runtime Type**
- Or run CPU-only tests (test_datasets.py)
- Or reduce batch size in evaluator tests

### "Model download is very slow"
- Normal on first run! DINOv2 is ~350 MB
- Subsequent runs will be faster (cached)
- Can take 5-10 minutes depending on connection

### "Tests timeout after 10 minutes"
- Notebook runtime limit reached
- Click **Runtime** → **Increase Timeout** (Colab Pro feature)
- Or run tests in smaller batches

### "GPU not detected in System Information"
- Colab might not have allocated a GPU
- Go to **Runtime** → **Change Runtime Type**
- Select **GPU**
- Restart runtime

## File Structure in Colab

After cloning, the directory structure will be:

```
/content/jepa-benchmark/
├── JEPA_Benchmarking_Tests.ipynb
├── scripts/
│   └── sanity_check.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_datasets.py
│   ├── test_evaluators.py
│   ├── test_models.py
│   └── test_reporting.py
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── dinov1.py
│   ├── dinov2.py
│   └── ...
├── evaluation/
│   ├── knn.py
│   ├── linear_probe.py
│   └── ...
└── utils/
    ├── datasets.py
    └── ...
```

All tests run from `/content/jepa-benchmark/` directory.

## Advanced: Run Tests Programmatically

You can also run tests directly in Python cells:

```python
import subprocess
import os

os.chdir('/content/jepa-benchmark')

# Run sanity check
result = subprocess.run(['python', 'scripts/sanity_check.py'])

# Run pytest
result = subprocess.run(['pytest', 'tests/', '-v'])
```

## Next Steps After Tests Pass

Once all tests pass in Colab:

1. **Download test results** to your computer
   - Scroll to **Save Results** section
   - Results are in `outputs/` directory

2. **Run CIFAR-100 benchmark** (if you want full results)
   ```bash
   python scripts/run_benchmark.py --config configs/default.yaml
   ```
   This takes 30-60 minutes but produces comprehensive results.

3. **Add your own model**
   - Implement `BaseSSLModel` in `models/`
   - Register in `models/__init__.py`
   - Run sanity check to validate
   - Benchmark against baselines

## Tips & Tricks

### Tip 1: Collapse Cell Output
Click the collapse arrow (▼) next to test output to save space.

### Tip 2: Skip to Specific Tests
You don't have to run setup every time. You can:
1. Run setup once
2. Skip to specific test cells to re-run them

### Tip 3: Download Notebook with Results
After running tests:
- Click **File** → **Download** → **Download .ipynb**
- All output is saved in the notebook

### Tip 4: Use Colab Pro
Colab Pro gives you:
- Longer runtimes (24 hours)
- Access to faster GPUs (A100, V100)
- Higher CPU/memory limits

### Tip 5: Monitor GPU Usage
Add this cell to check GPU usage:
```python
!nvidia-smi
```

Run it before and after tests to see memory consumption.

## Frequently Asked Questions

**Q: Do I need to download datasets?**
A: No! All tests use synthetic data.

**Q: Can I run this without a GPU?**
A: Yes! Tests auto-detect and run on CPU. Just slower.

**Q: How long do tests take?**
A: Sanity check: 2-5 min, Unit tests: 1-2 min, Total: ~10 min

**Q: What if tests fail?**
A: Check the error output and see TESTING.md for troubleshooting

**Q: Can I modify and test my own model?**
A: Yes! See models/ directory. Implement BaseSSLModel and register.

**Q: How do I save results?**
A: Use the "Save Results to Google Drive" section

**Q: Can I run this regularly (e.g., nightly)?**
A: Yes! Create a GitHub Actions workflow to run tests automatically

## Support

- **Documentation:** Check `TESTING.md` for detailed guide
- **Issues:** See `TESTING_QUICKSTART.md` for troubleshooting
- **Code:** `models/`, `evaluation/`, `utils/` directories
- **Examples:** Look at test files in `tests/` directory

## Summary

1. **Open** the notebook in Colab
2. **Run** Setup cell
3. **Run** Sanity Check cell
4. **Run** Unit Tests cell
5. **See** "✅ ALL TESTS PASSED!"

That's it! You now have a validated testing environment in Colab! 🎉
