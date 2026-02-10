"""Unit tests for reporting and visualization modules."""

import pytest
import json
from pathlib import Path
import tempfile

from utils.reporting import BenchmarkResults


class TestBenchmarkResults:
    """Test BenchmarkResults class."""

    @pytest.fixture
    def results(self):
        """Create sample benchmark results."""
        return BenchmarkResults(output_dir=tempfile.mkdtemp())

    def test_init(self, results):
        """Test BenchmarkResults initialization."""
        assert len(results.results) == 0

    def test_add_result(self, results):
        """Test adding results."""
        results.add_result(
            model_name="dinov2",
            arch="vit_base",
            dataset="cifar100",
            eval_type="knn",
            metrics={"top1_accuracy": 85.0, "top5_accuracy": 95.0},
        )

        assert len(results.results) == 1
        result = results.results[0]
        assert result["model_name"] == "dinov2"
        assert result["dataset"] == "cifar100"
        assert result["eval_type"] == "knn"
        assert result["top1_accuracy"] == 85.0

    def test_add_multiple_results(self, results):
        """Test adding multiple results."""
        for i in range(5):
            results.add_result(
                model_name=f"model_{i}",
                arch="vit_base",
                dataset="cifar100",
                eval_type="knn",
                metrics={"top1_accuracy": 80.0 + i},
            )

        assert len(results.results) == 5

    def test_filter_by_dataset(self, results):
        """Test filtering results by dataset."""
        results.add_result("dinov2", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})
        results.add_result("dinov2", "vit_base", "imagenet1k", "knn", {"top1_accuracy": 92.0})
        results.add_result("ijepa", "vit_huge", "cifar100", "knn", {"top1_accuracy": 80.0})

        cifar_results = results._filter(dataset="cifar100")
        assert len(cifar_results) == 2
        assert all(r["dataset"] == "cifar100" for r in cifar_results)

    def test_filter_by_eval_type(self, results):
        """Test filtering results by eval type."""
        results.add_result("dinov2", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})
        results.add_result("dinov2", "vit_base", "cifar100", "linear_probe", {"top1_accuracy": 88.0})

        knn_results = results._filter(eval_type="knn")
        assert len(knn_results) == 1
        assert knn_results[0]["eval_type"] == "knn"

    def test_comparison_table(self, results):
        """Test generating comparison table."""
        models = [("dinov2", "vit_base"), ("ijepa", "vit_huge"), ("mae", "vit_base")]

        for model_name, arch in models:
            results.add_result(
                model_name, arch, "cifar100", "knn",
                {"top1_accuracy": 80.0, "top5_accuracy": 95.0},
            )

        table = results.generate_comparison_table(dataset="cifar100", eval_type="knn")
        assert "Dinov2" in table
        assert "Ijepa" in table
        assert "Mae" in table
        assert "CIFAR-100" in table

    def test_comparison_table_empty(self, results):
        """Test comparison table with no matching results."""
        table = results.generate_comparison_table(dataset="cifar100", eval_type="knn")
        assert "No results found" in table

    def test_generate_full_report(self, results):
        """Test full report generation."""
        results.add_result("dinov2", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})
        results.add_result("ijepa", "vit_huge", "cifar100", "knn", {"top1_accuracy": 80.0})

        report = results.generate_full_report(
            datasets=["cifar100"],
            eval_types=["knn"],
        )
        assert isinstance(report, str)
        assert len(report) > 0
        assert "SSL Benchmark Report" in report

    def test_empty_results_report(self, results):
        """Test report with empty results."""
        report = results.generate_full_report(
            datasets=["cifar100"],
            eval_types=["knn"],
        )
        assert isinstance(report, str)
        assert len(report) > 0

    def test_len(self, results):
        """Test __len__."""
        assert len(results) == 0
        results.add_result("dinov2", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})
        assert len(results) == 1

    def test_repr(self, results):
        """Test __repr__."""
        r = repr(results)
        assert "BenchmarkResults" in r

    def test_save_and_load(self, results):
        """Test saving and loading results."""
        results.add_result("dinov2", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})
        results.add_result("ijepa", "vit_huge", "cifar100", "knn", {"top1_accuracy": 80.0})

        # Save
        path = results.save("test_results.json")
        assert Path(path).exists()

        # Load into new instance
        results2 = BenchmarkResults(output_dir=results.output_dir)
        results2.load("test_results.json")
        assert len(results2.results) == 2

    def test_save_json_content(self, results):
        """Test that saved JSON has correct content."""
        results.add_result("dinov2", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})

        path = results.save("test_results.json")

        with open(path) as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["model_name"] == "dinov2"


class TestBenchmarkResultsEdgeCases:
    """Test edge cases and error handling."""

    def test_add_result_with_none_metrics(self):
        """Test adding result with missing metrics."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())
        results.add_result("model", "vit_base", "dataset", "knn", {})

        assert len(results.results) == 1
        assert results.results[0]["top1_accuracy"] is None
        assert results.results[0]["top5_accuracy"] is None

    def test_special_characters_in_names(self):
        """Test handling special characters."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())
        results.add_result(
            "model-v2.0", "vit_base", "dataset/subset", "knn",
            {"top1_accuracy": 85.0},
        )

        assert len(results.results) == 1
        assert results.results[0]["model_name"] == "model-v2.0"
        assert results.results[0]["dataset"] == "dataset/subset"

    def test_unicode_in_names(self):
        """Test handling unicode characters."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())
        results.add_result(
            "模型", "vit_base", "数据集", "knn",
            {"top1_accuracy": 85.0},
        )

        assert len(results.results) == 1


class TestBenchmarkComparisonLogic:
    """Test comparison and ranking logic."""

    def test_rank_models_by_metric(self):
        """Test ranking models by metric."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())

        results.add_result("model_a", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})
        results.add_result("model_b", "vit_base", "cifar100", "knn", {"top1_accuracy": 92.0})
        results.add_result("model_c", "vit_base", "cifar100", "knn", {"top1_accuracy": 80.0})

        cifar_results = results._filter(dataset="cifar100")
        assert len(cifar_results) == 3
        models = {r["model_name"] for r in cifar_results}
        assert models == {"model_a", "model_b", "model_c"}

    def test_get_best_model_for_dataset(self):
        """Test finding best model for a dataset."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())

        results.add_result("model_a", "vit_base", "cifar100", "knn", {"top1_accuracy": 80.0})
        results.add_result("model_b", "vit_base", "cifar100", "knn", {"top1_accuracy": 92.0})
        results.add_result("model_c", "vit_base", "cifar100", "knn", {"top1_accuracy": 85.0})

        cifar_results = results._filter(dataset="cifar100")
        best = max(cifar_results, key=lambda x: x["top1_accuracy"])
        assert best["model_name"] == "model_b"
        assert best["top1_accuracy"] == 92.0

    def test_verify_against_published_within_tolerance(self):
        """Test verification against published baselines."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())

        # Add a result close to published baseline for dinov2 vit_base imagenet1k linear_probe (84.5)
        results.add_result(
            "dinov2", "vit_base", "imagenet1k", "linear_probe",
            {"top1_accuracy": 84.0},
        )

        flagged = results.verify_against_published(tolerance=2.0)
        assert len(flagged) == 0  # within tolerance

    def test_verify_against_published_exceeds_tolerance(self):
        """Test verification flags results outside tolerance."""
        results = BenchmarkResults(output_dir=tempfile.mkdtemp())

        # Add a result far from published baseline
        results.add_result(
            "dinov2", "vit_base", "imagenet1k", "linear_probe",
            {"top1_accuracy": 70.0},  # published is 84.5, delta = -14.5
        )

        flagged = results.verify_against_published(tolerance=2.0)
        assert len(flagged) == 1
        assert flagged[0]["exceeds_tolerance"] is True
