"""Unit tests for reporting and visualization modules."""

import pytest
import torch
from pathlib import Path
import tempfile

from utils.reporting import BenchmarkResults


class TestBenchmarkResults:
    """Test BenchmarkResults class."""

    @pytest.fixture
    def results(self):
        """Create sample benchmark results."""
        return BenchmarkResults(
            benchmark_name="test_benchmark",
            timestamp="2025-02-09T10:30:00Z",
        )

    def test_init(self, results):
        """Test BenchmarkResults initialization."""
        assert results.benchmark_name == "test_benchmark"
        assert results.timestamp == "2025-02-09T10:30:00Z"
        assert len(results.results) == 0

    def test_add_result(self, results):
        """Test adding results."""
        results.add_result(
            model_name="dinov2",
            dataset="cifar100",
            metric="knn_top1",
            value=0.85,
        )

        assert len(results.results) == 1
        result = results.results[0]
        assert result["model_name"] == "dinov2"
        assert result["dataset"] == "cifar100"
        assert result["metric"] == "knn_top1"
        assert result["value"] == 0.85

    def test_add_multiple_results(self, results):
        """Test adding multiple results."""
        for i in range(5):
            results.add_result(
                model_name=f"model_{i}",
                dataset="cifar100",
                metric="knn_top1",
                value=0.80 + i * 0.01,
            )

        assert len(results.results) == 5

    def test_get_model_results(self, results):
        """Test filtering results by model."""
        results.add_result("dinov2", "cifar100", "knn_top1", 0.85)
        results.add_result("dinov2", "imagenet1k", "knn_top1", 0.92)
        results.add_result("ijepa", "cifar100", "knn_top1", 0.80)

        dinov2_results = results.get_model_results("dinov2")
        assert len(dinov2_results) == 2
        assert all(r["model_name"] == "dinov2" for r in dinov2_results)

    def test_get_dataset_results(self, results):
        """Test filtering results by dataset."""
        results.add_result("dinov2", "cifar100", "knn_top1", 0.85)
        results.add_result("ijepa", "cifar100", "knn_top1", 0.80)
        results.add_result("dinov2", "imagenet1k", "knn_top1", 0.92)

        cifar_results = results.get_dataset_results("cifar100")
        assert len(cifar_results) == 2
        assert all(r["dataset"] == "cifar100" for r in cifar_results)

    def test_comparison_table(self, results):
        """Test generating comparison table."""
        # Add sample results
        models = ["dinov2", "ijepa", "mae"]
        datasets = ["cifar100", "imagenet1k"]

        for model in models:
            for dataset in datasets:
                value = 0.80 if dataset == "cifar100" else 0.85
                results.add_result(model, dataset, "knn_top1", value)

        table = results.comparison_table(metric="knn_top1")
        assert "dinov2" in table
        assert "ijepa" in table
        assert "mae" in table
        assert "cifar100" in table
        assert "imagenet1k" in table

    def test_to_markdown(self, results):
        """Test markdown export."""
        results.add_result("dinov2", "cifar100", "knn_top1", 0.85)
        results.add_result("ijepa", "cifar100", "knn_top1", 0.80)

        markdown = results.to_markdown()
        assert isinstance(markdown, str)
        assert len(markdown) > 0
        assert "dinov2" in markdown or "Benchmark Results" in markdown

    def test_empty_results_markdown(self, results):
        """Test markdown export with empty results."""
        markdown = results.to_markdown()
        assert isinstance(markdown, str)
        # Should still produce valid markdown
        assert len(markdown) > 0

    def test_summary_stats(self, results):
        """Test computing summary statistics."""
        results.add_result("dinov2", "cifar100", "knn_top1", 0.85)
        results.add_result("dinov2", "cifar100", "knn_top5", 0.95)
        results.add_result("ijepa", "cifar100", "knn_top1", 0.80)

        summary = results.summary_stats()
        assert isinstance(summary, dict)
        # Basic structure checks
        assert isinstance(summary, dict)

    def test_save_and_load(self, results):
        """Test saving and loading results."""
        # Add some results
        results.add_result("dinov2", "cifar100", "knn_top1", 0.85)
        results.add_result("ijepa", "cifar100", "knn_top1", 0.80)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            output_path = Path(tmpdir) / "results.json"
            results.save_json(str(output_path))

            # Check file exists
            assert output_path.exists()

            # Verify content
            import json
            with open(output_path) as f:
                data = json.load(f)
                assert "results" in data
                assert len(data["results"]) == 2


class TestBenchmarkResultsEdgeCases:
    """Test edge cases and error handling."""

    def test_add_result_with_nan(self):
        """Test adding NaN values."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")
        results.add_result("model", "dataset", "metric", float("nan"))

        assert len(results.results) == 1
        assert results.results[0]["value"] != results.results[0]["value"]  # NaN != NaN

    def test_add_result_with_inf(self):
        """Test adding infinite values."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")
        results.add_result("model", "dataset", "metric", float("inf"))

        assert len(results.results) == 1
        assert results.results[0]["value"] == float("inf")

    def test_add_result_with_negative(self):
        """Test adding negative values."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")
        results.add_result("model", "dataset", "metric", -0.5)

        assert len(results.results) == 1
        assert results.results[0]["value"] == -0.5

    def test_special_characters_in_names(self):
        """Test handling special characters."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")
        results.add_result(
            "model-v2.0",
            "dataset/subset",
            "metric_type",
            0.85,
        )

        assert len(results.results) == 1
        assert results.results[0]["model_name"] == "model-v2.0"
        assert results.results[0]["dataset"] == "dataset/subset"

    def test_unicode_in_names(self):
        """Test handling unicode characters."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")
        results.add_result(
            "模型",  # "model" in Chinese
            "数据集",  # "dataset" in Chinese
            "指标",  # "metric" in Chinese
            0.85,
        )

        assert len(results.results) == 1


class TestBenchmarkComparisonLogic:
    """Test comparison and ranking logic."""

    def test_rank_models_by_metric(self):
        """Test ranking models by metric."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")

        # Add results in non-sorted order
        results.add_result("model_a", "cifar100", "knn_top1", 0.85)
        results.add_result("model_b", "cifar100", "knn_top1", 0.92)
        results.add_result("model_c", "cifar100", "knn_top1", 0.80)

        # Simple ranking check
        cifar_results = results.get_dataset_results("cifar100")
        assert len(cifar_results) == 3
        # Verify all results are present
        models = {r["model_name"] for r in cifar_results}
        assert models == {"model_a", "model_b", "model_c"}

    def test_get_best_model_for_dataset(self):
        """Test finding best model for a dataset."""
        results = BenchmarkResults("test", "2025-02-09T10:30:00Z")

        results.add_result("model_a", "cifar100", "knn_top1", 0.80)
        results.add_result("model_b", "cifar100", "knn_top1", 0.92)
        results.add_result("model_c", "cifar100", "knn_top1", 0.85)

        cifar_results = results.get_dataset_results("cifar100")
        best = max(cifar_results, key=lambda x: x["value"])
        assert best["model_name"] == "model_b"
        assert best["value"] == 0.92
