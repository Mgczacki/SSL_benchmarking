"""Results aggregation, comparison tables, and visualization for SSL benchmarking.

Provides a unified :class:`BenchmarkResults` container for collecting evaluation
metrics (k-NN, linear probe, fine-tune) across multiple self-supervised learning
models and datasets.  Supports JSON persistence, pandas export, markdown
comparison tables with published baselines, and matplotlib visualizations.

Typical usage::

    results = BenchmarkResults(output_dir="./results")
    results.add_result(
        model_name="ijepa",
        arch="vit_huge",
        dataset="imagenet1k",
        eval_type="linear_probe",
        metrics={"top1_accuracy": 79.1, "top5_accuracy": 94.2, "best_lr": 0.001},
    )
    results.save()

    # Markdown comparison table
    table = results.generate_comparison_table("imagenet1k", "linear_probe")
    print(table)

    # Bar chart
    results.plot_comparison_bar_chart("imagenet1k", "linear_probe", save_path="bar.png")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Published baselines for verification.
# Keys: (model_name, arch, dataset, eval_type)
# Values: dict with "top1" (and optionally "top5") accuracy in percentage
#         points, as reported in the original papers.
# ---------------------------------------------------------------------------
PUBLISHED_BASELINES: Dict[Tuple[str, str, str, str], Dict[str, float]] = {
    # I-JEPA  (Assran et al., "Self-Supervised Learning from Images with a
    #          Joint-Embedding Predictive Architecture", CVPR 2023)
    ("ijepa", "vit_huge", "imagenet1k", "linear_probe"): {"top1": 79.3},
    ("ijepa", "vit_base", "imagenet1k", "linear_probe"): {"top1": 72.9},
    ("ijepa", "vit_large", "imagenet1k", "linear_probe"): {"top1": 77.5},
    # DINOv2  (Oquab et al., "DINOv2: Learning Robust Visual Features
    #          without Supervision", TMLR 2024)
    ("dinov2", "vit_small", "imagenet1k", "linear_probe"): {"top1": 81.1},
    ("dinov2", "vit_base", "imagenet1k", "linear_probe"): {"top1": 84.5},
    ("dinov2", "vit_large", "imagenet1k", "linear_probe"): {"top1": 86.3},
    # MAE  (He et al., "Masked Autoencoders Are Scalable Vision Learners",
    #        CVPR 2022)
    ("mae", "vit_base", "imagenet1k", "linear_probe"): {"top1": 68.0},
    ("mae", "vit_large", "imagenet1k", "linear_probe"): {"top1": 76.0},
    ("mae", "vit_huge", "imagenet1k", "linear_probe"): {"top1": 77.2},
    # DINO v1  (Caron et al., "Emerging Properties in Self-Supervised Vision
    #           Transformers", ICCV 2021)
    ("dinov1", "vit_small", "imagenet1k", "linear_probe"): {"top1": 76.0},
    ("dinov1", "vit_base", "imagenet1k", "linear_probe"): {"top1": 80.1},
    # iBOT  (Zhou et al., "iBOT: Image BERT Pre-Training with Online
    #         Tokenizer", ICLR 2022)
    ("ibot", "vit_small", "imagenet1k", "linear_probe"): {"top1": 77.9},
    ("ibot", "vit_base", "imagenet1k", "linear_probe"): {"top1": 79.5},
    ("ibot", "vit_large", "imagenet1k", "linear_probe"): {"top1": 81.0},
}

# ---------------------------------------------------------------------------
# Pretty-print helpers for architecture names shown in tables.
# ---------------------------------------------------------------------------
_ARCH_DISPLAY: Dict[str, str] = {
    "vit_tiny": "ViT-Ti/16",
    "vit_small": "ViT-S/14",
    "vit_base": "ViT-B/16",
    "vit_large": "ViT-L/16",
    "vit_huge": "ViT-H/14",
    "vit_giant": "ViT-G/14",
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "resnet152": "ResNet-152",
}

# Friendly dataset names for report headings.
_DATASET_DISPLAY: Dict[str, str] = {
    "imagenet1k": "ImageNet-1K",
    "imagenet100": "ImageNet-100",
    "imagenet1pct": "ImageNet-1%",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "stl10": "STL-10",
    "tinyimagenet": "Tiny ImageNet",
    "inaturalist2018": "iNaturalist 2018",
    "places205": "Places205",
}

# Friendly eval-type names for report headings.
_EVAL_TYPE_DISPLAY: Dict[str, str] = {
    "knn": "k-NN",
    "linear_probe": "Linear Probe",
    "finetune": "Fine-tune",
}


# ---------------------------------------------------------------------------
# BenchmarkResults
# ---------------------------------------------------------------------------

class BenchmarkResults:
    """Collect, persist, and report evaluation results for SSL models.

    Each result entry is a flat dictionary containing at least:

    * ``model_name`` -- logical model identifier (e.g. ``"ijepa"``).
    * ``arch``       -- architecture identifier (e.g. ``"vit_huge"``).
    * ``dataset``    -- dataset identifier (e.g. ``"imagenet1k"``).
    * ``eval_type``  -- one of ``"knn"``, ``"linear_probe"``, ``"finetune"``.
    * ``top1_accuracy`` -- top-1 accuracy in **percentage points**.
    * ``top5_accuracy`` -- top-5 accuracy in percentage points.
    * ``best_lr``       -- learning rate selected (linear probe / fine-tune).
    * ``k_value``       -- *k* used for k-NN evaluation.
    * ``training_time_seconds`` -- wall-clock time for the evaluation run.
    * ``timestamp``     -- ISO-8601 UTC timestamp of when the result was added.

    Parameters
    ----------
    output_dir : str
        Directory used for saving / loading JSON result files.
    """

    def __init__(self, output_dir: str = "./results") -> None:
        self.output_dir: str = output_dir
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Adding results
    # ------------------------------------------------------------------ #

    def add_result(
        self,
        model_name: str,
        arch: str,
        dataset: str,
        eval_type: str,
        metrics: Dict[str, Any],
    ) -> None:
        """Append a single evaluation result.

        Parameters
        ----------
        model_name : str
            Logical model name (e.g. ``"ijepa"``, ``"dinov2"``).
        arch : str
            Architecture identifier (e.g. ``"vit_huge"``).
        dataset : str
            Dataset identifier (e.g. ``"imagenet1k"``).
        eval_type : str
            Evaluation protocol.  One of ``"knn"``, ``"linear_probe"``,
            or ``"finetune"``.
        metrics : dict
            Arbitrary metric dictionary.  Common keys:

            * ``top1_accuracy`` (float, percentage points)
            * ``top5_accuracy`` (float, percentage points)
            * ``best_lr`` (float, for linear probe / fine-tune)
            * ``k_value`` (int, for k-NN)
            * ``training_time_seconds`` (float)
        """
        entry: Dict[str, Any] = {
            "model_name": model_name,
            "arch": arch,
            "dataset": dataset,
            "eval_type": eval_type,
            "top1_accuracy": metrics.get("top1_accuracy"),
            "top5_accuracy": metrics.get("top5_accuracy"),
            "best_lr": metrics.get("best_lr"),
            "k_value": metrics.get("k_value"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.results.append(entry)
        logger.info(
            "Recorded result: %s / %s / %s / %s  top1=%.2f%%",
            model_name,
            arch,
            dataset,
            eval_type,
            entry["top1_accuracy"] if entry["top1_accuracy"] is not None else float("nan"),
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, filename: str = "results.json") -> str:
        """Serialise all results to a JSON file.

        Parameters
        ----------
        filename : str
            Name of the output file (written inside ``self.output_dir``).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as fh:
            json.dump(self.results, fh, indent=2, default=str)
        logger.info("Saved %d result(s) to %s", len(self.results), path)
        return os.path.abspath(path)

    def load(self, filename: str = "results.json") -> None:
        """Load results from a JSON file, **replacing** any currently held data.

        Parameters
        ----------
        filename : str
            Name of the file to read (inside ``self.output_dir``).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "r") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array at top level in {path}, got {type(data).__name__}."
            )
        self.results = data
        logger.info("Loaded %d result(s) from %s", len(self.results), path)

    # ------------------------------------------------------------------ #
    # Pandas export
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> "pd.DataFrame":
        """Convert the results list to a :class:`pandas.DataFrame`.

        Returns
        -------
        pd.DataFrame
            One row per result entry.

        Raises
        ------
        ImportError
            If ``pandas`` is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install it with: pip install pandas"
            ) from exc

        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------ #
    # Filtering helpers
    # ------------------------------------------------------------------ #

    def _filter(
        self,
        dataset: Optional[str] = None,
        eval_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the subset of results matching the given filters."""
        out: List[Dict[str, Any]] = []
        for r in self.results:
            if dataset is not None and r.get("dataset") != dataset:
                continue
            if eval_type is not None and r.get("eval_type") != eval_type:
                continue
            out.append(r)
        return out

    # ------------------------------------------------------------------ #
    # Comparison table generation
    # ------------------------------------------------------------------ #

    def generate_comparison_table(
        self,
        dataset: str,
        eval_type: str,
        metric: str = "top1_accuracy",
    ) -> str:
        """Generate a markdown table comparing all models on a dataset / eval type.

        The table includes columns for the model name, architecture, top-1 and
        top-5 accuracy, the published baseline (if available), and the delta
        between the obtained and published top-1 values.

        Parameters
        ----------
        dataset : str
            Dataset identifier (e.g. ``"imagenet1k"``).
        eval_type : str
            Evaluation type (e.g. ``"linear_probe"``).
        metric : str
            Primary metric used for sorting rows (descending).

        Returns
        -------
        str
            A full markdown-formatted table string.
        """
        subset = self._filter(dataset=dataset, eval_type=eval_type)
        if not subset:
            return f"_No results found for dataset={dataset}, eval_type={eval_type}._\n"

        # Sort by the primary metric (descending), with None treated as -inf.
        subset.sort(key=lambda r: r.get(metric) if r.get(metric) is not None else float("-inf"), reverse=True)

        # -- heading ---------------------------------------------------------
        ds_display = _DATASET_DISPLAY.get(dataset, dataset)
        et_display = _EVAL_TYPE_DISPLAY.get(eval_type, eval_type)
        metric_display = metric.replace("_", " ").title().replace("Top1", "Top-1").replace("Top5", "Top-5")
        heading = f"## {ds_display} {et_display} ({metric_display})\n"

        # -- table header ----------------------------------------------------
        lines: List[str] = [
            heading,
            "| Method     | Architecture | Top-1 (%) | Top-5 (%) | Published | Delta |",
            "|------------|-------------|-----------|-----------|-----------|-------|",
        ]

        # -- table rows ------------------------------------------------------
        for entry in subset:
            model_name: str = entry.get("model_name", "unknown")
            arch: str = entry.get("arch", "unknown")
            top1 = entry.get("top1_accuracy")
            top5 = entry.get("top5_accuracy")

            arch_display = _ARCH_DISPLAY.get(arch, arch)

            top1_str = f"{top1:.1f}" if top1 is not None else "--"
            top5_str = f"{top5:.1f}" if top5 is not None else "--"

            # Look up published baseline
            baseline_key = (model_name.lower(), arch.lower(), dataset.lower(), eval_type.lower())
            baseline = PUBLISHED_BASELINES.get(baseline_key)

            if baseline is not None and top1 is not None:
                pub_top1 = baseline.get("top1")
                pub_str = f"{pub_top1:.1f}" if pub_top1 is not None else "--"
                delta = top1 - pub_top1 if pub_top1 is not None else None
                delta_str = f"{delta:+.1f}" if delta is not None else "--"
            else:
                pub_str = "--"
                delta_str = "--"

            # Capitalise model name for display
            method_display = model_name.replace("_", " ").title()

            lines.append(
                f"| {method_display:<10} | {arch_display:<11} | {top1_str:>9} | {top5_str:>9} | {pub_str:>9} | {delta_str:>5} |"
            )

        lines.append("")  # trailing newline
        return "\n".join(lines)

    def generate_full_report(
        self,
        datasets: List[str],
        eval_types: List[str],
    ) -> str:
        """Generate a complete markdown report with comparison tables.

        Iterates over every ``(dataset, eval_type)`` pair and concatenates
        the individual comparison tables.  Pairs with no matching results are
        silently skipped.

        Parameters
        ----------
        datasets : list[str]
            Dataset identifiers to include.
        eval_types : list[str]
            Evaluation types to include.

        Returns
        -------
        str
            Full markdown report string.
        """
        sections: List[str] = ["# SSL Benchmark Report\n"]
        sections.append(f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_\n")

        for ds in datasets:
            for et in eval_types:
                subset = self._filter(dataset=ds, eval_type=et)
                if not subset:
                    continue
                sections.append(self.generate_comparison_table(ds, et))

        # Summary statistics
        sections.append("---\n")
        sections.append(f"**Total results:** {len(self.results)}\n")

        return "\n".join(sections)

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def plot_comparison_bar_chart(
        self,
        dataset: str,
        eval_type: str,
        save_path: Optional[str] = None,
    ) -> None:
        """Create a grouped bar chart comparing model accuracies.

        Each model is shown as a cluster of two bars (top-1 and top-5).
        Published baselines, when available, are overlaid as horizontal
        dashed lines per bar group.

        Parameters
        ----------
        dataset : str
            Dataset identifier.
        eval_type : str
            Evaluation type.
        save_path : str, optional
            If provided, the figure is saved to this path.  Otherwise the
            plot is displayed interactively via ``plt.show()``.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        subset = self._filter(dataset=dataset, eval_type=eval_type)
        if not subset:
            logger.warning("No results to plot for dataset=%s, eval_type=%s", dataset, eval_type)
            return

        # Sort by top-1 accuracy (descending).
        subset.sort(
            key=lambda r: r.get("top1_accuracy") if r.get("top1_accuracy") is not None else float("-inf"),
            reverse=True,
        )

        labels: List[str] = []
        top1_vals: List[float] = []
        top5_vals: List[float] = []
        published_top1: List[Optional[float]] = []

        for entry in subset:
            model_name = entry.get("model_name", "unknown")
            arch = entry.get("arch", "unknown")
            arch_short = _ARCH_DISPLAY.get(arch, arch)
            labels.append(f"{model_name}\n{arch_short}")
            top1_vals.append(entry.get("top1_accuracy", 0.0) or 0.0)
            top5_vals.append(entry.get("top5_accuracy", 0.0) or 0.0)

            baseline_key = (model_name.lower(), arch.lower(), dataset.lower(), eval_type.lower())
            baseline = PUBLISHED_BASELINES.get(baseline_key)
            published_top1.append(baseline["top1"] if baseline else None)

        x = np.arange(len(labels))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 6))
        bars_top1 = ax.bar(x - bar_width / 2, top1_vals, bar_width, label="Top-1 (%)", color="#4C72B0")
        bars_top5 = ax.bar(x + bar_width / 2, top5_vals, bar_width, label="Top-5 (%)", color="#55A868", alpha=0.8)

        # Overlay published baselines as markers.
        for i, pub in enumerate(published_top1):
            if pub is not None:
                ax.plot(
                    i - bar_width / 2, pub,
                    marker="D", markersize=7, color="red", zorder=5,
                )

        # Add value labels on bars.
        for bar in bars_top1:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8,
                )

        ds_display = _DATASET_DISPLAY.get(dataset, dataset)
        et_display = _EVAL_TYPE_DISPLAY.get(eval_type, eval_type)
        ax.set_title(f"{ds_display} -- {et_display}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved comparison bar chart to %s", save_path)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_training_curves(
        training_histories: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training curves (loss and/or accuracy) for multiple models.

        Parameters
        ----------
        training_histories : dict[str, dict[str, list[float]]]
            Mapping of ``model_name`` to a dict with one or more of the
            following keys, each holding a list of per-epoch values:

            * ``"train_loss"``
            * ``"val_loss"``
            * ``"train_accuracy"``
            * ``"val_accuracy"``

            Example::

                {
                    "ijepa": {
                        "train_loss": [2.1, 1.8, ...],
                        "val_accuracy": [50.0, 60.0, ...],
                    },
                    "dinov2": { ... },
                }

        save_path : str, optional
            If provided, the figure is saved to this path.
        """
        import matplotlib.pyplot as plt

        has_loss = any("train_loss" in h or "val_loss" in h for h in training_histories.values())
        has_acc = any("train_accuracy" in h or "val_accuracy" in h for h in training_histories.values())

        n_panels = int(has_loss) + int(has_acc)
        if n_panels == 0:
            logger.warning("No recognised keys in training_histories; nothing to plot.")
            return

        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        # ---- Loss panel ----------------------------------------------------
        if has_loss:
            ax = axes[panel_idx]
            panel_idx += 1
            for model_name, history in training_histories.items():
                if "train_loss" in history:
                    epochs = range(1, len(history["train_loss"]) + 1)
                    ax.plot(epochs, history["train_loss"], label=f"{model_name} (train)")
                if "val_loss" in history:
                    epochs = range(1, len(history["val_loss"]) + 1)
                    ax.plot(epochs, history["val_loss"], linestyle="--", label=f"{model_name} (val)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training / Validation Loss")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        # ---- Accuracy panel ------------------------------------------------
        if has_acc:
            ax = axes[panel_idx]
            panel_idx += 1
            for model_name, history in training_histories.items():
                if "train_accuracy" in history:
                    epochs = range(1, len(history["train_accuracy"]) + 1)
                    ax.plot(epochs, history["train_accuracy"], label=f"{model_name} (train)")
                if "val_accuracy" in history:
                    epochs = range(1, len(history["val_accuracy"]) + 1)
                    ax.plot(epochs, history["val_accuracy"], linestyle="--", label=f"{model_name} (val)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Training / Validation Accuracy")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved training curves to %s", save_path)
            plt.close(fig)
        else:
            plt.show()

    @staticmethod
    def plot_knn_k_sensitivity(
        knn_results: Dict[str, Dict[int, float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot top-1 accuracy as a function of *k* for multiple models.

        Parameters
        ----------
        knn_results : dict[str, dict[int, float]]
            Mapping of ``model_name`` to a dict of ``{k_value: top1_accuracy}``.

            Example::

                {
                    "ijepa (ViT-H/14)": {10: 72.5, 20: 73.1, 50: 72.8, 200: 70.3},
                    "dinov2 (ViT-B/14)": {10: 81.0, 20: 81.5, 50: 81.2, 200: 80.0},
                }

        save_path : str, optional
            If provided, the figure is saved to this path.
        """
        import matplotlib.pyplot as plt

        if not knn_results:
            logger.warning("Empty knn_results dict; nothing to plot.")
            return

        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name, k_to_acc in knn_results.items():
            sorted_k = sorted(k_to_acc.keys())
            accuracies = [k_to_acc[k] for k in sorted_k]
            ax.plot(sorted_k, accuracies, marker="o", label=model_name)

        ax.set_xlabel("k (number of neighbours)")
        ax.set_ylabel("Top-1 Accuracy (%)")
        ax.set_title("k-NN Sensitivity Analysis")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Use log scale if k values span a wide range.
        all_k = set()
        for k_to_acc in knn_results.values():
            all_k.update(k_to_acc.keys())
        if len(all_k) >= 2 and max(all_k) / max(min(all_k), 1) > 20:
            ax.set_xscale("log")
            ax.set_xticks(sorted(all_k))
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        fig.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Saved k-NN sensitivity plot to %s", save_path)
            plt.close(fig)
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    # Verification against published baselines
    # ------------------------------------------------------------------ #

    def verify_against_published(
        self,
        tolerance: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Compare obtained results against published baselines.

        For each result that has a matching entry in :data:`PUBLISHED_BASELINES`,
        the absolute difference between the obtained top-1 accuracy and the
        published value is computed.  Entries where the delta exceeds
        *tolerance* (in percentage points) are returned and logged as potential
        issues.

        Parameters
        ----------
        tolerance : float
            Maximum acceptable absolute difference (in percentage points)
            between the obtained and published top-1 accuracy.

        Returns
        -------
        list[dict]
            Each dict contains:

            * ``model_name``, ``arch``, ``dataset``, ``eval_type``
            * ``obtained_top1`` -- the accuracy from our evaluation.
            * ``published_top1`` -- the reference value.
            * ``delta`` -- ``obtained - published`` (signed).
            * ``exceeds_tolerance`` -- bool, True when ``|delta| > tolerance``.

            Only entries where ``exceeds_tolerance`` is ``True`` are included.
        """
        flagged: List[Dict[str, Any]] = []

        for entry in self.results:
            model_name = entry.get("model_name", "")
            arch = entry.get("arch", "")
            dataset = entry.get("dataset", "")
            eval_type = entry.get("eval_type", "")
            top1 = entry.get("top1_accuracy")

            if top1 is None:
                continue

            baseline_key = (
                model_name.lower(),
                arch.lower(),
                dataset.lower(),
                eval_type.lower(),
            )
            baseline = PUBLISHED_BASELINES.get(baseline_key)
            if baseline is None:
                continue

            published_top1 = baseline.get("top1")
            if published_top1 is None:
                continue

            delta = top1 - published_top1

            if abs(delta) > tolerance:
                issue = {
                    "model_name": model_name,
                    "arch": arch,
                    "dataset": dataset,
                    "eval_type": eval_type,
                    "obtained_top1": top1,
                    "published_top1": published_top1,
                    "delta": round(delta, 2),
                    "exceeds_tolerance": True,
                }
                flagged.append(issue)
                logger.warning(
                    "VERIFICATION ISSUE: %s / %s / %s / %s -- "
                    "obtained %.2f%% vs published %.2f%% (delta %+.2f pp, tolerance %.1f pp)",
                    model_name, arch, dataset, eval_type,
                    top1, published_top1, delta, tolerance,
                )

        if not flagged:
            logger.info(
                "All %d verifiable result(s) are within %.1f pp of published baselines.",
                sum(
                    1 for r in self.results
                    if (r.get("model_name", "").lower(), r.get("arch", "").lower(),
                        r.get("dataset", "").lower(), r.get("eval_type", "").lower())
                    in PUBLISHED_BASELINES and r.get("top1_accuracy") is not None
                ),
                tolerance,
            )

        return flagged

    # ------------------------------------------------------------------ #
    # Dunder helpers
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Return the number of stored results."""
        return len(self.results)

    def __repr__(self) -> str:
        return (
            f"BenchmarkResults(output_dir={self.output_dir!r}, "
            f"num_results={len(self.results)})"
        )
