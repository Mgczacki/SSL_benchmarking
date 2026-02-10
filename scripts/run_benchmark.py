#!/usr/bin/env python3
"""Main benchmarking orchestrator for JEPA and SSL model evaluation.

This script drives the full evaluation pipeline:

1. Load a YAML configuration (with optional CLI overrides).
2. Set deterministic random seeds.
3. Load the requested dataset (train + val splits).
4. For every model listed in the configuration:
   a. Load the pretrained encoder.
   b. Extract (or load cached) features for both splits.
   c. Run all enabled evaluation protocols -- k-NN, quick linear probe
      (sklearn), full linear probe with LR sweep, fine-tuning, and
      low-shot evaluation.
   d. Log per-model results.
5. Aggregate results into a comparison table and persist to disk.

Usage examples::

    # Run with default config
    python scripts/run_benchmark.py

    # Evaluate specific models on ImageNet-1K
    python scripts/run_benchmark.py \\
        --config configs/imagenet_full.yaml \\
        --models dinov2,ijepa \\
        --eval knn,linear

    # Quick CIFAR-100 k-NN check on CPU
    python scripts/run_benchmark.py \\
        --dataset cifar100 --eval knn --device cpu
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# ---- project imports (relative to repo root) --------------------------------
# These modules are expected to live alongside this script in the jepa-benchmark
# package.  When running as ``python scripts/run_benchmark.py`` from the repo
# root, we insert the repo root into ``sys.path`` so that bare ``import``
# statements resolve correctly.

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.datasets import get_dataset, get_eval_transform, get_num_classes
from models.baselines import load_model
from evaluation.feature_extraction import extract_and_cache, load_cached_features
from evaluation.knn import KNNEvaluator, run_knn_evaluation
from evaluation.linear_probe import (
    LinearProbeTrainer,
    lr_sweep,
    quick_linear_probe,
)
from evaluation.finetune import FineTuner, low_shot_finetune

# ---- logging setup -----------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jepa_benchmark")

# ===========================================================================
# Configuration helpers
# ===========================================================================


def _load_yaml(path: str) -> Dict[str, Any]:
    """Read a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : str
        Filesystem path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *overrides* into *base* (non-destructive).

    Values in *overrides* take precedence.  Nested dictionaries are merged
    recursively rather than replaced wholesale.

    Parameters
    ----------
    base : dict
        Base configuration dictionary (will be deep-copied).
    overrides : dict
        Dictionary whose values override those in *base*.

    Returns
    -------
    dict
        Merged configuration.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI argument overrides on top of the loaded YAML config.

    Only non-``None`` CLI arguments are applied so that unset flags leave the
    YAML defaults intact.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from YAML.
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict
        Updated configuration dictionary.
    """
    if args.dataset is not None:
        cfg["datasets"]["primary"] = args.dataset

    if args.models is not None:
        requested = [m.strip() for m in args.models.split(",")]
        cfg["models"] = [m for m in cfg["models"] if m["name"] in requested]
        if not cfg["models"]:
            logger.warning(
                "None of the requested models (%s) matched entries in the "
                "config.  Available: check your YAML file.",
                args.models,
            )

    if args.eval is not None:
        eval_keys = {e.strip() for e in args.eval.split(",")}
        key_map = {
            "knn": "knn",
            "linear": "linear_probe",
            "quick_linear": "quick_linear_probe",
            "finetune": "finetune",
            "low_shot": "low_shot",
        }
        for yaml_key in ["knn", "linear_probe", "quick_linear_probe", "finetune", "low_shot"]:
            cfg["evaluation"][yaml_key]["enabled"] = yaml_key in eval_keys or key_map.get(yaml_key, yaml_key) in eval_keys

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    if args.cache_dir is not None:
        cfg["cache_dir"] = args.cache_dir

    if args.device is not None:
        cfg["device"] = args.device

    if args.seed is not None:
        cfg["seed"] = args.seed

    return cfg


# ===========================================================================
# Deterministic seeding
# ===========================================================================


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all RNG sources.

    Parameters
    ----------
    seed : int
        Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


# ===========================================================================
# Device resolution
# ===========================================================================


def _resolve_device(device_str: str) -> torch.device:
    """Resolve the ``device`` config string to a :class:`torch.device`.

    ``"auto"`` selects the best available accelerator (CUDA > MPS > CPU).

    Parameters
    ----------
    device_str : str
        One of ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns
    -------
    torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ===========================================================================
# Timestamp helper
# ===========================================================================


def _timestamp() -> str:
    """Return an ISO-8601 timestamp string for log messages."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ===========================================================================
# Results formatting
# ===========================================================================


def _format_results_table(all_results: Dict[str, Dict[str, Any]]) -> str:
    """Format aggregated results as a Markdown table.

    Parameters
    ----------
    all_results : dict
        Mapping of ``model_name -> {eval_name -> metrics}``.

    Returns
    -------
    str
        A Markdown-formatted comparison table.
    """
    if not all_results:
        return "(no results)"

    # Collect all unique metric columns across every model.
    columns: List[str] = []
    for model_metrics in all_results.values():
        for eval_name, metrics in model_metrics.items():
            if isinstance(metrics, dict):
                for metric_key, value in metrics.items():
                    col = f"{eval_name}/{metric_key}"
                    if col not in columns and isinstance(value, (int, float)):
                        columns.append(col)

    if not columns:
        return "(no numeric results to display)"

    # Build table header.
    header = "| Model | " + " | ".join(columns) + " |"
    separator = "|-------|" + "|".join(["--------"] * len(columns)) + "|"

    rows: List[str] = [header, separator]
    for model_name, model_metrics in all_results.items():
        values: List[str] = []
        for col in columns:
            eval_name, metric_key = col.split("/", 1)
            if eval_name in model_metrics and isinstance(model_metrics[eval_name], dict):
                val = model_metrics[eval_name].get(metric_key, "")
                if isinstance(val, float):
                    values.append(f"{val:.4f}")
                elif isinstance(val, int):
                    values.append(str(val))
                else:
                    values.append("")
            else:
                values.append("")
        rows.append(f"| {model_name} | " + " | ".join(values) + " |")

    return "\n".join(rows)


def _serializable(obj: Any) -> Any:
    """Convert *obj* to a JSON-serializable type (recursive).

    Handles :class:`torch.Tensor`, numpy scalars/arrays, and nested
    dicts / lists.
    """
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


# ===========================================================================
# W&B integration
# ===========================================================================


def _init_wandb(cfg: Dict[str, Any]) -> bool:
    """Conditionally initialise Weights & Biases.

    Parameters
    ----------
    cfg : dict
        Full benchmark configuration.

    Returns
    -------
    bool
        ``True`` if W&B was initialised successfully.
    """
    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return False

    try:
        import wandb  # noqa: F811

        wandb.init(
            project=wandb_cfg.get("project", "jepa-benchmark"),
            entity=wandb_cfg.get("entity"),
            config=cfg,
        )
        logger.info("Weights & Biases logging enabled (project=%s)", wandb_cfg.get("project"))
        return True
    except Exception as exc:
        logger.warning("Failed to initialise wandb: %s. Continuing without it.", exc)
        return False


def _log_wandb(metrics: Dict[str, Any], prefix: str = "") -> None:
    """Log flat metrics to W&B if available.

    Parameters
    ----------
    metrics : dict
        Metrics to log.  Nested dicts are flattened with ``/`` separators.
    prefix : str
        Optional prefix prepended to every key.
    """
    try:
        import wandb

        if wandb.run is None:
            return

        flat: Dict[str, Any] = {}

        def _flatten(d: Dict[str, Any], parent_key: str = "") -> None:
            for k, v in d.items():
                full_key = f"{parent_key}/{k}" if parent_key else k
                if isinstance(v, dict):
                    _flatten(v, full_key)
                elif isinstance(v, (int, float)):
                    flat[full_key] = v

        _flatten(metrics, prefix)
        wandb.log(flat)
    except Exception:
        pass


# ===========================================================================
# Per-model evaluation pipeline
# ===========================================================================


def _run_model_evaluation(
    model_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    dataset_name: str,
    num_classes: int,
    device: torch.device,
    cache_dir: str,
    output_dir: str,
    use_wandb: bool,
) -> Dict[str, Any]:
    """Run the complete evaluation pipeline for a single model.

    Parameters
    ----------
    model_cfg : dict
        Model entry from the YAML config (``name``, ``arch``, ``checkpoint``).
    eval_cfg : dict
        ``evaluation`` section of the YAML config.
    train_loader : DataLoader
        Training-set DataLoader (with eval transform applied).
    val_loader : DataLoader
        Validation-set DataLoader.
    dataset_name : str
        Name of the dataset being evaluated (used for cache keys).
    num_classes : int
        Number of target classes in the dataset.
    device : torch.device
        Device for model inference and training.
    cache_dir : str
        Root directory for feature caching.
    output_dir : str
        Root directory for persisted results.
    use_wandb : bool
        Whether W&B logging is active.

    Returns
    -------
    dict
        Mapping of ``eval_name -> metrics_dict`` for this model.
    """
    model_name: str = model_cfg["name"]
    arch: str = model_cfg["arch"]
    checkpoint: Optional[str] = model_cfg.get("checkpoint")
    results: Dict[str, Any] = {}

    logger.info("=" * 70)
    logger.info("Model: %s  (arch=%s)", model_name, arch)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    try:
        model = load_model(
            model_name=model_name,
            arch=arch,
            checkpoint_path=checkpoint,
            device=str(device),
        )
        logger.info("Successfully loaded model '%s'.", model_name)
    except Exception as exc:
        logger.warning(
            "Failed to load model '%s' (arch=%s, checkpoint=%s): %s. Skipping.",
            model_name,
            arch,
            checkpoint,
            exc,
        )
        results["_error"] = f"Model load failed: {exc}"
        return results

    # ------------------------------------------------------------------
    # 2. Extract features (with caching)
    # ------------------------------------------------------------------
    cache_key = f"{model_name}_{arch}"
    try:
        logger.info("Extracting training features ...")
        t0 = time.time()
        train_features, train_labels = extract_and_cache(
            model=model,
            dataloader=train_loader,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            model_name=cache_key,
            split="train",
            device=device,
        )
        logger.info(
            "Train features: %s  (%.1fs)",
            tuple(train_features.shape),
            time.time() - t0,
        )

        logger.info("Extracting validation features ...")
        t0 = time.time()
        val_features, val_labels = extract_and_cache(
            model=model,
            dataloader=val_loader,
            cache_dir=cache_dir,
            dataset_name=dataset_name,
            model_name=cache_key,
            split="val",
            device=device,
        )
        logger.info(
            "Val features:   %s  (%.1fs)",
            tuple(val_features.shape),
            time.time() - t0,
        )
    except Exception as exc:
        logger.error(
            "Feature extraction failed for '%s': %s. Skipping.", model_name, exc
        )
        results["_error"] = f"Feature extraction failed: {exc}"
        return results

    # ------------------------------------------------------------------
    # 3. k-NN evaluation
    # ------------------------------------------------------------------
    knn_cfg = eval_cfg.get("knn", {})
    if knn_cfg.get("enabled", False):
        logger.info("[%s] Running k-NN evaluation ...", model_name)
        try:
            knn_results = run_knn_evaluation(
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                num_classes=num_classes,
                k_values=knn_cfg.get("k_values", [10, 20, 200]),
                temperature=knn_cfg.get("temperature", 0.07),
            )
            # Flatten per-k results for the summary.  Drop per_class_accuracy
            # tensors to keep the summary JSON small.
            flat_knn: Dict[str, Any] = {}
            for k_key, k_metrics in knn_results.items():
                if isinstance(k_metrics, dict):
                    flat_knn[f"{k_key}/top1"] = k_metrics.get("top1_accuracy")
                    flat_knn[f"{k_key}/top5"] = k_metrics.get("top5_accuracy")
            results["knn"] = flat_knn

            if use_wandb:
                _log_wandb(flat_knn, prefix=f"{model_name}/knn")

            logger.info("[%s] k-NN results: %s", model_name, {
                k: f"{v:.4f}" for k, v in flat_knn.items() if v is not None
            })
        except Exception as exc:
            logger.error("[%s] k-NN evaluation failed: %s", model_name, exc)
            results["knn"] = {"_error": str(exc)}

    # ------------------------------------------------------------------
    # 4. Quick linear probe (sklearn)
    # ------------------------------------------------------------------
    qlp_cfg = eval_cfg.get("quick_linear_probe", {})
    if qlp_cfg.get("enabled", False):
        logger.info("[%s] Running quick linear probe (sklearn) ...", model_name)
        try:
            qlp_results = quick_linear_probe(
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                C_values=qlp_cfg.get("C_values", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]),
            )
            results["quick_linear_probe"] = qlp_results

            if use_wandb:
                _log_wandb(qlp_results, prefix=f"{model_name}/quick_linear_probe")

            logger.info("[%s] Quick linear probe: %s", model_name, {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in qlp_results.items()
            })
        except Exception as exc:
            logger.error("[%s] Quick linear probe failed: %s", model_name, exc)
            results["quick_linear_probe"] = {"_error": str(exc)}

    # ------------------------------------------------------------------
    # 5. Full linear probe with LR sweep
    # ------------------------------------------------------------------
    lp_cfg = eval_cfg.get("linear_probe", {})
    if lp_cfg.get("enabled", False):
        logger.info("[%s] Running full linear probe (LR sweep) ...", model_name)
        try:
            lp_results = lr_sweep(
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                num_classes=num_classes,
                feature_dim=train_features.shape[1],
                lr_values=lp_cfg.get("lr_sweep", [0.01, 0.1, 1.0]),
                epochs=lp_cfg.get("epochs", 100),
                batch_size=lp_cfg.get("batch_size", 256),
                momentum=lp_cfg.get("momentum", 0.9),
                weight_decay=lp_cfg.get("weight_decay", 0.0),
                lr_schedule=lp_cfg.get("lr_schedule", "cosine"),
                normalize_features=lp_cfg.get("normalize_features", True),
                device=str(device),
            )
            results["linear_probe"] = lp_results

            if use_wandb:
                _log_wandb(lp_results, prefix=f"{model_name}/linear_probe")

            logger.info("[%s] Linear probe best: %s", model_name, {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in lp_results.items()
                if k.startswith("best")
            })
        except Exception as exc:
            logger.error("[%s] Linear probe failed: %s", model_name, exc)
            results["linear_probe"] = {"_error": str(exc)}

    # ------------------------------------------------------------------
    # 6. Fine-tuning (end-to-end)
    # ------------------------------------------------------------------
    ft_cfg = eval_cfg.get("finetune", {})
    if ft_cfg.get("enabled", False):
        logger.info("[%s] Running fine-tuning ...", model_name)
        try:
            # FineTuner expects an nn.Module backbone; unwrap from BaseSSLModel
            backbone = model._model if hasattr(model, "_model") else model
            finetuner = FineTuner(
                model=backbone,
                num_classes=num_classes,
                lr=ft_cfg.get("lr", 0.001),
                weight_decay=ft_cfg.get("weight_decay", 0.05),
                epochs=ft_cfg.get("epochs", 100),
                batch_size=ft_cfg.get("batch_size", 64),
                layer_decay=ft_cfg.get("layer_decay", 0.65),
                warmup_epochs=ft_cfg.get("warmup_epochs", 5),
                device=str(device),
            )
            ft_results = finetuner.train(
                train_loader=train_loader,
                val_loader=val_loader,
            )
            results["finetune"] = ft_results

            if use_wandb:
                _log_wandb(ft_results, prefix=f"{model_name}/finetune")

            logger.info("[%s] Fine-tune results: %s", model_name, {
                k: f"{v:.4f}" if isinstance(v, float) else v
                for k, v in ft_results.items()
            })
        except Exception as exc:
            logger.error("[%s] Fine-tuning failed: %s", model_name, exc)
            results["finetune"] = {"_error": str(exc)}

    # ------------------------------------------------------------------
    # 7. Low-shot evaluation
    # ------------------------------------------------------------------
    ls_cfg = eval_cfg.get("low_shot", {})
    if ls_cfg.get("enabled", False):
        logger.info("[%s] Running low-shot evaluation ...", model_name)
        try:
            backbone = model._model if hasattr(model, "_model") else model
            ls_all: Dict[str, Any] = {}
            for pct in ls_cfg.get("percentages", [0.01, 0.1]):
                logger.info("[%s] Low-shot fine-tuning with %.1f%% data ...", model_name, pct * 100)
                pct_result = low_shot_finetune(
                    model=backbone,
                    train_dataset=train_loader.dataset,
                    val_dataset=val_loader.dataset,
                    num_classes=num_classes,
                    percent=pct,
                    device=str(device),
                )
                ls_all[f"{pct:.0%}"] = pct_result
            ls_results = ls_all
            results["low_shot"] = ls_results

            if use_wandb:
                _log_wandb(ls_results, prefix=f"{model_name}/low_shot")

            logger.info("[%s] Low-shot results: %s", model_name, ls_results)
        except Exception as exc:
            logger.error("[%s] Low-shot evaluation failed: %s", model_name, exc)
            results["low_shot"] = {"_error": str(exc)}

    return results


# ===========================================================================
# Main entry point
# ===========================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="JEPA Benchmark -- evaluate SSL models on standard protocols.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(_REPO_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override datasets.primary (e.g. cifar100, imagenet1k).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of model names to evaluate (e.g. dinov2,ijepa).",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help=(
            "Comma-separated list of evaluation protocols to enable "
            "(e.g. knn,linear,finetune).  Recognised keys: "
            "knn, linear, quick_linear, finetune, low_shot."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override cache_dir for extracted features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (auto, cuda, mps, cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed.",
    )
    return parser


def main() -> None:
    """Run the full JEPA benchmarking pipeline.

    Steps
    -----
    1. Parse CLI args and load YAML config.
    2. Set deterministic random seeds.
    3. Resolve the compute device.
    4. Load dataset (train + val splits with eval transform).
    5. For each model: load, extract features, run evaluations.
    6. Aggregate and save results.
    """
    parser = _build_parser()
    args = parser.parse_args()

    # -- Load config --------------------------------------------------------
    logger.info("Loading config from %s", args.config)
    cfg = _load_yaml(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    # -- Seed ---------------------------------------------------------------
    seed: int = cfg.get("seed", 42)
    _set_seed(seed)

    # -- Device -------------------------------------------------------------
    device = _resolve_device(cfg.get("device", "auto"))
    logger.info("Using device: %s", device)

    # -- W&B ----------------------------------------------------------------
    use_wandb = _init_wandb(cfg)

    # -- Output directory ---------------------------------------------------
    output_dir: str = cfg.get("output_dir", "./results")
    os.makedirs(output_dir, exist_ok=True)

    cache_dir: str = cfg.get("cache_dir", "./cache/features")
    os.makedirs(cache_dir, exist_ok=True)

    # -- Dataset ------------------------------------------------------------
    ds_cfg = cfg["datasets"]
    dataset_name: str = ds_cfg["primary"]
    image_size: int = ds_cfg.get("image_size", 224)
    batch_size: int = ds_cfg.get("batch_size", 256)
    num_workers: int = cfg.get("num_workers", 4)

    logger.info("Loading dataset '%s' (image_size=%d) ...", dataset_name, image_size)
    eval_transform = get_eval_transform(image_size)
    train_dataset, val_dataset = get_dataset(
        dataset_name=dataset_name,
        data_root=ds_cfg.get("data_root", "./data"),
        image_size=image_size,
        train_transform=eval_transform,   # use eval transform for feature extraction
        eval_transform=eval_transform,
        seed=seed,
    )
    num_classes = get_num_classes(dataset_name)
    logger.info(
        "Dataset loaded: train=%d  val=%d  classes=%d",
        len(train_dataset),
        len(val_dataset),
        num_classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # -- Evaluation loop ----------------------------------------------------
    eval_cfg = cfg.get("evaluation", {})
    model_configs: List[Dict[str, Any]] = cfg.get("models", [])

    if not model_configs:
        logger.warning("No models specified in the configuration.  Nothing to evaluate.")
        return

    all_results: Dict[str, Dict[str, Any]] = {}
    run_start = time.time()

    for model_cfg in model_configs:
        model_name = model_cfg["name"]
        model_start = time.time()

        model_results = _run_model_evaluation(
            model_cfg=model_cfg,
            eval_cfg=eval_cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            dataset_name=dataset_name,
            num_classes=num_classes,
            device=device,
            cache_dir=cache_dir,
            output_dir=output_dir,
            use_wandb=use_wandb,
        )

        elapsed = time.time() - model_start
        logger.info(
            "Finished %s in %.1fs",
            model_name,
            elapsed,
        )
        model_results["_elapsed_seconds"] = round(elapsed, 1)
        all_results[model_name] = model_results

        # Save per-model results incrementally so partial progress is preserved
        # even if a later model fails.
        per_model_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_results.json")
        try:
            with open(per_model_path, "w") as fh:
                json.dump(_serializable(model_results), fh, indent=2)
            logger.info("Saved per-model results to %s", per_model_path)
        except Exception as exc:
            logger.warning("Could not save per-model results: %s", exc)

    # -- Aggregate ----------------------------------------------------------
    total_elapsed = time.time() - run_start
    logger.info("=" * 70)
    logger.info("All models evaluated in %.1fs", total_elapsed)
    logger.info("=" * 70)

    # Save aggregate JSON
    aggregate_path = os.path.join(output_dir, f"benchmark_{dataset_name}_all.json")
    aggregate_payload = {
        "config": cfg,
        "results": _serializable(all_results),
        "timestamp": _timestamp(),
        "total_elapsed_seconds": round(total_elapsed, 1),
    }
    try:
        with open(aggregate_path, "w") as fh:
            json.dump(aggregate_payload, fh, indent=2)
        logger.info("Saved aggregate results to %s", aggregate_path)
    except Exception as exc:
        logger.warning("Could not save aggregate results: %s", exc)

    # Print Markdown summary table
    table = _format_results_table(all_results)
    logger.info("Results summary:\n\n%s\n", table)
    print("\n" + table + "\n")

    # W&B finish
    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass

    logger.info("Benchmark complete.")


if __name__ == "__main__":
    main()
