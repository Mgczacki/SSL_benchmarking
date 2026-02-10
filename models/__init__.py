"""SSL baseline model loaders.

Unified interface for loading and evaluating pre-trained self-supervised learning
models for benchmarking.

Usage::

    from models import load_model

    model = load_model("dinov2", arch="vit_base")
    features = model.extract_features(images)   # (B, D) normalised tensor
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

from models.base import BaseSSLModel
from models.dinov1 import DINOv1Model
from models.dinov2 import DINOv2Model
from models.ibot import iBOTModel
from models.ijepa import IJEPAModel
from models.lejepa import LeJEPAModel
from models.mae import MAEModel

__all__ = [
    "BaseSSLModel",
    "DINOv2Model",
    "DINOv1Model",
    "IJEPAModel",
    "MAEModel",
    "iBOTModel",
    "LeJEPAModel",
    "load_model",
    "MODEL_REGISTRY",
    "CHECKPOINT_INFO",
    "get_checkpoint_info",
    "list_available_models",
]

# ---------------------------------------------------------------------------
# Model registry & factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Type[BaseSSLModel]] = {
    "dinov2": DINOv2Model,
    "dinov1": DINOv1Model,
    "ijepa": IJEPAModel,
    "mae": MAEModel,
    "ibot": iBOTModel,
    "lejepa": LeJEPAModel,
}


def load_model(
    model_name: str,
    arch: str = "vit_base",
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> BaseSSLModel:
    """Factory function that builds and loads an SSL model from the registry.

    Parameters
    ----------
    model_name:
        Key into ``MODEL_REGISTRY``, e.g. ``"dinov2"``, ``"ijepa"``.
    arch:
        Architecture variant understood by the chosen model class.
    checkpoint_path:
        Optional path to a local checkpoint file.  Required for models that
        do not have publicly-hosted timm/HuggingFace weights (I-JEPA, iBOT,
        LeJEPA).
    **kwargs:
        Forwarded to the model constructor (e.g. ``concat_n_layers`` for
        I-JEPA, ``device``).

    Returns
    -------
    BaseSSLModel
        A fully loaded model ready for feature extraction.

    Raises
    ------
    ValueError
        If *model_name* is not found in the registry.

    Examples
    --------
    >>> model = load_model("dinov2", arch="vit_base")
    >>> feats = model.extract_features(batch)   # (B, 768)

    >>> model = load_model(
    ...     "ijepa",
    ...     arch="vit_huge",
    ...     checkpoint_path="/data/checkpoints/ijepa_vith14_1k.pth",
    ...     concat_n_layers=4,
    ... )
    >>> feats = model.extract_features(batch)   # (B, 5120)
    """
    model_name = model_name.lower().replace("-", "").replace("_", "")
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {available}"
        )

    cls = MODEL_REGISTRY[model_name]
    model = cls(arch=arch, checkpoint_path=checkpoint_path, **kwargs)
    model.load_checkpoint()
    return model


# ---------------------------------------------------------------------------
# Checkpoint information catalogue
# ---------------------------------------------------------------------------

CHECKPOINT_INFO: Dict[Tuple[str, str], Dict[str, Any]] = {
    # ---- DINOv2 ----
    ("dinov2", "vit_small"): {
        "url": "https://huggingface.co/facebook/dinov2-small",
        "embedding_dim": 384,
        "description": "DINOv2 ViT-S/14 -- 79.0% linear probe top-1 on ImageNet-1k.",
    },
    ("dinov2", "vit_base"): {
        "url": "https://huggingface.co/facebook/dinov2-base",
        "embedding_dim": 768,
        "description": "DINOv2 ViT-B/14 -- 82.1% linear probe top-1 on ImageNet-1k.",
    },
    ("dinov2", "vit_large"): {
        "url": "https://huggingface.co/facebook/dinov2-large",
        "embedding_dim": 1024,
        "description": "DINOv2 ViT-L/14 -- 83.5% linear probe top-1 on ImageNet-1k.",
    },
    ("dinov2", "vit_giant"): {
        "url": "https://huggingface.co/facebook/dinov2-giant",
        "embedding_dim": 1536,
        "description": "DINOv2 ViT-g/14 -- 83.5% linear probe top-1 on ImageNet-1k (distilled).",
    },
    # ---- DINOv1 ----
    ("dinov1", "dino_vits16"): {
        "url": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
        "embedding_dim": 384,
        "description": "DINOv1 ViT-S/16 -- 77.0% linear probe top-1 on ImageNet-1k.",
    },
    ("dinov1", "dino_vits8"): {
        "url": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
        "embedding_dim": 384,
        "description": "DINOv1 ViT-S/8 -- 78.3% linear probe top-1 on ImageNet-1k.",
    },
    ("dinov1", "dino_vitb16"): {
        "url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
        "embedding_dim": 768,
        "description": "DINOv1 ViT-B/16 -- 78.2% linear probe top-1 on ImageNet-1k.",
    },
    ("dinov1", "dino_vitb8"): {
        "url": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        "embedding_dim": 768,
        "description": "DINOv1 ViT-B/8 -- 80.1% linear probe top-1 on ImageNet-1k.",
    },
    # ---- I-JEPA ----
    ("ijepa", "vit_base"): {
        "url": "https://github.com/facebookresearch/ijepa/releases/download/v1.0/ijepa-vit.b.16-448px-300e.pth",
        "embedding_dim": 768,
        "description": "I-JEPA ViT-B/16 -- 76.5% linear probe top-1 on ImageNet-1k (300 epochs).",
    },
    ("ijepa", "vit_large"): {
        "url": "https://github.com/facebookresearch/ijepa/releases/download/v1.0/ijepa-vit.l.16-448px-300e.pth",
        "embedding_dim": 1024,
        "description": "I-JEPA ViT-L/16 -- 79.3% linear probe top-1 on ImageNet-1k (300 epochs).",
    },
    ("ijepa", "vit_huge"): {
        "url": "https://github.com/facebookresearch/ijepa/releases/download/v1.0/ijepa-vit.h.14-448px-300e.pth",
        "embedding_dim": 1280,
        "description": (
            "I-JEPA ViT-H/14 -- 81.1% linear probe top-1 on ImageNet-1k "
            "(300 epochs). With concat of last 4 layers: ~82%."
        ),
    },
    # ---- MAE ----
    ("mae", "vit_base"): {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth",
        "embedding_dim": 768,
        "description": "MAE ViT-B/16 -- 68.0% linear probe top-1 on ImageNet-1k (1600 epochs).",
    },
    ("mae", "vit_large"): {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth",
        "embedding_dim": 1024,
        "description": "MAE ViT-L/16 -- 76.0% linear probe top-1 on ImageNet-1k (1600 epochs).",
    },
    ("mae", "vit_huge"): {
        "url": "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth",
        "embedding_dim": 1280,
        "description": "MAE ViT-H/14 -- 78.0% linear probe top-1 on ImageNet-1k (1600 epochs).",
    },
    # ---- iBOT ----
    ("ibot", "vit_small"): {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth",
        "embedding_dim": 384,
        "description": "iBOT ViT-S/16 -- 77.9% linear probe top-1 on ImageNet-1k.",
    },
    ("ibot", "vit_base"): {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth",
        "embedding_dim": 768,
        "description": "iBOT ViT-B/16 -- 79.5% linear probe top-1 on ImageNet-1k.",
    },
    ("ibot", "vit_large"): {
        "url": "https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitl_16/checkpoint_teacher.pth",
        "embedding_dim": 1024,
        "description": "iBOT ViT-L/16 -- 81.7% linear probe top-1 on ImageNet-1k.",
    },
    # ---- LeJEPA ----
    ("lejepa", "vit_huge"): {
        "url": None,  # no public URL at time of writing
        "embedding_dim": 1280,
        "description": "LeJEPA ViT-H/14 -- local checkpoint required.",
    },
}


def get_checkpoint_info(
    model_name: str, arch: str
) -> Optional[Dict[str, Any]]:
    """Look up checkpoint metadata for a (model, arch) pair.

    Parameters
    ----------
    model_name:
        Registry key, e.g. ``"dinov2"``.
    arch:
        Architecture variant, e.g. ``"vit_base"``.

    Returns
    -------
    dict or None
        A dict with ``url``, ``embedding_dim``, and ``description`` keys,
        or ``None`` if the combination is unknown.
    """
    return CHECKPOINT_INFO.get((model_name, arch))


def list_available_models() -> Dict[str, list[str]]:
    """Return a mapping of model names to their supported architecture keys.

    Returns
    -------
    dict
        ``{ "dinov2": ["vit_small", "vit_base", ...], ... }``
    """
    result: Dict[str, list[str]] = {}
    for (model_name, arch) in sorted(CHECKPOINT_INFO):
        result.setdefault(model_name, []).append(arch)
    return result
