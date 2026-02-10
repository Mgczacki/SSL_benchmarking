"""DINOv1 model loader via torch.hub."""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch

from models.base import BaseSSLModel

logger = logging.getLogger(__name__)

_DINOV1_ARCHS: Dict[str, int] = {
    "dino_vits16": 384,
    "dino_vits8": 384,
    "dino_vitb16": 768,
    "dino_vitb8": 768,
}

# Allow standard arch names to map to DINO-specific hub names.
_DINOV1_ARCH_ALIASES: Dict[str, str] = {
    "vit_small": "dino_vits16",
    "vit_base": "dino_vitb16",
}


class DINOv1Model(BaseSSLModel):
    """DINOv1 models loaded via ``torch.hub`` from *facebookresearch/dino*.

    Supported *arch* values: ``dino_vits16``, ``dino_vits8``,
    ``dino_vitb16``, ``dino_vitb8``.  Aliases ``vit_small`` and ``vit_base``
    are also accepted for convenience.  Feature extraction uses the CLS token.
    """

    def __init__(self, arch: str = "dino_vitb16", **kwargs: Any) -> None:
        # Resolve aliases (e.g. "vit_base" -> "dino_vitb16").
        resolved = _DINOV1_ARCH_ALIASES.get(arch, arch)
        if resolved not in _DINOV1_ARCHS:
            raise ValueError(
                f"Unknown DINOv1 arch '{arch}'. "
                f"Choose from {list(_DINOV1_ARCHS)} or aliases {list(_DINOV1_ARCH_ALIASES)}"
            )
        super().__init__(arch=resolved, **kwargs)
        self._dim = _DINOV1_ARCHS[resolved]

    # -- BaseSSLModel interface ---------------------------------------------

    def load_checkpoint(self) -> None:
        """Load DINOv1 from ``torch.hub`` (downloads weights automatically)."""
        logger.info("Loading DINOv1 model '%s' via torch.hub ...", self.arch)
        self._model = torch.hub.load(
            "facebookresearch/dino:main", self.arch
        )
        self._model.eval()
        logger.info("DINOv1 '%s' loaded (dim=%d).", self.arch, self._dim)

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features.

        Parameters
        ----------
        images:
            ``(B, 3, 224, 224)`` tensor.

        Returns
        -------
        torch.Tensor
            ``(B, embedding_dim)`` L2-normalised features.
        """
        self._ensure_loaded()
        images = images.to(self.device)
        features = self._model(images)  # (B, D)
        return self._normalise(features)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"dinov1-{self.arch}"
