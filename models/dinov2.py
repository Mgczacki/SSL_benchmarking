"""DINOv2 model loader via HuggingFace."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from models.base import BaseSSLModel

logger = logging.getLogger(__name__)

_DINOV2_HF_MODELS: Dict[str, Tuple[str, int]] = {
    "vit_small": ("facebook/dinov2-small", 384),
    "vit_base": ("facebook/dinov2-base", 768),
    "vit_large": ("facebook/dinov2-large", 1024),
    "vit_giant": ("facebook/dinov2-giant", 1536),
}


class DINOv2Model(BaseSSLModel):
    """DINOv2 models loaded via HuggingFace ``transformers``.

    Supported *arch* values: ``vit_small``, ``vit_base``, ``vit_large``,
    ``vit_giant``.  Feature extraction uses the CLS token from the last
    hidden state.
    """

    def __init__(self, arch: str = "vit_base", **kwargs: Any) -> None:
        if arch not in _DINOV2_HF_MODELS:
            raise ValueError(
                f"Unknown DINOv2 arch '{arch}'. "
                f"Choose from {list(_DINOV2_HF_MODELS)}"
            )
        super().__init__(arch=arch, **kwargs)
        self._hf_name, self._dim = _DINOV2_HF_MODELS[arch]

    # -- BaseSSLModel interface ---------------------------------------------

    def load_checkpoint(self) -> None:
        """Load DINOv2 from HuggingFace Hub."""
        from transformers import AutoModel

        logger.info("Loading DINOv2 model '%s' from HuggingFace Hub ...", self._hf_name)
        self._model = AutoModel.from_pretrained(self._hf_name)
        self._model.eval()
        logger.info("DINOv2 '%s' loaded (dim=%d).", self._hf_name, self._dim)

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features from the last hidden state.

        Parameters
        ----------
        images:
            ``(B, 3, H, W)`` tensor, typically 224x224 for base/small or
            518x518 for the ``with_registers`` variants.

        Returns
        -------
        torch.Tensor
            ``(B, embedding_dim)`` L2-normalised features.
        """
        self._ensure_loaded()
        images = images.to(self.device)
        outputs = self._model(pixel_values=images)
        # CLS token is the first token of the last hidden state
        cls_token = outputs.last_hidden_state[:, 0]
        return self._normalise(cls_token)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"dinov2-{self.arch}"
