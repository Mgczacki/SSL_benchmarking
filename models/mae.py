"""MAE (Masked Autoencoder) model loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import timm
import torch

from models.base import BaseSSLModel
from models.utils import strip_module_prefix

logger = logging.getLogger(__name__)

_MAE_ARCHS: Dict[str, Tuple[str, int]] = {
    "vit_base": ("vit_base_patch16_224", 768),
    "vit_large": ("vit_large_patch16_224", 1024),
    "vit_huge": ("vit_huge_patch14_224", 1280),
}


class MAEModel(BaseSSLModel):
    """Masked Autoencoder (MAE) models loaded via ``timm``.

    Supported *arch* values: ``vit_base`` (ViT-B/16), ``vit_large``
    (ViT-L/16), ``vit_huge`` (ViT-H/14).

    If *checkpoint_path* is provided the weights are loaded from a local
    file; otherwise ``timm`` downloads the MAE-finetuned ImageNet weights
    (``vit_*_patch*_224.mae``).

    Feature extraction uses the CLS token of the **encoder** output (the
    decoder is not used).
    """

    # Map arch keys to timm pretrained model tags that carry MAE weights.
    _TIMM_PRETRAINED_TAGS: Dict[str, str] = {
        "vit_base": "vit_base_patch16_224.mae",
        "vit_large": "vit_large_patch16_224.mae",
        "vit_huge": "vit_huge_patch14_224.mae",
    }

    def __init__(self, arch: str = "vit_base", **kwargs: Any) -> None:
        if arch not in _MAE_ARCHS:
            raise ValueError(
                f"Unknown MAE arch '{arch}'. Choose from {list(_MAE_ARCHS)}"
            )
        super().__init__(arch=arch, **kwargs)
        self._timm_name, self._dim = _MAE_ARCHS[arch]

    # -- BaseSSLModel interface ---------------------------------------------

    def load_checkpoint(self) -> None:
        """Load MAE encoder weights."""
        if self.checkpoint_path is not None:
            logger.info("Building MAE encoder '%s' from local checkpoint ...", self.arch)
            self._model = timm.create_model(
                self._timm_name,
                pretrained=False,
                num_classes=0,
            )
            checkpoint = torch.load(
                self.checkpoint_path, map_location="cpu", weights_only=False
            )

            state_dict = checkpoint
            for key in ("model", "state_dict", "encoder"):
                if isinstance(state_dict, dict) and key in state_dict:
                    state_dict = state_dict[key]
                    break

            state_dict = strip_module_prefix(state_dict)
            # Filter out decoder keys so we only load the encoder.
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("decoder")
            }

            msg = self._model.load_state_dict(state_dict, strict=False)
            logger.info("MAE checkpoint loaded.  load_state_dict message: %s", msg)
        else:
            pretrained_tag = self._TIMM_PRETRAINED_TAGS.get(self.arch)
            logger.info(
                "Loading MAE model '%s' (timm tag: %s) ...",
                self.arch,
                pretrained_tag,
            )
            self._model = timm.create_model(
                pretrained_tag if pretrained_tag else self._timm_name,
                pretrained=True,
                num_classes=0,
            )
            logger.info("MAE '%s' loaded from timm (dim=%d).", self.arch, self._dim)

        self._model.eval()

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS token features from the MAE encoder.

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
        output = self._model.forward_features(images)
        if output.dim() == 3:
            cls_token = output[:, 0]
        else:
            cls_token = output
        return self._normalise(cls_token)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"mae-{self.arch}"
