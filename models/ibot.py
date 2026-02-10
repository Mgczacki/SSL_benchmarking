"""iBOT (Image BERT Pre-Training with Online Tokenizer) model loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import timm
import torch

from models.base import BaseSSLModel
from models.utils import strip_module_prefix

logger = logging.getLogger(__name__)

_IBOT_ARCHS: Dict[str, Tuple[str, int]] = {
    "vit_small": ("vit_small_patch16_224", 384),
    "vit_base": ("vit_base_patch16_224", 768),
    "vit_large": ("vit_large_patch16_224", 1024),
}


class iBOTModel(BaseSSLModel):
    """iBOT models (ByteDance) loaded via ``timm`` + local checkpoint.

    Supported *arch* values: ``vit_small`` (ViT-S/16), ``vit_base``
    (ViT-B/16), ``vit_large`` (ViT-L/16).

    A local ``checkpoint_path`` **must** be provided.  Checkpoints can be
    obtained from https://github.com/bytedance/ibot.  Feature extraction
    uses the CLS token.
    """

    def __init__(
        self,
        arch: str = "vit_base",
        checkpoint_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if arch not in _IBOT_ARCHS:
            raise ValueError(
                f"Unknown iBOT arch '{arch}'. "
                f"Choose from {list(_IBOT_ARCHS)}"
            )
        if checkpoint_path is None:
            raise ValueError(
                "iBOT requires a local checkpoint path.  Download one from "
                "https://github.com/bytedance/ibot"
            )
        super().__init__(arch=arch, checkpoint_path=checkpoint_path, **kwargs)
        self._timm_name, self._dim = _IBOT_ARCHS[arch]

    # -- BaseSSLModel interface ---------------------------------------------

    def load_checkpoint(self) -> None:
        """Instantiate a timm ViT and load iBOT weights from disk."""
        logger.info(
            "Building iBOT encoder '%s' (timm: %s) ...",
            self.arch,
            self._timm_name,
        )
        self._model = timm.create_model(
            self._timm_name,
            pretrained=False,
            num_classes=0,
        )

        logger.info("Loading iBOT checkpoint from '%s' ...", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint
        for key in ("state_dict", "model", "teacher"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
                break

        state_dict = strip_module_prefix(state_dict)

        # iBOT checkpoints sometimes contain keys from the projection head
        # that are not part of the backbone -- we filter them out.
        backbone_keys = set(self._model.state_dict().keys())
        state_dict = {
            k: v for k, v in state_dict.items() if k in backbone_keys
        }

        msg = self._model.load_state_dict(state_dict, strict=False)
        logger.info("iBOT checkpoint loaded.  load_state_dict message: %s", msg)
        self._model.eval()

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
        return f"ibot-{self.arch}"
