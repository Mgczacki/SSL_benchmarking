"""LeJEPA (Lightweight Joint-Embedding Predictive Architecture) model loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import timm
import torch

from models.base import BaseSSLModel
from models.utils import strip_module_prefix

logger = logging.getLogger(__name__)

_LEJEPA_ARCHS: Dict[str, Tuple[str, int, int]] = {
    "vit_huge": ("vit_huge_patch14_224", 1280, 14),
}


class LeJEPAModel(BaseSSLModel):
    """LeJEPA models loaded via ``timm`` + local checkpoint.

    Currently supports ``vit_huge`` (ViT-H/14).  The loading pattern is
    identical to I-JEPA: a ``timm`` ViT backbone is instantiated and then
    the encoder weights are loaded from a local file.

    A local ``checkpoint_path`` **must** be provided.
    """

    def __init__(
        self,
        arch: str = "vit_huge",
        checkpoint_path: Optional[str] = None,
        concat_n_layers: int = 0,
        **kwargs: Any,
    ) -> None:
        if arch not in _LEJEPA_ARCHS:
            raise ValueError(
                f"Unknown LeJEPA arch '{arch}'. "
                f"Choose from {list(_LEJEPA_ARCHS)}"
            )
        if checkpoint_path is None:
            raise ValueError("LeJEPA requires a local checkpoint path.")
        super().__init__(arch=arch, checkpoint_path=checkpoint_path, **kwargs)
        self._timm_name, self._base_dim, self._patch_size = _LEJEPA_ARCHS[arch]
        self.concat_n_layers = max(0, concat_n_layers)

    # -- BaseSSLModel interface ---------------------------------------------

    def load_checkpoint(self) -> None:
        """Instantiate a timm ViT and load LeJEPA encoder weights."""
        logger.info(
            "Building LeJEPA encoder '%s' (timm: %s) ...",
            self.arch,
            self._timm_name,
        )
        self._model = timm.create_model(
            self._timm_name,
            pretrained=False,
            num_classes=0,
        )

        logger.info("Loading LeJEPA checkpoint from '%s' ...", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        state_dict = checkpoint
        for key in ("target_encoder", "encoder", "model", "state_dict"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
                break

        state_dict = strip_module_prefix(state_dict)

        msg = self._model.load_state_dict(state_dict, strict=False)
        logger.info("LeJEPA checkpoint loaded.  load_state_dict message: %s", msg)
        self._model.eval()

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the LeJEPA encoder.

        Supports optional multi-layer CLS concatenation (same as I-JEPA)
        when ``concat_n_layers > 0``.

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

        if self.concat_n_layers > 0:
            features = self._extract_multi_layer(images)
        else:
            output = self._model.forward_features(images)
            features = output[:, 0] if output.dim() == 3 else output

        return self._normalise(features)

    def _extract_multi_layer(self, images: torch.Tensor) -> torch.Tensor:
        """Concatenate CLS tokens from the last *concat_n_layers* layers."""
        feats: list[torch.Tensor] = []
        blocks = self._model.blocks
        target_blocks = blocks[-self.concat_n_layers :]

        hooks = []
        for blk in target_blocks:
            hook = blk.register_forward_hook(
                lambda _module, _input, output, _feats=feats: _feats.append(
                    output[:, 0] if output.dim() == 3 else output
                )
            )
            hooks.append(hook)

        try:
            _ = self._model.forward_features(images)
        finally:
            for h in hooks:
                h.remove()

        return torch.cat(feats, dim=-1)

    @property
    def embedding_dim(self) -> int:
        if self.concat_n_layers > 0:
            return self._base_dim * self.concat_n_layers
        return self._base_dim

    @property
    def name(self) -> str:
        suffix = f"-cat{self.concat_n_layers}" if self.concat_n_layers > 0 else ""
        return f"lejepa-{self.arch}{suffix}"
