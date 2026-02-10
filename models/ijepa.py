"""I-JEPA (Image Joint-Embedding Predictive Architecture) model loader."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import timm
import torch

from models.base import BaseSSLModel
from models.utils import strip_module_prefix

logger = logging.getLogger(__name__)

_IJEPA_ARCHS: Dict[str, Tuple[str, int, int]] = {
    # arch_key -> (timm_model_name, embedding_dim, patch_size)
    "vit_base": ("vit_base_patch16_224", 768, 16),
    "vit_large": ("vit_large_patch16_224", 1024, 16),
    "vit_huge": ("vit_huge_patch14_224", 1280, 14),
}


class IJEPAModel(BaseSSLModel):
    """I-JEPA models (Meta AI) loaded via ``timm`` with a local checkpoint.

    Supported *arch* values: ``vit_base`` (ViT-B/16), ``vit_large``
    (ViT-L/16), ``vit_huge`` (ViT-H/14).

    Feature extraction supports two modes controlled by *concat_n_layers*:
    * ``concat_n_layers=0`` -- use only the CLS token of the last layer.
    * ``concat_n_layers=N`` (default 4) -- concatenate CLS tokens from the
      last *N* transformer layers for richer representations.  This is the
      recommended setting and matches the protocol used in the I-JEPA paper.

    A local ``checkpoint_path`` **must** be provided.  Checkpoint files can
    be downloaded from the facebookresearch/ijepa GitHub releases page.
    """

    def __init__(
        self,
        arch: str = "vit_huge",
        checkpoint_path: Optional[str] = None,
        concat_n_layers: int = 4,
        **kwargs: Any,
    ) -> None:
        if arch not in _IJEPA_ARCHS:
            raise ValueError(
                f"Unknown I-JEPA arch '{arch}'. "
                f"Choose from {list(_IJEPA_ARCHS)}"
            )
        if checkpoint_path is None:
            raise ValueError(
                "I-JEPA requires a local checkpoint path.  Download one from "
                "https://github.com/facebookresearch/ijepa/releases"
            )
        super().__init__(arch=arch, checkpoint_path=checkpoint_path, **kwargs)
        self._timm_name, self._base_dim, self._patch_size = _IJEPA_ARCHS[arch]
        self.concat_n_layers = max(0, concat_n_layers)

    # -- BaseSSLModel interface ---------------------------------------------

    def load_checkpoint(self) -> None:
        """Instantiate a timm ViT and load I-JEPA weights from disk."""
        logger.info(
            "Building I-JEPA encoder '%s' (timm: %s) ...",
            self.arch,
            self._timm_name,
        )
        self._model = timm.create_model(
            self._timm_name,
            pretrained=False,
            num_classes=0,  # remove classification head
        )

        logger.info("Loading I-JEPA checkpoint from '%s' ...", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # I-JEPA checkpoints typically nest encoder weights under a key such
        # as "target_encoder" or "encoder".  We try the common conventions.
        state_dict = checkpoint
        for key in ("target_encoder", "encoder", "model", "state_dict"):
            if isinstance(state_dict, dict) and key in state_dict:
                state_dict = state_dict[key]
                break

        # Strip module prefix that may come from DDP wrapping.
        state_dict = strip_module_prefix(state_dict)

        msg = self._model.load_state_dict(state_dict, strict=False)
        logger.info("I-JEPA checkpoint loaded.  load_state_dict message: %s", msg)
        self._model.eval()

    @torch.inference_mode()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from the I-JEPA encoder.

        When ``concat_n_layers > 0`` the CLS tokens of the last *N*
        transformer layers are concatenated, producing a feature vector of
        dimension ``N * base_embedding_dim``.  Otherwise only the CLS token
        of the last layer is returned.

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
            features = self._extract_last_cls(images)

        return self._normalise(features)

    def _extract_last_cls(self, images: torch.Tensor) -> torch.Tensor:
        """CLS token from the final transformer layer."""
        output = self._model.forward_features(images)
        # timm ViTs return (B, N, D); CLS is token 0.
        if output.dim() == 3:
            return output[:, 0]
        return output  # already pooled

    def _extract_multi_layer(self, images: torch.Tensor) -> torch.Tensor:
        """Concatenate CLS tokens from the last *concat_n_layers* layers.

        We temporarily register forward hooks on the transformer blocks to
        capture their outputs, then remove the hooks.
        """
        feats: list[torch.Tensor] = []

        blocks = self._model.blocks  # timm ViT transformer blocks
        n_layers = self.concat_n_layers
        target_blocks = blocks[-n_layers:]

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

        return torch.cat(feats, dim=-1)  # (B, n_layers * D)

    @property
    def embedding_dim(self) -> int:
        if self.concat_n_layers > 0:
            return self._base_dim * self.concat_n_layers
        return self._base_dim

    @property
    def name(self) -> str:
        suffix = f"-cat{self.concat_n_layers}" if self.concat_n_layers > 0 else ""
        return f"ijepa-{self.arch}{suffix}"
