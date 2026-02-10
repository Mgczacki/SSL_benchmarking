"""Abstract base class for SSL model wrappers."""

from __future__ import annotations

import abc
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseSSLModel(abc.ABC):
    """Abstract base class that every SSL baseline must implement.

    All models follow the same contract:
    1.  ``load_checkpoint()`` loads or downloads weights.
    2.  ``extract_features(images)`` returns L2-normalised feature vectors
        of shape ``(batch_size, embedding_dim)``.
    3.  ``embedding_dim`` reports the dimensionality of the feature space.
    4.  ``name`` returns a human-readable identifier.
    """

    def __init__(
        self,
        arch: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs: Any,
    ) -> None:
        self.arch = arch
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self._model: Optional[nn.Module] = None

    # -- public API ---------------------------------------------------------

    @abc.abstractmethod
    def load_checkpoint(self) -> None:
        """Load or download the pretrained checkpoint and set ``self._model``."""
        ...

    @abc.abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised features of shape ``(B, D)``.

        Parameters
        ----------
        images:
            Batch of preprocessed images, shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature matrix of shape ``(B, embedding_dim)`` with unit-norm rows.
        """
        ...

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the feature vectors returned by *extract_features*."""
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable name such as ``'dinov2-vit_base'``."""
        ...

    # -- helpers ------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Call ``load_checkpoint`` if not yet done and move model to device."""
        if self._model is None:
            self.load_checkpoint()
        assert self._model is not None
        self._model = self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _normalise(x: torch.Tensor) -> torch.Tensor:
        """L2-normalise along the feature dimension."""
        return F.normalize(x, p=2, dim=-1)

    def eval(self) -> "BaseSSLModel":
        """Set the underlying model to evaluation mode (compatibility shim)."""
        if self._model is not None:
            self._model.eval()
        return self

    def to(self, device: Any) -> "BaseSSLModel":
        """Move the underlying model to *device* (compatibility shim)."""
        self.device = torch.device(device) if isinstance(device, str) else device
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(arch={self.arch!r}, dim={self.embedding_dim})"
