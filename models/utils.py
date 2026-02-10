"""Utility functions for model loading."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove ``'module.'`` prefix that ``DistributedDataParallel`` adds.

    Also strips common backbone prefixes like ``'backbone.'`` that some
    checkpoints use.
    """
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        for prefix in ("module.", "backbone."):
            if k.startswith(prefix):
                k = k[len(prefix) :]
        cleaned[k] = v
    return cleaned
