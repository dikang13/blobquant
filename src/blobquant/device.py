"""Device selection.

``CUDA_VISIBLE_DEVICES`` must be set before torch initialises CUDA, so
:func:`configure_visible_devices` is called from the CLI before any torch import.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_GPU = "3"


def configure_visible_devices(gpu: str | None = DEFAULT_GPU) -> None:
    """Pin the process to a single physical GPU.

    Exposing exactly one device keeps inference deterministic: ``predict.py``
    upstream wraps the model in ``nn.DataParallel`` whenever more than one GPU is
    visible, and the resulting attribute shadowing silently suppresses the final
    softmax (see README). A pre-existing ``CUDA_VISIBLE_DEVICES`` is respected.
    """
    if gpu is None:
        return
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current is not None:
        logger.info("Honouring existing CUDA_VISIBLE_DEVICES=%s", current)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logger.info("Set CUDA_VISIBLE_DEVICES=%s", gpu)


def select_device(prefer: str = "auto"):
    """Return the torch device to run on, falling back to CPU when CUDA is absent.

    ``prefer`` is one of ``auto``, ``cuda`` or ``cpu``. Under ``auto`` a missing or
    unusable GPU is not an error -- the volume is simply processed on CPU.
    """
    import torch

    if prefer == "cpu":
        logger.info("Using device: cpu (forced)")
        return torch.device("cpu")

    if torch.cuda.is_available():
        # Only one device is visible, so it is always index 0 inside this process.
        device = torch.device("cuda:0")
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
        logger.info(
            "Using device: %s (physical GPU %s, %s)",
            device,
            physical,
            torch.cuda.get_device_name(0),
        )
        return device

    if prefer == "cuda":
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is False. "
            "Check that the GPU selected by CUDA_VISIBLE_DEVICES exists."
        )

    logger.info("Using device: cpu (CUDA not available)")
    return torch.device("cpu")
