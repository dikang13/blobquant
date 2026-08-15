"""Model construction and sliding-window inference."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Architecture of the shipped checkpoint. Overridable via --config.
DEFAULT_MODEL_CONFIG = {
    "name": "UNet3D",
    "in_channels": 1,
    "out_channels": 2,
    "layer_order": "gcr",
    "f_maps": 128,
    "num_groups": 16,
    "final_sigmoid": False,
    "is_segmentation": True,
}

DEFAULT_PATCH_SHAPE = (64, 120, 284)
DEFAULT_OVERLAP = (8, 16, 16)


def load_model_config(config_path: Path | None) -> tuple[dict, str | None]:
    """Read the ``model:`` block (and optional ``model_path``) from a YAML config."""
    if config_path is None:
        return dict(DEFAULT_MODEL_CONFIG), None
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}
    model_config = dict(DEFAULT_MODEL_CONFIG)
    model_config.update(config.get("model", {}))
    return model_config, config.get("model_path")


def build_model(model_config: dict, checkpoint: Path, device):
    """Instantiate the network and load weights onto *device*.

    The model is deliberately built with ``is_segmentation=False`` so ``forward``
    always returns raw logits; the final softmax/sigmoid is applied explicitly in
    :func:`predict_volume`. This makes the output independent of the upstream
    ``model.testing`` flag, which is silently lost when the model is wrapped in
    ``nn.DataParallel``.
    """
    from pytorch3dunet.unet3d.model import get_model
    from pytorch3dunet.unet3d.utils import load_checkpoint

    final_sigmoid = bool(model_config.get("final_sigmoid", False))
    build_config = dict(model_config)
    build_config["is_segmentation"] = False

    model = get_model({"model": build_config})
    logger.info("Loading checkpoint %s", checkpoint)
    load_checkpoint(str(checkpoint), model)
    model = model.to(device)
    model.eval()
    return model, final_sigmoid


def normalize(volume: np.ndarray) -> np.ndarray:
    """Min-max scale the whole volume into [-1, 1].

    Matches ``pytorch3dunet.augment.transforms.Normalize`` driven by global
    volume statistics, which is how the checkpoint was trained and validated.
    """
    array = volume.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        logger.warning("Volume is constant (min == max == %s); returning zeros", min_value)
        return np.zeros_like(array)
    scaled = (array - min_value) / (max_value - min_value)
    return np.clip(2.0 * scaled - 1.0, -1.0, 1.0)


def _window_starts(extent: int, patch: int, overlap: int) -> list[int]:
    """Start offsets tiling *extent* with the given patch size and overlap."""
    if patch >= extent:
        return [0]
    stride = max(1, patch - overlap)
    starts = list(range(0, extent - patch + 1, stride))
    if starts[-1] != extent - patch:
        starts.append(extent - patch)  # keep the final window flush with the edge
    return starts


def predict_volume(
    volume: np.ndarray,
    model,
    device,
    final_sigmoid: bool,
    patch_shape=DEFAULT_PATCH_SHAPE,
    overlap=DEFAULT_OVERLAP,
) -> np.ndarray:
    """Run the network over *volume*, returning per-class probabilities (C, Z, Y, X).

    Patches larger than the volume are clamped, so a volume that fits entirely in
    one patch is a single forward pass. Overlapping windows are averaged.
    """
    import torch

    normalized = normalize(volume)
    shape = normalized.shape
    patch = tuple(min(int(p), int(s)) for p, s in zip(patch_shape, shape))
    over = tuple(min(int(o), int(p) - 1) if p > 1 else 0 for o, p in zip(overlap, patch))

    starts = [_window_starts(s, p, o) for s, p, o in zip(shape, patch, over)]
    n_windows = int(np.prod([len(s) for s in starts]))
    logger.info(
        "Volume %s -> patch %s, overlap %s, %d window(s)", shape, patch, over, n_windows
    )

    # Voxels near a patch border are predicted with little surrounding context, so
    # discard a halo from every interior edge and keep only the well-supported core.
    halo = tuple(o // 2 for o in over)

    accumulator = None
    counts = np.zeros(shape, dtype=np.float32)

    with torch.no_grad():
        for zi in starts[0]:
            for yi in starts[1]:
                for xi in starts[2]:
                    origin = (zi, yi, xi)
                    window = tuple(
                        slice(o, o + p) for o, p in zip(origin, patch)
                    )
                    chunk = normalized[window]
                    batch = torch.from_numpy(chunk).to(device)[None, None]  # (1, 1, Z, Y, X)
                    logits = model(batch)
                    probs = (
                        torch.sigmoid(logits) if final_sigmoid else torch.softmax(logits, dim=1)
                    )
                    probs = probs[0].cpu().numpy().astype(np.float32)

                    if accumulator is None:
                        accumulator = np.zeros((probs.shape[0],) + shape, dtype=np.float32)

                    # Trim the halo only where the patch does not touch the volume edge.
                    keep_local, keep_global = [], []
                    for axis, (o, p, h) in enumerate(zip(origin, patch, halo)):
                        lo = h if o > 0 else 0
                        hi = p - h if o + p < shape[axis] else p
                        keep_local.append(slice(lo, hi))
                        keep_global.append(slice(o + lo, o + hi))
                    keep_local, keep_global = tuple(keep_local), tuple(keep_global)

                    accumulator[(slice(None),) + keep_global] += probs[(slice(None),) + keep_local]
                    counts[keep_global] += 1.0

    if accumulator is None:  # pragma: no cover - shape guards make this unreachable
        raise RuntimeError("No patches were produced for this volume")

    uncovered = int((counts == 0).sum())
    if uncovered:  # halo trimming must never leave holes; loudly reject a bad geometry
        raise RuntimeError(
            f"{uncovered} voxel(s) received no prediction with patch={patch}, "
            f"overlap={over}. Increase --overlap or --patch."
        )

    return accumulator / counts[None]


def foreground_probability(probability: np.ndarray) -> np.ndarray:
    """Extract the foreground channel from a (C, Z, Y, X) probability map."""
    if probability.shape[0] == 1:
        return probability[0]
    return probability[1]
