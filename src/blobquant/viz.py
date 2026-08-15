"""Quick-look visualisation of a segmentation result."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def mip(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """Maximum intensity projection along *axis* (0=Z/top, 1=Y/front, 2=X/side)."""
    return volume.max(axis=axis)


def to_rgb(plane: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Scale a 2D plane to an (H, W, 3) float RGB grayscale image in [0, 1]."""
    if vmin is None or vmax is None:
        vmin, vmax = np.percentile(plane, (1, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((plane.astype(np.float64) - vmin) / (vmax - vmin), 0, 1)
    return np.repeat(scaled[..., None], 3, axis=2)


def highlight(rgb: np.ndarray, flags: np.ndarray, color=(1.0, 0.0, 1.0), alpha: float = 1.0):
    """Paint *color* onto an RGB image wherever the 2D boolean *flags* are set."""
    out = rgb.copy()
    if flags.any():
        out[flags] = (1 - alpha) * out[flags] + alpha * np.asarray(color, dtype=float)
    return out


def _pick_slices(mask: np.ndarray, n_slices: int) -> list[int]:
    """Choose z-slices to show, biased toward those containing signal."""
    depth = mask.shape[0]
    occupancy = mask.reshape(depth, -1).sum(axis=1)
    populated = np.flatnonzero(occupancy)
    if populated.size == 0:
        return list(np.linspace(0, depth - 1, num=min(n_slices, depth), dtype=int))
    lo, hi = int(populated[0]), int(populated[-1])
    return list(np.linspace(lo, hi, num=min(n_slices, hi - lo + 1), dtype=int))


def save_overlay(
    path: Path,
    volume: np.ndarray,
    mask: np.ndarray,
    n_slices: int = 6,
    title: str | None = None,
) -> Path:
    """Write a PNG montage of raw slices with the mask outlined in red."""
    indices = _pick_slices(mask, n_slices)
    columns = min(3, len(indices))
    rows = int(np.ceil(len(indices) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 2.6 * rows), squeeze=False)
    # Shared contrast across slices so brightness differences are meaningful.
    vmin, vmax = np.percentile(volume, (1, 99.5))

    for ax, z in zip(axes.ravel(), indices):
        ax.imshow(volume[z], cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        if mask[z].any():
            ax.contour(mask[z].astype(float), levels=[0.5], colors="red", linewidths=0.6)
        ax.set_title(f"z={z}  ({int(mask[z].sum())} vox)", fontsize=8)
        ax.set_axis_off()
    for ax in axes.ravel()[len(indices):]:
        ax.set_axis_off()

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    logger.info("Wrote overlay %s", path)
    return path
