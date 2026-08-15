"""Thresholding and connected-component labelling of the probability map."""

from __future__ import annotations

import logging

import numpy as np
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5
DEFAULT_MIN_VOXELS = 6


def binarize(foreground: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Threshold a foreground probability map into a boolean mask.

    *foreground* is expected in [0, 1]; the threshold is a probability, not a logit.
    """
    return foreground > threshold


def label_blobs(
    mask: np.ndarray, min_voxels: int = DEFAULT_MIN_VOXELS
) -> tuple[np.ndarray, list[dict]]:
    """Label 26-connected blobs and drop those below *min_voxels*.

    Returns the relabelled volume (1..N, 0 = background) and per-blob properties.
    """
    labelled = label(mask, connectivity=3)
    regions = regionprops(labelled)
    kept = [r for r in regions if r.area >= min_voxels]
    logger.info(
        "Connected components: %d total, %d with >= %d voxels",
        len(regions),
        len(kept),
        min_voxels,
    )

    filtered = np.zeros_like(labelled, dtype=np.int32)
    blobs: list[dict] = []
    for new_label, region in enumerate(kept, start=1):
        filtered[labelled == region.label] = new_label
        blobs.append(
            {
                "label": new_label,
                "num_voxels": int(region.area),
                "centroid": tuple(float(c) for c in region.centroid),
            }
        )
    return filtered, blobs
