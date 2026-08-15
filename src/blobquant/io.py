"""Reading input volumes and writing prediction artifacts.

Inputs may be TIFF or HDF5; outputs are always written as a matching pair of
``*_prediction.h5`` and ``*_prediction.tif``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import tifffile as tiff

logger = logging.getLogger(__name__)

TIFF_SUFFIXES = {".tif", ".tiff"}
HDF5_SUFFIXES = {".h5", ".hdf5", ".hdf", ".hd5"}
INPUT_SUFFIXES = TIFF_SUFFIXES | HDF5_SUFFIXES

# Dataset names tried first when an HDF5 file has no explicit --dataset.
PREFERRED_KEYS = ("raw", "image", "images", "data", "volume")


class AmbiguousDatasetError(ValueError):
    """An HDF5 file holds several 3D datasets and none is an obvious default."""

OUTPUT_SUFFIX = "_prediction"
# "_predictions" is the legacy suffix written by pytorch-3dunet's predict.py.
DERIVED_SUFFIXES = (OUTPUT_SUFFIX, "_predictions", "_prediction_overlay")


def is_supported_input(path: Path) -> bool:
    """True if *path* looks like a volume this tool can read."""
    if path.name.startswith("."):
        return False
    # Never re-ingest our own outputs, or a previous pipeline's, when scanning.
    if path.stem.endswith(DERIVED_SUFFIXES):
        return False
    return path.suffix.lower() in INPUT_SUFFIXES


def collect_inputs(paths, recursive: bool = True) -> list[tuple[Path, bool]]:
    """Expand files and directories into a sorted list of ``(path, discovered)``.

    ``discovered`` is True for volumes found by scanning a directory, which the
    caller treats more leniently than an explicitly named file: a time-series
    HDF5 that needs ``--dataset`` is worth skipping during a sweep, but is an
    error when the user asked for it by name.
    """
    found: list[tuple[Path, bool]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            matches = sorted(p for p in candidates if p.is_file() and is_supported_input(p))
            if not matches:
                logger.warning("No supported volumes found in directory %s", path)
            found.extend((p, True) for p in matches)
        elif path.is_file():
            # An explicitly named file is honoured even if the filter would skip it.
            found.append((path, False))
        else:
            raise FileNotFoundError(f"Input path does not exist: {path}")
    return found


def _find_3d_datasets(handle) -> list[str]:
    """Collect paths of every dataset in an HDF5 file with 3 or more axes."""
    keys: list[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            keys.append(name)

    handle.visititems(visitor)
    return keys


def _squeeze_to_3d(array: np.ndarray, origin: str) -> np.ndarray:
    """Drop singleton axes so a (1, Z, Y, X) style volume becomes (Z, Y, X)."""
    if array.ndim > 3:
        squeezed = np.squeeze(array)
        if squeezed.ndim != 3:
            raise ValueError(
                f"{origin}: expected a 3D volume, got shape {array.shape} "
                f"which squeezes to {squeezed.ndim}D. Use --dataset to select one volume."
            )
        array = squeezed
    if array.ndim != 3:
        raise ValueError(f"{origin}: expected a 3D volume, got shape {array.shape}")
    return array


def load_volume(path: Path, dataset: str | None = None) -> tuple[np.ndarray, str]:
    """Load a 3D volume from TIFF or HDF5.

    Returns the array and a human-readable description of where it came from.
    For HDF5 without an explicit *dataset*, a well-known name such as ``raw`` is
    preferred, otherwise the sole 3D dataset is used; ambiguity is an error
    rather than a guess.
    """
    suffix = path.suffix.lower()

    if suffix in TIFF_SUFFIXES:
        array = tiff.imread(str(path))
        return _squeeze_to_3d(np.asarray(array), str(path)), "tiff"

    if suffix in HDF5_SUFFIXES:
        with h5py.File(path, "r") as handle:
            if dataset is not None:
                if dataset not in handle:
                    raise KeyError(f"{path}: no dataset named '{dataset}'")
                key = dataset
            else:
                available = _find_3d_datasets(handle)
                if not available:
                    raise ValueError(f"{path}: contains no 3D dataset to predict on")
                preferred = [k for k in available if k.split("/")[-1] in PREFERRED_KEYS]
                if preferred:
                    key = preferred[0]
                elif len(available) == 1:
                    key = available[0]
                else:
                    raise AmbiguousDatasetError(
                        f"{path}: found {len(available)} candidate 3D datasets "
                        f"({', '.join(available[:5])}...). Pass --dataset to choose one."
                    )
            array = handle[key][()]
        return _squeeze_to_3d(np.asarray(array), f"{path}:{key}"), f"hdf5:{key}"

    raise ValueError(
        f"{path}: unsupported input format '{suffix}'. "
        f"Expected one of {sorted(INPUT_SUFFIXES)}."
    )


def output_paths(input_path: Path, outdir: Path | None) -> tuple[Path, Path]:
    """Map an input volume to its ``*_prediction.h5`` / ``*_prediction.tif`` pair."""
    stem = input_path.stem + OUTPUT_SUFFIX
    directory = Path(outdir) if outdir is not None else input_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{stem}.h5", directory / f"{stem}.tif"


def save_prediction_h5(
    path: Path,
    probability: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    attrs: dict | None = None,
) -> None:
    """Write the probability map, binary mask and instance labels to one HDF5 file."""
    with h5py.File(path, "w") as handle:
        handle.create_dataset("probability", data=probability.astype(np.float32), compression="gzip")
        handle.create_dataset("mask", data=mask.astype(np.uint8), compression="gzip")
        handle.create_dataset("labels", data=labels.astype(np.int32), compression="gzip")
        for key, value in (attrs or {}).items():
            handle.attrs[key] = value


def save_mask_tif(path: Path, mask: np.ndarray) -> None:
    """Write the binary mask as an ImageJ-readable 8-bit TIFF stack (0 / 255)."""
    tiff.imwrite(str(path), (mask.astype(np.uint8) * 255), imagej=True)
