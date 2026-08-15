#!/usr/bin/env python
"""Segment the green channel using the mask derived from the red channel.

The red (marker) channel is the one the network is trained on, so blobs are
detected there once and the *same* mask is transferred to the green (activity)
channel. Both channels are then segmented identically, which is what makes a
per-blob green/red ratio meaningful.

Typical use::

    uv run blobquant data/hdf5/worm_0_ch_red.h5 -o output/

    uv run python examples/apply_red_mask_to_green.py \\
        --prediction output/worm_0_ch_red_prediction.h5 \\
        --red data/hdf5/worm_0_ch_red.h5 \\
        --green data/hdf5/worm_0_ch_green.h5 \\
        -o output/

Writes ``worm_0_ch_green_prediction.h5`` / ``.tif`` (the transferred mask), an
overlay PNG, and a per-blob intensity table as CSV.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import h5py
import numpy as np

from blobquant import io, viz

logger = logging.getLogger("apply_red_mask_to_green")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prediction", required=True,
        help="Red-channel *_prediction.h5 produced by `blobquant` (source of the mask).",
    )
    parser.add_argument("--red", required=True, help="Red channel volume (.h5 or .tif).")
    parser.add_argument("--green", required=True, help="Green channel volume (.h5 or .tif).")
    parser.add_argument("--red-dataset", default=None, help="HDF5 dataset inside --red.")
    parser.add_argument("--green-dataset", default=None, help="HDF5 dataset inside --green.")
    parser.add_argument("-o", "--outdir", default=".", help="Output directory.")
    parser.add_argument(
        "--bkg-percentile", type=float, default=20.0,
        help="Per-channel background estimated as this intensity percentile (default: 20).",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip the overlay PNG.")
    parser.add_argument(
        "--roi", type=int, default=None,
        help="Inspect this ROI number: report its voxels and write an isolated mask/CSV. "
             "Default: the largest ROI, as a self-demonstration.",
    )
    parser.add_argument(
        "--at", type=int, nargs=3, metavar=("Z", "Y", "X"), default=None,
        help="Look up which ROI contains this voxel. Default: the inspected ROI's centroid.",
    )
    return parser.parse_args(argv)


def roi_voxel_coords(labels: np.ndarray, roi: int) -> np.ndarray:
    """All (z, y, x) coordinates belonging to *roi*, shape (N, 3)."""
    return np.argwhere(labels == roi)


def roi_at_coordinate(labels: np.ndarray, z: int, y: int, x: int) -> int | None:
    """ROI number containing voxel (z, y, x), or None if it is background.

    Labels are a plain integer volume, so the lookup is a single index -- no search.
    """
    if not all(0 <= c < s for c, s in zip((z, y, x), labels.shape)):
        raise IndexError(f"({z}, {y}, {x}) is outside the volume {labels.shape}")
    roi = int(labels[z, y, x])
    return roi if roi > 0 else None


def nearest_roi(labels: np.ndarray, z: int, y: int, x: int) -> tuple[int, float]:
    """Nearest non-background ROI to a coordinate, and the distance to it."""
    filled = np.argwhere(labels > 0)
    distances = np.linalg.norm(filled - np.array([z, y, x]), axis=1)
    closest = filled[int(np.argmin(distances))]
    return int(labels[tuple(closest)]), float(distances.min())


def per_blob_sums(labels: np.ndarray, volume: np.ndarray, n_labels: int) -> np.ndarray:
    """Sum *volume* within each label 1..n_labels (index 0 is background)."""
    return np.bincount(
        labels.ravel(), weights=volume.ravel().astype(np.float64), minlength=n_labels + 1
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load the mask and instance labels computed from the red channel.
    with h5py.File(args.prediction, "r") as f:
        mask = f["mask"][()].astype(bool)
        labels = f["labels"][()].astype(np.int64)
        source_attrs = dict(f.attrs)
    n_labels = int(labels.max())
    logger.info("Loaded mask from %s: %d blobs, %d voxels", args.prediction, n_labels, mask.sum())

    # 2. Load both channels.
    red, red_src = io.load_volume(Path(args.red), args.red_dataset)
    green, green_src = io.load_volume(Path(args.green), args.green_dataset)
    logger.info("red   %s from %s", red.shape, red_src)
    logger.info("green %s from %s", green.shape, green_src)

    # Transferring a mask between channels is only valid if they are co-registered.
    if red.shape != mask.shape or green.shape != mask.shape:
        raise SystemExit(
            f"Shape mismatch: mask {mask.shape}, red {red.shape}, green {green.shape}. "
            "The mask can only be transferred between co-registered volumes of equal shape."
        )

    # 3. Write the green channel's segmentation -- the identical mask, same layout
    #    as blobquant's own output so downstream code does not care which channel it is.
    green_h5 = outdir / "worm_0_ch_green_prediction.h5"
    green_tif = outdir / "worm_0_ch_green_prediction.tif"
    with h5py.File(green_h5, "w") as f:
        f.create_dataset("mask", data=mask.astype(np.uint8), compression="gzip")
        f.create_dataset("labels", data=labels.astype(np.int32), compression="gzip")
        f.attrs["mask_source"] = str(args.prediction)
        f.attrs["mask_transferred_from"] = source_attrs.get("source", "red channel")
        f.attrs["source"] = str(args.green)
        f.attrs["num_blobs"] = n_labels
    io.save_mask_tif(green_tif, mask)
    logger.info("Wrote %s and %s", green_h5, green_tif)

    # 4. Background-subtract each channel, then quantify per blob.
    red_bkg = float(np.percentile(red, args.bkg_percentile))
    green_bkg = float(np.percentile(green, args.bkg_percentile))
    logger.info("Background (p%.0f): red=%.1f green=%.1f", args.bkg_percentile, red_bkg, green_bkg)
    red_corr = np.clip(red.astype(np.float64) - red_bkg, 0, None)
    green_corr = np.clip(green.astype(np.float64) - green_bkg, 0, None)

    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
    red_sums = per_blob_sums(labels, red_corr, n_labels)
    green_sums = per_blob_sums(labels, green_corr, n_labels)

    csv_path = outdir / "worm_0_ch_blob_intensities.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["label", "num_voxels", "red_sum", "green_sum", "red_mean", "green_mean", "ratio"]
        )
        for lab in range(1, n_labels + 1):
            n = int(sizes[lab])
            r, g = float(red_sums[lab]), float(green_sums[lab])
            # Guard the ratio: a blob with no marker signal has no meaningful ratio.
            ratio = g / r if r > 0 else float("nan")
            writer.writerow([lab, n, f"{r:.1f}", f"{g:.1f}", f"{r/n:.2f}", f"{g/n:.2f}", f"{ratio:.4f}"])
    logger.info("Wrote %s (%d blobs)", csv_path, n_labels)

    if not args.no_viz:
        overlay = outdir / "worm_0_ch_green_prediction_overlay.png"
        viz.save_overlay(
            overlay, green, mask,
            title=f"green channel with red-derived mask  |  {n_labels} blobs",
        )

    ratios = np.divide(
        green_sums[1:], red_sums[1:],
        out=np.full(n_labels, np.nan), where=red_sums[1:] > 0,
    )
    print(f"\n{n_labels} blobs quantified in both channels")
    print(f"  median green/red ratio: {np.nanmedian(ratios):.4f}")
    print(f"  ratio range:            {np.nanmin(ratios):.4f} - {np.nanmax(ratios):.4f}")
    print(f"  table: {csv_path}")

    # ------------------------------------------------------------------
    # A) Given an ROI number, isolate every pixel belonging to it.
    # ------------------------------------------------------------------
    roi = args.roi if args.roi is not None else int(np.argmax(sizes[1:])) + 1
    if not 1 <= roi <= n_labels:
        raise SystemExit(f"ROI {roi} is out of range; this volume has ROIs 1..{n_labels}")

    coords = roi_voxel_coords(labels, roi)
    roi_only = labels == roi  # boolean mask isolating just this ROI
    centroid = coords.mean(axis=0)
    zmin, ymin, xmin = coords.min(axis=0)
    zmax, ymax, xmax = coords.max(axis=0)

    print(f"\n--- ROI {roi}: isolating its pixels ---")
    print(f"  voxels:   {len(coords)}")
    print(f"  centroid: z={centroid[0]:.1f} y={centroid[1]:.1f} x={centroid[2]:.1f}")
    print(f"  bbox:     z {zmin}-{zmax}, y {ymin}-{ymax}, x {xmin}-{xmax}")
    print(f"  first 5 voxel coords (z, y, x): {[tuple(int(v) for v in c) for c in coords[:5]]}")
    print(f"  red intensity   in ROI: sum={red_corr[roi_only].sum():.1f} mean={red_corr[roi_only].mean():.1f}")
    print(f"  green intensity in ROI: sum={green_corr[roi_only].sum():.1f} mean={green_corr[roi_only].mean():.1f}")

    # Everything outside this ROI is zeroed -- "filter out all other pixels".
    roi_tif = outdir / f"roi_{roi}_mask.tif"
    io.save_mask_tif(roi_tif, roi_only)
    roi_csv = outdir / f"roi_{roi}_voxels.csv"
    with open(roi_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["z", "y", "x", "red", "green"])
        for z, y, x in coords:
            writer.writerow([z, y, x, f"{red_corr[z, y, x]:.1f}", f"{green_corr[z, y, x]:.1f}"])
    print(f"  wrote {roi_tif}")
    print(f"  wrote {roi_csv}")

    # ------------------------------------------------------------------
    # B) Given a pixel coordinate, identify which ROI it belongs to.
    # ------------------------------------------------------------------
    probe = tuple(args.at) if args.at else tuple(int(round(c)) for c in centroid)
    print(f"\n--- Pixel {probe}: identifying its ROI ---")
    found = roi_at_coordinate(labels, *probe)
    if found is None:
        near, distance = nearest_roi(labels, *probe)
        print(f"  background (no ROI); nearest is ROI {near} at {distance:.1f} voxels")
    else:
        print(f"  belongs to ROI {found} ({int(sizes[found])} voxels)")
        print(f"  red={red_corr[probe]:.1f}  green={green_corr[probe]:.1f}")

    # A background probe, to show the other branch.
    corner = (0, 0, 0)
    if roi_at_coordinate(labels, *corner) is None:
        near, distance = nearest_roi(labels, *corner)
        print(f"  pixel {corner}: background; nearest is ROI {near} at {distance:.1f} voxels")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
