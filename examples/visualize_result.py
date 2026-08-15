#!/usr/bin/env python
"""Visualise a blobquant segmentation: projections, ROI statistics, ROI locator.

Reads an input volume and its ``*_prediction.h5`` and produces one figure with:

* maximum intensity projection (MIP) of the input volume
* MIP of the output binary mask
* MIP of the segmented foreground (input masked to the blobs)
* the distribution of ROI sizes in voxels

Given ``--roi N`` it additionally reports that ROI's 3D coordinates and paints its
voxels in a contrasting colour on three orthogonal projections, so the ROI can be
located by eye in the volume.

Usage::

    uv run python examples/visualize_result.py \\
        --input data/hdf5/worm_0_ch_red.h5 \\
        --prediction output/worm_0_ch_red_prediction.h5 \\
        --roi 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from blobquant import io, viz  # noqa: E402

ROI_COLOR = (1.0, 0.0, 1.0)  # magenta: absent from the grayscale data, so unambiguous
MASK_COLOR = (1.0, 0.25, 0.25)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, help="Input volume (.h5 or .tif).")
    parser.add_argument("--prediction", required=True, help="Matching *_prediction.h5.")
    parser.add_argument("--dataset", default=None, help="HDF5 dataset inside --input.")
    parser.add_argument(
        "--roi", type=int, default=None,
        help="Highlight this ROI number and report its 3D location.",
    )
    parser.add_argument(
        "-o", "--outdir", default="output", help="Where to write the figure (default: output)."
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open the figure interactively instead of only writing a PNG.",
    )
    return parser.parse_args(argv)


def describe_sizes(sizes: np.ndarray) -> str:
    """One-line summary of the ROI size distribution."""
    q = np.percentile(sizes, [25, 50, 75])
    return (
        f"n={len(sizes)}  min={sizes.min()}  q25={q[0]:.0f}  "
        f"median={q[1]:.0f}  q75={q[2]:.0f}  max={sizes.max()}  total={sizes.sum()}"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    volume, source = io.load_volume(Path(args.input), args.dataset)
    with h5py.File(args.prediction, "r") as f:
        mask = f["mask"][()].astype(bool)
        labels = f["labels"][()].astype(np.int32)

    if volume.shape != mask.shape:
        raise SystemExit(f"Shape mismatch: input {volume.shape} vs prediction {mask.shape}")

    n_roi = int(labels.max())
    # Index 0 is background, so drop it to get per-ROI voxel counts.
    sizes = np.bincount(labels.ravel(), minlength=n_roi + 1)[1:]

    print(f"Input:      {args.input} ({source}), shape {volume.shape}")
    print(f"Prediction: {args.prediction}")
    print(f"\nROIs segmented: {n_roi}")
    print(f"Foreground voxels: {int(mask.sum())} "
          f"({100 * mask.sum() / mask.size:.3f}% of the volume)")
    print(f"ROI size (voxels): {describe_sizes(sizes)}")
    # `mask` is the raw threshold; `labels` additionally drops sub-min-voxel blobs,
    # so the two totals differ by whatever the size filter removed.
    unassigned = int(mask.sum()) - int(sizes.sum())
    if unassigned:
        print(f"  note: {unassigned} masked voxel(s) belong to no ROI "
              f"(components below the --min-voxels filter used at prediction time)")

    foreground = np.where(mask, volume, 0)
    vmin, vmax = np.percentile(volume, (1, 99.5))

    has_roi = args.roi is not None
    if has_roi:
        if not 1 <= args.roi <= n_roi:
            raise SystemExit(f"ROI {args.roi} out of range; this volume has ROIs 1..{n_roi}")
        roi_mask = labels == args.roi
        coords = np.argwhere(roi_mask)
        centroid = coords.mean(axis=0)
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        print(f"\n--- ROI {args.roi} ---")
        print(f"  voxels:   {len(coords)}")
        print(f"  centroid: z={centroid[0]:.1f}  y={centroid[1]:.1f}  x={centroid[2]:.1f}")
        print(f"  bbox:     z {lo[0]}-{hi[0]}   y {lo[1]}-{hi[1]}   x {lo[2]}-{hi[2]}")
        print(f"  extent:   {hi[0]-lo[0]+1} x {hi[1]-lo[1]+1} x {hi[2]-lo[2]+1} voxels (z, y, x)")
        print(f"  intensity in ROI: mean={volume[roi_mask].mean():.1f} max={volume[roi_mask].max()}")

    rows = 2 if has_roi else 1
    # The Z projection is wide and short; matching the other panels to its aspect keeps
    # the row from being stretched to the height of a square plot.
    z_mip_shape = viz.mip(volume).shape
    panel_aspect = z_mip_shape[0] / z_mip_shape[1]
    fig_height = 4.2 * panel_aspect * rows + 1.6
    fig = plt.figure(figsize=(16, fig_height))
    # Reserve a fixed strip at the top for the suptitle so it never lands on a panel
    # title, whatever the volume's aspect ratio works out to be.
    grid = fig.add_gridspec(rows, 4, hspace=0.42, wspace=0.20, top=1 - 0.95 / fig_height)

    def show(ax, image, **kwargs):
        """Draw a projection anchored to the top of its cell, to avoid dead space."""
        ax.imshow(image, interpolation="nearest", **kwargs)
        ax.set_anchor("N")

    # --- Row 1: projections along Z, plus the size distribution. ---
    ax = fig.add_subplot(grid[0, 0])
    show(ax, viz.mip(volume), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Input volume\nmax intensity projection (Z)", fontsize=9)
    ax.set_axis_off()

    ax = fig.add_subplot(grid[0, 1])
    show(ax, viz.mip(mask.astype(np.uint8)), cmap="gray")
    ax.set_title(f"Output mask\nMIP (Z) - {n_roi} ROIs", fontsize=9)
    ax.set_axis_off()

    ax = fig.add_subplot(grid[0, 2])
    show(ax, viz.mip(foreground), cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title("Segmented foreground\nMIP (Z) of input x mask", fontsize=9)
    ax.set_axis_off()

    ax = fig.add_subplot(grid[0, 3])
    ax.hist(sizes, bins=min(40, max(5, n_roi // 4)), color="#4878a8", edgecolor="white")
    ax.axvline(np.median(sizes), color="crimson", ls="--", lw=1.2,
               label=f"median {np.median(sizes):.0f}")
    ax.set_xlabel("ROI size (voxels)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(f"ROI size distribution\nn={n_roi}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    ax.set_box_aspect(panel_aspect)
    ax.set_anchor("N")

    # --- Row 2: locate one ROI in 3D via three orthogonal projections. ---
    if has_roi:
        # (title, projection axis, horizontal data axis, vertical data axis)
        views = [
            ("Top (Z projection)", 0, 2, 1, "x", "y"),
            ("Front (Y projection)", 1, 2, 0, "x", "z"),
            ("Side (X projection)", 2, 1, 0, "y", "z"),
        ]
        for column, (label, axis, h_axis, v_axis, xlabel, ylabel) in enumerate(views):
            ax = fig.add_subplot(grid[1, column])
            base = viz.to_rgb(viz.mip(volume, axis=axis), vmin, vmax)
            # Draw the full mask faintly, then this ROI opaquely on top.
            base = viz.highlight(base, viz.mip(mask, axis=axis), MASK_COLOR, alpha=0.30)
            base = viz.highlight(base, viz.mip(roi_mask, axis=axis), ROI_COLOR, alpha=1.0)
            show(ax, base)
            # A few-voxel ROI is invisible at this scale, so ring it and add crosshairs.
            cx, cy = centroid[h_axis], centroid[v_axis]
            ax.add_patch(plt.Circle((cx, cy), 9, fill=False, color=ROI_COLOR, lw=1.4))
            ax.axhline(cy, color=ROI_COLOR, lw=0.5, alpha=0.45)
            ax.axvline(cx, color=ROI_COLOR, lw=0.5, alpha=0.45)
            ax.set_title(f"ROI {args.roi} - {label}", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.tick_params(labelsize=7)

        ax = fig.add_subplot(grid[1, 3])
        ax.axis("off")
        rank = int((sizes > len(coords)).sum()) + 1
        ax.text(
            0.0, 0.95,
            "\n".join([
                f"ROI {args.roi} of {n_roi}",
                "",
                f"voxels:    {len(coords)}",
                f"size rank: {rank} of {n_roi}",
                "",
                "centroid (z, y, x)",
                f"  {centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}",
                "",
                "bounding box",
                f"  z {lo[0]}-{hi[0]}",
                f"  y {lo[1]}-{hi[1]}",
                f"  x {lo[2]}-{hi[2]}",
                "",
                f"mean intensity: {volume[roi_mask].mean():.1f}",
                f"max intensity:  {volume[roi_mask].max()}",
                "",
                "magenta = this ROI",
                "faint red = all other ROIs",
            ]),
            va="top", ha="left", fontsize=8.5, family="monospace", transform=ax.transAxes,
        )

    title = f"{Path(args.input).name}  |  {n_roi} ROIs  |  {int(mask.sum())} foreground voxels"
    fig.suptitle(title, fontsize=11, y=1 - 0.22 / fig_height)

    stem = Path(args.input).stem
    suffix = f"_roi{args.roi}" if has_roi else ""
    figure_path = outdir / f"{stem}_summary{suffix}.png"
    fig.savefig(figure_path, dpi=130, bbox_inches="tight")
    print(f"\nWrote {figure_path}")
    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
