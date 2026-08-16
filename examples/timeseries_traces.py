#!/usr/bin/env python
"""Extract per-ROI activity traces from a two-channel time series.

Blobs move very little between frames of an immobilised recording, so the volume is
segmented **once** and the resulting ROIs are reused for every timepoint. For each
frame this script sums the background-subtracted red (marker) and green (activity)
signal inside every ROI and reports the green/red ratio -- the trace.

A time-series HDF5 stores one 3D dataset per frame, e.g. ``t0/channel0`` ...
``t4/channel0``. Inspect an unfamiliar file first::

    uv run python examples/timeseries_traces.py --list data/cropped/worm2_2_red.h5

Then segment one frame and quantify all of them::

    uv run blobquant data/cropped/worm2_2_red.h5 --dataset t0/channel0 -o output/

    uv run python examples/timeseries_traces.py \\
        --prediction output/worm2_2_red_prediction.h5 \\
        --red   data/cropped/worm2_2_red.h5 \\
        --green data/cropped/worm2_2_green.h5 \\
        -o output/

Writes ``<name>_traces.csv`` (one row per ROI per timepoint) and ``<name>_traces.png``
(all traces as a heatmap, plus the most variable ones as line plots).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger("timeseries_traces")

# Frame groups are named t0, t1, ... -- sorted numerically, not as strings, so t10
# does not land between t1 and t2.
TIMEPOINT_RE = re.compile(r"^t(\d+)$")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--list", metavar="FILE", default=None,
        help="Print the structure of an HDF5 file and exit (no other arguments needed).",
    )
    parser.add_argument(
        "--prediction",
        help="Red-channel *_prediction.h5 from `blobquant`; its labels define the ROIs.",
    )
    parser.add_argument("--red", help="Red/marker channel time series (.h5).")
    parser.add_argument("--green", help="Green/activity channel time series (.h5).")
    parser.add_argument(
        "--channel", default="channel0",
        help="Dataset inside each timepoint group (default: channel0).",
    )
    parser.add_argument("-o", "--outdir", default="output", help="Output directory.")
    parser.add_argument(
        "--bkg-percentile", type=float, default=20.0,
        help="Background estimated as this intensity percentile (default: 20).",
    )
    parser.add_argument(
        "--bkg", choices=("each", "first"), default="each",
        help="Estimate background per timepoint ('each', default) or once on the first "
             "frame and hold it fixed ('first', what the original notebook did).",
    )
    parser.add_argument(
        "--roi", type=int, nargs="*", default=None,
        help="ROI numbers to plot as line traces (default: the 6 most variable).",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip the figure.")
    return parser.parse_args(argv)


def print_structure(path: Path) -> None:
    """Print every group and dataset in an HDF5 file, with shapes and dtypes."""
    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"  Group:   {name}")
        elif isinstance(obj, h5py.Dataset):
            print(f"  Dataset: {name} | shape {obj.shape} | dtype {obj.dtype}")
        else:
            # Committed datatypes, e.g. the __DATA_TYPES__ entries some writers add.
            print(f"  Type:    {name}")

    with h5py.File(path, "r") as handle:
        print(f"Structure of {path}:")
        handle.visititems(visit)


def timepoint_keys(handle, channel: str) -> list[str]:
    """Dataset paths for every frame, ordered by timepoint index.

    Groups that do not look like ``t<N>`` are ignored, which skips the
    ``__DATA_TYPES__`` bookkeeping group written by some acquisition software.
    """
    frames: list[tuple[int, str]] = []
    for name in handle:
        match = TIMEPOINT_RE.match(name)
        if match and isinstance(handle[name], h5py.Group) and channel in handle[name]:
            frames.append((int(match.group(1)), f"{name}/{channel}"))
    if not frames:
        raise SystemExit(
            f"No t<N>/{channel} datasets found. Run with --list to see the file's contents, "
            "and set --channel if the frames use another dataset name."
        )
    return [key for _, key in sorted(frames)]


def load_series(path: Path, channel: str) -> tuple[np.ndarray, list[str]]:
    """Load every frame of a time-series HDF5 as a (T, Z, Y, X) array."""
    with h5py.File(path, "r") as handle:
        keys = timepoint_keys(handle, channel)
        frames = [handle[key][()] for key in keys]
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise SystemExit(f"{path}: timepoints have differing shapes {sorted(shapes)}")
    return np.stack(frames), keys


def per_roi_sums(labels: np.ndarray, volume: np.ndarray, n_labels: int) -> np.ndarray:
    """Sum *volume* within each label, returning an array indexed 1..n_labels."""
    sums = np.bincount(
        labels.ravel(), weights=volume.ravel().astype(np.float64), minlength=n_labels + 1
    )
    return sums[1:]


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.list:
        print_structure(Path(args.list))
        return 0
    missing = [name for name in ("prediction", "red", "green") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required argument(s): {', '.join('--' + m for m in missing)}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. The ROIs, segmented once on a single frame of the red channel.
    with h5py.File(args.prediction, "r") as f:
        labels = f["labels"][()].astype(np.int64)
    n_roi = int(labels.max())
    sizes = np.bincount(labels.ravel(), minlength=n_roi + 1)[1:]
    logger.info("Loaded %d ROIs from %s", n_roi, args.prediction)

    # 2. Both channels, every timepoint.
    red, red_keys = load_series(Path(args.red), args.channel)
    green, green_keys = load_series(Path(args.green), args.channel)
    logger.info("red   %s from %s (%s ... %s)", red.shape, args.red, red_keys[0], red_keys[-1])
    logger.info("green %s from %s", green.shape, args.green)

    if red.shape != green.shape:
        raise SystemExit(f"Channel shape mismatch: red {red.shape} vs green {green.shape}")
    if red.shape[1:] != labels.shape:
        raise SystemExit(
            f"Volume shape {red.shape[1:]} does not match the ROI labels {labels.shape}. "
            "The ROIs can only be reused across co-registered volumes of equal shape."
        )
    n_t = red.shape[0]

    # 3. Background-subtract, then sum each channel inside every ROI, frame by frame.
    if args.bkg == "first":
        fixed_bkg = (
            float(np.percentile(red[0], args.bkg_percentile)),
            float(np.percentile(green[0], args.bkg_percentile)),
        )
        logger.info("Background fixed from frame 0: red=%.1f green=%.1f", *fixed_bkg)

    red_sums = np.zeros((n_roi, n_t))
    green_sums = np.zeros((n_roi, n_t))
    for t in range(n_t):
        if args.bkg == "first":
            red_bkg, green_bkg = fixed_bkg
        else:
            red_bkg = float(np.percentile(red[t], args.bkg_percentile))
            green_bkg = float(np.percentile(green[t], args.bkg_percentile))
        red_corr = np.clip(red[t].astype(np.float64) - red_bkg, 0, None)
        green_corr = np.clip(green[t].astype(np.float64) - green_bkg, 0, None)
        red_sums[:, t] = per_roi_sums(labels, red_corr, n_roi)
        green_sums[:, t] = per_roi_sums(labels, green_corr, n_roi)

    # The ratio is only meaningful where the marker channel has signal to divide by.
    traces = np.divide(
        green_sums, red_sums, out=np.full_like(green_sums, np.nan), where=red_sums > 0
    )

    stem = Path(args.red).stem
    csv_path = outdir / f"{stem}_traces.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["label", "timepoint", "num_voxels", "red_sum", "green_sum",
             "red_mean", "green_mean", "ratio"]
        )
        for roi in range(n_roi):
            n = int(sizes[roi])
            for t in range(n_t):
                r, g = red_sums[roi, t], green_sums[roi, t]
                writer.writerow(
                    [roi + 1, t, n, f"{r:.1f}", f"{g:.1f}",
                     f"{r / n:.2f}", f"{g / n:.2f}", f"{traces[roi, t]:.4f}"]
                )
    logger.info("Wrote %s (%d ROIs x %d timepoints)", csv_path, n_roi, n_t)

    # Coefficient of variation over time: which ROIs actually changed.
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.nanstd(traces, axis=1) / np.nanmean(traces, axis=1)
    ranked = np.argsort(np.nan_to_num(cv, nan=-1))[::-1]

    print(f"\n{n_roi} ROIs x {n_t} timepoints")
    print(f"  median green/red ratio: {np.nanmedian(traces):.4f}")
    print(f"  ratio range:            {np.nanmin(traces):.4f} - {np.nanmax(traces):.4f}")
    print("  most variable ROIs (CV of the trace):")
    for roi in ranked[:5]:
        values = " ".join(f"{v:.3f}" for v in traces[roi])
        print(f"    ROI {roi + 1:>4} ({int(sizes[roi]):>4} vox)  CV={cv[roi]:.3f}  [{values}]")
    print(f"  table: {csv_path}")

    if args.no_viz:
        return 0

    selected = (
        [r - 1 for r in args.roi] if args.roi else list(ranked[: min(6, n_roi)])
    )
    for index in selected:
        if not 0 <= index < n_roi:
            raise SystemExit(f"ROI {index + 1} out of range; this volume has ROIs 1..{n_roi}")

    fig, (ax_map, ax_lines) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # Normalise each ROI to its own mean so the heatmap shows relative change rather
    # than which ROIs are brightest.
    normalised = traces / np.nanmean(traces, axis=1, keepdims=True)
    order = np.argsort(np.nanmean(traces, axis=1))
    image = ax_map.imshow(
        normalised[order], aspect="auto", cmap="magma", interpolation="nearest",
        vmin=np.nanpercentile(normalised, 2), vmax=np.nanpercentile(normalised, 98),
    )
    ax_map.set_xlabel("timepoint")
    ax_map.set_ylabel("ROI (sorted by mean ratio)")
    ax_map.set_title(f"green/red ratio, normalised per ROI\n{n_roi} ROIs x {n_t} timepoints",
                     fontsize=10)
    ax_map.set_xticks(range(n_t))
    fig.colorbar(image, ax=ax_map, label="ratio / mean ratio")

    for index in selected:
        ax_lines.plot(
            range(n_t), traces[index], marker="o", ms=4, lw=1.4,
            label=f"ROI {index + 1} ({int(sizes[index])} vox)",
        )
    ax_lines.set_xlabel("timepoint")
    ax_lines.set_ylabel("green / red")
    ax_lines.set_title(
        "selected ROIs" if args.roi else "6 most variable ROIs", fontsize=10
    )
    ax_lines.set_xticks(range(n_t))
    ax_lines.legend(fontsize=8)
    ax_lines.grid(alpha=0.25)

    fig.suptitle(f"{Path(args.red).name} / {Path(args.green).name}", fontsize=11)
    fig.tight_layout()
    figure_path = outdir / f"{stem}_traces.png"
    fig.savefig(figure_path, dpi=130)
    plt.close(fig)
    print(f"  figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
