"""Command-line entry point: volume in, prediction + mask + overlay out."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from blobquant import io
from blobquant.device import DEFAULT_GPU, configure_visible_devices, select_device

logger = logging.getLogger("blobquant")

DEFAULT_CHECKPOINT = "model_best_checkpoint.pytorch"
DEFAULT_INDIR = "data"
DEFAULT_OUTDIR = "output"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="blobquant",
        description=(
            "Segment 3D blobs from TIFF or HDF5 volumes. Each input produces "
            "<name>_prediction.h5 and <name>_prediction.tif."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=[DEFAULT_INDIR],
        help=f"Input .tif/.tiff/.h5 files, or directories to scan (default: {DEFAULT_INDIR}/).",
    )
    parser.add_argument(
        "-o", "--outdir", default=DEFAULT_OUTDIR,
        help=f"Directory for outputs (default: {DEFAULT_OUTDIR}/).",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Do not descend into subdirectories when scanning.",
    )
    parser.add_argument(
        "-c", "--checkpoint", default=DEFAULT_CHECKPOINT,
        help=f"Model weights (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML file whose 'model:' block overrides the built-in architecture.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="HDF5 dataset holding the volume (default: autodetect, preferring 'raw').",
    )
    parser.add_argument(
        "--device", choices=("auto", "cuda", "cpu"), default="auto",
        help="Compute device (default: auto -- GPU if available, else CPU).",
    )
    parser.add_argument(
        "--gpu", default=DEFAULT_GPU,
        help=f"Physical GPU index to pin via CUDA_VISIBLE_DEVICES (default: {DEFAULT_GPU}).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Foreground probability threshold in [0, 1] (default: 0.5).",
    )
    parser.add_argument(
        "--min-voxels", type=int, default=6,
        help="Drop connected components smaller than this (default: 6).",
    )
    parser.add_argument(
        "--patch", type=int, nargs=3, metavar=("Z", "Y", "X"), default=None,
        help="Sliding-window patch size; clamped to the volume (default: 64 120 284).",
    )
    parser.add_argument(
        "--overlap", type=int, nargs=3, metavar=("Z", "Y", "X"), default=None,
        help="Overlap between patches (default: 8 16 16).",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip writing the <name>_prediction_overlay.png montage.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute inputs whose outputs already exist.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return parser


def _process_one(path, model, device, final_sigmoid, args, inference, postprocess):
    """Run one volume end to end and return a summary dict."""
    volume, source = io.load_volume(path, args.dataset)
    logger.info("Loaded %s from %s -> shape %s, dtype %s", path, source, volume.shape, volume.dtype)

    kwargs = {}
    if args.patch:
        kwargs["patch_shape"] = tuple(args.patch)
    if args.overlap:
        kwargs["overlap"] = tuple(args.overlap)

    probability = inference.predict_volume(volume, model, device, final_sigmoid, **kwargs)
    foreground = inference.foreground_probability(probability)
    mask = postprocess.binarize(foreground, args.threshold)
    labels, blobs = postprocess.label_blobs(mask, args.min_voxels)

    h5_path, tif_path = io.output_paths(path, args.outdir)
    io.save_prediction_h5(
        h5_path,
        probability=probability,
        mask=mask,
        labels=labels,
        attrs={
            "source": str(path),
            "source_dataset": source,
            "threshold": args.threshold,
            "min_voxels": args.min_voxels,
            "num_blobs": len(blobs),
            "device": str(device),
        },
    )
    io.save_mask_tif(tif_path, mask)
    logger.info("Wrote %s and %s", h5_path, tif_path)

    overlay = None
    if not args.no_viz:
        from blobquant import viz

        overlay = viz.save_overlay(
            h5_path.with_name(h5_path.stem + "_overlay.png"),
            volume,
            mask,
            title=f"{path.name}  |  {len(blobs)} blobs  |  thr={args.threshold}",
        )

    return {
        "input": path,
        "h5": h5_path,
        "tif": tif_path,
        "overlay": overlay,
        "num_blobs": len(blobs),
        "foreground_voxels": int(mask.sum()),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Must happen before torch is imported, hence the deferred imports below.
    if args.device != "cpu":
        configure_visible_devices(args.gpu)

    try:
        inputs = io.collect_inputs(args.inputs, recursive=not args.no_recursive)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    if not inputs:
        logger.error("No supported input volumes found.")
        return 2

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint not found: %s (pass --checkpoint)", checkpoint)
        return 2

    from blobquant import inference, postprocess

    device = select_device(args.device)
    model_config, config_checkpoint = inference.load_model_config(
        Path(args.config) if args.config else None
    )
    if args.checkpoint == DEFAULT_CHECKPOINT and config_checkpoint:
        checkpoint = Path(config_checkpoint)  # config wins only if -c was left default
    model, final_sigmoid = inference.build_model(model_config, checkpoint, device)

    # A recursive sweep can turn two same-named files in different folders into one
    # output path; refuse rather than let the second silently overwrite the first.
    seen: dict[Path, Path] = {}
    for path, _ in inputs:
        target, _tif = io.output_paths(path, args.outdir)
        if target in seen:
            logger.error(
                "Output collision: %s and %s both map to %s. Use separate --outdir runs.",
                seen[target], path, target,
            )
            return 2
        seen[target] = path

    results, failures = [], 0
    for path, discovered in inputs:
        h5_path, tif_path = io.output_paths(path, args.outdir)
        if not args.overwrite and h5_path.exists() and tif_path.exists():
            logger.info("Skipping %s (outputs exist; use --overwrite)", path)
            continue
        try:
            results.append(
                _process_one(path, model, device, final_sigmoid, args, inference, postprocess)
            )
        except io.AmbiguousDatasetError as exc:
            # Tolerated while sweeping a directory; an error if the user named the file.
            if discovered:
                logger.warning("Skipping %s: %s", path, exc)
            else:
                failures += 1
                logger.error("%s", exc)
        except Exception as exc:  # keep going so one bad file cannot abort a batch
            failures += 1
            logger.error("Failed on %s: %s", path, exc)
            if args.verbose:
                logger.exception("Traceback:")

    if results:
        print(f"\nProcessed {len(results)} volume(s) on {device}:")
        for r in results:
            print(
                f"  {r['input'].name}: {r['num_blobs']} blobs, "
                f"{r['foreground_voxels']} foreground voxels"
            )
            print(f"    -> {r['h5']}")
            print(f"    -> {r['tif']}")
            if r["overlay"]:
                print(f"    -> {r['overlay']}")
    if failures:
        logger.error("%d input(s) failed.", failures)
        return 1
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
