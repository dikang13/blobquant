#!/usr/bin/env python
"""Shrink a training checkpoint to the weights inference actually needs.

A checkpoint saved by the training loop carries the Adam optimizer state
(``exp_avg`` / ``exp_avg_sq`` per parameter) alongside the weights. That state is
only needed to *resume training* -- inference never touches it, and it is twice the
size of the weights themselves:

    783 MB checkpoint = 261 MB weights + 522 MB optimizer state

Dropping it produces bit-identical predictions. ``--half`` additionally stores the
weights as float16, halving the file again at a cost of a few voxels per volume.

Usage::

    uv run python scripts/strip_checkpoint.py \\
        model_best_checkpoint.pytorch model_weights.pytorch

    uv run blobquant -c model_weights.pytorch data/hdf5/worm_0_ch_red.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Kept because they identify which training run the weights came from; all are
# scalars, so they cost nothing.
PROVENANCE_KEYS = ("epoch", "num_iterations", "best_eval_score", "eval_score_higher_is_better")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", type=Path, help="Training checkpoint to shrink.")
    parser.add_argument("output", type=Path, help="Where to write the weights-only file.")
    parser.add_argument(
        "--half", action="store_true",
        help="Store weights as float16 (half the size; changes ~5 voxels in 2.2M).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; remove it or choose another name.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise SystemExit(f"{args.checkpoint}: no 'model_state_dict' -- not a training checkpoint?")

    state = checkpoint["model_state_dict"]
    if args.half:
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in state.items()}

    slim = {"model_state_dict": state}
    for key in PROVENANCE_KEYS:
        if key in checkpoint:
            slim[key] = checkpoint[key]
    torch.save(slim, args.output)

    before = args.checkpoint.stat().st_size / 1e6
    after = args.output.stat().st_size / 1e6
    dropped = sorted(set(checkpoint) - set(slim))
    print(f"{args.checkpoint}: {before:.1f} MB")
    print(f"{args.output}: {after:.1f} MB  ({before / after:.1f}x smaller)")
    print(f"dropped: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
