# Implementation notes

Background on how `blobquant` behaves internally. Not needed to use the tool — see
`README.md` for installation and usage.

## Device selection

By default the process pins itself to **physical GPU 3** by setting
`CUDA_VISIBLE_DEVICES=3` before torch initialises, then runs on it if CUDA is
available. If it is not — no GPU, no driver, or GPU 3 does not exist — it falls back
to CPU automatically and says so. A full `64 × 120 × 284` volume takes ~1 s on an RTX
6000 Ada and ~6 s on CPU.

Pick a different GPU with `--gpu 0`, or force a device with `--device cpu` /
`--device cuda`. An explicit `--device cuda` that cannot be satisfied is an error
rather than a silent fallback. A `CUDA_VISIBLE_DEVICES` you set yourself is always
respected.

### Why a single GPU is pinned

Upstream `pytorch-3dunet`'s `predict.py` wraps the model in `nn.DataParallel` whenever
more than one GPU is visible. The predictor then sets `model.testing = True` to enable
the final softmax, but on a `DataParallel` wrapper that attribute lands on the wrapper
instead of the inner module, so the replicas never see it and **the softmax is silently
skipped** — the saved file contains raw logits instead of probabilities.

`blobquant` sidesteps this entirely: it builds the network with `is_segmentation=False`
and applies the softmax explicitly, so the output is probabilities regardless of device
count.

This was confirmed empirically on a 4-GPU host:

| Run | Output range |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=0` | `0 … 1.0`, channels sum to 1 → probabilities |
| `CUDA_VISIBLE_DEVICES=0,1` | `−17.050 … 11.600` → logits |

Legacy `*_predictions.h5` files produced by the old multi-GPU path hold logits, and
thresholding those at `0.5` is not a probability cut — it corresponds to p ≈ 0.62.

## What is in the checkpoint

The file the training loop saved, `model_best_checkpoint.pytorch`, is 783 MB — three
times the size of the network. Most of it is Adam's per-parameter momentum, which only
matters if you intend to resume training:

| Contents | Size |
| --- | --- |
| `model_state_dict` (44 tensors, float32) | 261 MB |
| `optimizer_state_dict` (`exp_avg`, `exp_avg_sq`) | 522 MB |

`scripts/strip_checkpoint.py` drops the optimizer state and the training bookkeeping
(`h5_dir`, `max_num_iterations`, …), keeping the weights plus the scalars that identify
the run — epoch 527, `best_eval_score` 0.885. The result is what is published as
`model_weights.pytorch`; on the sample volume it reproduces the original mask, labels
and probabilities exactly (max absolute difference 0.0).

`--half` stores the weights as float16 for a further 2× (130 MB). That is *not* what is
published: it changes 5 voxels out of 2 181 120 on the sample volume — the same 166
blobs, but no longer bit-identical.

Neither variant fits GitHub's 100 MB per-file limit, so the weights ride along as a
release asset fetched by `scripts/fetch_weights.sh`. Git LFS would let a plain
`git clone` bring them, but GitHub Free allows only 1 GiB of LFS traffic per month —
roughly four clones of a 261 MB file — after which downloads fail until the quota
resets. Release assets have no such cap.

## Sliding-window inference

If the volume is larger than `--patch`, it is tiled with `--overlap` between windows.
A halo of `overlap // 2` is trimmed from every interior patch edge, since voxels near a
patch border are predicted with little surrounding context; remaining overlap is
averaged. The tool errors out rather than emitting zeros if a geometry would leave any
voxel uncovered.

The patch default matches the shape the checkpoint was trained on, so typical volumes
are a single forward pass and need no tiling. Tiling much smaller than the training
patch measurably changes results:

| Patch | Mask agreement with single-pass |
| --- | --- |
| `32 × 64 × 128` | 99.69 % |
| `64 × 120 × 200` | 99.89 % |
| `64 × 120 × 240` | 99.92 % |

Agreement improves monotonically as the patch approaches the training shape, so the
residual difference is the model's context sensitivity rather than a tiling artifact.
Prefer the largest patch that fits in memory.

## Normalisation

Input volumes are min-max scaled into `[-1, 1]` using **global volume statistics**,
matching `pytorch3dunet.augment.transforms.Normalize` as driven by the training
pipeline. Patch-local normalisation would not reproduce the checkpoint's behaviour.

## Mask vs labels

The two datasets in `*_prediction.h5` are not interchangeable:

- `mask` is the raw threshold on the foreground probability.
- `labels` additionally drops connected components below `--min-voxels`, then relabels
  the survivors `1..N`.

They differ by exactly what the size filter removed — for the sample volume, 33 voxels
out of 20 769. Use `labels` when you mean "the ROIs"; use `mask` when you mean "every
voxel above threshold".

## Repository layout

```
src/blobquant/
  cli.py           argument parsing, per-file orchestration
  device.py        CUDA_VISIBLE_DEVICES pinning and CPU fallback
  io.py            TIFF/HDF5 loading, output naming, writers
  inference.py     model construction, normalisation, sliding-window prediction
  postprocess.py   thresholding, connected components, size filtering
  viz.py           overlay montage and projection helpers
examples/
  visualize_result.py          projections, ROI size distribution, 3D ROI locator
  apply_red_mask_to_green.py   two-channel mask transfer + ROI lookup demo
  timeseries_traces.py         per-ROI green/red traces across timepoints
data/hdf5/
  worm_0_ch_red.h5    red/marker channel (network input)
  worm_0_ch_green.h5  green/activity channel, co-registered with the red
data/cropped/
  worm2_2_{red,green}.h5  five-timepoint series, one volume per t<N>/channel0 group
scripts/
  fetch_weights.sh       download model_weights.pytorch from the GitHub release
  strip_checkpoint.py    shrink a training checkpoint to inference weights
pytorch-3dunet/       vendored network (git submodule)
model_inference.yaml  legacy config for the upstream predict.py; optional --config source
```

The model architecture is built into `inference.py`, so no config file is required.
Pass `--config model_inference.yaml` to override it from YAML.

`data/hdf5/worm_0_ch_red_legacy_predictions.h5` is an artifact of the old multi-GPU
pipeline and holds logits rather than probabilities; it is kept only for reference and
is skipped when scanning directories.

## Reusing ROIs across timepoints

`timeseries_traces.py` segments one frame and applies those labels to every frame,
rather than segmenting each frame independently. This is deliberate: ROI *k* must mean
the same cell at every timepoint for a trace to exist at all, and independent
segmentations do not produce corresponding label numbers. It assumes the recording is
immobilised — for a moving sample the labels would need tracking between frames.
