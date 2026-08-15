# blobquant

Segment 3D blobs (fluorescent neurons) from a volume using a 3D U-Net, and write a
probability map, a binary mask and a quick-look overlay.

One command takes a TIFF or HDF5 volume and produces:

| Output | Contents |
| --- | --- |
| `<name>_prediction.h5` | `probability` `(C, Z, Y, X)` float32, `mask` `(Z, Y, X)` uint8, `labels` `(Z, Y, X)` int32 instance IDs |
| `<name>_prediction.tif` | binary mask as an 8-bit ImageJ stack (0 / 255) |
| `<name>_prediction_overlay.png` | montage of raw z-slices with the mask outlined in red |

## Installation

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12. The segmentation network
is vendored as a git submodule, so clone recursively:

```bash
git clone --recurse-submodules <repo-url>
cd prj_immob
uv sync
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
uv sync
```

`uv sync` creates `.venv/` and installs everything, including the `pytorch-3dunet`
submodule as an editable dependency. No conda environment is needed.

You also need the model weights, `model_best_checkpoint.pytorch` (~780 MB), in the
repository root. It is not tracked in git.

## Quick start

With volumes in `data/` and the checkpoint in the repository root:

```bash
uv run blobquant
```

That is the whole pipeline. It scans `data/` recursively, segments every volume it can
read, and writes results into `output/`. Files it cannot interpret unambiguously (a
time series needing `--dataset`) are reported and skipped rather than failing the run.

Then visualise a result:

```bash
uv run python examples/visualize_result.py \
    --input      data/hdf5/worm_0_ch_red.h5 \
    --prediction output/worm_0_ch_red_prediction.h5
```

## Usage

To be explicit about inputs and outputs:

```bash
uv run blobquant data/hdf5/worm_0_ch_red.h5 -o output/
```

which writes `output/worm_0_ch_red_prediction.h5`, `..._prediction.tif` and
`..._prediction_overlay.png`.

Any mix of files and directories works; previously generated `*_prediction.*` files are
skipped, so re-running is safe:

```bash
uv run blobquant data/hdf5 my_stack.tif -o output/
```

Pass `--no-recursive` to scan only the top level of a directory. If a recursive sweep
would map two same-named files onto one output path, the run stops rather than letting
the second silently overwrite the first.

Outputs land next to each input unless you pass `-o/--outdir`. Inputs whose outputs
already exist are skipped — pass `--overwrite` to recompute.

### Input formats

- **TIFF** (`.tif`, `.tiff`) — read as a `(Z, Y, X)` stack.
- **HDF5** (`.h5`, `.hdf5`, `.hdf`, `.hd5`) — the dataset is auto-detected, preferring
  a name like `raw`. If a file holds several 3D datasets (e.g. one per timepoint), the
  tool refuses to guess and asks you to choose:

  ```bash
  uv run blobquant data/cropped/worm2_2_red.h5 --dataset t2/channel0 -o output/
  ```

### Common options

| Flag | Default | Purpose |
| --- | --- | --- |
| `inputs` | `data/` | Files or directories to segment |
| `-o, --outdir` | `output/` | Where to write outputs |
| `-c, --checkpoint` | `model_best_checkpoint.pytorch` | Model weights |
| `--dataset` | autodetect | HDF5 dataset holding the volume |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--gpu` | `3` | Physical GPU index to pin |
| `--threshold` | `0.5` | Foreground **probability** cutoff |
| `--min-voxels` | `6` | Drop blobs smaller than this |
| `--patch` / `--overlap` | `64 120 284` / `8 16 16` | Sliding-window geometry |
| `--no-viz` | off | Skip the overlay PNG |
| `--overwrite` | off | Recompute existing outputs |

Run `uv run blobquant --help` for the full list.

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

**Why a single GPU is pinned.** Upstream `pytorch-3dunet`'s `predict.py` wraps the
model in `nn.DataParallel` whenever more than one GPU is visible. The predictor then
sets `model.testing = True` to enable the final softmax, but on a `DataParallel`
wrapper that attribute lands on the wrapper instead of the inner module, so the
replicas never see it and **the softmax is silently skipped** — the saved file
contains raw logits instead of probabilities. `blobquant` sidesteps this entirely: it
builds the network with `is_segmentation=False` and applies the softmax explicitly, so
the output is probabilities regardless of device count. Legacy
`*_predictions.h5` files in `data/` produced by the old multi-GPU path hold logits, and
thresholding those at `0.5` is not a probability cut (it corresponds to p ≈ 0.62).

## Sliding-window inference

If the volume is larger than `--patch`, it is tiled with `--overlap` between windows.
A halo of `overlap // 2` is trimmed from every interior patch edge, since voxels near a
patch border are predicted with little surrounding context; remaining overlap is
averaged. The tool errors out rather than emitting zeros if a geometry would leave any
voxel uncovered.

The patch default matches the shape the checkpoint was trained on, so typical volumes
are a single forward pass and need no tiling. Tiling much smaller than the training
patch measurably changes results — at `32 × 64 × 128` the mask agrees with the
single-pass result on 99.69 % of voxels, rising to 99.92 % at `64 × 120 × 240`. Prefer
the largest patch that fits in memory.

## Using the results

```python
import h5py

with h5py.File("output/worm_0_ch_red_prediction.h5", "r") as f:
    probability = f["probability"][()]  # (2, Z, Y, X); channel 1 is foreground
    mask = f["mask"][()].astype(bool)   # (Z, Y, X)
    labels = f["labels"][()]            # 0 = background, 1..N per blob
    print(dict(f.attrs))                # source, threshold, min_voxels, num_blobs, device
```

`labels` is already filtered by `--min-voxels` and relabelled `1..N`, so
`labels == k` is the mask for blob *k*.

## Visualising a result

`examples/visualize_result.py` renders one figure summarising a segmentation:

```bash
uv run python examples/visualize_result.py \
    --input      data/hdf5/worm_0_ch_red.h5 \
    --prediction output/worm_0_ch_red_prediction.h5
```

The top row shows, side by side, the **maximum intensity projection of the input
volume**, the **MIP of the output mask**, the **MIP of the segmented foreground**
(input × mask), and the **distribution of ROI sizes** in voxels. It also prints the
summary to the terminal:

```
ROIs segmented: 166
Foreground voxels: 20769 (0.952% of the volume)
ROI size (voxels): n=166  min=6  q25=61  median=119  q75=176  max=378  total=20736
  note: 33 masked voxel(s) belong to no ROI (components below the --min-voxels filter)
```

That last note reflects a real distinction in the output file: `mask` is the raw
threshold, while `labels` additionally drops components below `--min-voxels`. The two
totals differ by exactly what the size filter removed.

### Locating one ROI in 3D

Add `--roi N` to locate a single ROI:

```bash
uv run python examples/visualize_result.py \
    --input      data/hdf5/worm_0_ch_red.h5 \
    --prediction output/worm_0_ch_red_prediction.h5 \
    --roi 140
```

A second row appears with three orthogonal projections — top (Z), front (Y) and side
(X). The chosen ROI's voxels are painted **magenta**, every other ROI sits behind in
faint red, and a ring plus crosshairs mark the centroid so even a 12-voxel ROI is
findable. A side panel lists its voxel count, size rank, centroid, bounding box and
intensity, which is also printed:

```
--- ROI 140 ---
  voxels:   378
  centroid: z=53.8  y=27.9  x=73.1
  bbox:     z 45-62   y 24-31   x 70-76
  extent:   18 x 8 x 7 voxels (z, y, x)
  intensity in ROI: mean=288.9 max=471
```

Figures are written to `--outdir` (default `output/`) as `<name>_summary[_roiN].png`.
Pass `--show` to open the figure interactively instead.

## Two channels, one mask

The network is trained on the **red** (marker) channel, so blobs are detected there
once and the *same* mask is transferred to the **green** (activity) channel. Both
channels end up segmented identically, which is what makes a per-blob green/red ratio
meaningful — a ratio computed from two independently-segmented masks would compare
different sets of voxels. (Running the network directly on this green channel finds
172 blobs against the red channel's 166, so the two masks genuinely do not correspond.)

`examples/apply_red_mask_to_green.py` is a runnable end-to-end demonstration:

```bash
# 1. Segment the red channel.
uv run blobquant data/hdf5/worm_0_ch_red.h5 -o output/

# 2. Transfer that mask to the green channel and quantify both.
uv run python examples/apply_red_mask_to_green.py \
    --prediction output/worm_0_ch_red_prediction.h5 \
    --red   data/hdf5/worm_0_ch_red.h5 \
    --green data/hdf5/worm_0_ch_green.h5 \
    -o output/
```

This writes `worm_0_ch_green_prediction.h5` / `.tif` (the transferred mask, same layout
as blobquant's own output), an overlay of the green channel with the red-derived
contours, and `worm_0_ch_blob_intensities.csv`:

```
label,num_voxels,red_sum,green_sum,red_mean,green_mean,ratio
1,218,67252.0,71355.0,308.50,327.32,1.0610
2,163,14268.0,29383.0,87.53,180.26,2.0594
```

Each channel is background-subtracted at its own 20th intensity percentile
(`--bkg-percentile`) before summing, following the original prototype. Mask transfer
requires co-registered volumes of equal shape; the script refuses to run otherwise
rather than silently misattributing signal.

### Working with a single ROI

The script also demonstrates both directions of ROI lookup. **Given an ROI number, get
its pixels** (`--roi 7`, defaulting to the largest ROI):

```
--- ROI 7: isolating its pixels ---
  voxels:   67
  centroid: z=19.3 y=31.9 x=58.9
  bbox:     z 16-22, y 30-34, x 58-60
  first 5 voxel coords (z, y, x): [(16, 32, 59), (17, 30, 59), (17, 31, 58), ...]
  red intensity   in ROI: sum=7916.0 mean=118.1
  green intensity in ROI: sum=17790.0 mean=265.5
  wrote output/roi_7_mask.tif
  wrote output/roi_7_voxels.csv
```

**Given a pixel coordinate, get its ROI** (`--at 20 60 150`). Because `labels` is a
plain integer volume, this is a single array index, not a search:

```python
roi = int(labels[z, y, x])   # 0 means background
```

The script reports the nearest ROI when a probe lands on background:

```
--- Pixel (20, 60, 150): identifying its ROI ---
  background (no ROI); nearest is ROI 28 at 3.7 voxels
```

The underlying one-liners, if you would rather work in your own notebook:

```python
import numpy as np, h5py

with h5py.File("output/worm_0_ch_red_prediction.h5") as f:
    labels = f["labels"][()]

roi_only = labels == 7            # boolean mask isolating ROI 7
coords = np.argwhere(labels == 7) # (N, 3) array of its (z, y, x) voxels
green_in_roi = green[roi_only]    # every green pixel belonging to ROI 7
roi_here = int(labels[54, 28, 73])# which ROI covers this voxel (0 = background)
```

To extend this to a time series, load each timepoint and reuse the same `labels`
volume — see `check_images.ipynb` for the prototype of that analysis.

## Repository layout

```
src/blobquant/
  cli.py           argument parsing, per-file orchestration
  device.py        CUDA_VISIBLE_DEVICES pinning and CPU fallback
  io.py            TIFF/HDF5 loading, output naming, writers
  inference.py     model construction, normalisation, sliding-window prediction
  postprocess.py   thresholding, connected components, size filtering
  viz.py           overlay montage
examples/
  visualize_result.py          projections, ROI size distribution, 3D ROI locator
  apply_red_mask_to_green.py   two-channel mask transfer + ROI lookup demo
data/hdf5/
  worm_0_ch_red.h5    red/marker channel (network input)
  worm_0_ch_green.h5  green/activity channel, co-registered with the red
pytorch-3dunet/    vendored network (git submodule)
model_inference.yaml  legacy config for the upstream predict.py; optional --config source
```

`data/hdf5/worm_0_ch_red_legacy_predictions.h5` is an artifact of the old multi-GPU
pipeline and holds logits rather than probabilities; it is kept only for reference and
is skipped when scanning directories.

The model architecture is built into `inference.py`, so no config file is required.
Pass `--config model_inference.yaml` to override it from YAML.
