# blobquant

Segment 3D blobs (fluorescent neurons) from a volume using a 3D U-Net, then quantify
per-blob fluorescence.

One command takes a TIFF or HDF5 volume and produces:

| Output | Contents |
| --- | --- |
| `<name>_prediction.h5` | `probability` `(C, Z, Y, X)` float32, `mask` `(Z, Y, X)` uint8, `labels` `(Z, Y, X)` int32 instance IDs |
| `<name>_prediction.tif` | binary mask as an 8-bit ImageJ stack (0 / 255) |
| `<name>_prediction_overlay.png` | montage of raw z-slices with the mask outlined in red |

For internals — device selection, sliding-window inference, repository layout — see
[`notes.md`](notes.md).

## Installation

You need:

- [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12.
- **GitHub SSH access to `flavell-lab/private_pytorch-3dunet`.** The segmentation
  network is vendored as a submodule pointing at that private lab repository over SSH,
  so the recursive clone below fails at the submodule step without it. If you are
  outside the lab, ask for read access and add an SSH key to your GitHub account first.

Then copy and paste the whole block:

```bash
git clone --recurse-submodules https://github.com/dikang13/blobquant.git
cd blobquant
uv sync
./scripts/fetch_weights.sh
uv run blobquant --help
```

`uv sync` creates `.venv/` and installs everything, including the `pytorch-3dunet`
submodule as an editable dependency. No conda environment is needed.
`fetch_weights.sh` downloads the 261 MB `model_weights.pytorch` and checks its
SHA-256; it is a no-op if the file is already there, so re-running it is harmless.
The final line should print the usage message.

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
uv sync
```

### The model weights

`model_weights.pytorch` (261 MB) is over GitHub's 100 MB per-file limit, so it is
**attached to a release** rather than committed, and `git clone` alone does not bring
it. `./scripts/fetch_weights.sh` is the one extra step; it pulls the asset from
[the `weights-v1` release](https://github.com/dikang13/blobquant/releases/tag/weights-v1)
and verifies the checksum. Without it every run stops with:

```
ERROR blobquant - No model weights found (looked for model_weights.pytorch and
model_best_checkpoint.pytorch). Run ./scripts/fetch_weights.sh to download them.
```

Point `--checkpoint` anywhere else to use a different file. These weights are the
original 783 MB training checkpoint with the optimizer state removed — see
[`notes.md`](notes.md) — and produce bit-identical output, so an existing
`model_best_checkpoint.pytorch` in the repository root is still picked up automatically.

### The sample volumes are not in the clone either

The use cases below refer to `data/hdf5/worm_0_ch_red.h5`, `worm_0_ch_green.h5` and the
time series `data/cropped/worm2_2_*.h5` as concrete examples — substitute your own
paths, or ask the maintainer for the samples if you want to reproduce the output shown
here exactly.

---

## Use cases

### 1. Segment every volume in `data/`

With volumes in `data/` and the weights in the repository root:

```bash
uv run blobquant
```

That is the whole pipeline. It scans `data/` recursively, segments every volume it can
read, and writes results into `output/`. Files it cannot interpret unambiguously (a
time series needing `--dataset`) are reported and skipped rather than failing the run.
Inputs whose outputs already exist are skipped, so re-running is safe — pass
`--overwrite` to recompute.

### 2. Segment specific files

```bash
uv run blobquant data/hdf5/worm_0_ch_red.h5 -o output/
```

Any mix of files and directories works:

```bash
uv run blobquant data/hdf5 my_stack.tif -o output/
```

**Input formats.** TIFF (`.tif`, `.tiff`) is read as a `(Z, Y, X)` stack. For HDF5
(`.h5`, `.hdf5`, `.hdf`, `.hd5`) the dataset is auto-detected, preferring a name like
`raw`. If a file holds several 3D datasets — one per timepoint, say — the tool refuses
to guess and asks you to choose:

```bash
uv run blobquant data/cropped/worm2_2_red.h5 --dataset t2/channel0 -o output/
```

**Common options:**

| Flag | Default | Purpose |
| --- | --- | --- |
| `inputs` | `data/` | Files or directories to segment |
| `-o, --outdir` | `output/` | Where to write outputs |
| `-c, --checkpoint` | `model_weights.pytorch` | Model weights |
| `--dataset` | autodetect | HDF5 dataset holding the volume |
| `--device` | `auto` | `auto`, `cuda` or `cpu` |
| `--gpu` | `3` | Physical GPU index to pin |
| `--threshold` | `0.5` | Foreground **probability** cutoff |
| `--min-voxels` | `6` | Drop blobs smaller than this |
| `--patch` / `--overlap` | `64 120 284` / `8 16 16` | Sliding-window geometry |
| `--no-recursive` | off | Scan only the top level of a directory |
| `--no-viz` | off | Skip the overlay PNG |
| `--overwrite` | off | Recompute existing outputs |

Run `uv run blobquant --help` for the full list.

### 3. Visualise a result

```bash
uv run python examples/visualize_result.py \
    --input      data/hdf5/worm_0_ch_red.h5 \
    --prediction output/worm_0_ch_red_prediction.h5
```

Renders one figure showing, side by side, the **maximum intensity projection of the
input volume**, the **MIP of the output mask**, the **MIP of the segmented foreground**
(input × mask), and the **distribution of ROI sizes** in voxels. The summary is also
printed:

```
ROIs segmented: 166
Foreground voxels: 20769 (0.952% of the volume)
ROI size (voxels): n=166  min=6  q25=61  median=119  q75=176  max=378  total=20736
  note: 33 masked voxel(s) belong to no ROI (components below the --min-voxels filter)
```

### 4. Locate one ROI in 3D

```bash
uv run python examples/visualize_result.py \
    --input      data/hdf5/worm_0_ch_red.h5 \
    --prediction output/worm_0_ch_red_prediction.h5 \
    --roi 140
```

Adds a second row with three orthogonal projections — top (Z), front (Y) and side (X).
The chosen ROI's voxels are painted **magenta**, every other ROI sits behind in faint
red, and a ring plus crosshairs mark the centroid so even a 12-voxel ROI is findable.
A side panel lists its voxel count, size rank, centroid, bounding box and intensity:

```
--- ROI 140 ---
  voxels:   378
  centroid: z=53.8  y=27.9  x=73.1
  bbox:     z 45-62   y 24-31   x 70-76
  extent:   18 x 8 x 7 voxels (z, y, x)
  intensity in ROI: mean=288.9 max=471
```

Figures go to `--outdir` (default `output/`) as `<name>_summary[_roiN].png`. Pass
`--show` to open the figure interactively instead.

### 5. Segment two channels with one mask

The network is trained on the **red** (marker) channel, so blobs are detected there
once and the *same* mask is transferred to the **green** (activity) channel. Both
channels end up segmented identically, which is what makes a per-blob green/red ratio
meaningful — a ratio computed from two independently-segmented masks would compare
different sets of voxels. (Running the network directly on this green channel finds
172 blobs against the red channel's 166, so the two masks genuinely do not correspond.)

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
(`--bkg-percentile`) before summing. Mask transfer requires co-registered volumes of
equal shape; the script refuses to run otherwise rather than silently misattributing
signal.

### 6. Work with a single ROI

`examples/apply_red_mask_to_green.py` demonstrates both directions of ROI lookup.

**Given an ROI number, get its pixels** (`--roi 7`, defaulting to the largest ROI).
Writes `roi_7_mask.tif` and `roi_7_voxels.csv`:

```
--- ROI 7: isolating its pixels ---
  voxels:   67
  centroid: z=19.3 y=31.9 x=58.9
  bbox:     z 16-22, y 30-34, x 58-60
  red intensity   in ROI: sum=7916.0 mean=118.1
  green intensity in ROI: sum=17790.0 mean=265.5
```

**Given a pixel coordinate, get its ROI** (`--at 20 60 150`). The script reports the
nearest ROI when a probe lands on background:

```
--- Pixel (20, 60, 150): identifying its ROI ---
  background (no ROI); nearest is ROI 28 at 3.7 voxels
```

### 7. Quantify a time series

An immobilised recording stores one 3D volume per frame, e.g. `t0/channel0` …
`t4/channel0`. Blobs barely move between frames, so the volume is segmented **once**
and the same ROIs are reused for every timepoint. Inspect an unfamiliar file first:

```bash
uv run python examples/timeseries_traces.py --list data/cropped/worm2_2_red.h5
```

```
Structure of data/cropped/worm2_2_red.h5:
  Group:   t0
  Dataset: t0/channel0 | shape (64, 120, 284) | dtype uint16
  ...
```

Then segment one frame and quantify all of them:

```bash
# 1. Segment a single timepoint of the red channel.
uv run blobquant data/cropped/worm2_2_red.h5 --dataset t0/channel0 -o output/

# 2. Sum both channels inside those ROIs, frame by frame.
uv run python examples/timeseries_traces.py \
    --prediction output/worm2_2_red_prediction.h5 \
    --red   data/cropped/worm2_2_red.h5 \
    --green data/cropped/worm2_2_green.h5 \
    -o output/
```

```
162 ROIs x 5 timepoints
  median green/red ratio: 1.6593
  ratio range:            0.0797 - 6.2142
  most variable ROIs (CV of the trace):
    ROI    7 ( 121 vox)  CV=0.213  [2.051 1.668 1.353 1.153 1.298]
    ROI  147 ( 181 vox)  CV=0.148  [1.568 1.429 1.450 1.851 2.065]
```

Writes `worm2_2_red_traces.csv` — one row per ROI per timepoint, with the voxel count,
both background-subtracted channel sums and their ratio — and `worm2_2_red_traces.png`,
showing every trace as a per-ROI–normalised heatmap alongside the most variable ones as
line plots. Pass `--roi 7 147` to plot chosen ROIs instead.

Background is estimated per frame at the `--bkg-percentile` (default 20th) of each
channel; `--bkg first` instead fixes it from the first frame. Use `--channel` if the
frames are not named `channel0`.

### 8. Use the results in your own code

```python
import h5py, numpy as np

with h5py.File("output/worm_0_ch_red_prediction.h5", "r") as f:
    probability = f["probability"][()]  # (2, Z, Y, X); channel 1 is foreground
    mask = f["mask"][()].astype(bool)   # (Z, Y, X)
    labels = f["labels"][()]            # 0 = background, 1..N per blob
    print(dict(f.attrs))                # source, threshold, min_voxels, num_blobs, device

roi_only = labels == 7             # boolean mask isolating ROI 7
coords = np.argwhere(labels == 7)  # (N, 3) array of its (z, y, x) voxels
roi_here = int(labels[54, 28, 73]) # which ROI covers this voxel (0 = background)
```

`labels` is already filtered by `--min-voxels` and relabelled `1..N`, so `labels == k`
is the mask for blob *k*. Because `labels` is a plain integer volume, looking up the
ROI at a coordinate is a single array index, not a search.

To extend this to a time series, load each timepoint and reuse the same `labels`
volume — `examples/timeseries_traces.py` (use case 7) does exactly that.
