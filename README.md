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

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.12. The segmentation network
is vendored as a git submodule, so clone recursively. Copy and paste the whole block:

```bash
git clone --recurse-submodules https://github.com/dikang13/blobquant.git
cd blobquant
uv sync
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
uv sync
```

`uv sync` creates `.venv/` and installs everything, including the `pytorch-3dunet`
submodule as an editable dependency. No conda environment is needed.

Finally, place the model weights `model_best_checkpoint.pytorch` (~780 MB) in the
repository root. They are too large for git and are **not** included in the clone —
ask the maintainer for a copy.

Verify the install:

```bash
uv run blobquant --help
```

---

## Use cases

### 1. Segment every volume in `data/`

With volumes in `data/` and the checkpoint in the repository root:

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
| `-c, --checkpoint` | `model_best_checkpoint.pytorch` | Model weights |
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

### 7. Use the results in your own code

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
volume — see `check_images.ipynb` for the prototype of that analysis.
