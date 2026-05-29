# ifcb-features

A Python implementation of segmentation and feature extraction for
Imaging FlowCytobot (IFCB) imagery.

IFCB is a submersible imaging flow cytometer that captures images of individual
plankton cells and other particles. Each raw IFCB sample (a "bin") contains many
regions of interest (ROIs) — small grayscale images, one per imaged particle.
This library takes those ROIs and, for each one:

1. **Segments** the particle from the background, producing a binary "blob" mask.
2. **Extracts features** from the blob and the original ROI — morphological
   measurements (area, biovolume, axis lengths, perimeter, convexity, Feret
   diameters, …).

The implementation is designed to reproduce the numerical output of the original
MATLAB IFCB feature extraction code as closely as possible, so that features
computed here are comparable with historical IFCB datasets.

## Installation

The package targets Python 3.10+ and depends on `numpy`, `scipy`,
`scikit-image`, and `scikit-learn`, plus two WHOI packages installed directly
from GitHub ([`pyifcb`](https://github.com/joefutrelle/pyifcb) for reading IFCB
data and [`phasepack`](https://github.com/WHOIGit/phasepack) for phase
congruency used during segmentation).

```bash
pip install git+https://github.com/WHOIGit/ifcb-features.git
```

Or, for local development:

```bash
git clone https://github.com/WHOIGit/ifcb-features.git
cd ifcb-features
pip install -e .
```

## Usage

The main entry point is [`extract_slim_features.py`](extract_slim_features.py).
It reads whole IFCB bins (via `pyifcb`), computes the per-ROI feature set, and
writes the results to disk:

```bash
python extract_slim_features.py <data_directory> <output_directory> [--bins BIN1 BIN2 ...]
```

- `data_directory` — directory of IFCB data (read via `pyifcb`).
- `output_directory` — where the outputs are written.
- `--bins` — optional list of bin names (e.g. `D20240423T115846_IFCB127`) to
  process; if omitted, every bin in the data directory is processed.

For each sample this produces two files in the output directory:

- `<bin>_features_v4.csv` — one row per ROI, with a `roi_number` column and one
  column per feature. See [FEATURES.md](FEATURES.md) for a description of each
  feature.
- `<bin>_blobs_v4.zip` — the segmented blob masks, one 1-bit PNG per ROI.

### Docker

A container image is built and published to the GitHub Container Registry, with
the batch extractor as its entry point:

```bash
docker run --rm \
  -v /path/to/ifcb/data:/data \
  -v /path/to/output:/output \
  ghcr.io/whoigit/ifcb-features:latest \
  /data /output --bins D20240423T115846_IFCB127
```

You can also build it locally:

```bash
docker build -t ifcb-features .
```

## License

MIT — see [LICENSE](LICENSE).

---

## Note: deprecated "non-slim" features

Earlier versions of this code also computed a larger set of features — among
them Histogram of Oriented Gradients (HOG), ring/wedge power spectra, invariant
moments, and texture and symmetry statistics. These are **deprecated** and
retained only for historical reasons; they are not part of the output of
`extract_slim_features.py`.

The underlying machinery still exists on the `RoiFeatures` and `BlobFeatures`
classes in [`ifcb_features/all.py`](ifcb_features/all.py) for anyone who needs
to reproduce older results, but new work should rely on the slim feature set
produced by the batch extractor.
