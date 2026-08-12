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
from GitHub ([`ifcbkit`](https://github.com/WHOIGit/ifcbkit) for reading IFCB
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

The main entry point is
[`extract_features_batch.py`](extract_features_batch.py). It reads IFCB bins
(via `ifcbkit`), computes the per-ROI feature set in parallel, and writes the
results into a MATLAB-compatible directory structure:

```bash
python extract_features_batch.py <data_directory> <output_directory> [--workers 4] [--bins BIN1 BIN2 ...]
```

- `data_directory` — directory of IFCB data (read via `ifcbkit`).
- `output_directory` — root directory for outputs.
- `--workers` — number of parallel worker processes (default: 4).
- `--bins` — optional list of bin names (e.g. `D20240423T115846_IFCB127`) to
  process; if omitted, every bin in the data directory is processed.

Outputs are organized into day-based subdirectories:

```
<output_directory>/
    features/<day>/<bin>_features_v4.csv
    features/<day>/multiblob/<bin>_multiblob_v4.csv
    blobs/<day>/<bin>_blobs_v4.zip
```

- `<bin>_features_v4.csv` — one row per ROI, with a `roi_number` column and one
  column per feature. See [FEATURES.md](FEATURES.md) for a description of each
  feature.
- `<bin>_blobs_v4.zip` — the segmented blob masks, one 1-bit PNG per ROI.
- `<bin>_multiblob_v4.csv` — per-blob features for ROIs containing more than one
  blob, with `roi_number`, `blob_number`, and individual blob measurements.

For simpler use cases that don't need the directory structure,
[`extract_slim_features.py`](extract_slim_features.py) writes all outputs flat
into a single directory:

```bash
python extract_slim_features.py <data_directory> <output_directory> [--bins BIN1 BIN2 ...]
```

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

## Tests

The test suite is a regression test: it runs the extractor over a small sample
bin committed in `tests/data` and compares both outputs against a golden
baseline in `tests/golden` — the features CSV column by column with a numeric
tolerance, and the blob masks pixel for pixel.

```bash
pip install -e ".[test]"
pytest
```

The baseline is a snapshot of this implementation's own output, not MATLAB
reference output, so the tests detect drift rather than proving correctness.
When a numeric change is intended, regenerate the baseline and review the diff
before committing it:

```bash
python tests/generate_golden.py
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
