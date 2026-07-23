"""End-to-end regression test for the slim feature extractor.

Runs `extract_and_save_all_features` over the sample bin in tests/data and
compares both outputs against the committed golden baseline in tests/golden:

  * the features CSV, column by column, with a numeric tolerance
  * the blob masks in the zip, decoded to boolean arrays, exactly

The baseline is a snapshot of this implementation's output, not MATLAB
reference output -- these tests detect drift, they do not prove correctness.
Regenerate with `python tests/generate_golden.py` when a change is intended.
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from extract_slim_features import (
    FEATURE_COLUMNS,
    configure_output,
    extract_and_save_all_features,
)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(TESTS_DIR, 'data')
GOLDEN_DIR = os.path.join(TESTS_DIR, 'golden')

BIN_LID = 'IFCB5_2012_028_081515'
FEATURES_FILENAME = f'{BIN_LID}_features_v4.csv'
BLOBS_FILENAME = f'{BIN_LID}_blobs_v4.zip'

# Columns that are exact counts, compared without tolerance.
EXACT_COLUMNS = ['roi_number', 'numBlobs']
APPROX_COLUMNS = [c for c in FEATURE_COLUMNS if c not in EXACT_COLUMNS]

# Loose enough to absorb BLAS/platform float differences, tight enough that a
# real change in the segmentation or measurement code shows up.
RTOL = 1e-5
ATOL = 1e-8


def _read_features(path):
    """Read a features CSV as {column: float64 array}, preserving order."""
    df = pd.read_csv(path, dtype=str)
    return df.columns.tolist(), {
        column: df[column].astype(np.float64).to_numpy()
        for column in df.columns
    }


def _read_blobs(path):
    """Read a blobs zip as {member name: boolean mask array}."""
    masks = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with Image.open(io.BytesIO(zf.read(name))) as img:
                masks[name] = np.array(img) > 0
    return masks


@pytest.fixture(scope='session')
def produced_dir(tmp_path_factory):
    """Run the extractor once over the sample bin; yield the output directory."""
    configure_output(verbose=False)
    output_dir = tmp_path_factory.mktemp('features_output')
    extract_and_save_all_features(DATA_DIR, str(output_dir), bins=[BIN_LID])
    return output_dir


@pytest.fixture(scope='session')
def features(produced_dir):
    produced_path = produced_dir / FEATURES_FILENAME
    assert produced_path.is_file(), f'extractor did not write {FEATURES_FILENAME}'
    return (
        _read_features(str(produced_path)),
        _read_features(os.path.join(GOLDEN_DIR, FEATURES_FILENAME)),
    )


@pytest.fixture(scope='session')
def blobs(produced_dir):
    produced_path = produced_dir / BLOBS_FILENAME
    assert produced_path.is_file(), f'extractor did not write {BLOBS_FILENAME}'
    return (
        _read_blobs(str(produced_path)),
        _read_blobs(os.path.join(GOLDEN_DIR, BLOBS_FILENAME)),
    )


def test_columns_match_golden(features):
    (produced_columns, _), (golden_columns, _) = features
    assert produced_columns == golden_columns
    assert produced_columns == ['roi_number'] + FEATURE_COLUMNS


def test_row_count_matches_golden(features):
    (_, produced), (_, golden) = features
    assert len(produced['roi_number']) == len(golden['roi_number'])


@pytest.mark.parametrize('column', EXACT_COLUMNS)
def test_exact_column_matches_golden(features, column):
    (_, produced), (_, golden) = features
    np.testing.assert_array_equal(produced[column], golden[column])


@pytest.mark.parametrize('column', APPROX_COLUMNS)
def test_numeric_column_matches_golden(features, column):
    (_, produced), (_, golden) = features
    p, g = produced[column], golden[column]
    close = np.isclose(p, g, rtol=RTOL, atol=ATOL, equal_nan=True)
    if not close.all():
        rois = produced['roi_number'].astype(int)
        detail = '\n'.join(
            f'  roi {rois[i]}: produced={p[i]} golden={g[i]}'
            for i in np.flatnonzero(~close)
        )
        pytest.fail(
            f'{column} drifted from golden (rtol={RTOL}, atol={ATOL}):\n{detail}'
        )


def test_blob_members_match_golden(blobs):
    produced, golden = blobs
    assert sorted(produced) == sorted(golden)


def test_blob_masks_match_golden(blobs):
    produced, golden = blobs
    mismatched = []
    for name in sorted(golden):
        p, g = produced[name], golden[name]
        if p.shape != g.shape:
            mismatched.append(f'  {name}: shape {p.shape} != golden {g.shape}')
        elif not np.array_equal(p, g):
            mismatched.append(f'  {name}: {int((p != g).sum())} pixels differ')
    if mismatched:
        pytest.fail('blob masks drifted from golden:\n' + '\n'.join(mismatched))
