#!/usr/bin/env python
"""Regenerate the golden regression outputs.

Runs the feature extractor over the sample bin in tests/data and writes
`<lid>_features_v4.csv` and `<lid>_blobs_v4.zip` into tests/golden, overwriting
whatever is there.

Run this ONLY when a numeric change is intended, and review the resulting diff
before committing it -- the golden files are the regression baseline.

    python tests/generate_golden.py
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_ROOT)

from extract_slim_features import configure_output, extract_and_save_all_features

DATA_DIR = os.path.join(TESTS_DIR, 'data')
GOLDEN_DIR = os.path.join(TESTS_DIR, 'golden')


def main():
    configure_output(verbose=False)
    extract_and_save_all_features(DATA_DIR, GOLDEN_DIR)
    for name in sorted(os.listdir(GOLDEN_DIR)):
        print(f'wrote {os.path.join("tests", "golden", name)}')


if __name__ == '__main__':
    main()
