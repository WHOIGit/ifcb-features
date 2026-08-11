"""Batch feature extraction with MATLAB-compatible output directory structure.

Organizes outputs into day-based subdirectories matching the MATLAB pipeline:

    <output_directory>/
        features/<day>/<bin>_features_v4.csv
        features/<day>/multiblob/<bin>_multiblob_v4.csv
        blobs/<day>/<bin>_blobs_v4.zip

Usage:
    python extract_features_batch.py data_dir output_dir [--workers 4] [--bins BIN1 BIN2 ...]
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from ifcbkit import SyncIfcbDataDirectory, parse_pid

from extract_slim_features import (
    configure_output,
    extract_and_save_all_features,
)


def day_dir(lid):
    """First 9 characters of the bin LID, e.g. 'D20230109'."""
    return lid[:9]


def process_bin(data_directory, output_directory, bin_name, verbose=False):
    lid = parse_pid(bin_name)['lid']
    day = day_dir(lid)

    features_dir = os.path.join(output_directory, 'features', day)
    blobs_dir = os.path.join(output_directory, 'blobs', day)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(blobs_dir, exist_ok=True)

    extract_and_save_all_features(
        data_directory, features_dir, bins=[lid], verbose=verbose,
    )

    # Move blobs ZIP from features dir to blobs dir
    blobs_src = os.path.join(features_dir, f'{lid}_blobs_v4.zip')
    if os.path.exists(blobs_src):
        shutil.move(blobs_src, os.path.join(blobs_dir, f'{lid}_blobs_v4.zip'))

    return lid


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('data_directory', help='Path to the directory containing IFCB data.')
    parser.add_argument('output_directory', help='Root output directory.')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel worker processes (default: 4).')
    parser.add_argument('--bins', nargs='+', help='Bin names to process. If omitted, all bins are processed.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit per-ROI error messages and library warnings.')
    args = parser.parse_args()

    configure_output(args.verbose)

    if args.bins:
        bin_names = args.bins
    else:
        data_dir = SyncIfcbDataDirectory(args.data_directory)
        bin_names = [fileset['pid'] for fileset in data_dir.list()]

    print(f'Processing {len(bin_names)} bins with {args.workers} workers.')
    beginning = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_bin, args.data_directory, args.output_directory, name, args.verbose): name
            for name in bin_names
        }
        completed = 0
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f'Error processing bin {name!r}: {e}')
            completed += 1
            print(f'[{completed}/{len(bin_names)}] {name}')

    elapsed = time.time() - beginning
    print(f'Total extract time: {elapsed:.2f} seconds')


if __name__ == '__main__':
    main()
