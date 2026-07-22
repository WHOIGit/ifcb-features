import argparse
import csv
import warnings
from ifcbkit import SyncIfcbDataDirectory, parse_pid
from ifcb_features import RoiFeatures
import os
import pandas as pd
import numpy as np
import io
import zipfile
import time
from PIL import Image
import traceback

from ifcb_features.all import compute_features


def configure_output(verbose):
    """Quiet by default: suppress the library's skimage/numpy warnings and
    silence numpy divide/invalid errors. --verbose restores everything."""
    if verbose:
        return
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    # numpy ComplexWarning (all.py:448) — location moved across numpy versions
    for exc in (getattr(np, "ComplexWarning", None),
                getattr(getattr(np, "exceptions", None), "ComplexWarning", None)):
        if exc is not None:
            warnings.filterwarnings("ignore", category=exc)
    np.seterr(divide="ignore", invalid="ignore")

FEATURE_COLUMNS = [
    'Area',
    'Biovolume',
    'BoundingBox_xwidth',
    'BoundingBox_ywidth',
    'ConvexArea',
    'ConvexPerimeter',
    'Eccentricity',
    'EquivDiameter',
    'Extent',
    'MajorAxisLength',
    'MinorAxisLength',
    'Orientation',
    'Perimeter',
    'RepresentativeWidth',
    'Solidity',
    'SurfaceArea',
    'maxFeretDiameter',
    'minFeretDiameter',
    'numBlobs',
    'summedArea',
    'summedBiovolume',
    'summedConvexArea',
    'summedConvexPerimeter',
    'summedMajorAxisLength',
    'summedMinorAxisLength',
    'summedPerimeter',
    'summedSurfaceArea',
    'Area_over_PerimeterSquared',
    'Area_over_Perimeter',
    'summedConvexPerimeter_over_Perimeter' 
]

def extract_and_save_all_features(data_directory, output_directory, bins=None, verbose=False):
    """
    Extracts slim features from IFCB images in the given directory
    and saves them to a CSV file.

    Args:
        data_directory (str): Path to the directory containing IFCB data.
        output_directory (str): Path to the directory where the CSV file will be saved.
        bins (list, optional): A list of bin names (e.g., 'D20240423T115846_IFCB127') to process.
            If None, all bins in the data directory are processed. Defaults to None.
    """
    if not os.path.isdir(data_directory):
        print(f"Error: Data directory not found at '{data_directory}'.")
        return
    try:
        data_dir = SyncIfcbDataDirectory(data_directory)
    except Exception as e:
        print(f"Error loading data directory: {e}")
        return

    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_directory}': {e}")
        return

    pids_to_process = []
    if bins:
        for bin_name in bins:
            try:
                if data_dir.exists(bin_name):
                    pids_to_process.append(bin_name)
                else:
                    print(f"Warning: Bin '{bin_name}' not found in data directory. Skipping.")
            except Exception as e:
                print(f"Error accessing bin '{bin_name}': {e}")
                traceback.print_exc()
    else:
        for fileset in data_dir.list():
            pids_to_process.append(fileset['pid'])

    for pid in pids_to_process:
        lid = parse_pid(pid)['lid']
        all_features = []
        all_blobs = {}
        features_output_filename = os.path.join(output_directory, f"{lid}_features_v4.csv")
        blobs_output_filename = os.path.join(output_directory, f"{lid}_blobs_v4.zip")
        for number, image in data_dir.read_images(pid).items():
            features = {
                'roi_number': number,
            }
            try:
                blobs_image, roi_features = compute_features(image)
                features.update(roi_features)

                img_buffer = io.BytesIO()
                Image.fromarray((blobs_image > 0).astype(np.uint8) * 255).save(img_buffer, format="PNG")
                all_blobs[number] = img_buffer.getvalue()
            except Exception as e:
                if verbose:
                    print(f"Error processing ROI {number} in sample {pid}: {e}")

            all_features.append(features)

        if all_features:
            df = pd.DataFrame.from_records(all_features, columns=['roi_number'] + FEATURE_COLUMNS)
            df.to_csv(features_output_filename, index=False, float_format="%.10g")

        if all_blobs:
            with zipfile.ZipFile(blobs_output_filename, 'w') as zf:
                for roi_number, blob_data in all_blobs.items():
                    filename = f"{lid}_{roi_number:05d}.png"
                    zf.writestr(filename, blob_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract various ROI features and save blobs as 1-bit PNGs.")
    parser.add_argument("data_directory", help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")
    parser.add_argument("--bins", nargs='+', help="List of bin names to process (space-separated). If not provided, all bins are processed.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Emit per-ROI error messages and library warnings (skimage/numpy). Quiet by default.")

    args = parser.parse_args()

    configure_output(args.verbose)

    beginning = time.time()
    extract_and_save_all_features(args.data_directory, args.output_directory, args.bins, verbose=args.verbose)
    elapsed = time.time() - beginning

    if args.verbose:
        print(f'Total extract time: {elapsed:.2f} seconds')
