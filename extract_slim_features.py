import argparse
import csv
from ifcb import DataDirectory
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

def extract_and_save_all_features(data_directory, output_directory, bins=None):
    """
    Extracts slim features from IFCB images in the given directory
    and saves them to a CSV file.

    Args:
        data_directory (str): Path to the directory containing IFCB data.
        output_directory (str): Path to the directory where the CSV file will be saved.
        bins (list, optional): A list of bin names (e.g., 'D20240423T115846_IFCB127') to process.
            If None, all bins in the data directory are processed. Defaults to None.
    """
    try:
        data_dir = DataDirectory(data_directory)
    except FileNotFoundError:
        print(f"Error: Data directory not found at '{data_directory}'.")
        return
    except Exception as e:
        print(f"Error loading data directory: {e}")
        return

    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_directory}': {e}")
        return

    samples_to_process = []
    if bins:
        for bin_name in bins:
            try:
                sample = data_dir[bin_name]
                samples_to_process.append(sample)
            except KeyError:
                print(f"Warning: Bin '{bin_name}' not found in data directory. Skipping.")
            except Exception as e:
                print(f"Error accessing bin '{bin_name}': {e}")
                traceback.print_exc()
    else:
        for sample in data_dir:
            samples_to_process.append(sample)

    for sample in samples_to_process:
        all_features = []
        all_blobs = {}
        features_output_filename = os.path.join(output_directory, f"{sample.lid}_features_v4.csv")
        blobs_output_filename = os.path.join(output_directory, f"{sample.lid}_blobs.zip")
        for number, image in sample.images.items():
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
                print(f"Error processing ROI {number} in sample {sample.pid}: {e}")

            all_features.append(features)

        if all_features:
            df = pd.DataFrame.from_records(all_features, columns=['roi_number'] + FEATURE_COLUMNS)
            df.to_csv(features_output_filename, index=False)
        
        if all_blobs:
            with zipfile.ZipFile(blobs_output_filename, 'w') as zf:
                for roi_number, blob_data in all_blobs.items():
                    filename = f"{sample.lid}_{roi_number:05d}.png"
                    zf.writestr(filename, blob_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract various ROI features and save blobs as 1-bit PNGs.")
    parser.add_argument("data_directory", help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")
    parser.add_argument("--bins", nargs='+', help="List of bin names to process (space-separated). If not provided, all bins are processed.")

    args = parser.parse_args()

    beginning = time.time()
    extract_and_save_all_features(args.data_directory, args.output_directory, args.bins)
    elapsed = time.time() - beginning

    print(f'Total extract time: {elapsed:.2f} seconds')