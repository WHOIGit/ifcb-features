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
from storage import S3Config, create_blob_storage

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

def extract_and_save_all_features(data_directory, output_directory, bins=None, storage_mode="local", s3_config=None):
    """
    Extracts slim features from IFCB images in the given directory
    and saves them to a CSV file.

    Args:
        data_directory (str): Path to the directory containing IFCB data.
        output_directory (str): Path to the directory where the CSV file will be saved.
        bins (list, optional): A list of bin names (e.g., 'D20240423T115846_IFCB127') to process.
            If None, all bins in the data directory are processed. Defaults to None.
        storage_mode (str): Storage mode - "local" or "s3". Defaults to "local".
        s3_config (S3Config, optional): S3 configuration when using S3 storage.
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

    # Validate storage configuration
    if storage_mode == "s3" and s3_config is None:
        raise ValueError("S3 configuration required for S3 storage mode")
    if storage_mode not in ["local", "s3"]:
        raise ValueError(f"Invalid storage mode '{storage_mode}'. Use 'local' or 's3'")

    # Create blob storage backend
    blob_storage = create_blob_storage(
        storage_mode=storage_mode,
        output_directory=output_directory,
        s3_config=s3_config
    )

    print(f"Using {storage_mode} storage mode")

    try:
        for sample in samples_to_process:
            all_features = []
            features_output_filename = os.path.join(output_directory, f"{sample.lid}_features_v4.csv")
        for number, image in sample.images.items():
            features = {
                'roi_number': number,
            }
            try:
                blobs_image, roi_features = compute_features(image)
                features.update(roi_features)

                # Store blob using the configured storage backend
                img_buffer = io.BytesIO()
                Image.fromarray((blobs_image > 0).astype(np.uint8) * 255).save(img_buffer, format="PNG")
                blob_data = img_buffer.getvalue()
                
                blob_storage.store_blob(sample.lid, number, blob_data)
                
            except Exception as e:
                print(f"Error processing ROI {number} in sample {sample.pid}: {e}")

            all_features.append(features)

        if all_features:
            df = pd.DataFrame.from_records(all_features, columns=['roi_number'] + FEATURE_COLUMNS)
            df.to_csv(features_output_filename, index=False)

        # Finalize blob storage for this sample
        blob_storage.finalize_sample(sample.lid)

    finally:
        # Cleanup storage resources
        blob_storage.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract various ROI features and save blobs as 1-bit PNGs.")
    parser.add_argument("data_directory", help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")
    parser.add_argument("--bins", nargs='+', help="List of bin names to process (space-separated). If not provided, all bins are processed.")
    
    # S3 storage options
    parser.add_argument("--storage-mode", choices=["local", "s3"], default="local", 
                       help="Storage mode for blob images (default: local)")
    parser.add_argument("--s3-bucket", help="S3 bucket name (required when storage-mode=s3)")
    parser.add_argument("--s3-url", help="S3 endpoint URL (required when storage-mode=s3)")
    parser.add_argument("--s3-prefix", default="ifcb-blobs/", 
                       help="S3 key prefix for blob storage (default: ifcb-blobs/)")

    args = parser.parse_args()
    
    # Set up S3 configuration if using S3 storage
    if args.storage_mode == "s3":
        if not args.s3_bucket or not args.s3_url:
            parser.error("--s3-bucket and --s3-url are required when using --storage-mode=s3")
        s3_config = S3Config(
            bucket_name=args.s3_bucket,
            s3_url=args.s3_url,
            prefix=args.s3_prefix
        )
    else:
        s3_config = None

    beginning = time.time()
    extract_and_save_all_features(args.data_directory, args.output_directory, args.bins, args.storage_mode, s3_config)
    elapsed = time.time() - beginning

    print(f'Total extract time: {elapsed:.2f} seconds')