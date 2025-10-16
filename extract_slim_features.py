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
from blob_storage import S3Config, create_blob_storage
from feature_storage import VastDBFeatureStorage, VastDBConfig

FEATURE_COLUMNS = [
    'sample_id',
    'roi_number',
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

def extract_and_save_all_features(data_directory, output_directory, bins=None, blob_storage_mode="local", s3_config=None, feature_storage_mode="local", vastdb_config=None):
    """
    Extracts slim features from IFCB images in the given directory
    and saves them to CSV or VastDB.

    Args:
        data_directory (str): Path to the directory containing IFCB data.
        output_directory (str): Path to the directory where the CSV file will be saved (if feature_storage_mode=local).
        bins (list, optional): A list of bin names (e.g., 'D20240423T115846_IFCB127') to process.
            If None, all bins in the data directory are processed. Defaults to None.
        blob_storage_mode (str): Storage mode for blobs - "local" or "s3". Defaults to "local".
        s3_config (S3Config, optional): S3 configuration when using S3 blob storage.
        feature_storage_mode (str): Storage mode for features - "local" or "vastdb". Defaults to "local".
        vastdb_config (VastDBConfig, optional): VastDB configuration when using VastDB feature storage.
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

    # Validate blob storage configuration
    if blob_storage_mode == "s3" and s3_config is None:
        raise ValueError("S3 configuration required for S3 blob storage mode")
    if blob_storage_mode not in ["local", "s3"]:
        raise ValueError(f"Invalid blob storage mode '{blob_storage_mode}'. Use 'local' or 's3'")

    # Validate feature storage configuration
    if feature_storage_mode == "vastdb" and vastdb_config is None:
        raise ValueError("VastDB configuration required for VastDB feature storage mode")
    if feature_storage_mode not in ["local", "vastdb"]:
        raise ValueError(f"Invalid feature storage mode '{feature_storage_mode}'. Use 'local' or 'vastdb'")

    # Create blob storage backend
    blob_storage = create_blob_storage(
        storage_mode=blob_storage_mode,
        output_directory=output_directory,
        s3_config=s3_config
    )

    print(f"Blob storage: {blob_storage_mode}")

    # Initialize feature storage
    vastdb_storage = None
    if feature_storage_mode == "vastdb":
        vastdb_storage = VastDBFeatureStorage(vastdb_config)
        print(f"Feature storage: vastdb ({vastdb_config.schema_name}.{vastdb_config.table_name})")
    else:
        print(f"Feature storage: local CSV")

    try:
        for sample in samples_to_process:
            all_features = []
            features_output_filename = os.path.join(output_directory, f"{sample.lid}_features_v4.csv")

            try:
                with sample:  # Open ROI file
                    for number, image in sample.images.items():
                        features = {
                            'sample_id': sample.lid,
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
                    df = pd.DataFrame.from_records(all_features, columns=FEATURE_COLUMNS)

                    # Save features based on storage mode
                    if feature_storage_mode == "local":
                        df.to_csv(features_output_filename, index=False)
                    elif feature_storage_mode == "vastdb":
                        vastdb_storage.insert_features(df)

                # Finalize blob storage for this sample
                blob_storage.finalize_sample(sample.lid)

            except Exception as e:
                print(f"Error processing sample {sample.pid}: {e}")
                traceback.print_exc()
                continue

    finally:
        # Cleanup storage resources
        blob_storage.cleanup()
        if vastdb_storage:
            vastdb_storage.cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract various ROI features and save blobs as 1-bit PNGs.")
    parser.add_argument("data_directory", help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")
    parser.add_argument("--bins", nargs='+', help="List of bin names to process (space-separated). If not provided, all bins are processed.")
    
    # Blob storage options
    parser.add_argument("--blob-storage-mode", choices=["local", "s3"], default="local",
                       help="Storage mode for blob images (default: local)")
    parser.add_argument("--s3-bucket", help="S3 bucket name (required when blob-storage-mode=s3)")
    parser.add_argument("--s3-url", help="S3 endpoint URL (required when blob-storage-mode=s3)")
    parser.add_argument("--s3-prefix", default="ifcb-blobs-slim-features/",
                       help="S3 key prefix for blob storage (default: ifcb-blobs-slim-features/)")

    # Feature storage options
    parser.add_argument("--feature-storage-mode", choices=["local", "vastdb"], default="local",
                       help="Storage mode for features (default: local)")
    parser.add_argument("--vastdb-bucket", help="VastDB bucket name (required when feature-storage-mode=vastdb)")
    parser.add_argument("--vastdb-schema", help="VastDB schema name (required when feature-storage-mode=vastdb)")
    parser.add_argument("--vastdb-table", help="VastDB table name (required when feature-storage-mode=vastdb)")
    parser.add_argument("--vastdb-url", help="VastDB endpoint URL (defaults to s3-url if not provided)")
    parser.add_argument("--vastdb-access-key", help="VastDB access key (uses AWS_ACCESS_KEY_ID env var if not provided)")
    parser.add_argument("--vastdb-secret-key", help="VastDB secret key (uses AWS_SECRET_ACCESS_KEY env var if not provided)")

    args = parser.parse_args()
    
    # Set up S3 configuration if using S3 blob storage
    if args.blob_storage_mode == "s3":
        if not args.s3_bucket or not args.s3_url:
            parser.error("--s3-bucket and --s3-url are required when using --blob-storage-mode=s3")
        s3_config = S3Config(
            bucket_name=args.s3_bucket,
            s3_url=args.s3_url,
            prefix=args.s3_prefix
        )
    else:
        s3_config = None

    # Set up VastDB configuration if using VastDB feature storage
    vastdb_config = None
    if args.feature_storage_mode == "vastdb":
        if not args.vastdb_bucket or not args.vastdb_schema or not args.vastdb_table:
            parser.error("--vastdb-bucket, --vastdb-schema, and --vastdb-table are required when using --feature-storage-mode=vastdb")

        # Use provided endpoint or fall back to S3 URL
        vastdb_url = args.vastdb_url or args.s3_url
        if not vastdb_url:
            parser.error("--vastdb-url or --s3-url must be provided when using --feature-storage-mode=vastdb")

        # Get credentials from args or environment
        access_key = args.vastdb_access_key or os.environ.get('AWS_ACCESS_KEY_ID')
        secret_key = args.vastdb_secret_key or os.environ.get('AWS_SECRET_ACCESS_KEY')

        if not access_key or not secret_key:
            parser.error("VastDB credentials required: provide --vastdb-access-key/--vastdb-secret-key or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables")

        vastdb_config = VastDBConfig(
            bucket_name=args.vastdb_bucket,
            schema_name=args.vastdb_schema,
            table_name=args.vastdb_table,
            endpoint_url=vastdb_url,
            access_key=access_key,
            secret_key=secret_key
        )

    beginning = time.time()
    extract_and_save_all_features(args.data_directory, args.output_directory, args.bins, args.blob_storage_mode, s3_config, args.feature_storage_mode, vastdb_config)
    elapsed = time.time() - beginning

    print(f'Total extract time: {elapsed:.2f} seconds')