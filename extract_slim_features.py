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
from tqdm import tqdm

from ifcb_features.all import compute_features
from ifcb_features.batch_processing import BatchedFeatureExtractor

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

def extract_and_save_all_features(data_directory, output_directory, bins=None, 
                                  batch_processing=False, min_batch_size=4, max_batch_size=64, 
                                  gpu_device=None):
    """
    Extracts slim features from IFCB images in the given directory
    and saves them to a CSV file.

    Args:
        data_directory (str): Path to the directory containing IFCB data.
        output_directory (str): Path to the directory where the CSV file will be saved.
        bins (list, optional): A list of bin names (e.g., 'D20240423T115846_IFCB127') to process.
            If None, all bins in the data directory are processed. Defaults to None.
        batch_processing (bool): Enable batched GPU processing for phase congruency.
        min_batch_size (int): Minimum ROIs needed to form a batch.
        max_batch_size (int): Maximum batch size for memory management.
        gpu_device (int): GPU device index to use (e.g., 0, 1, 2).
    """
    # Set GPU device if specified
    if gpu_device is not None:
        import os
        os.environ['IFCB_GPU_DEVICE'] = str(gpu_device)
        print(f"Using GPU device: {gpu_device}")
    
    # Initialize batch extractor if requested
    batch_extractor = None
    if batch_processing:
        try:
            batch_extractor = BatchedFeatureExtractor(
                min_batch_size=min_batch_size, 
                max_batch_size=max_batch_size
            )
            print(f"Batch processing enabled: batch_size={min_batch_size}-{max_batch_size}")
        except Exception as e:
            print(f"Warning: Could not initialize batch processing: {e}")
            print("Falling back to individual processing")
            batch_processing = False
    
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

    for i, sample in enumerate(tqdm(samples_to_process, desc="Processing samples")):
        print(f"Processing sample: {sample.lid}")
        features_output_filename = os.path.join(output_directory, f"{sample.lid}_features_v4.csv")
        blobs_output_filename = os.path.join(output_directory, f"{sample.lid}_blobs.zip")
        
        try:
            if batch_processing and batch_extractor:
                # Use batched processing
                all_features, all_blobs_raw = batch_extractor.process_sample_batched(
                    sample, compute_features
                )
                
                # Convert blob images to PNG format for storage
                all_blobs = {}
                for roi_number, blobs_image in all_blobs_raw.items():
                    img_buffer = io.BytesIO()
                    Image.fromarray((blobs_image > 0).astype(np.uint8) * 255).save(img_buffer, format="PNG")
                    all_blobs[roi_number] = img_buffer.getvalue()
                
            else:
                # Use original individual processing
                all_features = []
                all_blobs = {}
                with sample:  # Open ROI file
                    # Add progress bar for individual ROI processing
                    roi_items = list(sample.images.items())
                    for number, image in tqdm(roi_items, desc=f"Processing ROIs (individual)", leave=False):
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

            # Save results (same for both processing modes)
            if all_features:
                df = pd.DataFrame.from_records(all_features, columns=['roi_number'] + FEATURE_COLUMNS)
                df.to_csv(features_output_filename, index=False, float_format='%.8f')
            
            if all_blobs:
                with zipfile.ZipFile(blobs_output_filename, 'w') as zf:
                    for roi_number, blob_data in all_blobs.items():
                        filename = f"{sample.lid}_{roi_number:05d}.png"
                        zf.writestr(filename, blob_data)
                        
        except Exception as e:
            print(f"Error processing sample {sample.pid}: {e}")
            traceback.print_exc()
            continue
        
    
    # Print performance statistics if using batch processing
    if batch_processing and batch_extractor:
        stats = batch_extractor.get_performance_stats()
        print(f"\nBatch Processing Statistics:")
        print(f"  Total ROIs: {stats['total_rois']:,}")
        print(f"  Batched ROIs: {stats['batched_rois']:,} ({stats['batch_efficiency']:.1f}%)")
        print(f"  Individual ROIs: {stats['individual_rois']:,}")
        print(f"  Total batches: {stats['total_batches']:,}")
        print(f"  Average batch size: {stats['avg_batch_size']:.1f}")
        print(f"  Unique dimensions: {stats['unique_dimensions']}")
        print(f"\nDimension Distribution (top 25):")
        for dimension, count in stats['dimension_distribution'].items():
            print(f"    {dimension}: {count:,} ROIs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract various ROI features and save blobs as 1-bit PNGs.")
    parser.add_argument("data_directory", help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")
    parser.add_argument("--bins", nargs='+', help="List of bin names to process (space-separated). If not provided, all bins are processed.")
    
    # Batch processing options
    parser.add_argument("--batch-processing", action='store_true', 
                       help="Enable batched GPU processing for phase congruency (requires JAX)")
    parser.add_argument("--min-batch-size", type=int, default=4,
                       help="Minimum ROIs needed to form a batch (default: 4)")
    parser.add_argument("--max-batch-size", type=int, default=64,
                       help="Maximum batch size for memory management (default: 64)")
    parser.add_argument("--gpu-device", type=int,
                       help="GPU device index to use (e.g., 0, 1, 2)")

    args = parser.parse_args()

    beginning = time.time()
    extract_and_save_all_features(
        args.data_directory, 
        args.output_directory, 
        args.bins,
        batch_processing=args.batch_processing,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        gpu_device=args.gpu_device
    )
    elapsed = time.time() - beginning

    print(f'Total extract time: {elapsed:.2f} seconds')
