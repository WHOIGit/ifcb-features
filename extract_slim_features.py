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
from ifcb_features.batch_processing import BatchedFeatureExtractor, CrossSampleBatchedFeatureExtractor

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
                                  batch_processing=False, batch_strategy='within-sample',
                                  min_batch_size=4, max_batch_size=64, gpu_device=None,
                                  accumulation_samples=10, max_memory_mb=1000):
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
    
    # Initialize batch extractor based on strategy
    batch_extractor = None
    if batch_processing:
        try:
            if batch_strategy == 'cross-sample':
                batch_extractor = CrossSampleBatchedFeatureExtractor(
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    accumulation_samples=accumulation_samples,
                    max_memory_mb=max_memory_mb
                )
                print(f"Cross-sample batch processing enabled: batch_size={min_batch_size}-{max_batch_size}, accumulation={accumulation_samples} samples")
            else:  # within-sample
                batch_extractor = BatchedFeatureExtractor(
                    min_batch_size=min_batch_size, 
                    max_batch_size=max_batch_size
                )
                print(f"Within-sample batch processing enabled: batch_size={min_batch_size}-{max_batch_size}")
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

    # Choose processing strategy
    if batch_processing and batch_extractor and batch_strategy == 'cross-sample':
        # Cross-sample strategy: accumulate across samples, then process in large batches
        for i, sample in enumerate(tqdm(samples_to_process, desc="Accumulating samples")):
            print(f"Accumulating sample: {sample.lid}")
            features_output_filename = os.path.join(output_directory, f"{sample.lid}_features_v4.csv")
            blobs_output_filename = os.path.join(output_directory, f"{sample.lid}_blobs.zip")
            
            try:
                # Accumulate ROIs from this sample
                roi_count = batch_extractor.accumulate_sample(sample, features_output_filename, blobs_output_filename)
                print(f"  Accumulated {roi_count} ROIs from {sample.lid}")
                
                # Process accumulated batches if threshold reached
                if batch_extractor.should_process_batches():
                    print(f"Processing accumulated batches (triggered after {batch_extractor.accumulated_samples} samples)")
                    batch_extractor.process_accumulated_batches()
                    
            except Exception as e:
                print(f"Error accumulating sample {sample.pid}: {e}")
                traceback.print_exc()
                continue
                
        # Process any remaining accumulated batches
        if batch_extractor.accumulated_samples > 0:
            print(f"Processing final accumulated batches ({batch_extractor.accumulated_samples} samples)")
            batch_extractor.process_accumulated_batches()
            
        # Finalize all remaining results
        batch_extractor.finalize_all_results()
        
    else:
        # Within-sample strategy or individual processing
        for i, sample in enumerate(tqdm(samples_to_process, desc="Processing samples")):
            print(f"Processing sample: {sample.lid}")
            features_output_filename = os.path.join(output_directory, f"{sample.lid}_features_v4.csv")
            blobs_output_filename = os.path.join(output_directory, f"{sample.lid}_blobs.zip")
            
            try:
                if batch_processing and batch_extractor:
                    # Use within-sample batched processing
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

                # Save results (for within-sample strategy)
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
        strategy_name = "Cross-Sample" if batch_strategy == 'cross-sample' else "Within-Sample"
        print(f"\n{strategy_name} Batch Processing Statistics:")
        print(f"  Total ROIs: {stats['total_rois']:,}")
        print(f"  Batched ROIs: {stats['batched_rois']:,} ({stats['batch_efficiency']:.1f}%)")
        print(f"  Individual ROIs: {stats['individual_rois']:,}")
        print(f"  Total batches: {stats['total_batches']:,}")
        print(f"  Average batch size: {stats['avg_batch_size']:.1f}")
        print(f"  Unique dimensions: {stats['unique_dimensions']}")
        
        # Cross-sample specific stats
        if batch_strategy == 'cross-sample':
            print(f"  Completed samples: {stats['completed_samples']:,}")
            print(f"  Memory usage: {stats['memory_usage_mb']:.1f} MB")
        
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
    parser.add_argument("--batch-strategy", 
                       choices=['within-sample', 'cross-sample'],
                       default='within-sample',
                       help="Batching strategy: within-sample (default) processes each sample independently, cross-sample accumulates ROIs across samples for larger batches")
    parser.add_argument("--min-batch-size", type=int, default=4,
                       help="Minimum ROIs needed to form a batch (default: 4)")
    parser.add_argument("--max-batch-size", type=int, default=64,
                       help="Maximum batch size for memory management (default: 64)")
    parser.add_argument("--gpu-device", type=int,
                       help="GPU device index to use (e.g., 0, 1, 2)")
    
    # Cross-sample specific options
    parser.add_argument("--accumulation-samples", type=int, default=10,
                       help="For cross-sample strategy: number of samples to accumulate before processing (default: 10)")
    parser.add_argument("--max-memory-mb", type=int, default=1000,
                       help="For cross-sample strategy: memory limit in MB for accumulated ROIs/results (default: 1000)")

    args = parser.parse_args()
    
    # Adjust batch size defaults for cross-sample strategy
    min_batch_size = args.min_batch_size
    max_batch_size = args.max_batch_size
    
    if args.batch_processing and args.batch_strategy == 'cross-sample':
        # Use larger default batch sizes for cross-sample strategy if not explicitly set
        if args.min_batch_size == 4:  # Default value, user didn't override
            min_batch_size = 32
        if args.max_batch_size == 64:  # Default value, user didn't override
            max_batch_size = 256

    beginning = time.time()
    extract_and_save_all_features(
        args.data_directory, 
        args.output_directory, 
        args.bins,
        batch_processing=args.batch_processing,
        batch_strategy=args.batch_strategy,
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        gpu_device=args.gpu_device,
        accumulation_samples=args.accumulation_samples,
        max_memory_mb=args.max_memory_mb
    )
    elapsed = time.time() - beginning

    print(f'Total extract time: {elapsed:.2f} seconds')
