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
import imageio
import traceback

def extract_and_save_all_features(data_directory, output_directory):
    """
    Extracts slim features from IFCB images in the given directory
    and saves them to a CSV file.

    Args:
        data_directory (str): Path to the directory containing IFCB data.
        output_directory (str): Path to the directory where the CSV file will be saved.
    """
    try:
        data_dir = DataDirectory(data_directory)
    except FileNotFoundError:
        print(f"Error: Data directory not found at '{data_directory}'.")
        return
    except Exception as e:
        print(f"Error loading data directory: {e}")
        return

    output_filename = os.path.join(output_directory, "all_roi_features.csv")
    blobs_output_filename = os.path.join(output_directory, "blobs.zip")

    try:
        os.makedirs(output_directory, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_directory}': {e}")
        return

    all_features_data = []
    blobs_data = {}

    for sample in data_dir:
        print(f"Processing sample: {sample}, {len(sample.images)} ROIs")
        for number, image in sample.images.items():
            try:
                roi_features = RoiFeatures(image)
                features = {
                    'sample_name': sample.pid,
                    'roi_number': number,
                    'summed_feret_diameter': f"{roi_features.summed_feret_diameter:.2f}",
                    # 'BoundingBox_xwidth': roi_features.bbox_xwidth,
                    # 'BoundingBox_ywidth': roi_features.bbox_ywidth,
                    # Add other features
                }
                all_features_data.append(features)

                blobs = roi_features.blobs
                if blobs:
                    blobs_data_sample = {}
                    for i, blob in enumerate(blobs):
                        blob_image = (blob.image > 0).astype(np.uint8)
                        img_buffer = io.BytesIO()
                        try:
                            imageio.imwrite(img_buffer, blob_image, format="png")
                        except Exception as e:
                            print(f"Error saving blob {i} of ROI {number} in sample {sample.pid}: {e}")
                            traceback.print_exc()
                        blob_filename = f"{str(sample.pid)}_roi_{number}_blob_{i+1}.png"
                        blobs_data_sample[blob_filename] = img_buffer.getvalue()
                    blobs_data[(str(sample.pid), number)] = blobs_data_sample
            except Exception as e:
                print(f"Error processing ROI {number} in sample {sample.pid}: {e}")

    if all_features_data:
        df = pd.DataFrame(all_features_data)
        df.to_csv(output_filename, index=False)
        print(f"\nAll features saved to: {output_filename}")
    else:
        print("\nNo features were extracted.")
    
    if blobs_data:
        with zipfile.ZipFile(blobs_output_filename, 'w') as zf:
            for (sample_pid, roi_number), blob_files in blobs_data.items():
                for filename, data in blob_files.items():
                    zf.writestr(filename, data)
        print(f"Blobs saved as 1-bit PNGs (using scikit-image) in: {blobs_output_filename}")
    else:
        print("No blobs were segmented to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract various ROI features and save blobs as 1-bit PNGs.")
    parser.add_argument("data_directory", help="Path to the directory containing IFCB data.")
    parser.add_argument("output_directory", help="Path to the directory to save the output CSV file and blobs.")

    args = parser.parse_args()

    beginning = time.time()
    extract_and_save_all_features(args.data_directory, args.output_directory)
    elapsed = time.time() - beginning

    print(f'Total extract time: {elapsed:.2f} seconds')