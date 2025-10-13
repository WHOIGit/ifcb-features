"""
ROI batching infrastructure for GPU-accelerated phase congruency processing.

This module provides utilities for grouping ROIs by dimensions and processing
them in batches to maximize GPU utilization while maintaining mathematical
equivalence to individual processing.
"""

import numpy as np
# jax.numpy not needed at module level - JAX is imported in phasecong_Mm_batch when needed
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Iterator
import os
import logging
from tqdm import tqdm

# Configure JAX for GPU if available
try:
    import jax

    # Enable 64-bit precision in JAX (required for numerical accuracy)
    jax.config.update("jax_enable_x64", True)

    # Set GPU device if specified
    gpu_device = os.environ.get('IFCB_GPU_DEVICE', None)
    if gpu_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_device)
        jax.config.update('jax_platform_name', 'gpu')

    # Check if GPU is available
    try:
        devices = jax.devices('gpu')
        if devices:
            logging.info(f"JAX GPU devices available: {devices}")
        else:
            logging.info("No GPU devices found, using CPU")
    except:
        logging.info("GPU not available, using CPU")

except ImportError:
    logging.warning("JAX not available, falling back to CPU processing")

from .phasecong import phasecong_Mm_batch

class ROIBatcher:
    """
    Groups ROIs by identical dimensions for efficient batch processing.
    """
    
    def __init__(self, min_batch_size: int = 4, max_batch_size: int = 64):
        """
        Args:
            min_batch_size: Minimum number of ROIs needed to form a batch
            max_batch_size: Maximum batch size (for memory management)
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.roi_groups = defaultdict(list)  # {(height, width): [(roi, metadata), ...]}
        self.dimension_stats = Counter()
        
    def add_roi(self, roi: np.ndarray, metadata: dict = None):
        """
        Add a ROI to the appropriate dimension group.
        
        Args:
            roi: 2D numpy array representing the ROI image
            metadata: Optional metadata to associate with the ROI (e.g., roi_number, sample_id)
        """
        if roi.ndim != 2:
            raise ValueError(f"ROI must be 2D, got shape {roi.shape}")
        
        height, width = roi.shape
        dimension = (height, width)
        
        if metadata is None:
            metadata = {}
        
        roi_number = metadata.get('roi_number', 'unknown')
        # print(f"Debug: Adding ROI {roi_number} with shape {roi.shape} to dimension group {dimension}")
            
        self.roi_groups[dimension].append((roi, metadata))
        self.dimension_stats[dimension] += 1
        
    def get_batchable_stats(self) -> Dict:
        """
        Get statistics about batchable ROIs.
        
        Returns:
            Dictionary with batching statistics
        """
        total_rois = sum(self.dimension_stats.values())
        batchable_counts = {}
        
        for batch_size in [4, 8, 16, 32, 64, 128]:
            batchable_rois = sum(
                count for count in self.dimension_stats.values() 
                if count >= batch_size
            )
            batchable_counts[batch_size] = {
                'rois': batchable_rois,
                'percentage': batchable_rois / total_rois * 100 if total_rois > 0 else 0
            }
            
        return {
            'total_rois': total_rois,
            'unique_dimensions': len(self.dimension_stats),
            'batchable_by_size': batchable_counts,
            'dimension_distribution': dict(self.dimension_stats.most_common(25))
        }
        
    def get_batches(self) -> Iterator[Tuple[np.ndarray, List[dict]]]:
        """
        Yield batches of ROIs with identical dimensions.
        
        Yields:
            Tuple of (roi_batch, metadata_list) where:
            - roi_batch: np.ndarray of shape [batch_size, height, width]
            - metadata_list: List of metadata dicts corresponding to each ROI in the batch
        """
        for dimension, roi_list in self.roi_groups.items():
            height, width = dimension
            
            # Process ROIs in this dimension group
            i = 0
            while i < len(roi_list):
                # Determine batch size (don't exceed max_batch_size)
                remaining = len(roi_list) - i
                batch_size = min(remaining, self.max_batch_size)
                
                # Only create batch if we have enough ROIs or this is the last batch
                if batch_size >= self.min_batch_size or i + batch_size == len(roi_list):
                    # Extract ROIs and metadata for this batch
                    batch_rois = []
                    batch_metadata = []
                    
                    for j in range(i, i + batch_size):
                        roi, metadata = roi_list[j]
                        batch_rois.append(roi)
                        batch_metadata.append(metadata)
                    
                    # Stack ROIs into batch array
                    roi_batch = np.stack(batch_rois, axis=0)  # Shape: [batch_size, height, width]
                    yield roi_batch, batch_metadata
                    
                i += batch_size
        
        # Clear processed groups
        self.roi_groups.clear()
        self.dimension_stats.clear()

class BatchedFeatureExtractor:
    """
    Coordinates batched ROI feature extraction for improved GPU utilization.
    """
    
    def __init__(self, min_batch_size: int = 4, max_batch_size: int = 64, 
                 enable_cross_sample: bool = False):
        """
        Args:
            min_batch_size: Minimum ROIs needed to form a batch
            max_batch_size: Maximum batch size for memory management
            enable_cross_sample: If True, batch ROIs across multiple samples
        """
        self.batcher = ROIBatcher(min_batch_size, max_batch_size)
        self.enable_cross_sample = enable_cross_sample
        self.stats = {
            'batched_rois': 0,
            'individual_rois': 0,
            'total_batches': 0
        }
        
    def process_sample_batched(self, sample, compute_features_func) -> Tuple[List, Dict]:
        """
        Process all ROIs in a sample using batched phase congruency.
        
        Args:
            sample: IFCB sample object with .images attribute
            compute_features_func: Function to compute features from (blobs_image, roi)
            
        Returns:
            Tuple of (all_features, all_blobs) matching original extract_slim_features format
        """
        # Collect all ROIs from the sample first
        sample_rois = []
        
        with sample:  # Open ROI file
            # Add progress bar for ROI loading
            roi_items = list(sample.images.items())
            for roi_number, roi_image in tqdm(roi_items, desc="Loading ROIs", leave=False, mininterval=1.0, ncols=80):
                roi_array = np.asarray(roi_image, dtype=np.float32)
                metadata = {
                    'roi_number': roi_number,
                    'sample_id': sample.pid,
                    'original_index': len(sample_rois)
                }
                sample_rois.append((roi_array, metadata))
                self.batcher.add_roi(roi_array, metadata)
        
        # Process in batches and collect results
        all_features = [None] * len(sample_rois)  # Pre-allocate to maintain order
        all_blobs = {}
        
        # Process batches
        # Convert to list to get total count for progress bar
        batch_list = list(self.batcher.get_batches())
        for roi_batch, metadata_batch in tqdm(batch_list, desc="Processing batches", leave=False, mininterval=1.0, ncols=80):
            batch_size = len(metadata_batch)
            
            if batch_size >= self.batcher.min_batch_size:
                # Process as batch
                Mm_batch = phasecong_Mm_batch(roi_batch)  # Shape: [batch_size, height, width]
                self.stats['batched_rois'] += batch_size
                self.stats['total_batches'] += 1

                # Process each ROI result in the batch
                for i, metadata in enumerate(metadata_batch):
                    roi_number = metadata['roi_number']
                    original_index = metadata['original_index']
                    original_roi = sample_rois[original_index][0]

                    # Index into the batch explicitly (JAX arrays don't iterate well with zip)
                    Mm_result = Mm_batch[i]

                    # Convert JAX result back to numpy for downstream processing
                    Mm_np = np.asarray(Mm_result)
                    # Ensure it's 2D (squeeze any extra dimensions from batch of size 1)
                    while Mm_np.ndim > 2:
                        Mm_np = np.squeeze(Mm_np, axis=0)
                    
                    # Use the batched compute_features function with precomputed phase congruency
                    blobs_image, roi_features = compute_features_with_batched_phasecong(
                        original_roi, precomputed_Mm=Mm_np
                    )
                    
                    # Store results in original order
                    features = {'roi_number': roi_number}
                    features.update(roi_features)
                    all_features[original_index] = features
                    all_blobs[roi_number] = blobs_image
            else:
                # Process individually (fallback for small groups)
                for i, (roi, metadata) in enumerate(zip(roi_batch, metadata_batch)):
                    roi_number = metadata['roi_number']
                    original_index = metadata['original_index']
                    
                    blobs_image, roi_features = compute_features_with_batched_phasecong(roi)
                    
                    features = {'roi_number': roi_number}
                    features.update(roi_features)
                    all_features[original_index] = features
                    all_blobs[roi_number] = blobs_image
                    
                self.stats['individual_rois'] += len(metadata_batch)
        
        return all_features, all_blobs
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for this extraction session."""
        total_rois = self.stats['batched_rois'] + self.stats['individual_rois']
        batcher_stats = self.batcher.get_batchable_stats()
        return {
            **self.stats,
            'total_rois': total_rois,
            'batch_efficiency': self.stats['batched_rois'] / total_rois * 100 if total_rois > 0 else 0,
            'avg_batch_size': self.stats['batched_rois'] / self.stats['total_batches'] if self.stats['total_batches'] > 0 else 0,
            'dimension_distribution': batcher_stats['dimension_distribution'],
            'unique_dimensions': batcher_stats['unique_dimensions']
        }

def compute_features_with_batched_phasecong(roi_image, precomputed_Mm=None, raw_stitch=None):
    """
    Modified compute_features function that can use pre-computed phase congruency results.
    
    This allows us to batch the expensive phase congruency computation while keeping
    the rest of the feature extraction pipeline unchanged.
    
    Args:
        roi_image: Original ROI image  
        precomputed_Mm: Pre-computed phase congruency result (M + m)
        raw_stitch: Optional stitch mask
        
    Returns:
        Tuple of (blobs_image, roi_features) matching compute_features format
    """
    from .all import compute_features, RoiFeatures
    from .segmentation import segment_roi_with_precomputed_Mm
    
    if precomputed_Mm is not None:
        # Use pre-computed phase congruency result for segmentation
        blobs_image = segment_roi_with_precomputed_Mm(roi_image, precomputed_Mm, raw_stitch)
        
        # Create RoiFeatures object with pre-computed blobs
        roi_features = RoiFeatures(roi_image, blobs_image, raw_stitch)
        
        # Extract features using the same logic as compute_features
        if roi_features.num_blobs > 0:
            b = roi_features.blobs[0]
        else:
            from .all import ZeroMock
            b = ZeroMock()
            
        feature_dict = [
            ('Area', b.area),
            ('Biovolume', b.biovolume),
            ('BoundingBox_xwidth', b.bbox_xwidth),
            ('BoundingBox_ywidth', b.bbox_ywidth),
            ('ConvexArea', b.convex_area),
            ('ConvexPerimeter', b.convex_perimeter),
            ('Eccentricity', b.eccentricity),
            ('EquivDiameter', b.equiv_diameter),
            ('Extent', b.extent),
            ('MajorAxisLength', b.major_axis_length),
            ('MinorAxisLength', b.minor_axis_length),
            ('Orientation', b.orientation),
            ('Perimeter', b.perimeter),
            ('RepresentativeWidth', b.representative_width),
            ('Solidity', b.solidity),
            ('SurfaceArea', b.surface_area),
            ('maxFeretDiameter', b.max_feret_diameter),
            ('minFeretDiameter', b.min_feret_diameter),
            ('numBlobs', roi_features.num_blobs),
            ('summedArea', roi_features.summed_area),
            ('summedBiovolume', roi_features.summed_biovolume),
            ('summedConvexArea', roi_features.summed_convex_area),
            ('summedConvexPerimeter', roi_features.summed_convex_perimeter),
            ('summedMajorAxisLength', roi_features.summed_major_axis_length),
            ('summedMinorAxisLength', roi_features.summed_minor_axis_length),
            ('summedPerimeter', roi_features.summed_perimeter),
            ('summedSurfaceArea', roi_features.summed_surface_area),
            ('Area_over_PerimeterSquared', b.area_over_perimeter_squared),
            ('Area_over_Perimeter', b.area_over_perimeter),
            ('summedConvexPerimeter_over_Perimeter', roi_features.summed_convex_perimeter_over_perimeter),
        ]
        
        return blobs_image, dict(feature_dict)
    else:
        # Fall back to original compute_features
        return compute_features(roi_image, raw_stitch=raw_stitch)