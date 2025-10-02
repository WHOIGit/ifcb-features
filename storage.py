from abc import ABC, abstractmethod
import zipfile
import os
import io
import boto3
import botocore
from typing import Dict, Any
import traceback
from dataclasses import dataclass


@dataclass
class S3Config:
    """Configuration for S3 blob storage."""
    bucket_name: str
    s3_url: str
    prefix: str = "ifcb-blobs/"


class BlobStorage(ABC):
    """Abstract interface for blob storage backends."""
    
    @abstractmethod
    def store_blob(self, sample_id: str, roi_number: int, blob_data: bytes) -> bool:
        """Store a single blob."""
        pass
    
    @abstractmethod
    def finalize_sample(self, sample_id: str) -> bool:
        """Finalize storage for a sample (e.g., close ZIP file)."""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup resources."""
        pass


class LocalZipStorage(BlobStorage):
    """Local ZIP file storage backend (original behavior)."""
    
    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.zip_files: Dict[str, zipfile.ZipFile] = {}
        self.blob_counts: Dict[str, int] = {}
    
    def store_blob(self, sample_id: str, roi_number: int, blob_data: bytes) -> bool:
        """Store a blob in the ZIP file for this sample."""
        try:
            if sample_id not in self.zip_files:
                zip_filename = os.path.join(self.output_directory, f"{sample_id}_blobs_v4.zip")
                self.zip_files[sample_id] = zipfile.ZipFile(zip_filename, 'w')
                self.blob_counts[sample_id] = 0
            
            filename = f"{sample_id}_{roi_number:05d}.png"
            self.zip_files[sample_id].writestr(filename, blob_data)
            self.blob_counts[sample_id] += 1
            return True
            
        except Exception as e:
            print(f"Error storing blob {roi_number} for sample {sample_id}: {e}")
            return False
    
    def finalize_sample(self, sample_id: str) -> bool:
        """Close the ZIP file for this sample."""
        if sample_id in self.zip_files:
            try:
                self.zip_files[sample_id].close()
                print(f"Stored {self.blob_counts[sample_id]} blobs for sample {sample_id} in ZIP")
                del self.zip_files[sample_id]
                del self.blob_counts[sample_id]
                return True
            except Exception as e:
                print(f"Error finalizing ZIP for sample {sample_id}: {e}")
                return False
        return True
    
    def cleanup(self):
        """Close any remaining ZIP files."""
        for sample_id in list(self.zip_files.keys()):
            self.finalize_sample(sample_id)


class S3BlobStorage(BlobStorage):
    """S3 storage backend using boto3."""
    
    def __init__(self, s3_config: S3Config):
        self.config = s3_config
        self.s3_client = None
        self.blob_counts: Dict[str, int] = {}
        self._setup_s3_client()
    
    def _setup_s3_client(self):
        """Initialize S3 client."""
        try:
            session = boto3.Session()
            self.s3_client = session.client(
                's3',
                endpoint_url=self.config.s3_url
            )
            print(f"Connected to S3 at {self.config.s3_url}")
        except Exception as e:
            print(f"Failed to setup S3 client: {e}")
            raise
    
    def store_blob(self, sample_id: str, roi_number: int, blob_data: bytes) -> bool:
        """Store a blob in S3."""
        try:
            if sample_id not in self.blob_counts:
                self.blob_counts[sample_id] = 0
            
            key = f"{self.config.prefix}{sample_id}/{roi_number:05d}.png"
            
            self.s3_client.put_object(
                Bucket=self.config.bucket_name,
                Key=key,
                Body=blob_data,
                ContentType='image/png'
            )
            
            self.blob_counts[sample_id] += 1
            return True
            
        except Exception as e:
            print(f"Error storing blob {roi_number} for sample {sample_id} to S3: {e}")
            traceback.print_exc()
            return False
    
    def finalize_sample(self, sample_id: str) -> bool:
        """Log completion for this sample."""
        if sample_id in self.blob_counts:
            print(f"Stored {self.blob_counts[sample_id]} blobs for sample {sample_id} in S3")
            del self.blob_counts[sample_id]
        return True
    
    def cleanup(self):
        """Close S3 client."""
        if self.s3_client:
            try:
                self.s3_client.close()
            except:
                pass


def create_blob_storage(storage_mode: str, output_directory: str, s3_config: S3Config = None) -> BlobStorage:
    """Factory function to create appropriate blob storage backend."""
    if storage_mode == "local":
        return LocalZipStorage(output_directory)
    elif storage_mode == "s3":
        if s3_config is None:
            raise ValueError("S3 configuration required for S3 storage mode")
        return S3BlobStorage(s3_config)
    else:
        raise ValueError(f"Unknown storage mode: {storage_mode}. Use 'local' or 's3'")
