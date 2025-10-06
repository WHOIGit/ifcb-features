"""VastDB storage for IFCB features."""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import pyarrow as pa
import vastdb
from blob_storage import S3Config


@dataclass
class VastDBConfig:
    """Configuration for VastDB feature storage."""
    bucket_name: str
    schema_name: str
    table_name: str
    endpoint_url: str
    access_key: str
    secret_key: str


class VastDBFeatureStorage:
    """VastDB storage backend for features."""

    def __init__(self, config: VastDBConfig):
        self.config = config
        self.session = None
        self._setup_session()

    def _setup_session(self):
        """Initialize VastDB session."""
        try:
            self.session = vastdb.connect(
                endpoint=self.config.endpoint_url,
                access=self.config.access_key,
                secret=self.config.secret_key
            )
            print(f"Connected to VastDB at {self.config.endpoint_url}")
        except Exception as e:
            print(f"Failed to connect to VastDB: {e}")
            raise

    def _get_features_schema(self) -> pa.Schema:
        """Define PyArrow schema for features table."""
        # Composite key: sample_id (string) + roi_number (int64)
        # All feature columns are float64
        return pa.schema([
            ('sample_id', pa.string()),
            ('roi_number', pa.int64()),
            ('Area', pa.float64()),
            ('Biovolume', pa.float64()),
            ('BoundingBox_xwidth', pa.float64()),
            ('BoundingBox_ywidth', pa.float64()),
            ('ConvexArea', pa.float64()),
            ('ConvexPerimeter', pa.float64()),
            ('Eccentricity', pa.float64()),
            ('EquivDiameter', pa.float64()),
            ('Extent', pa.float64()),
            ('MajorAxisLength', pa.float64()),
            ('MinorAxisLength', pa.float64()),
            ('Orientation', pa.float64()),
            ('Perimeter', pa.float64()),
            ('RepresentativeWidth', pa.float64()),
            ('Solidity', pa.float64()),
            ('SurfaceArea', pa.float64()),
            ('maxFeretDiameter', pa.float64()),
            ('minFeretDiameter', pa.float64()),
            ('numBlobs', pa.float64()),
            ('summedArea', pa.float64()),
            ('summedBiovolume', pa.float64()),
            ('summedConvexArea', pa.float64()),
            ('summedConvexPerimeter', pa.float64()),
            ('summedMajorAxisLength', pa.float64()),
            ('summedMinorAxisLength', pa.float64()),
            ('summedPerimeter', pa.float64()),
            ('summedSurfaceArea', pa.float64()),
            ('Area_over_PerimeterSquared', pa.float64()),
            ('Area_over_Perimeter', pa.float64()),
            ('summedConvexPerimeter_over_Perimeter', pa.float64()),
        ])

    def _ensure_table_exists(self, tx):
        """Ensure schema and table exist, create if they don't."""
        try:
            # Get or create bucket
            bucket = tx.bucket(self.config.bucket_name)

            # Try to get existing schema, create if doesn't exist
            try:
                schema = bucket.schema(self.config.schema_name)
                print(f"Using existing schema: {self.config.schema_name}")
            except Exception:
                schema = bucket.create_schema(self.config.schema_name)
                print(f"Created new schema: {self.config.schema_name}")

            # Try to get existing table, create if doesn't exist
            try:
                table = schema.table(self.config.table_name)
                print(f"Using existing table: {self.config.table_name}")
            except Exception:
                columns = self._get_features_schema()
                table = schema.create_table(self.config.table_name, columns)
                print(f"Created new table: {self.config.table_name}")

            return table

        except Exception as e:
            print(f"Error ensuring table exists: {e}")
            raise

    def insert_features(self, features_df: pd.DataFrame) -> bool:
        """Insert features DataFrame into VastDB table."""
        try:
            with self.session.transaction() as tx:
                table = self._ensure_table_exists(tx)

                # Convert pandas DataFrame to PyArrow Table
                arrow_table = pa.Table.from_pandas(features_df, schema=self._get_features_schema())

                # Insert data
                table.insert(arrow_table)

                print(f"Inserted {len(features_df)} rows into {self.config.table_name}")
                return True

        except Exception as e:
            print(f"Error inserting features into VastDB: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cleanup(self):
        """Close VastDB session."""
        if self.session:
            try:
                # VastDB session cleanup if needed
                pass
            except Exception as e:
                print(f"Error during cleanup: {e}")


def create_vastdb_storage_from_s3_config(
    s3_config: S3Config,
    bucket_name: str,
    schema_name: str,
    table_name: str,
    access_key: str,
    secret_key: str
) -> VastDBFeatureStorage:
    """Create VastDB storage using S3 config endpoint."""
    vastdb_config = VastDBConfig(
        bucket_name=bucket_name,
        schema_name=schema_name,
        table_name=table_name,
        endpoint_url=s3_config.s3_url,
        access_key=access_key,
        secret_key=secret_key
    )
    return VastDBFeatureStorage(vastdb_config)
