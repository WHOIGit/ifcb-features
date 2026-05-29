# Slim features

This document describes the per-ROI features produced by
[`extract_slim_features.py`](extract_slim_features.py) — the columns of each
`<bin>_features_v4.csv` file (alongside the `roi_number` column that identifies
the ROI within the bin).

A few conventions apply throughout:

- Each ROI is segmented into one or more **blobs** (connected regions). The
  blobs are ordered largest-area first.
- Features whose names are *not* prefixed with `summed` describe the **largest
  blob** in the ROI.
- Features prefixed with `summed` are aggregated across **all** blobs in the ROI.
- All lengths and areas are in **pixels** (or pixels² / pixels³ as appropriate);
  No physical scaling is applied here.
- Definitions are chosen to match the original MATLAB IFCB feature code so that
  values are comparable with historical datasets.

## Largest-blob features

| Feature | Description |
| --- | --- |
| `Area` | Number of pixels in the blob. |
| `Biovolume` | Estimated 3-D volume of the particle (pixels³), computed with the Moberg & Sosik algorithm — either a solid-of-revolution model or a distance-map model is selected automatically based on the blob's shape. |
| `BoundingBox_xwidth` | Width (in x) of the blob's axis-aligned bounding box. |
| `BoundingBox_ywidth` | Height (in y) of the blob's axis-aligned bounding box. |
| `ConvexArea` | Area (pixels²) enclosed by the blob's convex hull. |
| `ConvexPerimeter` | Perimeter length of the blob's convex hull. |
| `Eccentricity` | Eccentricity of the ellipse with the same second moments as the blob (0 = circle, approaching 1 = elongated). |
| `EquivDiameter` | Diameter of a circle with the same area as the blob. |
| `Extent` | Fraction of the bounding box filled by the blob (area ÷ bounding-box area). |
| `MajorAxisLength` | Length of the major axis of the best-fit ellipse. |
| `MinorAxisLength` | Length of the minor axis of the best-fit ellipse. |
| `Orientation` | Angle (degrees) of the ellipse's major axis relative to the horizontal. |
| `Perimeter` | Perimeter length of the blob boundary (Benkrid perimeter estimate). |
| `RepresentativeWidth` | Representative width of the particle from the Moberg & Sosik biovolume model. |
| `Solidity` | Area ÷ convex area; how completely the blob fills its convex hull (1 = fully convex). |
| `SurfaceArea` | Estimated 3-D surface area (pixels²) from the Moberg & Sosik biovolume model. |
| `maxFeretDiameter` | Maximum Feret (caliper) diameter — the largest distance across the blob over all orientations. |
| `minFeretDiameter` | Minimum Feret (caliper) diameter — the smallest such distance. |

## Whole-ROI and summed features

| Feature | Description |
| --- | --- |
| `numBlobs` | Number of blobs segmented from the ROI. |
| `summedArea` | Sum of `Area` over all blobs. |
| `summedBiovolume` | Sum of `Biovolume` over all blobs. |
| `summedConvexArea` | Sum of `ConvexArea` over all blobs. |
| `summedConvexPerimeter` | Sum of `ConvexPerimeter` over all blobs. |
| `summedMajorAxisLength` | Sum of `MajorAxisLength` over all blobs. |
| `summedMinorAxisLength` | Sum of `MinorAxisLength` over all blobs. |
| `summedPerimeter` | Sum of `Perimeter` over all blobs. |
| `summedSurfaceArea` | Sum of `SurfaceArea` over all blobs. |

## Derived ratios

These are computed from the features above. Each is set to `NaN` when its
denominator is zero (e.g. an empty ROI with no blobs).

| Feature | Description |
| --- | --- |
| `Area_over_PerimeterSquared` | `Area` ÷ `Perimeter²` — a dimensionless compactness measure (largest blob). |
| `Area_over_Perimeter` | `Area` ÷ `Perimeter` (largest blob). |
| `summedConvexPerimeter_over_Perimeter` | `summedConvexPerimeter` ÷ `summedPerimeter`, aggregated across all blobs — a measure of overall boundary roughness. |