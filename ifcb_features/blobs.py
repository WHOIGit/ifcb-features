import numpy as np

from scipy.ndimage import measurements

from .morphology import SE2, SE3, EIGHT, bwmorph_thin

def label_blobs(B):
    B = np.array(B).astype(np.bool)
    labeled, _ = measurements.label(B,structure=EIGHT)
    objects = measurements.find_objects(labeled)
    return labeled, objects
    
def find_blobs(B):
    """find and return all blobs in the image, using
    eight-connectivity. returns a labeled image, the
    bounding boxes of the blobs, and the blob masks cropped
    to those bounding boxes"""
    B = np.array(B).astype(np.bool)
    labeled, objects = label_blobs(B)
    # Match MATLAB blob_geomprop: sort connected components by area (descending).
    labeled_objects = [(ix + 1, obj) for ix, obj in enumerate(objects)]
    def sort_key(item):
        label_id, obj = item
        comp = labeled[obj] == label_id
        area = int(np.sum(comp))
        return (-area, obj[1].start, obj[0].start)
    labeled_objects.sort(key=sort_key)
    objects_sorted = [obj for _, obj in labeled_objects]
    blobs = [labeled[obj] == label for label, obj in labeled_objects]
    return labeled, objects_sorted, blobs

def center_blob(B):
    """Center blob on centroid, matching MATLAB center_blob.m."""
    B = np.array(B).astype(np.bool)
    # MATLAB uses regionprops centroid (x, y) with 1-based coordinates.
    ys, xs = np.where(B)
    if ys.size == 0:
        return B.copy()
    h, w = B.shape
    # Match MATLAB center_blob.m:
    # centroid is 1-based, then shift to 0-based for padding math.
    xc = (np.mean(xs) + 1.0) - 1.0
    yc = (np.mean(ys) + 1.0) - 1.0
    s = max(yc, h - yc, xc, w - xc)
    m = int(np.ceil(s * 2))
    C = np.zeros((m, m), dtype=np.bool)
    # Integer-exact offsets to avoid float drift.
    n = ys.size
    sum_y = int(np.sum(ys))
    sum_x = int(np.sum(xs))
    hN = h * n
    wN = w * n
    ycN = sum_y
    xcN = sum_x
    sN = max(ycN, hN - ycN, xcN, wN - xcN)
    m = int((2 * sN + n - 1) // n)
    y0 = int((sN - ycN) // n)
    x0 = int((sN - xcN) // n)
    C[y0:y0 + h, x0:x0 + w] = B
    return C

def rotate_blob(blob, theta):
    """rotate a blob counterclockwise"""
    blob = center_blob(blob)
    # Match MATLAB imrotate(centered, theta, 'nearest', 'crop')
    blob = imrotate_nearest_crop(blob, theta)
    # note that v2 does morphological post-processing and v3 does not
    return blob

def imrotate_nearest_crop(img, angle_deg):
    """MATLAB-compatible imrotate(..., 'nearest', 'crop') for binary masks."""
    img = np.array(img).astype(np.bool)
    h, w = img.shape

    # MATLAB blob_rotate calls imrotate(centered, -Orientation, ...).  The
    # Python caller passes the already sign-adjusted orientation, so use the
    # opposite angle for MATLAB's affine2d/imwarp coordinate convention.
    ang = np.deg2rad(-angle_deg)
    cos_a = np.cos(ang)
    sin_a = np.sin(ang)

    # imrotate's general-rotation crop path is:
    #   RA = imref2d(size(A));
    #   Rout = applyGeometricTransformToSpatialRef(RA, tform);
    #   Rout.ImageSize = RA.ImageSize;
    #   Rout limits are shifted to preserve the center;
    #   imwarp(A, tform, 'nearest', 'OutputView', Rout)
    # Default imref2d world limits are [0.5, size + 0.5].
    x_limits = (0.5, w + 0.5)
    y_limits = (0.5, h + 0.5)
    corners = np.array(
        [
            [x_limits[0], y_limits[0]],
            [x_limits[0], y_limits[1]],
            [x_limits[1], y_limits[0]],
            [x_limits[1], y_limits[1]],
        ],
        dtype=np.float64,
    )
    x_out = corners[:, 0] * cos_a + corners[:, 1] * sin_a
    y_out = -corners[:, 0] * sin_a + corners[:, 1] * cos_a
    x_trans = (float(np.min(x_out)) + float(np.max(x_out))) / 2.0 - (
        x_limits[0] + x_limits[1]
    ) / 2.0
    y_trans = (float(np.min(y_out)) + float(np.max(y_out))) / 2.0 - (
        y_limits[0] + y_limits[1]
    ) / 2.0
    x_world_min = x_limits[0] + x_trans
    y_world_min = y_limits[0] + y_trans

    # MATLAB's imwarp path lands infinitesimally below exact half-pixel
    # boundaries for the observed tie cases. Nudge the crop reference down
    # by two ULPs so nearest-neighbor rounding follows the same side.
    x_world_min = np.nextafter(np.nextafter(x_world_min, -np.inf), -np.inf)
    y_world_min = np.nextafter(np.nextafter(y_world_min, -np.inf), -np.inf)

    rows, cols = np.indices((h, w), dtype=np.float64)
    x_world = x_world_min + (cols + 1.0 - 0.5)
    y_world = y_world_min + (rows + 1.0 - 0.5)

    # transformPointsInverse for affine2d rotation matrix
    # [cos -sin 0; sin cos 0; 0 0 1], then worldToIntrinsic(RA).
    # With default RA limits and unit pixel extents, world == intrinsic.
    x_in = x_world * cos_a - y_world * sin_a
    y_in = x_world * sin_a + y_world * cos_a

    # MATLAB round: ties away from zero (works for negative too).
    x_idx = (np.sign(x_in) * np.floor(np.abs(x_in) + 0.5)).astype(np.int64)
    y_idx = (np.sign(y_in) * np.floor(np.abs(y_in) + 0.5)).astype(np.int64)

    out = np.zeros_like(img, dtype=np.bool)
    mask = (x_idx >= 1) & (x_idx <= w) & (y_idx >= 1) & (y_idx <= h)
    out[mask] = img[y_idx[mask] - 1, x_idx[mask] - 1]
    return out
    
def blob_shape(b0):
    h, w = b0.shape
    blr = np.fliplr(b0)
    bud = np.flipud(b0)

    # reproduce MATLAB's center-of-pixel approach
    x0 = np.argmax(np.sum(b0,axis=0)>0) + 0.5
    x1 = w - np.argmax(np.sum(blr,axis=0)>0)
    y0 = np.argmax(np.sum(b0,axis=1)>0) + 0.5
    y1 = h - np.argmax(np.sum(bud,axis=1)>0)
    h = int((y1-y0) + 0.5)
    w = int((x1-x0) + 0.5)
    
    return h,w
