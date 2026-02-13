import numpy as np

from scipy.ndimage import measurements

from scipy.ndimage import rotate as nd_rotate

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
    # Avoid precision drift at integer boundaries only when very close.
    val_y = s - yc
    val_x = s - xc
    # Snap values extremely close to integers to the integer boundary.
    if abs(val_y - round(val_y)) < 1e-9:
        val_y = round(val_y)
    if abs(val_x - round(val_x)) < 1e-9:
        val_x = round(val_x)
    y0 = int(np.floor(val_y))
    x0 = int(np.floor(val_x))
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
    # MATLAB uses 1-based coordinates, center at (w+1)/2, (h+1)/2
    cx = (w + 1) / 2.0
    cy = (h + 1) / 2.0

    yy, xx = np.indices((h, w))
    x = xx + 1.0
    y = yy + 1.0
    x0 = x - cx
    y0 = y - cy

    ang = np.deg2rad(angle_deg)
    cos_a = np.cos(-ang)
    sin_a = np.sin(-ang)
    x_in = cos_a * x0 - sin_a * y0 + cx
    y_in = sin_a * x0 + cos_a * y0 + cy

    # MATLAB round halves away from zero for positive coords.
    x_idx = np.floor(x_in + 0.5).astype(np.int64)
    y_idx = np.floor(y_in + 0.5).astype(np.int64)

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
