import numpy as np

from scipy.ndimage import label
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import binary_closing, binary_dilation, binary_erosion, diamond
import skimage.filters as imfilters

from .phasecong import phasecong_Mm
from .morphology import SE3, hysthresh, bwmorph_thin, EIGHT

SE2 = diamond(2)
SED = diamond(1)

# parameters
HT_T1, HT_T2 = 0.3, 0.09
BLOB_MIN = 40
DARK_THRESHOLD_ADJUSTMENT = 0.75


def _kmeans_1d_strict(values, max_iter=100):
    """MATLAB-style 1D k-means (batch) with singleton empty handling."""
    values = np.asarray(values, dtype=np.float32)
    n = values.shape[0]
    if n == 0:
        return np.array([0.0, 1.0], dtype=np.float32), np.zeros(0, dtype=np.int8)
    row_indices = np.arange(n)

    def center_for_cluster(idx, cluster):
        members = idx == cluster
        count = int(np.count_nonzero(members))
        if count == 0:
            return np.float32(np.nan), 0
        total = np.cumsum(values[members], dtype=np.float32)[-1]
        count_float = np.float32(count)
        return np.float32(total / count_float), count

    def distances_to_center(center):
        delta = values - np.float32(center)
        return np.asarray(delta * delta, dtype=np.float32)

    def assigned_total(distances, idx):
        assigned = distances[row_indices, idx]
        return np.cumsum(assigned, dtype=np.float32)[-1]

    centers = np.array([0.0, 1.0], dtype=np.float32)
    # Initial distances/assignments from provided centers
    D = np.empty((n, 2), dtype=np.float32)
    D[:, 0] = distances_to_center(centers[0])
    D[:, 1] = distances_to_center(centers[1])
    idx = np.argmin(D, axis=1).astype(np.int8)

    changed = np.array([0, 1], dtype=np.int64)
    previdx = np.zeros(n, dtype=np.int8)
    prevtotsumD = np.float32(np.inf)

    for _iter in range(max_iter):
        # Recompute centers and counts for changed clusters
        counts = np.bincount(idx, minlength=2).astype(np.int64)
        for c in changed:
            if counts[c] > 0:
                centers[c], counts[c] = center_for_cluster(idx, c)

        # Update distances for changed clusters
        for c in changed:
            D[:, c] = distances_to_center(centers[c])

        # Handle empty clusters (singleton)
        empties = [c for c in changed if counts[c] == 0]
        if empties:
            d_assigned = D[row_indices, idx]
            for empty in empties:
                lonely = int(np.argmax(d_assigned))
                from_cluster = int(idx[lonely])
                if counts[from_cluster] < 2:
                    from_cluster = int(np.argmax(counts > 1))
                    lonely = int(np.argmax(idx == from_cluster))

                centers[empty] = values[lonely]
                idx[lonely] = empty
                counts[empty] = 1
                counts[from_cluster] -= 1
                D[:, empty] = distances_to_center(centers[empty])

                # Update donor cluster center/distance
                if counts[from_cluster] > 0:
                    centers[from_cluster], counts[from_cluster] = center_for_cluster(idx, from_cluster)
                D[:, from_cluster] = distances_to_center(centers[from_cluster])
                changed = np.unique(np.concatenate([changed, np.array([from_cluster], dtype=np.int64)]))

        # Compute objective and check for improvement
        totsumD = assigned_total(D, idx)
        if prevtotsumD <= totsumD:
            idx = previdx
            counts = np.bincount(idx, minlength=2).astype(np.int64)
            for c in changed:
                if counts[c] > 0:
                    centers[c], counts[c] = center_for_cluster(idx, c)
            break

        previdx = idx.copy()
        prevtotsumD = totsumD

        # Reassign using nearest centroid, tie -> stay
        nidx = np.argmin(D, axis=1).astype(np.int8)
        dmin = D[row_indices, nidx]
        moved = np.where(nidx != previdx)[0]
        if moved.size:
            stay_mask = D[moved, previdx[moved]] > dmin[moved]
            moved = moved[stay_mask]
        if moved.size == 0:
            break
        idx[moved] = nidx[moved]
        changed = np.unique(np.concatenate([idx[moved], previdx[moved]]))

    return centers, idx.astype(np.int8)

def kmeans_segment(roi):
    # Match MATLAB im2single for uint8 by explicit float32 scaling.
    if roi.dtype == np.uint8:
        r = roi.astype(np.float32) / np.float32(255.0)
    else:
        r = roi.astype(np.float32)
    # Use column-major order to match MATLAB img(:) traversal.
    values = r.reshape(-1, order="F")
    C, J = _kmeans_1d_strict(values, max_iter=100)
    C = C.reshape(-1)
    J = J.reshape(-1)

    # reshape labels to image using MATLAB order
    J = J.reshape(r.shape, order="F")
    bg_label = np.argmax(C)
    # find the darkest pixel value in the bright (background) cluster
    darkest_background = np.min(r[J == bg_label])
    # use it to compute a threshold
    threshold = darkest_background * DARK_THRESHOLD_ADJUSTMENT
    # extend the background using that threshold (MATLAB uses >)
    J[r > threshold] = bg_label
    # return True for non-background pixels
    return (J != bg_label).reshape(roi.shape)


def apply_blob_min(roi):
    # Match MATLAB bwareaopen(img, 41): remove objects with size < 41 (keep 41+).
    roi_bool = roi.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    labeled, num = label(roi_bool, structure=structure)
    if num == 0:
        return roi_bool
    counts = np.bincount(labeled.ravel())
    keep_labels = np.where(counts >= (BLOB_MIN + 1))[0]
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(roi_bool)
    return np.isin(labeled, keep_labels)

def segment_roi(roi, raw_stitch=None):
    # step 1. phase congruency (edge detection)
    Mm = phasecong_Mm(roi)
    # step 1a: for stitched images, chop the raw stitch mask
    # after growing it one pixel
    if raw_stitch is not None:
        mask = binary_dilation(raw_stitch.mask)
        Mm[mask] = 0
    # step 2. hysteresis thresholding (of edges)
    B = hysthresh(Mm,HT_T1,HT_T2)
    # step 3. trim pixels off border
    B[B[:,1]==0,0]=0
    B[B[:,-2]==0,-1]=0
    B[0,B[1,:]==0]=0
    B[-1,B[-2,:]==0]=0
    # step 4. binary closing
    padded = np.pad(B, 2)
    B = binary_closing(padded, SE2)[2:-2,2:-2]
    # step 5. morphological thinning
    B = bwmorph_thin(B, 3)
    # step 6. background/foreground thresholding
    dark = kmeans_segment(roi)
    B = np.logical_or(B, dark)
    # step 7. fill holes (surrounded by target pixels)
    B = binary_fill_holes(B)
    # step 8. erode and remove small blobs
    B_eroded = binary_erosion(B, SED)
    if np.sum(apply_blob_min(B_eroded)) > 0:
        B = B_eroded
    B = apply_blob_min(B)
    return B
