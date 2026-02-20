import os
import numpy as np

from scipy.ndimage import binary_fill_holes, distance_transform_edt

from .morphology import find_perimeter

def bottom_top_area(X,Y,Z,ignore_ground=False):
    """computes top quad and bottom quad areas for distmap
    and SOR algorithms"""
    """ignore_ground is an adjustment used in distmap
    but not in SOR"""
    h, w = Z.shape

    i2 = slice(0,h-1)
    i1 = slice(1,h)
    ia2 = slice(0,w-1)
    ia1 = slice(1,w)
    
   # create linesegs AB for all quadrilaterals
    AB1, AB2, AB3 = [xyz[i2,ia2] - xyz[i1,ia2] for xyz in [X,Y,Z]]
    # create linesegs AD for all quadrilaterals
    AD1, AD2, AD3 = [xyz[i2,ia2] - xyz[i1,ia1] for xyz in [X,Y,Z]]
    # create linesegs AD for all quadrilaterals
    CD1, CD2, CD3 = [xyz[i2,ia1] - xyz[i1,ia1] for xyz in [X,Y,Z]]

    # triangle formed by AB and AD for all quadrilaterals
    leg1 = ((AB2 * AD3) - (AB3 * AD2)) ** 2
    leg2 = ((AB3 * AD1) - (AB1 * AD3)) ** 2
    leg3 = ((AB1 * AD2) - (AB2 * AD1)) ** 2
    # bottom area
    area_bot = 0.5 * np.sqrt(leg1 + leg2 + leg3)

    # triangle formed by CD and AD for all quadrilaterals
    leg1 = ((CD2 * AD3) - (CD3 * AD2)) ** 2
    leg2 = ((CD3 * AD1) - (CD1 * AD3)) ** 2
    leg3 = ((CD1 * AD2) - (CD2 * AD1)) ** 2
    # top area
    area_top = 0.5 * np.sqrt(leg1 + leg2 + leg3)
    
    if ignore_ground:
        ind = np.abs(AB3) + np.abs(AD3) + np.abs(CD3) + Z[i2,ia2]
        area_bot[ind==0] = 0
        area_top[ind==0] = 0
        
    return area_bot, area_top

USE_EDT_INDICES = True  # recompute distances from indices to better match MATLAB bwdist
# Match MATLAB SOR center logic (old top-edge + radius) when True.
SOR_USE_OLD_CENTER = True
# Use Heidi_explore distmap implementation (bwdist + heidi surface area) when True.
USE_HEIDI_DISTMAP = False

def distmap_volume_surface_area(B,perimeter_image=None):
    """Moberg & Sosik biovolume algorithm
    returns volume and representative transect"""
    if USE_HEIDI_DISTMAP:
        return distmap_volume_surface_area_heidi(B, perimeter_image)
    if perimeter_image is None:
        perimeter_image = find_perimeter(B)
    # elementwise distance to perimeter + 1
    if USE_EDT_INDICES:
        _, inds = distance_transform_edt(1 - perimeter_image, return_indices=True)
        coords = np.indices(perimeter_image.shape, dtype=np.int64)
        deltas = coords - inds.astype(np.int64, copy=False)
        dist2 = np.sum(deltas * deltas, axis=0, dtype=np.int64)
        D = np.sqrt(dist2.astype(np.float32)) + np.float32(1.0)
    else:
        D = distance_transform_edt(1 - perimeter_image) + 1
    # mask distances outside filled perimeter (match MATLAB imfill on boundary image)
    fill = binary_fill_holes(np.array(perimeter_image, dtype=bool))
    D = D.astype(np.float32)
    D[~fill] = np.nan
    # representative transect (match MATLAB float32 sum/mean in column-major order)
    flat = D.ravel(order="F")
    nan_mask = np.isnan(flat)
    count = np.int64(np.sum(~nan_mask))
    flat = np.where(nan_mask, np.float32(0.0), flat.astype(np.float32))
    use_deterministic_sum = True
    if count == 0:
        sum_val = np.float32(0.0)
        mean_val = np.float32(np.nan)
    else:
        if use_deterministic_sum:
            # Deterministic column-major sum in float64.
            sum_acc = 0.0
            cnt = 0
            for v in flat:
                if not np.isnan(v):
                    sum_acc += float(v)
                    cnt += 1
            sum_val = np.float64(sum_acc)
            mean_val = np.float64(sum_acc / float(cnt)) if cnt else np.float64(np.nan)
        else:
            # Match MATLAB sum for single arrays: column-major, float32 accumulation.
            sum_val = np.float32(np.sum(flat, dtype=np.float32))
            mean_val = np.float32(sum_val / np.float32(count))
    if use_deterministic_sum:
        x = np.float64(4.0) * mean_val - np.float64(2.0)
    else:
        x = np.float32(4.0) * mean_val - np.float32(2.0)
    # diamond correction
    c1 = x**2 / (x**2 + 2*x + 0.5)
    # circle correction
    # c2 = np.pi / 2 
    # volume = c1 * c2 * 2 * np.sum(D)
    if use_deterministic_sum:
        volume = np.float64(c1 * np.float64(np.pi) * sum_val)
    else:
        # compute volume in float32 to match MATLAB single-precision path
        c1 = np.float32(c1)
        volume = np.float32(c1 * np.float32(np.pi) * sum_val)
    if os.getenv("DISTMAP_DEBUG") == "1":
        print("distmap_debug sum_val", float(sum_val), "mean_val", float(mean_val), "x", float(x), "volume", float(volume))
    # surface area
    # surface area uses NaN-masked distances as zero
    D_sa = np.nan_to_num(D, nan=0.0)
    h, w = D_sa.shape
    # MATLAB uses 1-based X/Y indices for surface area geometry.
    Y, X = np.mgrid[1:h + 1, 1:w + 1]
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    area_bot, area_top = bottom_top_area(X, Y, D_sa, ignore_ground=True)
    # final correction of the diamond cross-section
    # inherent in the distance map to be circular instead
    c = (np.float32(np.pi) * x / np.float32(2.0)) / (
        np.float32(2.0) * np.float32(np.sqrt(2.0)) * x / np.float32(2.0)
        + (np.float32(1.0) + np.float32(np.sqrt(2.0))) / np.float32(2.0)
    )
    sa = np.float32(2.0) * c * np.float32(
        np.nansum(area_bot.astype(np.float32), dtype=np.float32)
        + np.nansum(area_top.astype(np.float32), dtype=np.float32)
    )
    # return volume, representative transect, and surface area
    return volume, x, sa


def distmap_volume_surface_area_heidi(B, perimeter_image=None):
    """Heidi_explore distmap_volume (bwdist + surface area) implementation."""
    if perimeter_image is None:
        perimeter_image = find_perimeter(B)
    # calculate distance map (MATLAB bwdist)
    D = distance_transform_edt(1 - perimeter_image) + 1
    # mask distances outside filled perimeter
    fill = binary_fill_holes(np.array(perimeter_image, dtype=bool))
    D = D.astype(np.float64)
    D[~fill] = np.nan
    # representative transect length
    x = 4 * np.nanmean(D) - 2
    # correction factors
    c1 = (x**2) / (x**2 + 2 * x + 0.5)
    c2 = np.pi / 2
    volume = c1 * c2 * 2 * np.nansum(D)
    # surface area
    D_sa = np.nan_to_num(D, nan=0.0)
    h, w = D_sa.shape
    Y, X = np.mgrid[1:h + 1, 1:w + 1]
    area_bot, area_top = bottom_top_area(X, Y, D_sa, ignore_ground=True)
    c = (np.pi * x / 2.0) / (2.0 * np.sqrt(2.0) * x / 2.0 + (1.0 + np.sqrt(2.0)) / 2.0)
    sa = 2 * c * (np.sum(area_bot) + np.sum(area_top))
    return volume, x, sa

def sor_volume_surface_area(B):
    """pass in rotated blob"""
    """Sosik and Kilfoyle surface area / volume algorithm"""
    # compute center using median (current) or bottom+radius (legacy MATLAB)
    h, w = B.shape
    r = np.sum(B, axis=0).astype(np.float64)
    ri = r > 0
    r = (r / 2.0)[ri]
    if SOR_USE_OLD_CENTER:
        y1 = np.argmax(B, axis=0).astype(np.float64) + 1.0
        y1 = y1[ri]
        center = y1 + r
    else:
        rowind = np.arange(1, h + 1, dtype=np.float64)[:, None]
        temp = rowind * B
        temp = temp.astype(np.float64, copy=False)
        temp[temp == 0] = np.nan
        center = np.nanmedian(temp, axis=0) + 0.5
        center = center[ri]
    n_slices = r.size
    # compute angles between 0 and 180 degrees inclusive, in radians
    da = 0.25
    angvec = np.arange(0, 180 + da / 2, da, dtype=np.float64)
    n_angles = angvec.size
    angR = angvec * (np.pi / 180)
    
    # make everything the same shape: (nslices, nangles)
    angR = np.vstack([angR] * n_slices)
    r = np.vstack([r] * n_angles).T
    center = np.vstack([center] * n_angles).T
    # correct for edge effects
    if n_slices >= 2:
        center[0, :] = center[1, :]
        center[-1, :] = center[-2, :]
    
    # y coordinates of all angles on all slices
    Y = center + np.cos(angR) * r
    # z coordinates of all angles on all slices
    Z = np.sin(angR) * r
    
    # compute index of slice in y matrix
    x = np.array(range(r.shape[0])) + 1.
    # half-pixel adjustment of edges
    x[0]-=0.5
    x[-1]+=0.5
    X = np.vstack([x] * n_angles).T
    
    # compute bottom and top area
    area_bot, area_top = bottom_top_area(X,Y,Z)
    
    # surface area
    # multiply sum of areas of quadrilaterals by 2 to account for angles 180-360
    sa = 2 * (np.sum(area_bot) + np.sum(area_top))
    # add flat end caps
    sa += np.sum(np.pi * r[[0,-1],0]**2)
    
    # compute height of cone slices
    b1 = np.pi * r[1:n_slices,0] ** 2
    b2 = np.pi * r[0:n_slices-1,0] ** 2
    h = np.diff(x)
    # volume
    v = np.sum((h/3) * (b1 + b2 + np.sqrt(b1 * b2)))
    
    # representative width
    xr = np.mean(r[:,0]*2)
    
    # return volume, representative width, and surface area
    return v, xr, sa
    
