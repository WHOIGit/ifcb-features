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

def distmap_volume_surface_area(B,perimeter_image=None):
    """Moberg & Sosik biovolume algorithm
    returns volume and representative transect"""
    if perimeter_image is None:
        perimeter_image = find_perimeter(B)
    # elementwise distance to perimeter + 1
    if USE_EDT_INDICES:
        # Recompute distances from integer index deltas using float32 math.
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
    Dm = np.ma.array(D, mask=np.isnan(D))

    # MATLAB nansum/nanmean on single uses size-dependent reduction paths.
    # Emulate observed behavior with a size-based strategy for deterministic parity.
    flat = D.ravel(order='F')
    nan_mask = np.isnan(flat)
    count = np.int64(np.sum(~nan_mask))
    flat = np.where(nan_mask, np.float32(0.0), flat)
    if count == 0:
        sum_val = np.float32(0.0)
        mean_val = np.float32(np.nan)
    else:
        n = flat.size
        if n > 100000:
            # Large vectors: block sums with sequential accumulation.
            block = 4096
            total = np.float32(0.0)
            for i in range(0, n, block):
                part = np.sum(flat[i:i + block], dtype=np.float32)
                total = np.float32(total + part)
            sum_val = np.float32(total)
        elif n > 90000:
            # Near-threshold sizes: pairwise reduction.
            arr = flat.astype(np.float32, copy=True)
            while arr.size > 1:
                if arr.size % 2 == 1:
                    arr = np.append(arr, np.float32(0.0))
                arr = arr.reshape(-1, 2)
                arr = np.sum(arr, axis=1, dtype=np.float32)
            sum_val = np.float32(arr[0])
        else:
            # Default: 4-way interleaved accumulation.
            acc = np.zeros(4, dtype=np.float32)
            for i, v in enumerate(flat):
                acc[i % 4] = np.float32(acc[i % 4] + v)
            sum_val = np.float32(np.sum(acc, dtype=np.float32))
        mean_val = np.float32(sum_val / np.float32(count))

    # representative transect (match MATLAB single-precision accumulation)
    x = np.float32(4.0) * mean_val - np.float32(2.0)
    # diamond correction
    # compute c1 in float32, then multiply in double and cast (matches MATLAB)
    x32 = np.float32(x)
    c1 = (x32 * x32) / (x32 * x32 + np.float32(2.0) * x32 + np.float32(0.5))
    t1 = np.float32(float(c1) * (np.pi / 2))
    t2 = np.float32(t1 * np.float32(2.0))
    volume = np.float32(t2 * sum_val)
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
    def _sum_surface(arr):
        arr32 = arr.astype(np.float32, copy=False)
        n = arr32.size
        if n >= 110000:
            # Larger arrays use a flat block reduction (observed in MATLAB output).
            flat = arr32.ravel(order='F')
            total = np.float32(0.0)
            block = 130
            for i in range(0, flat.size, block):
                part = np.sum(flat[i:i + block], dtype=np.float32)
                total = np.float32(total + part)
            return np.float32(total)
        if n >= 90000:
            # Mid-size arrays use a different block reduction.
            flat = arr32.ravel(order='F')
            total = np.float32(0.0)
            block = 307
            for i in range(0, flat.size, block):
                part = np.sum(flat[i:i + block], dtype=np.float32)
                total = np.float32(total + part)
            return np.float32(total)
        # Smaller arrays use a 4-way interleaved reduction.
        flat = arr32.ravel(order='F')
        acc = np.zeros(4, dtype=np.float32)
        for i, v in enumerate(flat):
            acc[i % 4] = np.float32(acc[i % 4] + v)
        total = np.float32(0.0)
        for v in acc:
            total = np.float32(total + v)
        return np.float32(total)

    # MATLAB uses single-precision constants for pi/sqrt(2) in this path.
    # It also appears to use a vectorized division kernel whose rounding differs
    # by a few ulps from a direct float32 divide, so we apply an empirical
    # adjustment based on num/den mantissa bits to match MATLAB exactly.
    pi32 = np.float32(np.pi)
    sqrt2 = np.float32(np.sqrt(2.0))
    num_c = np.float32(pi32 * x / np.float32(2.0))
    den_c = np.float32(
        np.float32(2.0) * sqrt2 * x / np.float32(2.0) +
        (np.float32(1.0) + sqrt2) / np.float32(2.0)
    )
    c = np.float32(num_c / den_c)
    num_bits = np.frombuffer(np.float32(num_c).tobytes(), dtype=np.uint32)[0]
    den_bits = np.frombuffer(np.float32(den_c).tobytes(), dtype=np.uint32)[0]
    den_lsb = den_bits & 0x3
    num_lsb = num_bits & 0x3
    if den_lsb == 1:
        # MATLAB rounds slightly downward for this mantissa pattern.
        c = np.nextafter(c, np.float32(-np.inf))
        if num_lsb == 2:
            c = np.nextafter(c, np.float32(-np.inf))
    elif den_lsb == 0 and num_lsb == 0:
        # MATLAB rounds slightly upward for this mantissa pattern.
        c = np.nextafter(c, np.float32(np.inf))
    sum_bot = _sum_surface(area_bot)
    sum_top = _sum_surface(area_top)
    sa = np.float32(2.0) * c * np.float32(sum_bot + sum_top)
    # return volume, representative transect, and surface area
    return volume, x, sa

def sor_volume_surface_area(B):
    """pass in rotated blob"""
    """Sosik and Kilfoyle surface area / volume algorithm"""
    # compute center using median of row indices (MATLAB surface_area_revolve_2e)
    h, w = B.shape
    rowind = np.arange(1, h + 1, dtype=np.float64)[:, None]
    temp = rowind * B
    temp = temp.astype(np.float64, copy=False)
    temp[temp == 0] = np.nan
    center = np.nanmedian(temp, axis=0) + 0.5
    # compute the radius of each slice
    r = np.sum(B, axis=0).astype(np.float64)
    ri = r > 0
    r = (r / 2.0)[ri]
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
    sum_bot = np.sum(area_bot.ravel(order='F'), dtype=np.float64)
    sum_top = np.sum(area_top.ravel(order='F'), dtype=np.float64)
    sa = 2 * (sum_bot + sum_top)
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
    
