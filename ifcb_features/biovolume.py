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

def _det_sum32(arr):
    """Deterministic float32 sum in column-major order."""
    sum_acc = np.float32(0.0)
    flat = arr.ravel(order="F")
    for v in flat:
        sum_acc = np.float32(sum_acc + np.float32(v))
    return np.float32(sum_acc)


def distmap_volume_surface_area(B, perimeter_image=None):
    """Moberg & Sosik biovolume algorithm (Heidi_explore implementation).
    Returns volume, representative transect, and surface area."""
    if perimeter_image is None:
        perimeter_image = find_perimeter(B)
    # calculate distance map (MATLAB bwdist)
    D = distance_transform_edt(1 - perimeter_image) + 1
    # mask distances outside filled perimeter
    fill = binary_fill_holes(np.array(perimeter_image, dtype=bool))
    D = D.astype(np.float64)
    D[~fill] = np.nan
    # deterministic sum/mean over column-major order to match MATLAB loop
    flat = D.ravel(order="F")
    sum_acc = np.float32(0.0)
    cnt = 0
    for v in flat:
        if not np.isnan(v):
            sum_acc = np.float32(sum_acc + np.float32(v))
            cnt += 1
    sum_val = np.float32(sum_acc)
    mean_val = np.float32(sum_acc / np.float32(cnt)) if cnt else np.float32(np.nan)
    # representative transect length
    x = np.float32(4.0) * mean_val - np.float32(2.0)
    # correction factors
    c1 = (x**2) / (x**2 + np.float32(2.0) * x + np.float32(0.5))
    c2 = np.float32(np.pi / 2.0)
    volume = np.float32(c1 * c2 * np.float32(2.0) * sum_val)
    # surface area
    D_sa = np.nan_to_num(D, nan=0.0).astype(np.float32, copy=False)
    h, w = D_sa.shape
    Y, X = np.mgrid[1:h + 1, 1:w + 1]
    X = X.astype(np.float32, copy=False)
    Y = Y.astype(np.float32, copy=False)
    area_bot, area_top = bottom_top_area(X, Y, D_sa, ignore_ground=True)
    # compute c fully in float32 to match MATLAB single-precision path
    c = (np.float32(np.pi) * np.float32(x) / np.float32(2.0)) / (
        np.float32(2.0) * np.float32(np.sqrt(2.0)) * np.float32(x) / np.float32(2.0)
        + (np.float32(1.0) + np.float32(np.sqrt(2.0))) / np.float32(2.0)
    )
    sum_bot = _det_sum32(area_bot.astype(np.float32))
    sum_top = _det_sum32(area_top.astype(np.float32))
    sa = np.float32(2.0) * np.float32(c) * np.float32(sum_bot + sum_top)
    return volume, x, sa

def sor_volume_surface_area(B):
    """pass in rotated blob"""
    """Sosik and Kilfoyle surface area / volume algorithm"""
    # compute center as top-edge + radius to match MATLAB
    r = np.sum(B, axis=0).astype(np.float64)
    ri = r > 0
    r = (r / 2.0)[ri]
    y1 = np.argmax(B, axis=0).astype(np.float64) + 1.0
    y1 = y1[ri]
    center = y1 + r
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
    
