# The following uses the optimized phasecong function from phasepack v1.6.1+
# with covariance_only=True for significant speedup in IFCB segmentation

# Original license reproduced below

# MIT License:

# Permission is hereby  granted, free of charge, to any  person obtaining a
# copy of this software and associated  documentation files (the "Software"),
# to deal in the Software without restriction, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# The software is provided "as is", without warranty of any kind.

# Original MATLAB version by Peter Kovesi
# <http://www.csse.uwa.edu.au/~pk/research/matlabfns/PhaseCongruency/phasecong3.m>

#Python translation by Alistair Muldal
# <alistair muldal@pharm ox ac uk>

# IFCB-specific optimizations and utilites by Joe Futrelle @ WHOI 2016

import numpy as np
import jax.numpy as jnp
from phasepack import phasecong

# IFCB-specific utility

PC_NSCALE=4
PC_NORIENT=6
PC_MIN_WAVELENGTH=2
PC_MULT=2.5
PC_SIGMA_ONF=0.55
PC_K=2.0
PC_CUTOFF=0.3
PC_G=5
PC_NOISEMETHOD=-1

def phasecong_Mm_batch(roi_batch):
    """
    IFCB-specific phase congruency function for batched processing.
    
    Args:
        roi_batch: JAX array of shape [N, H, W] where all ROIs have identical dimensions
    
    Returns:
        JAX array of shape [N, H, W] containing M + m for each ROI
    """
    # Ensure roi_batch is a JAX array
    roi_batch = jnp.asarray(roi_batch, dtype=jnp.float32)
    
    M_batch, m_batch = phasecong(roi_batch, 
                                nscale=PC_NSCALE,
                                norient=PC_NORIENT, 
                                minWaveLength=PC_MIN_WAVELENGTH,
                                mult=PC_MULT,
                                sigmaOnf=PC_SIGMA_ONF,
                                k=PC_K,
                                cutOff=PC_CUTOFF,
                                g=PC_G,
                                noiseMethod=PC_NOISEMETHOD,
                                covariance_only=True)
    
    return M_batch + m_batch

def phasecong_Mm(roi):
    """
    IFCB-specific phase congruency function using optimized phasepack.
    
    Uses phasepack.phasecong with covariance_only=True and IFCB-optimized parameters.
    Returns M + m (sum of maximum and minimum covariance moments).
    
    Supports both single ROIs and batched ROIs for backward compatibility.
    """
    roi = jnp.asarray(roi, dtype=jnp.float32)
    
    if roi.ndim == 2:
        # Single ROI - add batch dimension, process, then remove batch dimension
        roi_batch = roi[None, :, :]  # Shape: [1, H, W]
        result_batch = phasecong_Mm_batch(roi_batch)
        # Check if batch dimension still exists before trying to remove it
        if result_batch.ndim == 3:
            return result_batch[0]  # Remove batch dimension, return [H, W]
        else:
            return result_batch  # Already 2D, return as is
    elif roi.ndim == 3:
        # Already batched - process directly
        return phasecong_Mm_batch(roi)
    else:
        raise ValueError(f"ROI must be 2D (single) or 3D (batch), got shape {roi.shape}")
