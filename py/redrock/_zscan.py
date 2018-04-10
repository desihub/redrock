"""
Low-level optimized functions for chi^2 fitting.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import numba

# This code is purposely written in a very "C-like" way.  The logic
# being that it may help numba optimization and also makes it easier
# if it ever needs to be ported to Cython.  Actually Cython versions
# of this code have already been tested and shown to perform no better
# than numba on Intel haswell and KNL architectures.

#@numba.jit
def _zchi2_one(Tb, weights, flux, wflux, zcoeff):
    """Calculate a single chi2.

    For one redshift and a set of spectral data, compute the chi2 for template
    data that is already on the correct grid.
    """
    zchi2 = 0.0
    M = np.zeros_like(flux)
    y = np.zeros_like(zcoeff)
    model = np.zeros_like(flux)

    M = Tb.T.dot(np.multiply(weights[:,None], Tb))
    y = Tb.T.dot(wflux)
    
    try:
        zcoeff[:] = np.linalg.solve(M, y)
    except np.linalg.LinAlgError:
        return 9e99

    model = Tb.dot(zcoeff)

    zchi2 = np.dot( (flux - model)**2, weights )

    return zchi2
