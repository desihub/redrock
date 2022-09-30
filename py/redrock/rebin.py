"""
redrock.rebin
=============

Tools for binning data.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import numba

#Since CUPY import is moved, vars block_size and cuda_source will always be
#set but this should cost no real time.
block_size = 512 #Default block size, should work on all modern nVidia GPUs

# cuda_source contains raw CUDA kernels to be loaded as CUPY module
cuda_source = r'''
        extern "C" {
            __global__ void batch_trapz_rebin(const double* x, const double* y, const double* edges, const double* myz, const int* idx, double* result, int nz, int nbin, int nbasis, int nt) {
                // This kernel performs a trapezoidal rebinning for all
                // redshifts and all bases of a template with either evenly
                // or unevenly spaced input wavelength grid (QSOs).
                //** Args:
                //       x = input x values (1-d array)
                //       y = input y values (2-d array containing all bases)
                //       edges = edges of each output bin
                //       myz = array of redshifts
                //       idx = index in input x for each boundray of output
                //           bins (2d array, nbin+1 x nz)
                //       nz, nbin, nbasis, nt = array dimensions
                //
                //** Returns:
                //       result = 3d array (nz x nbin x nbasis)

                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i - corresponds to the output array index
                if (i >= nz*nbin*nbasis) return;

                double area = 0; //define a local var to accumulate
                double yedge = 0;
                double r = 0;
                double ylo, yhi;
                //ibasis, ibin, iz = index in 3d representation of output tb array
                int ibasis = i % nbasis;
                int ibin = (i % (nbasis*nbin)) / nbasis;
                int iz = i / (nbasis*nbin);
                //idx array is used rather than calculating the start and end
                //indices of input wavelengths contributing to each output bin
                int iidx = i / nbasis + iz; //idx = nbin+1 x nz so have to add iz

                //create local vars for (1+z) and wavelength stride
                double z = (1+myz[iz]);

                //Calculate first sample beyond this bin edge
                int j = idx[iidx];
                int end_idx = idx[iidx+1];
                double xj = x[j]*z;
                double xj1 = x[j-1]*z;
                int y_idx = j+ibasis*nt;

                // - What is the y value where the interpolation crossed the edge?
                yedge = y[y_idx-1] + (edges[ibin]-xj1) * (y[y_idx]-y[y_idx-1]) / (xj-xj1);

                //r = yedge;
                // - Is this sample inside this bin?
                if (xj < edges[ibin+1]) {
                    area = 0.5 * (y[y_idx] + yedge) * (xj - edges[ibin]);
                    r += area;

                    //- Continue with interior bins
                    while (j+1 < end_idx) {
                        j++;
                        y_idx++;
                        xj = x[j]*z;
                        xj1 = x[j-1]*z;
                        area = 0.5 * (y[y_idx] + y[y_idx-1]) * (xj - xj1);
                        r += area;
                    }
                    //- Next sample will be outside this bin; handle upper edge
                    yedge = y[y_idx] + (edges[ibin+1]-xj) * (y[y_idx+1]-y[y_idx]) / (x[j+1]*z-xj);
                    area = 0.5 * (yedge + y[y_idx]) * (edges[ibin+1] - xj);
                    r += area;
                } else {
                    //- Otherwise the samples span over this bin
                    ylo = y[y_idx] + (edges[ibin]-xj) * (y[y_idx] - y[y_idx-1]) / (xj - xj1);
                    yhi = y[y_idx] + (edges[ibin+1]-xj) * (y[y_idx] - y[y_idx-1]) / (xj - xj1);
                    area = 0.5 * (ylo+yhi) * (edges[ibin+1]-edges[ibin]);
                    r += area;
                }
                result[i] = r / (edges[ibin+1]-edges[ibin]);
            }
        }
'''

def centers2edges(centers):
    """Convert bin centers to bin edges, guessing at what you probably meant

    Args:
        centers (array): bin centers,

    Returns:
        array: bin edges, lenth = len(centers) + 1

    """
    centers = np.asarray(centers)
    edges = np.zeros(len(centers)+1)
    #- Interior edges are just points half way between bin centers
    edges[1:-1] = (centers[0:-1] + centers[1:]) / 2.0
    #- edge edges are extrapolation of interior bin sizes
    edges[0] = centers[0] - (centers[1]-edges[1])
    edges[-1] = centers[-1] + (centers[-1]-edges[-2])

    return edges

# This code is purposely written in a very "C-like" way.  The logic
# being that it may help numba optimization and also makes it easier
# if it ever needs to be ported to Cython.  Actually Cython versions
# of this code have already been tested and shown to perform no better
# than numba on Intel haswell and KNL architectures.

@numba.jit
def _trapz_rebin(x, y, edges, results):
    '''
    Numba-friendly version of trapezoidal rebinning

    See redrock.rebin.trapz_rebin() for input descriptions.
    `results` is pre-allocated array of length len(edges)-1 to keep results
    '''
    nbin = len(edges) - 1
    i = 0  #- index counter for output
    j = 0  #- index counter for inputs
    yedge = 0.0
    area = 0.0

    while i < nbin:
        #- Seek next sample beyond bin edge
        while x[j] <= edges[i]:
            j += 1

        #- What is the y value where the interpolation crossed the edge?
        yedge = y[j-1] + (edges[i]-x[j-1]) * (y[j]-y[j-1]) / (x[j]-x[j-1])

        #- Is this sample inside this bin?
        if x[j] < edges[i+1]:
            area = 0.5 * (y[j] + yedge) * (x[j] - edges[i])
            results[i] += area

            #- Continue with interior bins
            while x[j+1] < edges[i+1]:
                j += 1
                area = 0.5 * (y[j] + y[j-1]) * (x[j] - x[j-1])
                results[i] += area

            #- Next sample will be outside this bin; handle upper edge
            yedge = y[j] + (edges[i+1]-x[j]) * (y[j+1]-y[j]) / (x[j+1]-x[j])
            area = 0.5 * (yedge + y[j]) * (edges[i+1] - x[j])
            results[i] += area

        #- Otherwise the samples span over this bin
        else:
            ylo = y[j] + (edges[i]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            yhi = y[j] + (edges[i+1]-x[j]) * (y[j] - y[j-1]) / (x[j] - x[j-1])
            area = 0.5 * (ylo+yhi) * (edges[i+1]-edges[i])
            results[i] += area

        i += 1

    for i in range(nbin):
        results[i] /= edges[i+1] - edges[i]

    return

def trapz_rebin(x, y, xnew=None, edges=None, myz=None, use_gpu=False):
    """Rebin y(x) flux density using trapezoidal integration between bin edges
    Optionally use GPU helper method to rebin in batch, see trapz_rebin_batch_gpu

    Notes:
        y is interpreted as a density, as is the output, e.g.

        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, edges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])
        >>> y = np.ones((2,10)) #nbasis = 2
        >>> y[1,:] = np.arange(10)
        >>> rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], use_gpu=True) #nbasis=2, GPU mode
        array([[1., 1.],
               [1., 3.],
               [1., 5.],
               [1., 7.]])
        >>> rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], myz=[0,0.5],use_gpu=True) #nbasis=2, multiple redshifts, GPU mode
        array([[[1. , 1. ],
                [1. , 3. ],
                [1. , 5. ],
                [1. , 7. ]],
               [[1. , 0.5],
                [1. , 1.5],
                [1. , 2.5],
                [1. , 3.5]]])

    Args:
        x (array): input x values.
        y (1-d or 2-d array): input y values (GPU mode allows 2-d array
            with multiple bases). 
        edges (array): (optional) new bin edges.
        myz (array): (optional) redshift array to rebin in batch on GPU,
            applying redshifts on-the-fly to x
        use_gpu (boolean): whether or not to use GPU batch mode 

    Returns:
        array: integrated results with len(results) = len(edges)-1
            In GPU batch mode, returns cp.array with shape (nz, nbin, nbasis)
            where nbin = len(results) = len(edges)-1
            if nz or nbasis is 1, this dimension will be squeezed,
            e.g. for nbasis = 1 and nz = 100, the shape will be (100, nbin)
            while for nbasis = 1 and nz = 1 the shape will be (nbin), the
            same as CPU mode.

    Raises:
        ValueError: if edges are outside the range of x or if len(x) != len(y)

    """
    if edges is None:
        edges = centers2edges(xnew)
    else:
        edges = np.asarray(edges)

    if (myz is not None and not use_gpu):
        raise ValueError('redshifts only allowed in GPU mode')
    if (use_gpu):
        return trapz_rebin_batch_gpu(x, y, edges=edges, myz=myz)

    if edges[0] < x[0] or x[-1] < edges[-1]:
        raise ValueError('edges must be within input x range')

    result = np.zeros(len(edges)-1, dtype=np.float64)

    _trapz_rebin(x, y, edges, result)

    return result

def trapz_rebin_batch_gpu(x, y, xnew=None, edges=None, myz=None):
    """Rebin y(x) flux density using trapezoidal integration between bin edges
    GPU algorithm can rebin in batch for multiple redshifts and bases and
    returns 1-d, 2-d, or 3-d array (n redshifts x n bins x n basis) where
    n basis is the optional second dimension of the y input array and
    n redshifts is the length of the optional myz array.

    Notes:
        y is interpreted as a density, as is the output, e.g.

        >>> x = np.arange(10)
        >>> y = np.ones(10)
        >>> trapz_rebin(x, y, edges=[0,2,4,6,8])  #- density still 1, not 2
        array([ 1.,  1.,  1.,  1.])
        >>> y = np.ones((2,10)) #nbasis = 2
        >>> y[1,:] = np.arange(10)
        >>> rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], use_gpu=True)
        array([[1., 1.],
               [1., 3.],
               [1., 5.],
               [1., 7.]])
        >>> rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], myz=[0,0.5],use_gpu=True)
        array([[[1. , 1. ],
                [1. , 3. ],
                [1. , 5. ],
                [1. , 7. ]],
               [[1. , 0.5],
                [1. , 1.5],
                [1. , 2.5],
                [1. , 3.5]]])


    Args:
        x (1-d array): input x values.
        y (1-d or 2-d array): input y values (for all bases).
        edges (1-d array): (optional) new bin edges.
        myz (array): (optional) redshift array to rebin in batch, applying
            redshifts on-the-fly to x


    Returns:
        cp.array: integrated results with shape (nz, nbin, nbasis)
            where nbin = len(results) = len(edges)-1
            if nz or nbasis is 1, this dimension will be squeezed,
            e.g. for nbasis = 1 and nz = 100, the shape will be (100, nbin)
            while for nbasis = 1 and nz = 1 the shape will be (nbin).

    Raises:
        ValueError: if edges are outside the range of x or if len(x) != len(y)

    """
    #import cupy here
    import cupy as cp

    if edges is None:
        edges = centers2edges(xnew)
    else:
        edges = np.asarray(edges)
    edges = cp.array(edges, dtype=cp.float64)

    if (myz is None):
        myz = cp.array([0], dtype=cp.float64)
    myz = cp.array(myz, dtype=cp.float64)

    x = cp.array(x, dtype=cp.float64)
    y = cp.array(y, dtype=cp.float64)

    #Must multiply x by 1+z for comparison, only need to look at max/min cases
    if (edges[0] < x[0]*(1+myz.max()) or edges[-1] > x[-1]*(1+myz.min())):
        raise ValueError('edges must be within input x range')

    if (len(y.shape) == 2):
        nbasis = y.shape[0]
    else:
        nbasis = 1

    #Divide edges output wavelength array by (1+myz) to get 2d array
    #of input wavelengths at each boundary in edges and
    #use serachsorted to find index for each boundary
    e2d = cp.array(edges/(1+myz[:,None]))
    idx = cp.searchsorted(x, e2d, side='right').astype(np.int32)

    # Load CUDA kernel
    cp_module = cp.RawModule(code=cuda_source)
    batch_trapz_rebin_kernel = cp_module.get_function('batch_trapz_rebin')

    nbin = cp.int32(len(edges)-1)
    nz = cp.int32(len(myz))
    n = nbin*nbasis*nz
    nt = cp.int32(len(x))

    blocks = (n+block_size-1)//block_size
    result = cp.empty((nz, nbin, nbasis), dtype=cp.float64)

    #Launch kernel and syncrhronize
    batch_trapz_rebin_kernel((blocks,), (block_size,), (x, y, edges, myz, idx, result, nz, nbin, nbasis, nt))
    #cp.cuda.Stream.null.synchronize()

    #Squeeze array and remove dims of 1
    if (result.size == 1):
        #Use flatten because squeeze results in array dim 0
        result = result.flatten()
    elif (nz == 1 or nbasis == 1):
        result = result.squeeze()
    return result

def rebin_template(template, myz, dwave, use_gpu=False):
    """Rebin a template to a set of wavelengths.

    Given a template and a single redshift - or an array of redshifts,
    rebin the template to a set of wavelength arrays.

    Args:
        template (Template): the template object
        myz (float or array of float): the redshift(s)
        dwave (dict): the keys are the "wavehash" and the values
            are a 1D array containing the wavelength grid.

    Returns:
        dict:  The rebinned template for every basis function and wavelength
            grid in dwave.  This is a dict of np or cp 3d arrays
            (nz x nlambda x nbasis)

    """
    nbasis = template.flux.shape[0]  #- number of template basis vectors
    result = dict()
    if (use_gpu):
        #In GPU mode, rebin all z and all bases in batch in parallel
        #and return dict of 3-d CUPY arrays
        for hs, wave in dwave.items():
            result[hs] = trapz_rebin(template.wave, template.flux, xnew=wave, myz=myz, use_gpu=True)
    elif (type(myz) == np.ndarray):
        #If myz is numpy array, process all redshifts and all bases (one at
        #a time) and collect results in dict of 3-d numpy arrays
        nz = len(myz)
        result = { hs:np.empty((nz, len(wave), nbasis)) for hs, wave in dwave.items() }
        for i, z in enumerate(myz):
            x = (1.+z)*template.wave
            for hs, wave in dwave.items():
                for b in range(nbasis):
                    result[hs][i,:,b] = trapz_rebin(x, template.flux[b], wave)
    else:
        #Legacy mode - a single redshift passed - return a dict with rebinned
        #template for every basis for this one redshift.
        result = { hs:np.array([ trapz_rebin((1.+myz)*template.wave, template.flux[b], wave) for b in range(nbasis) ]).transpose() for hs, wave in dwave.items() }
    return result

