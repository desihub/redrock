"""
redrock.zscan
=============

Algorithms for scanning redshifts.
"""

from __future__ import division, print_function

import time
import sys
import traceback
import numpy as np
from scipy.optimize import lsq_linear, nnls

#try:
#    import cupy as cp
#    import cupyx.scipy
#    import cupyx
#    cupy_available = cp.is_available()
#except ImportError:
#    cupy_available = False
#Moved CUPY import to local within method.  Set global var to None here
cp = None
#On consumer grade GPUs with lower memory this will be set to True
#and attempts will be made to minimize memory usage to under 4 GB
cp_memcheck = False

from .utils import elapsed

from .targets import distribute_targets

#Since CUPY import is moved, vars block_size and cuda_source will always be
#set but this should cost no real time.
block_size = 512 #Default block size, should work on all modern nVidia GPUs

# cuda_source contains raw CUDA kernels to be loaded as CUPY module

cuda_source = r'''
        extern "C" {
            __global__ void batch_dot_product_sparse(const double* Rcsr_values, const int* Rcsr_cols, const int* Rcsr_indptr, const double* tdata, double* tb, int nrows, int ncols, int nbasis, int nt) {
                // This kernel performs a batch dot product of a sparse matrix Rcsr
                // with a set of redshifted templates
                //** Args:
                //       Rcsr_values, Rcsr_cols, Rcsr_indptr = individualarrays from sparse matrix
                //           (Rcsr = sparse array, ncols x nrows)
                //       tdata = redshifted templates, nt x ncols x nbasis
                //       tb = output array = nt x ncols x nbasis
                //       nrows, ncols, nbasis, nt = array dimensions

                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i - corresponds to the output array index
                if (i >= ncols*nbasis*nt) return;
                double x = 0; //define a local var to accumulate
                //ibatch, icol, it = index in 3d representation of output tb array
                //icol also == row in Rcsr input
                int ibatch = i % nbasis;
                int icol = (i % (nbasis*ncols)) / nbasis;
                int it = i / (nbasis*ncols);
                int t_start = it*nbasis*ncols; //first index in tdata for this thread
                int row = icol;

                int col;
                //loop over all nonzero entries in sparse matrix and compute dot product
                for (int j = Rcsr_indptr[row]; j < Rcsr_indptr[row+1]; j++) {
                    col = Rcsr_cols[j];
                    x += Rcsr_values[j] * tdata[t_start+nbasis*col+ibatch];
                }
                tb[i] = x;
                return;
            }

            __global__ void batch_dot_product_3d3d(const double* a, const double* b, double* M, int nrows, int nbasis, int nt, int nparallel, int transpose_a) {
                // This kernel computes a batch dot product of two 3-d arrays,
                // a and b such that for every i
                //     M[i,:,:] = a[i,:,:].dot(b[i,:,:])
                // This replicates the CUPY code M = a @ b but more efficiently
                // It will use nparallel threads to compute the dot product
                // for each output element in M.  Each thread will handle
                //  nrows/nparallel products and sums into an intermediate
                // local variable // and then an atomicAdd will be used to add
                // this intermediate sum to the output array.

                // This replicates the python command:
                //     M = Tb.T.dot(np.multiply(weights[:,None], Tb))
                // for all M where a = Tb.T and b = weights[:,None]*Tb
                //
                //** Args:
                //       a = a 3-d array
                //       b = a 3-d array
                //       weights = the weights array for this target (1d, size = nrows)
                //       wflux = the wflux array for this target (1d, size = nrows)
                //       M = the output M array (nt x nbasis x nbasis)
                //       y = the output y array (nt x nbasis)
                //       nrows, nbasis, nt = array dimensions
                //       nparallel = number of parallel threads to used for each output

                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i
                if (i >= nbasis*nbasis*nt*nparallel) return;

                int m_idx = i / nparallel; //index in output M array
                int t = m_idx / (nbasis*nbasis); //target number
                int a_row = (m_idx % (nbasis*nbasis)) % nbasis; //row in a array
                int b_row = (m_idx % (nbasis*nbasis)) / nbasis; //row in b array

                int stride = nrows/nparallel; //stride to divide up nparallel threads

                int start = (threadIdx.x % nparallel)*stride; //start index in nrows dim
                int end = ((threadIdx.x % nparallel)+1)*stride; //end index in nrows dim
                if (threadIdx.x % nparallel == (nparallel-1)) end = nrows;
                int a_idx = t*nrows*nbasis + a_row*nrows; // 1-d index for first element to be processed by this thread
                if (transpose_a) a_idx = t*nrows*nbasis + a_row;
                int b_idx = t*nrows*nbasis + b_row; // 1-d index for first element to be processed by this thread
                double x = 0; //define local var to accumulate

                //perform intermediate sum dot product for this thread
                if (transpose_a) {
                    for (int j = start; j < end; j++) {
                        //stride by nbasis
                        x += a[a_idx+j*nbasis] * b[b_idx+j*nbasis];
                    }
                } else {
                    for (int j = start; j < end; j++) {
                        //stride by nbasis
                        x += a[a_idx+j] * b[b_idx+j*nbasis];
                    }
                }
                //use atomic add to avoid collisions between threads
                atomicAdd(&M[m_idx], x);
            }

            __global__ void batch_dot_product_3d2d(const double* tb, const double* zc, double* model, int nrows, int nbasis, int nt) {
                // This kernel computes a batch dot product of Tb (a 3-d array)
                // and zc (a 2-d array), the result of the matrix solution of
                // M and y, for all templates (nt) in parallel.  It results in
                // the 2-d model array.  Each thread computes an element in the
                // output model array.  It replaces the python code:
                //     model = Tb.dot(cupy.array(zc))
                //** Args:
                //       tb = the Tb array, the stacked output from all 3 filters from
                //           batch_dot_product_sparse, for all redshift templates (nt x nrows x nbasis)
                //       zc = the zc array, the output of
                //           zc = cp.linalg.solve(all_M, all_y)
                //           shape = (nt x nbasis)
                //       model = the output of the dot product, (nt x nrows)
                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i
                if (i >= nrows*nt) return;
                int it = i / nrows; //target num
                int row = i % nrows; //row num
                int i_tb = it * nrows * nbasis + row * nbasis; //start index in Tb array
                int i_zc = it * nbasis; //start index in zc array
                double x = 0; //use local var to accumulate
                //compute dot product
                for (int j = 0; j < nbasis; j++) {
                    x += tb[i_tb+j] * zc[i_zc+j];
                }
                //copy to output
                model[i] = x;
            }

        }
'''

#This is used by original CPU algorithm
#It is called by archeypes so keep for now
def _zchi2_one(Tb, weights, flux, wflux, zcoeff, solve_matrices_algorithm):
    """Calculate a single chi2.

    For one redshift and a set of spectral data, compute the chi2 for template
    data that is already on the correct grid.
    """

    M = Tb.T.dot(np.multiply(weights[:,None], Tb))
    y = Tb.T.dot(wflux)

    try:
        zcoeff[:] = solve_matrices(M, y, solve_algorithm=solve_matrices_algorithm, use_gpu=False)
    except np.linalg.LinAlgError:
        return 9e99
    except NotImplementedError:
        return 9e99

    model = Tb.dot(zcoeff)

    zchi2 = np.dot( (flux - model)**2, weights )

    return zchi2

def spectral_data(spectra):
    """Compute concatenated spectral data products.

    This helper function builds full length array quantities needed for the
    chi2 fit.

    Args:
        spectra (list): list of Spectrum objects.

    Returns:
        tuple: (weights, flux, wflux) concatenated values used for single
            redshift chi^2 fits.

    """
    weights = np.concatenate([ s.ivar for s in spectra ])
    flux = np.concatenate([ s.flux for s in spectra ])
    wflux = weights * flux
    return (weights, flux, wflux)

#This is used by original CPU algorithm
#It is called by archeypes so keep for now
def calc_zchi2_one(spectra, weights, flux, wflux, tdata, solve_matrices_algorithm):
    """Calculate a single chi2.

    For one redshift and a set of spectra, compute the chi2 for template
    data that is already on the correct grid.

    Args:
        spectra (list): list of Spectrum objects.
        weights (array): concatenated spectral weights (ivar).
        flux (array): concatenated flux values.
        wflux (array): concatenated weighted flux values.
        tdata (dict): dictionary of interpolated template values for each
            wavehash.

    Returns:
        tuple: chi^2 and coefficients.

    """
    Tb = list()
    nbasis = None
    for s in spectra:
        key = s.wavehash
        if nbasis is None:
            nbasis = tdata[key].shape[1]
            #print("using ",nbasis," basis vectors", flush=True)
        Tb.append(s.Rcsr.dot(tdata[key]))
    Tb = np.vstack(Tb)
    zcoeff = np.zeros(nbasis, dtype=np.float64)
    zchi2 = _zchi2_one(Tb, weights, flux, wflux, zcoeff, solve_matrices_algorithm)

    return zchi2, zcoeff

def per_camera_coeff_with_least_square_batch(target, tdata, weights, flux, wflux, nleg, narch, method=None, n_nbh=None, prior=None, use_gpu=False, bands=None):
    
    """This function calculates coefficients for archetype mode in each camera using normal linear algebra matrix solver or BVLS (bounded value least square) method

    BVLS described in : https://www.stat.berkeley.edu/~stark/Preprints/bvls.pdf

    Scipy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html

    Args:
        target (object): target object
        tdata (dict): template data for model fit for ALL archetypes
        weights (array): concatenated spectral weights (ivar).
        flux (array): concatenated flux values.
        wflux (array): concatenated weighted flux values.
        nleg (int): number of Legendre polynomials
        narch (int): number of archetypes
        method (string): 'PCA', 'BVLS', 'NMF', or 'NNLS' (same as NMF)
        n_nbh (int): number of nearest best archetypes
        prior (array): prior matrix added to the Legendre coefficients (1/sigma^2)
        use_gpu (bool): use GPU or not
        bands (list): list of wavelength bands
    
    Returns:
        coefficients and chi2

    """

    ### TODO - implement BVLS on GPU
    # number of cameras in DESI: b, r, z
    spectra = target.spectra
    ncam = len(bands)
    nbasis = n_nbh+nleg*ncam # n_nbh : for actual physical archetype(s), nleg: number of legendre polynomials, ncamera: number of cameras

    ret_zcoeff = {}
    ret_zcoeff['alpha'] = []
    for b in bands: # bands as save in targets object
        ret_zcoeff[b] = []

    new_bands = sorted(bands) # saves as correct order

    #Setup dict of solver args to pass bounds to solver
    method = method.upper()
    solver_args = dict()
    if (method == 'BVLS'):
        #only positive coefficients are allowed for the archetypes
        bounds = np.zeros((2, nbasis))
        bounds[0][n_nbh:]=-np.inf #constant and slope terms in archetype method (can be positive or negative)
        bounds[1] = np.inf
        solver_args['bounds'] = bounds

    #Use branching options because GPU is faster in batch in 3d
    #but due to timing weirdness in numpy, CPU is faster looping over
    #narch and calling calc_zchi2_batch one arch at a time.
    if (use_gpu):
        #Use batch 3d array on GPU with CuPy to calculate new tdata array
        for i, hs in enumerate(tdata):
            tdata2 = np.zeros_like(tdata[hs], shape=(tdata[hs].shape[0], tdata[hs].shape[1], nbasis))
            tdata2[:,:,:n_nbh] = tdata[hs][:,:,:n_nbh]
            tdata2[:,:,n_nbh+i*nleg:n_nbh+(i+1)*nleg] = tdata[hs][:,:,n_nbh:]
            tdata[hs] = tdata2
        (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, tdata, weights, flux, wflux, narch, nbasis, solve_matrices_algorithm=method, solver_args=solver_args, prior=prior, use_gpu=use_gpu)
    else:
        #Create zzchi2, zcoeff, and tdata2 dict here
        tdata2 = dict()
        zzchi2 = np.zeros(narch, dtype=np.float64)
        zzcoeff = np.zeros((narch,  n_nbh+ncam*(nleg)), dtype=np.float64)
        #Loop over all arch
        for j in range(narch):
            #Create a 2d array in each color for tdata2 then add 3rd dimension of rank 1
            for i, hs in enumerate(tdata):
                tdata2[hs] = np.zeros((tdata[hs].shape[1], nbasis))
                for k in range(n_nbh):
                    tdata2[hs][:,k] = tdata[hs][j,:,k] # these are nearest archetype
                if nleg>0:
                    for k in range(n_nbh, n_nbh+nleg):
                        tdata2[hs][:,k+nleg*i] = tdata[hs][j,:,k] # Legendre polynomials terms
                tdata2[hs] = tdata2[hs][None,:,:]
            zzchi2[j], zzcoeff[j] = calc_zchi2_batch(spectra, tdata2, weights, flux, wflux, 1, nbasis, solve_matrices_algorithm=method, solver_args=solver_args, prior=prior, use_gpu=use_gpu)

    # saving leading archetype coefficients in correct order
    ret_zcoeff['alpha'] = [zzcoeff[:,k] for k in range(n_nbh)] # archetype coefficient(s)
    # logic in case nearest neighbour is used
    if n_nbh>1:
        ret_zcoeff['alpha'] = np.array([ret_zcoeff['alpha'][jj][0] for jj in range(n_nbh)])[None,:]
    else:
        ret_zcoeff['alpha'] = ret_zcoeff['alpha'][0][:,None]
    if nleg>=1:
        split_coeff =  np.split(zzcoeff[:,n_nbh:], ncam, axis=1) # n_camera = 3
        # In target spectra redrock saves values as 'b', 'z', 'r'.
        # So just re-ordering them here to 'b', 'r', 'z' for easier reading
        old_coeff = {band: split_coeff[i] for i, band in enumerate(bands)}

        for band in new_bands:# 3 cameras
            ret_zcoeff[band] = old_coeff[band]
    coeff = np.concatenate(list(ret_zcoeff.values()), axis=1)
    #print(f'{time.time()-start} [sec] took for per camera BVLS method\n')
    return zzchi2, coeff

def batch_dot_product_sparse(spectra, tdata, nz, use_gpu):
    """Calculate a batch dot product of the 3 sparse matrices in spectra
    with every template in tdata.  Sparse matrix libraries are used
    to perform the dot products.

    Args:
        spectra (list): list of Spectrum objects.
        tdata (dict): dictionary of interpolated template values for each
            wavehash.
        nz (int): number of templates
        use_gpu (bool): use GPU or not

    Returns:
        Tbs (list): dot products of these 3 spectra with all templates

    """

    if (use_gpu):
        #Use GPU to do dot products in batch
        return _batch_dot_product_sparse_gpu(spectra, tdata)

    #Need to find shape of output array of batch dot product
    nrows = 0
    nbasis = None
    for key in tdata:
        nrows += tdata[key].shape[1]
        if (nbasis is None):
            nbasis = tdata[key].shape[2]

    #Create empty array rather than stacking a list - faster
    Tbs = np.empty((nz, nrows, nbasis))
    #Loop over all templates
    for i in range(nz):
        irow = 0
        for s in spectra:
            key = s.wavehash
            curr_tb = s.Rcsr.dot(tdata[key][i,:,:])
            #Copy this dot product result into the Tbs array
            Tbs[i, irow:irow+curr_tb.shape[0],:] = curr_tb
            irow += curr_tb.shape[0]
    return Tbs

def _batch_dot_product_sparse_gpu(spectra, tdata):
    """GPU implementation.
    Calculate a batch dot product of the 3 sparse matrices in spectra
    with every template in tdata.  A CUDA kernel replicates the functionality
    of the scipy sparse matrix dot product but done for every template
    in parallel so that the kernel is only called once per spectrum.

    Args:
        spectra (list): list of Spectrum objects.
        tdata (dict): dictionary of interpolated template values for each
            wavehash.

    Returns:
        Tbs (cp.array): dot products of these 3 spectra with all templates

    """

    # Load CUDA kernel
    cp_module = cp.RawModule(code=cuda_source)
    batch_dot_product_sparse_kernel = cp_module.get_function('batch_dot_product_sparse')
    Tbs = list()

    for s in spectra:
        key = s.wavehash
        #Array dimensions needed by CUDA kernel
        nrows = cp.int32(s.Rcsr.shape[1])
        ncols = cp.int32(s.Rcsr.shape[0])
        nbasis = cp.int32(tdata[key].shape[2])
        nt = cp.int32(tdata[key].shape[0])

        #Use actual numpy arrays that represent sparse array - .data, .indices, and .indptr
        #Use batch_dot_product_sparse kernel to perform dot product in parallel for all templates
        #for this (target, spectrum) combination.
        #Allocate CUPY arrays and calculate number of blocks to use.
        n = tdata[key].size
        blocks = (n+block_size-1)//block_size
        Rcsr_values = cp.asarray(s.Rcsr.data, cp.float64)
        Rcsr_cols = cp.asarray(s.Rcsr.indices, cp.int32)
        Rcsr_indptr = cp.asarray(s.Rcsr.indptr, cp.int32)
        curr_tb = cp.empty((nt, ncols, nbasis))
        #Launch kernel and syncrhronize
        batch_dot_product_sparse_kernel((blocks,), (block_size,), (Rcsr_values, Rcsr_cols, Rcsr_indptr, tdata[key], curr_tb, nrows, ncols, nbasis, nt))

        #Commented out synchronize - needed for timing kernels but we still
        #get execution of this kernel before data is needed for next kernel
        #so output is the same and slightly faster without synchronize
        #cp.cuda.Stream.null.synchronize()
        #Append to list
        Tbs.append(curr_tb)
    #Use CUPY.hstack to combine into one nt x ncols x nbasis array
    Tbs = cp.hstack(Tbs)
    #cp.cuda.Stream.null.synchronize()
    return Tbs

def dot_product_sparse_one(spectra, tdata, i):
    """Calculate a dot product of the 3 sparse matrices in spectra
    with ONE template in tdata.  Sparse matrix libraries are used
    to perform the dot products.

    Args:
        spectra (list): list of Spectrum objects.
        tdata (dict): dictionary of interpolated template values for each
            wavehash - arrays are 3d (nz x nlambda x nbasis)
        i (int): index of this redshift

    Returns:
        Tb (array): dot products of these 3 spectra with ONE templates

    """

    Tb = list()
    for s in spectra:
        key = s.wavehash
        Tb.append(s.Rcsr.dot(tdata[key][i,:,:]))
    Tb = np.vstack(Tb)
    return Tb


def calc_batch_dot_product_3d2d(Tbs, zc, use_gpu):
    """Calculate a batch dot product of the 3d array Tbs with the 2d
    array zc.  The 3-d array shape is A x B x C and the 2-d array
    shape is A x C.  The resulting output 2-d array shape is A x B.
    E.g., for all A a dot product of a 2d array of shape B x C
    is performed with a 1-d array of shape C.
    These are non-sparse numpy arrays.

    Args:
        Tbs (array): the stacked output from all 3 filters from
            batch_dot_product_sparse, for all redshift templates
            (nz x nrows x nbasis)
        zc (array): zcoeffs, the 2-d array output of
            zc = linalg.solve(all_M, all_y)
            (nz x nbasis)
        use_gpu (bool): use GPU or not

    Returns:
        model (array): the output of the dot product, (nz x nrows)

    """

    if (use_gpu):
        return _calc_batch_dot_product_3d2d_gpu(Tbs, zc)

    #Get array dims to reshape model array to 2d
    nz = zc.shape[0]
    nrows = Tbs[0].shape[0]
    model = (Tbs@zc[:, :, None]).reshape((nz, nrows))
    return model


def _calc_batch_dot_product_3d2d_gpu(Tbs, zc):
    """GPU implementation.
    Calculate a batch dot product of the 3d array Tbs with the 2d
    array zc.  The 3-d array shape is A x B x C and the 2-d array
    shape is A x C.  The resulting output 2-d array shape is A x B.
    E.g., for all A a dot product of a 2d array of shape B x C
    is performed with a 1-d array of shape C.
    These are non-sparse numpy arrays.

    Args:
        Tbs (array): the stacked output from all 3 filters from
            batch_dot_product_sparse, for all redshift templates
            (nz x nrows x nbasis)
        zc (array): zcoeffs, the 2-d array output of
            zc = linalg.solve(all_M, all_y)
            (nz x nbasis)

    Returns:
        model (array): the output of the dot product, (nz x nrows)

    """

    #Use batch_dot_product_3d2d kernel to compute model array
    # Load CUDA kernel
    cp_module = cp.RawModule(code=cuda_source)
    batch_dot_product_3d2d_kernel = cp_module.get_function('batch_dot_product_3d2d')

    #Array dims needed by CUDA:
    nz = zc.shape[0]
    nrows = Tbs[0].shape[0]
    n = nrows * nz
    nbasis = zc.shape[1]

    #Allocate CUPY array and calc blocks to be used
    blocks = (n+block_size-1)//block_size
    model = cp.empty((nz, nrows), cp.float64)
    #Launch kernel and synchronize
    batch_dot_product_3d2d_kernel((blocks,), (block_size,), (Tbs, zc, model, nrows, nbasis, nz))
    #cp.cuda.Stream.null.synchronize()
    return model

###!!! NOTE - used in v2 and v3 algorithms as an alternative to straight CuPy
###    computation of M and y or the calc_M_y_batch method as a middle ground
###    between maximum speed and maximum maintainability.
###    This only offloads the computationally expensive dot product itself
###    (and optionally the transpose) because the CuPy @ matrix multiplication
###    seems to have a bug on Volta architecure GPUs.
###    This is the equivalent of M = a @ b
###    (Or if transpose_a is true, M = a.swapaxes(-2, -1) @ b)
def calc_batch_dot_product_3d3d_gpu(a, b, transpose_a=False, fullprecision=True):
    """GPU implementation.
    Calculate a batch dot product of the 3d array a with the 3d
    array b.  The 3-d array shape is A x B x C and the 2-d array
    shape is A x C.  The resulting output 2-d array shape is A x B.
    E.g., for all A a dot product of a 2d array of shape B x C
    is performed with a 1-d array of shape C.
    These are non-sparse numpy arrays.

    Args:
        a (array): a 3-d array (nz x ncols x nrows)
            In practice, the Tb array, the stacked output from all 3 filters
            from batch_dot_product_sparse, for all redshift templates
            (nz x nrows x nbasis) which should have its transpose
            performed yielding shape (nz x nbasis x nrows).
        b (array): another 3-d array (nz x nrows x ncols)
            In practice, the Tb array multiplied y weights, shape
            (nz x nrows x nbasis)
        transpose_a (bool): Whether or not to transpose the a array
            before performing the dot product
        fullprecision (bool): Whether or not to ensure reproducibly identical
            results.  The context is that it can be faster to use many
            parallel threads to compute a single output array element of the
            dot product result, but due to floating point rounding issues,
            the random order of the summation can change the result by
            order 1e-16 (e.g. a+b+c != c+a+b).  When set to true, the
            number of parallel threads is decreased to ensure reproducibility
            at a trade-off of speed.

    Returns:
        M (array): the output of the dot product, (nz x ncols x ncols)
            such that M[i,:,:] = a[i,:,:].dot(b[i,:,:])

    """

    #Use batch_dot_product_3d3d kernel to compute model array
    # Load CUDA kernel
    import cupy as cp
    cp_module = cp.RawModule(code=cuda_source)
    batch_dot_product_3d3d_kernel = cp_module.get_function('batch_dot_product_3d3d')

    #Array dims needed by CUDA:
    nz = cp.int32(a.shape[0])
    if (transpose_a):
        nrows = cp.int32(a.shape[1])
        ncols = cp.int32(a.shape[2])
    else:
        nrows = cp.int32(a.shape[2])
        ncols = cp.int32(a.shape[1])
    transpose_a = cp.int32(transpose_a)

    nparallel = cp.int32(4)
    if (fullprecision):
        nparallel = cp.int32(4)
    elif (nz <= 512):
        #With smaller arrays, use more parallel threads in order to maximize
        #parallelism and leverage the full resources of the GPU
        nparallel = cp.int32(64)
    #Create CUPY arrays and calculate number of blocks
    n = nz*ncols*ncols*nparallel
    blocks = (n+block_size-1)//block_size
    all_M = cp.zeros((nz, ncols, ncols))

    #Launch kernel and synchronize
    batch_dot_product_3d3d_kernel((blocks,), (block_size,), (a, b, all_M, nrows, ncols, nz, nparallel, transpose_a))
    #cp.cuda.Stream.null.synchronize()
    return all_M


###!!! NOTE - this is called in the v3 algorithm
###    In this version, everything is done in batch on the GPU but the
###    templates are looped over on the CPU.  The operations performed
###    are very obviously analagous though and should be highly
###    maintainable.  The main difference is the extra loop on the CPU version

def calc_zchi2_batch(spectra, tdata, weights, flux, wflux, nz, nbasis, solve_matrices_algorithm=None, solver_args=None, use_gpu=False, fullprecision=True, prior=None):
    
    """Calculate a batch of chi2.
    For many redshifts and a set of spectral data, compute the chi2 for
    template data that is already on the correct grid.

    Args:
        spectra (list): list of Spectrum objects.
        tdata (dict): dictionary of interpolated template values for each
            wavehash - arrays are 3d (nz x nlambda x nbasis)
        weights (array): concatenated spectral weights (ivar).
        flux (array): concatenated flux values.
        wflux (array): concatenated weighted flux values.
        nz (int): number of templates
        nbasis (int): nbasis
        solve_matrices_algorithm (string): PCA, BLVS, or NMF - the algorithm
            used to solve matrix equation
        solver_args (dict): Optional args to pass to solver, such as bounds
            array for BVLS.
        use_gpu (bool): use GPU or not
        fullprecision (bool): Whether or not to ensure reproducibly identical
            results.  See calc_batch_dot_product_3d3d_gpu.
        prior (2d array): prior matrix on coefficients (1/sig**2)     

    Returns:
        zchi2 (array): array with one element per redshift for this target
        zcoeff (array): array of best fit template coefficients

    """
    zchi2 = np.zeros(nz)
    if (weights.sum() == 0):
        zchi2[:] = 9e99
        zcoeff = np.zeros((nz, nbasis))
        return (zchi2, zcoeff)
    if (use_gpu):
        global cp
        import cupy as cp
        #On the GPU, all operations are batch operations for all templates
        #in parallel.

        #1) batch_dot_product_sparse will compute dot products of all
        #spectra with all templates in batch and return a 3D array of
        #size (nz x ncols x nbasis).
        Tbs = batch_dot_product_sparse(spectra, tdata, nz, use_gpu)
        if (cp_memcheck):
            #Free memory on consumer grade GPUs with low resources
            mpool = cp.get_default_memory_pool()
            mpool.free_all_blocks()

        #2) On the GPU, M and y are computed for all templates at once
        #CUPY swapaxes is the equivalent of the transpose in CPU mode
        #and the @ matrix multiplication operator performs a dot
        #product for each template.

        ###!!! NOTE - there are 3 different options for calculating the
        ###    M and y arrays -
        ###    A) Straight CUPY, which works well on perlmutter with a
        ###        runtime of 6.2s on 1 GPU and 2.0s on 4 GPUs, but is
        ###        unusably slow on Volta generation GPUs (16.8s for only
        ###        10 targets on a 1660 Super).
        ###    B) calc_M_y_batch, the custom CUDA kernel, which is the
        ###        fastest at 2.9s on 1 GPU and 0.7s on 4 GPUs (and 0.7s
        ###        for 10 targets on a 1660 Super) but is the most difficult
        ###        from a maintenance perspective
        ###    C) Use the calc_batch_dot_product_3d3d_gpu kernel to offload
        ###        only the matrix multiplication for M (and transpose of
        ###        Tbs) but use CUPY for everything else.  This strikes a
        ###        middle ground that is very maintainable but removes the
        ###        bottleneck of the CUPY Volta issue.  5.7s on 1 GPU and
        ###        1.8s on 4 GPUs on Perlmutter; 1.6s for 10 targets on
        ###        1660 Super.
        ###!!! NOTE - uncomment the 2 lines below to run (A)
        #all_M = Tbs.swapaxes(-2, -1) @ (weights[None, :, None] * Tbs)
        #all_y = (Tbs.swapaxes(-2, -1) @ wflux)
        ###!!! NOTE - uncomment the below line to run (B)
        #(all_M, all_y) = calc_M_y_batch(Tbs, weights, wflux, nz, nbasis)
        ###!!! NOTE - uncomment the 2 lines below to run (C)
        all_M = calc_batch_dot_product_3d3d_gpu(Tbs, (weights[None, :, None] * Tbs), transpose_a=True, fullprecision=fullprecision)
        all_y = (Tbs.swapaxes(-2, -1) @ wflux)
        ###!!! NOTE - uncomment the 2 lines below to run an alternative
        ###    version of (C) that does the transpose on the CPU - this seems
        ###    to needlessly waste time though
        #all_M = calc_batch_dot_product_3d3d_gpu(cp.ascontiguousarray(Tbs.swapaxes(-2, -1)), (weights[None, :, None] * Tbs))
        #all_y = (Tbs.swapaxes(-2, -1) @ wflux)
        if prior is not None:
            all_M += cp.asarray(prior)

        #3) Use new helper method solve_matrices to use appropriate method
        #for this template to solve for zcoeff in batch for all_M and all_y.
        #There is no Error thrown by cupy's version of linalg.solve so just
        #need to catch NotImplementedError.
        try:
            zcoeff = solve_matrices(all_M, all_y, solve_algorithm=solve_matrices_algorithm,
                                    solver_args=solver_args, use_gpu=True)
        except NotImplementedError:
            zchi2[:] = 9e99
            zcoeff = np.zeros((nz, nbasis))
            return (zchi2, zcoeff)

        #4) calc_batch_dot_product_3d2d will compute the dot product
        #of Tbs and zcoeff for all templates in parallel.
        #It is the same as model[i,:,:] = Tbs[i,:,:].dot(zcoeff[i,:])
        model = calc_batch_dot_product_3d2d(Tbs, zcoeff, use_gpu)

        #5) On the GPU, (flux-model)*(flux-model) is faster than
        #(flux-model)**2.  The @ matrix multiplication operator performs
        #a dot product for each template.  get() copies the data back
        #from the GPU to the numpy array allocated for zchi2.
        zchi2[:] = (((flux - model)*(flux-model)) @ weights).get()
        #Copy data from GPU to numpy arrays
        zcoeff = zcoeff.get()

        if (cp_memcheck):
            #Free memory on consumer grade GPUs with low resources
            del Tbs
            del all_M
            del all_y
            del model
            mpool = cp.get_default_memory_pool()
            mpool.free_all_blocks()
    else:
        zcoeff = np.zeros((nz, nbasis))
        #On the CPU, the templates are looped over and all operations
        #are performed on one template at a time.

        for i in range(nz):
            #1) dot_product_sparse_one will compute dot products of all
            #spectra with ONE template and return a 2D array of size
            #(ncols x nbasis)
            Tb = dot_product_sparse_one(spectra, tdata, i)

            #2) On the CPU, M and y are computed for each template
            M = Tb.T.dot(np.multiply(weights[:,None], Tb))
            y = Tb.T.dot(wflux)
            if prior is not None:
                M += prior

            #3) Use new helper method solve_matrices to use appropriate method
            #for this template to solve for zcoeff for each M, y.
            #Catch LinAlgError and NotImplementedError
            try:
                zcoeff[i,:] = solve_matrices(M, y, solve_algorithm=solve_matrices_algorithm, solver_args=solver_args, use_gpu=False)
            except np.linalg.LinAlgError:
                zchi2[i] = 9e99
                continue
            except NotImplementedError:
                zchi2[i] = 9e99
                continue

            #4) Calculate dot products individually for each template
            model = Tb.dot(zcoeff[i,:])

            #5) Calculate this zchi2 element individually for each template
            zchi2[i] = np.dot( (flux - model)**2, weights )
    return (zchi2, zcoeff)


###!!! NOTE - this is the main method for the v3 algorithm
###    In this version, everything is done in batch on the GPU but the
###    templates are looped over on the CPU.  The operations performed
###    are very obviously analagous though and should be highly
###    maintainable.  The main difference is the extra loop on the CPU version
def calc_zchi2(target_ids, target_data, dtemplate, progress=None, use_gpu=False):
    """Calculate chi2 vs. redshift for a given PCA template.

    New CPU/GPU algorithms June 2022

    Args:
        target_ids (list): targets IDs.
        target_data (list): list of Target objects.
        dtemplate (DistTemplate): distributed template data
        progress (multiprocessing.Queue): optional queue for tracking
            progress, only used if MPI is disabled.

    Returns:
        tuple: (zchi2, zcoeff, zchi2penalty) with:
            - zchi2[ntargets, nz]: array with one element per target per
                redshift
            - zcoeff[ntargets, nz, ncoeff]: array of best fit template
                coefficients for each target at each redshift
            - zchi2penalty[ntargets, nz]: array of penalty priors per target
                and redshift, e.g. to penalize unphysical fits

    """
    if (use_gpu):
        #Use global to import cupy here - it will not be needed to be imported
        #in any other method
        global cp
        import cupy as cp
        #Get CUDA device and check available memory.
        #If < 3 GB free set cp_memcheck
        d = cp.cuda.Device()
        if (d.mem_info[0] < 3*1024*1024*1024):
            global cp_memcheck
            cp_memcheck = True

    nz = len(dtemplate.local.redshifts)
    ntargets = len(target_ids)
    nbasis = dtemplate.template.nbasis

    zchi2 = np.zeros( (ntargets, nz) )
    zchi2penalty = np.zeros( (ntargets, nz) )
    zcoeff = np.zeros( (ntargets, nz, nbasis) )

    # Redshifts near [OII]; used only for galaxy templates
    if dtemplate.template.template_type == 'GALAXY':
        isOII = (3724 <= dtemplate.template.wave) & \
            (dtemplate.template.wave <= 3733)
        OIItemplate = dtemplate.template.flux[:,isOII].T

    ## Redshifted templates are now already in format needed - dict of 3d
    # arrays (CUPY or numpy).
    tdata = dtemplate.local.data

    for j in range(ntargets):
        if (use_gpu):
            #Use new helper method gpu_spectral_data() that will copy to GPU
            #on first access and recall data in subsequent calls instead of
            #copying every time
            (weights, flux, wflux) = target_data[j].gpu_spectral_data()
        else:
            #Use spectral_data() to get numpy arrays
            (weights, flux, wflux) = spectral_data(target_data[j].spectra)
        if np.sum(weights) == 0:
            zchi2[j,:] = 9e99
            #Update progress for multiprocessing!!
            if dtemplate.comm is None and progress is not None:
                progress.put(1)
            continue

        # Solving for template fit coefficients for all redshifts.
        # We use the pre-interpolated templates for each
        # unique wavelength range.

        # Use helper method calc_zchi2_batch to calculate zchi2 and zcoeff
        # for all templates for all three spectra for this target

        # For coarse z scan, use fullprecision = False to maximize speed
        (zchi2[j,:], zcoeff[j,:,:]) = calc_zchi2_batch(target_data[j].spectra, tdata, weights, flux, wflux, nz, nbasis, dtemplate.template.solve_matrices_algorithm, use_gpu=use_gpu, fullprecision=False)

        #- Penalize chi2 for negative [OII] flux; ad-hoc
        if dtemplate.template.template_type == 'GALAXY':
            OIIflux = np.sum(zcoeff[j] @ OIItemplate.T, axis=1)
            zchi2penalty[j][OIIflux < 0] = -OIIflux[OIIflux < 0]

        if dtemplate.comm is None and progress is not None:
            progress.put(1)

    return zchi2, zcoeff, zchi2penalty


def _mp_calc_zchi2(indx, target_ids, target_data, t, use_gpu, qout, qprog):
    """Wrapper for multiprocessing version of calc_zchi2.
    """
    try:
        # Unpack targets from shared memory
        for tg in target_data:
            tg.sharedmem_unpack()
        tzchi2, tzcoeff, tpenalty = calc_zchi2(target_ids, target_data, t,
            use_gpu=use_gpu, progress=qprog)
        qout.put( (indx, tzchi2, tzcoeff, tpenalty) )
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "MP calc_zchi2: {}".format(x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()


def calc_zchi2_targets(targets, templates, mp_procs=1, use_gpu=False):
    """Compute all chi2 fits for the local set of targets and collect.

    Given targets and templates distributed across a set of MPI processes,
    compute the coarse-binned chi^2 fit for all redshifts and our local set
    of targets.  Each process computes the fits for a slice of redshift range
    and then cycles through redshift slices by passing the interpolated
    templates along to the next process in order.

    Args:
        targets (DistTargets): distributed targets.
        templates (list): list of DistTemplate objects.
        mp_procs (int): if not using MPI, this is the number of multiprocessing
            processes to use.
        gpu (bool): (optional) use gpu for calc_zchi2

    Returns:
        dict: dictionary of results for each local target ID.

    """

    # Find most likely candidate redshifts by scanning over the
    # pre-interpolated templates on a coarse redshift spacing.

    # See if we are the process that should be printing stuff...
    am_root = False
    if targets.comm is None:
        am_root = True
    elif targets.comm.rank == 0:
        am_root = True

    # If we are not using MPI, our DistTargets object will have all the targets
    # on the main process.  In that case, we would like to distribute our
    # targets across multiprocesses.  Here we compute that distribution, if
    # needed.

    mpdist = None
    if targets.comm is None:
        mpdist = distribute_targets(targets.local(), mp_procs)

    results = dict()
    for tid in targets.local_target_ids():
        results[tid] = dict()

    if am_root:
        print("Computing redshifts")
        sys.stdout.flush()

    for t in templates:
        ft = t.template.full_type

        if am_root:
            print("  Scanning redshifts for template {}"\
                .format(t.template.full_type))
            sys.stdout.flush()

        start = elapsed(None, "", comm=targets.comm)

        # There are 2 parallelization techniques supported here (MPI and
        # multiprocessing).

        zchi2 = None
        zcoeff = None
        penalty = None

        if targets.comm is not None or mp_procs == 1:
            # MPI case.
            # The following while-loop will cycle through the redshift slices
            # (one per MPI process) until all processes have computed the chi2
            # for all redshifts for their local targets.

            if am_root:
                sys.stdout.write("    Progress: {:3d} %\n".format(0))
                sys.stdout.flush()

            zchi2 = dict()
            zcoeff = dict()
            penalty = dict()

            mpi_prog_frac = 1.0
            prog_chunk = 10
            if t.comm is not None:
                mpi_prog_frac = 1.0 / t.comm.size
                if t.comm.size < prog_chunk:
                    prog_chunk = 100 // t.comm.size
            proglast = 0
            prog = 1

            done = False
            #CW 04/25/22 - when running in GPU mode, all non-GPU procs should
            #have 0 targets.  Set done to true in these cases so it skips the
            #while loop - this saves ~2s on 500 targets on 64 CPU / 4 GPU
            #and no need to call calc_zchi2 on empty target list
            if (use_gpu and len(targets.local_target_ids()) == 0):
                done = True
            while not done:
                # Compute the fit for our current redshift slice.
                tzchi2, tzcoeff, tpenalty = \
                    calc_zchi2(targets.local_target_ids(), targets.local(), t, use_gpu=use_gpu)

                # Save the results into a dict keyed on targetid
                tids = targets.local_target_ids()
                for i, tid in enumerate(tids):
                    if tid not in zchi2:
                        zchi2[tid] = {}
                        zcoeff[tid] = {}
                        penalty[tid] = {}
                    zchi2[tid][t.local.index] = tzchi2[i]
                    zcoeff[tid][t.local.index] = tzcoeff[i]
                    penalty[tid][t.local.index] = tpenalty[i]

                prg = int(100.0 * prog * mpi_prog_frac)
                if prg >= proglast + prog_chunk:
                    proglast += prog_chunk
                    if am_root and (t.comm is not None):
                        sys.stdout.write("    Progress: {:3d} %\n"\
                            .format(proglast))
                        sys.stdout.flush()
                prog += 1

                # Cycle through the redshift slices
                done = t.cycle()

            for tid in zchi2.keys():
                zchi2[tid] = np.concatenate([ zchi2[tid][p] for p in sorted(zchi2[tid].keys()) ])
                zcoeff[tid] = np.concatenate([ zcoeff[tid][p] for p in sorted(zcoeff[tid].keys()) ])
                penalty[tid] = np.concatenate([ penalty[tid][p] for p in sorted(penalty[tid].keys()) ])
        else:
            # Multiprocessing case.
            import multiprocessing as mp

            # Ensure that all targets are packed into shared memory
            for tg in targets.local():
                tg.sharedmem_pack()

            # We explicitly spawn processes here (rather than using a pool.map)
            # so that we can communicate the read-only objects once and send
            # a whole list of redshifts to each process.

            qout = mp.Queue()
            qprog = mp.Queue()

            procs = list()
            for i in range(mp_procs):
                if len(mpdist[i]) == 0:
                    continue
                target_ids = mpdist[i]
                target_data = [ x for x in targets.local() if x.id in mpdist[i] ]
                p = mp.Process(target=_mp_calc_zchi2,
                    args=(i, target_ids, target_data, t, use_gpu, qout, qprog))
                procs.append(p)
                p.start()

            # Track progress
            sys.stdout.write("    Progress: {:3d} %\n".format(0))
            sys.stdout.flush()
            ntarget = len(targets.local_target_ids())
            progincr = 10
            if mp_procs > ntarget:
                progincr = int(100.0 / ntarget)
            tot = 0
            proglast = 0
            while (tot < ntarget):
                cnt = qprog.get()
                tot += cnt
                prg = int(100.0 * tot / ntarget)
                if prg >= proglast + progincr:
                    proglast += progincr
                    sys.stdout.write("    Progress: {:3d} %\n".format(proglast))
                    sys.stdout.flush()

            # Extract the output
            zchi2 = dict()
            zcoeff = dict()
            penalty = dict()
            for _ in range(len(procs)):
                res = qout.get()
                tids = mpdist[res[0]]
                for j,tid in enumerate(tids):
                    zchi2[tid] = res[1][j]
                    zcoeff[tid] = res[2][j]
                    penalty[tid] = res[3][j]

        elapsed(start, "    Finished in", comm=targets.comm)

        for tid in sorted(zchi2.keys()):
            results[tid][ft] = dict()
            results[tid][ft]['redshifts'] = t.template.redshifts
            results[tid][ft]['zchi2'] = zchi2[tid]
            results[tid][ft]['penalty'] = penalty[tid]
            results[tid][ft]['zcoeff'] = zcoeff[tid]

    return results

def solve_matrices(M, y, solve_algorithm="PCA", solver_args=None, use_gpu=False):
    """Solve the matrix equation y = M*x for the unknown x using the
    specified algorithm.  The default is to use PCA via numpy or cupy
    linalg.solve.  But non-negative matrix factorization (NMF) and other
    algorithms may be integrated here and selected based on the template.

    Args:
        M (array): 2d array on CPU; 3d array on GPU for all redshifts
        y (array): 1d array on CPU; 2d array on GPU for all redshifts
        solve_algorithm (string): which algorithm to use
        solver_args (dict): Optional args to pass to solver, such as bounds
            array for BVLS.
        use_gpu (bool): use GPU or not

    Returns:
        x (array): solution to y = M*x
            (1-d array on CPU, 2-d array on GPU)

    Raises:
        LinAlgError is passed up if raised by np.linalg.solve
        NotImplementedError if algorithm is not implemented or undefined

    """

    if solve_algorithm.upper() == "PCA":
        #Use PCA via linalg.solve in either numpy or cupy
        if (use_gpu):
            #Use cupy linalg.solve to solve for zcoeff in batch for all_M and
            #all_y where all_M and all_y are 3d and 2d arrays representing
            #M and y at every redshift bin for the given template.
            #There is no Error thrown by cupy's version.
            return cp.linalg.solve(M, y)
        else:
            #Use numpy linalg.solve which throws exception
            try:
                return np.linalg.solve(M, y)
            except np.linalg.LinAlgError:
                raise
    elif solve_algorithm in ("NMF", "NNLS"):
        if (use_gpu):
            nz = y.shape[0]
            Mcpu = M.get()
            ycpu = y.get()
            zcoeff = np.zeros(y.shape)
            #Copy to CPU, run scipy.optimize.nnls, copy back to GPU
            for j in range(nz):
                try:
                    res = nnls(Mcpu[j,:,:], ycpu[j,:])
                    zcoeff[j,:] = res[0]
                except Exception:
                    zcoeff[j,:] = 9e99
            return cp.asarray(zcoeff)
        else:
            try:
                res = nnls(M, y)
                zcoeff = res[0]
            except Exception:
                raise np.linalg.LinAlgError
            return zcoeff
    elif solve_algorithm == "BVLS":
        if (solver_args is not None and 'bounds' in solver_args):
            bounds = solver_args['bounds']
        else:
            nbasis = y.shape[-1]
            bounds = np.zeros((2, nbasis))
            bounds[0]=-np.inf
            bounds[1]=np.inf
        if (use_gpu):
            nz = y.shape[0]
            Mcpu = M.get()
            ycpu = y.get()
            zcoeff = np.zeros(y.shape)
            #Copy to CPU, run scipy.optimize.lsq_linear, copy back to GPU
            for j in range(nz):
                try:
                    res = lsq_linear(Mcpu[j,:,:], ycpu[j,:], bounds=bounds, method='bvls')
                    zcoeff[j,:] = res.x
                except np.linalg.LinAlgError:
                    zcoeff[j,:] = 9e99
            return cp.asarray(zcoeff)
        else:
            try:
                res = lsq_linear(M, y, bounds=bounds, method='bvls')
                zcoeff = res.x
            except np.linalg.LinAlgError:
                raise
            return zcoeff
    elif solve_algorithm == "NMF":
        raise NotImplementedError("NMF is not yet implemented.")
    else:
        raise NotImplementedError("The solve_algorithm "+solve_algorithm+" is not implemented.")
