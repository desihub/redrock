"""
redrock.zscan
=============

Algorithms for scanning redshifts.
"""

from __future__ import division, print_function

import sys
import traceback
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy
    import cupyx
    cupy_available = cp.is_available()
except ImportError:
    cupy_available = False

from .utils import elapsed

from .targets import distribute_targets


if (cupy_available):
    #set this flag to use cupy's linalg.solve as below:
    #            zc = cp.linalg.solve(all_M, all_y)
    #to solve the matrix for all templates in parallel for each object.
    #set this to 0 to copy data to the host and use numpy's linalg.solve,
    #looping over each template.  For some reason this is faster on
    #consumer grade GeForce 1660, likely due to double precision being slow,
    #but CUPY solve is much faster on A100s on perlmutter so this should
    #always be 1 unless running on a consumer grade chip
    use_cupy_linalg = 1

    # cuda_source contains raw CUDA kernels to be loaded as CUPY module

    cuda_source = r'''
        extern "C" {
            __global__ void batch_dot_product(const double* Rcsr_values, const int* Rcsr_cols, const int* Rcsr_indptr, const double* tdata, double* tb, int nrows, int ncols, int nbatch, int nt) {
                // This kernel performs a batch dot product of a sparse matrix Rcsr
                // with a set of redshifted templates
                //** Args:
                //       Rcsr_values, Rcsr_cols, Rcsr_indptr = individualarrays from sparse matrix
                //           (Rcsr = sparse array, ncols x nrows)
                //       tdata = redshifted templates, nt x ncols x nbatch
                //       tb = output array = nt x ncols x nbatch
                //       nrows, ncols, nbatch, nt = array dimensions

                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i - corresponds to the output array index
                if (i >= ncols*nbatch*nt) return;
                double x = 0; //define a local var to accumulate
                //ibatch, icol, it = index in 3d representation of output tb array
                //icol also == row in Rcsr input
                int ibatch = i % nbatch;
                int icol = (i % (nbatch*ncols)) / nbatch;
                int it = i / (nbatch*ncols);
                int t_start = it*nbatch*ncols; //first index in tdata for this thread
                int row = icol;

                int col;
                //loop over all nonzero entries in sparse matrix and compute dot product
                for (int j = Rcsr_indptr[row]; j < Rcsr_indptr[row+1]; j++) {
                    col = Rcsr_cols[j];
                    x += Rcsr_values[j] * tdata[t_start+nbatch*col+ibatch];
                }
                tb[i] = x;
                return;
            }

            __global__ void calc_M_and_y_atomic(const double* all_Tb, const double* weights, const double* wflux, double* M, double* y, int nrows, int nbatch, int nt, int nparallel) {
                // This kernel computes the dot products resulting in the M and y arrays in parallel
                // The y array is small compared to the M array so rather than launching a separate kernel,
                // a small number of threads will be diverted to compute y in parallel since the Tb array
                // is used to compute both.
                // It will use nparallel threads to compute the dot product for each output element in M and y.
                // Each thread will handle nrows/nparallel products and sums into an intermediate local variable
                // and then an atomicAdd will be used to add this intermediate sum to the output array.

                // This replicates the python commands:
                //     M = Tb.T.dot(np.multiply(weights[:,None], Tb))
                //     y = Tb.T.dot(wflux)
                //** Args:
                //       all_Tb = the Tb array, the stacked output from all 3 filters from
                //           batch_dot_product, for all redshift templates (nt x nrows x nbatch)
                //       weights = the weights array for this target (1d, size = nrows)
                //       wflux = the wflux array for this target (1d, size = nrows)
                //       M = the output M array (nt x nbatch x nbatch)
                //       y = the output y array (nt x nbatch)
                //       nrows, nbatch, nt = array dimensions
                //       nparallel = number of parallel threads to used for each output

                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i
                if (i >= nbatch*nbatch*nt*nparallel+nbatch*nt*nparallel) return;

                if (i < nbatch*nbatch*nt*nparallel) {
                    //These threads compute M
                    int m_idx = i / nparallel; //index in output M array
                    int t = m_idx / (nbatch*nbatch); //target number
                    int allTb_row = (m_idx % (nbatch*nbatch)) % nbatch; //row in all_Tb array
                    int wTb_row = (m_idx % (nbatch*nbatch)) / nbatch; // row in (weights*Tb)

                    int stride = nrows/nparallel; //stride to divide up nparallel threads

                    int start = (threadIdx.x % nparallel)*stride; //start index in nrows dim
                    int end = ((threadIdx.x % nparallel)+1)*stride; //end index in nrows dim
                    if (threadIdx.x % nparallel == (nparallel-1)) end = nrows;
                    int allTb_idx = t*nrows*nbatch + allTb_row; // 1-d index for first element to be processed by this thread
                    int wTb_idx = t*nrows*nbatch + wTb_row; // 1-d index for first element to be processed by this thread

                    double x = 0; //define local var to accumulate

                    //perform intermediate sum dot product for this thread
                    for (int j = start; j < end; j++) {
                        //stride by nbatch
                        x += all_Tb[allTb_idx+j*nbatch] * (all_Tb[wTb_idx+j*nbatch] * weights[j]);
                    }
                    //use atomic add to avoid collisions between threads
                    atomicAdd(&M[m_idx], x);
                } else {
                    //These threads compute y
                    int i2 = (i-nbatch*nbatch*nt*nparallel); //index among y-threads
                    int y_idx = i2 / nparallel; //index in output y array
                    int t = y_idx / nbatch; //target number
                    int allTb_row = y_idx % nbatch; //row in all_Tb array

                    int stride = nrows/nparallel; //stride to divide up nparallel threads

                    int start = (threadIdx.x % nparallel)*stride; //start index in nrows dim
                    int end = ((threadIdx.x % nparallel)+1)*stride; //end index in nrows dim
                    if (threadIdx.x % nparallel == (nparallel-1)) end = nrows;
                    int allTb_idx = t*nrows*nbatch + allTb_row; // 1-d index for first element to be processed by this thread

                    double x = 0; //define local var to accumulate

                    //perform intermediate sum dot product for this thread
                    for (int j = start; j < end; j++) {
                        //stride by nbatch
                        x += all_Tb[allTb_idx+j*nbatch] * wflux[j];
                    }
                    //use atomic add to avoid collisions between threads
                    atomicAdd(&y[y_idx], x);
                }
            }

            __global__ void tb_zc_dot(const double* tb, const double* zc, double* model, int nrows, int nbatch, int nt) {
                // This kernel computes the dot product of Tb and zc, the result of the
                // matrix solution of M and y, for all templates in parallel.  It
                // results in the model array.  Each thread computes an element in the
                // output model array.  It replaces the python code:
                //     model = Tb.dot(cupy.array(zc))
                //** Args:
                //       tb = the Tb array, the stacked output from all 3 filters from
                //           batch_dot_product, for all redshift templates (nt x nrows x nbatch)
                //       zc = the zc array, the output of
                //           zc = cp.linalg.solve(all_M, all_y)
                //           shape = (nt x nbatch)
                //       model = the output of the dot product, (nt x nrows)
                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i
                if (i >= nrows*nt) return;
                int it = i / nrows; //target num
                int row = i % nrows; //row num
                int i_tb = it * nrows * nbatch + row * nbatch; //start index in Tb array
                int i_zc = it * nbatch; //start index in zc array
                double x = 0; //use local var to accumulate
                //compute dot product
                for (int j = 0; j < nbatch; j++) {
                    x += tb[i_tb+j] * zc[i_zc+j];
                }
                //copy to output
                model[i] = x;
            }

            __global__ void calc_z_prod(const double* flux, const double* model, const double* weights, double* z_product, int nrows, int nt) {
                // This kernel computes the dot product of (flux-model)^2 and weights
                // that results in the final zchi2 for all templates and one target.
                // It replaces the python code:
                //     zchi2[i,j] = cupy.dot((flux-model)**2, weights)
                //** Args:
                //       flux = the flux array for this target (1d, size = nrows)
                //       model = the output of tb_zc_dot (nt x nrows)
                //       weights = the weights array for this target (1d, size = nrows)
                //       z_product = the output of the dot product (nt x nrows)
                const int i = blockDim.x*blockIdx.x + threadIdx.x; //thread index i
                if (i >= nrows*nt) return;
                int it = i / nrows; //target num
                int row = i % nrows; //row num
                int i_model = it * nrows + row; //index in model array
                double x = flux[row]-model[i_model];
                z_product[i] = x*x*weights[row];
            }

        }
    '''


def _zchi2_batch(Tb, weights, flux, wflux, zcoeff):
    """Calculate a batch of chi2.

    For many redshifts and a set of spectral data, compute the chi2 for template
    data that is already on the correct grid.
    """

    M = Tb.swapaxes(-2, -1) @ (weights[None, :, None] * Tb)
    y = (Tb.swapaxes(-2, -1) @ wflux)
    # TODO: use cholesky solve here?
    zcoeff[:] = np.linalg.solve(M, y)
    model = np.squeeze((Tb @ zcoeff[:, :, None]))
    zchi2 = ((flux - model)**2 @ weights)
    return zchi2

def _zchi2_one(Tb, weights, flux, wflux, zcoeff):
    """Calculate a single chi2.

    For one redshift and a set of spectral data, compute the chi2 for template
    data that is already on the correct grid.
    """

    M = Tb.T.dot(np.multiply(weights[:,None], Tb))
    y = Tb.T.dot(wflux)

    try:
        zcoeff[:] = np.linalg.solve(M, y)
    except np.linalg.LinAlgError:
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

def calc_zchi2_one(spectra, weights, flux, wflux, tdata):
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
    zchi2 = _zchi2_one(Tb, weights, flux, wflux, zcoeff)

    return zchi2, zcoeff

def calc_zchi2(target_ids, target_data, dtemplate, progress=None, use_gpu=False):
    """Calculate chi2 vs. redshift for a given PCA template.

    Args:
        target_ids (list): targets IDs.
        target_data (list): list of Target objects.
        dtemplate (DistTemplate): distributed template data
        progress (multiprocessing.Queue): optional queue for tracking
            progress, only used if MPI is disabled.
        use_gpu (bool): (optional) use gpu for calc_zchi2

    Returns:
        tuple: (zchi2, zcoeff, zchi2penalty) with:
            - zchi2[ntargets, nz]: array with one element per target per
                redshift
            - zcoeff[ntargets, nz, ncoeff]: array of best fit template
                coefficients for each target at each redshift
            - zchi2penalty[ntargets, nz]: array of penalty priors per target
                and redshift, e.g. to penalize unphysical fits

    """
    if use_gpu:
        #return calc_zchi2_gpu(target_ids, target_data, dtemplate, progress)
        #Call new calc_zchi2_gpu_new method
        return calc_zchi2_gpu_new(target_ids, target_data, dtemplate, progress)
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

    for j in range(ntargets):
        (weights, flux, wflux) = spectral_data(target_data[j].spectra)

        # Loop over redshifts, solving for template fit
        # coefficients.  We use the pre-interpolated templates for each
        # unique wavelength range.
        for i, _ in enumerate(dtemplate.local.redshifts):
            zchi2[j,i], zcoeff[j,i] = calc_zchi2_one(target_data[j].spectra,
                weights, flux, wflux, dtemplate.local.data[i])

            #- Penalize chi2 for negative [OII] flux; ad-hoc
            if dtemplate.template.template_type == 'GALAXY':
                OIIflux = np.sum( OIItemplate.dot(zcoeff[j,i]) )
                if OIIflux < 0:
                    zchi2penalty[j,i] = -OIIflux

        if dtemplate.comm is None:
            progress.put(1)

    return zchi2, zcoeff, zchi2penalty

def calc_zchi2_gpu(target_ids, target_data, dtemplate, progress=None):
    """Calculate chi2 vs. redshift for a given PCA template.

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
    nz = len(dtemplate.local.redshifts)
    ntargets = len(target_ids)
    nbasis = dtemplate.template.nbasis

    zchi2 = cp.zeros( (ntargets, nz) )
    zchi2penalty = cp.zeros( (ntargets, nz) )
    zcoeff = cp.zeros( (ntargets, nz, nbasis) )

    # Redshifts near [OII]; used only for galaxy templates
    if dtemplate.template.template_type == 'GALAXY':
        isOII = (3724 <= dtemplate.template.wave) & \
            (dtemplate.template.wave <= 3733)
        OIItemplate = cp.array(dtemplate.template.flux[:,isOII].T)

    # Combine redshifted templates
    tdata = dict()
    for key in dtemplate.local.data[0].keys():
        tdata[key] = cp.array([tdata[key] for tdata in dtemplate.local.data])

    for j in range(ntargets):
        (weights, flux, wflux) = spectral_data(target_data[j].spectra)
        if np.sum(weights) == 0:
            zchi2[j] = 9e99
            continue
        weights = cp.array(weights)
        flux = cp.array(flux)
        wflux = cp.array(wflux)

        # Solving for template fit coefficients for all redshifts.
        # We use the pre-interpolated templates for each
        # unique wavelength range.
        Tbs = []
        for s in target_data[j].spectra:
            key = s.wavehash
            R = cupyx.scipy.sparse.csr_matrix(s.Rcsr).toarray()
            Tbs.append(cp.einsum('mn,jnk->jmk', R, tdata[key]))
        Tbs = cp.concatenate(Tbs, axis=1)
        zchi2[j] = _zchi2_batch(Tbs, weights, flux, wflux, zcoeff[j])

        #- Penalize chi2 for negative [OII] flux; ad-hoc
        if dtemplate.template.template_type == 'GALAXY':
            OIIflux = np.sum(zcoeff[j] @ OIItemplate.T, axis=1)
            zchi2penalty[j][OIIflux < 0] = -OIIflux[OIIflux < 0]

        if dtemplate.comm is None:
            progress.put(1)

    return zchi2.get(), zcoeff.get(), zchi2penalty.get()


def calc_zchi2_gpu_new(target_ids, target_data, dtemplate, progress=None):
    """Calculate chi2 vs. redshift for a given PCA template.

    New GPU algorithms 4/22/22

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
    nz = len(dtemplate.local.redshifts)
    ntargets = len(target_ids)
    nbasis = dtemplate.template.nbasis

    zchi2 = np.zeros( (ntargets, nz) )
    zchi2penalty = np.zeros( (ntargets, nz) )
    zcoeff = np.zeros( (ntargets, nz, nbasis) )

    block_size = 512 #Default block size, should work on all modern nVidia GPUs
    # Load CUDA kernels
    cp_module = cp.RawModule(code=cuda_source)
    batch_dot_product = cp_module.get_function('batch_dot_product')
    calc_M_y = cp_module.get_function('calc_M_and_y_atomic')
    tb_zc_dot = cp_module.get_function('tb_zc_dot')
    calc_z_prod = cp_module.get_function('calc_z_prod')

    # Redshifts near [OII]; used only for galaxy templates
    if dtemplate.template.template_type == 'GALAXY':
        isOII = (3724 <= dtemplate.template.wave) & \
            (dtemplate.template.wave <= 3733)
        OIItemplate = np.array(dtemplate.template.flux[:,isOII].T)

    # Combine redshifted templates
    tdata = dict()
    for key in dtemplate.local.data[0].keys():
        tdata[key] = cp.array([tdata[key] for tdata in dtemplate.local.data])

    for j in range(ntargets):
        (weights, flux, wflux) = spectral_data(target_data[j].spectra)
        if np.sum(weights) == 0:
            zchi2[j,:] = 9e99
            continue
        weights = cp.array(weights)
        flux = cp.array(flux)
        wflux = cp.array(wflux)

        # Solving for template fit coefficients for all redshifts.
        # We use the pre-interpolated templates for each
        # unique wavelength range.
        Tbs = []
        for s in target_data[j].spectra:
            key = s.wavehash
            #R = cupyx.scipy.sparse.csr_matrix(s.Rcsr).toarray()

            #Use actual numpy arrays that represent sparse array - .data, .indices, and .indptr
            #Use batch_dot_product array to perform dot product in parallel for all templates
            #for this (target, spectrum) combination.
            #Allocate CUPY arrays and calculate number of blocks to use.
            n = tdata[key].size
            blocks = (n+block_size-1)//block_size
            Rcsr_values = cp.array(s.Rcsr.data, cp.float64)
            Rcsr_cols = cp.array(s.Rcsr.indices, cp.int32)
            Rcsr_indptr = cp.array(s.Rcsr.indptr, cp.int32)
            #Array dimensions
            nrows = cp.int32(s.Rcsr.shape[1])
            ncols = cp.int32(s.Rcsr.shape[0])
            nbatch = cp.int32(tdata[key].shape[2])
            nt = cp.int32(tdata[key].shape[0])
            curr_tb = cp.empty((nz, ncols, nbatch))
            #Launch kernel and syncrhronize
            batch_dot_product((blocks,), (block_size,), (Rcsr_values, Rcsr_cols, Rcsr_indptr, tdata[key], curr_tb, nrows, ncols, nbatch, nt))
            #Commented out synchronize - needed for timing kernels but we still
            #get execution of this kernel before data is needed for next kernel
            #so output is the same and slightly faster without synchronize
            #cp.cuda.Stream.null.synchronize()
            #Append to list
            Tbs.append(curr_tb)
        #Use CUPY.hstack to combine into one nt x ncols x nbatch array
        Tbs = cp.hstack(Tbs)

        #Use calc_M_y_atomic kernel to compute M and y arrays
        #nparallel - number of parallel threads for each output array element
        #For larger input Tbs arrays - e.g., GALAXY, QSO, 4 parallel threads
        #is faster because we don't want to create too many total threads
        #But for smaller Tb arrays - STARS - we can use more parallel threads
        #to maximize parallelism - this can be dynamically tuned but in test
        #data, 4 and 64 were optimal.  Needs to be power of 2.
        if (nt > 512):
            nparallel = cp.int32(4)
        else:
            nparallel = cp.int32(64)
        #Create CUPY arrays and calculate number of blocks
        nrows = cp.int32(Tbs.shape[1])
        n = nt*nbatch*nbatch*nparallel + nt*nbatch*nparallel
        blocks = (n+block_size-1)//block_size
        if (j == 0):
            #Only need to allocate first iteration through loop
            #provided the shape is constant which it should be
            #all elements will be overwritten by kernel each iteration
            #empty is super fast but this saves us time on cudaFree
            all_M = cp.empty((nt, nbatch, nbatch))
            all_y = cp.empty((nt, nbatch))
        #Launch kernel and syncrhonize
        calc_M_y((blocks,), (block_size,), (Tbs, weights, wflux, all_M, all_y, nrows, nbatch, nt, nparallel))
        #Commented out synchronize - needed for timing kernels but we still
        #get execution of this kernel before data is needed for next kernel
        #so output is the same and slightly faster without synchronize
        #cp.cuda.Stream.null.synchronize()

        ### CUPY ####
        #Commented out because above code is slightly faster but leaving for
        #future reference because this is simpler code that does the same
        #all_M = Tbs.swapaxes(-2, -1) @ (weights[None, :, None] * Tbs)
        #all_y = (Tbs.swapaxes(-2, -1) @ wflux)
        #cp.cuda.Stream.null.synchronize()

        #bool array to track elements with LinAlgError from np.linalg.solve
        iserr = np.zeros(nt, dtype=np.bool)
        #use_cupy_linalg should alwyas be 1 for running on Perlmutter or other
        #supercomputer with A100s.  For some reason on GeForce 1660, looping
        #over templates and running np.linalg.solve is faster than
        #cp.linalg.solve.  Possibly because consumer grade GPUs are slower with
        #double precision. use_cupy_linalg = 0 should be run on consumer grade
        #GPUs.
        if (use_cupy_linalg == 1):
            #Normal case on Perlmutter - solve for all templates at once
            zc = cp.linalg.solve(all_M, all_y)
        else:
            #For consumer grade GPUs, copy data to host, loop over
            #templates, and run np.linalg.solve
            M1 = all_M.get()
            y1 = all_y.get()
            zc = cp.zeros((nt, nbatch), dtype=cp.float64)
            for i, _ in enumerate(dtemplate.local.redshifts):
                M = M1[i,:,:]
                y = y1[i,:]
                try:
                    zc[i,:] = cp.array(np.linalg.solve(M, y))
                except np.linalg.LinAlgError:
                    iserr[i] = True
                    continue

        #Use tb_zc_dot kernel to computer model array
        #Allocate CUPY array and calc blocks to be used
        n = nrows * nt
        blocks = (n+block_size-1)//block_size
        if (j == 0):
            #Again only allocate first iteration through loop
            model = cp.empty((nt, nrows), cp.float64)
        #Launch kernel and synchronize
        tb_zc_dot((blocks,), (block_size,), (Tbs, zc, model, nrows, nbatch, nt))
        #cp.cuda.Stream.null.synchronize()

        #Use calc_z_prod kernel to calculate all zchi2 for this target in parallel
        #Allocate temp array to hold results - blocks is the same as in tb_zc_dot kernel above.
        if (j == 0):
            #Again only allocate first iteration through loop
            z_product = cp.empty((nt, nrows), cp.float64)
        #Launch kernel
        calc_z_prod((blocks,), (block_size,), (flux, model, weights, z_product, nrows, nt))
        #Copy data from GPU to numpy arrays
        zchi2[j,:] = z_product.sum(1).get()
        zchi2[j,:][iserr] = 9e99
        zcoeff[j,:,:] = zc.get()
        #Free data from GPU
        del zc
        del Tbs
        #Moved freeing these to after loop, only allocate and free once
        #del model
        #del z_product
        #del all_M
        #del all_y

        #- Penalize chi2 for negative [OII] flux; ad-hoc
        if dtemplate.template.template_type == 'GALAXY':
            OIIflux = np.sum(zcoeff[j] @ OIItemplate.T, axis=1)
            zchi2penalty[j][OIIflux < 0] = -OIIflux[OIIflux < 0]

        if dtemplate.comm is None:
            progress.put(1)
    #Free all_M and all_y here since only allocating once
    del all_M
    del all_y
    del model
    del z_product

    return zchi2, zcoeff, zchi2penalty

def _mp_calc_zchi2(indx, target_ids, target_data, t, qout, qprog):
    """Wrapper for multiprocessing version of calc_zchi2.
    """
    try:
        # Unpack targets from shared memory
        for tg in target_data:
            tg.sharedmem_unpack()
        tzchi2, tzcoeff, tpenalty = calc_zchi2(target_ids, target_data, t,
            progress=qprog)
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

        if targets.comm is not None:
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
            if (len(targets.local_target_ids()) == 0):
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
                    args=(i, target_ids, target_data, t, qout, qprog))
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
