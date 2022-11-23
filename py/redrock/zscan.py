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
        return calc_zchi2_gpu(target_ids, target_data, dtemplate, progress)
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
