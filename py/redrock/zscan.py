from __future__ import division, print_function
import time

import os,sys
import numpy as np
import scipy.sparse

from . import rebin

from .dataobj import (MultiprocessingSharedSpectrum, 
    SimpleSpectrum, MPISharedTargets, Target)


def calc_zchi2(redshifts, spectra, template):
    '''
    Calculates chi2 vs. redshift for these spectra and this template.

    returns chi2[nz], coeff[nz,ncoeff], penalty[nz]

    `chi2[i]` is best fit chi2 at `redshifts[i]`.
    `coeff[i]` is array of best fit template coefficients at `redshifts[i]`.
    `penalty` is array of penalty priors, e.g. to penalize unphysical fits.
    '''
    targets = [Target(0, spectra), ]
    zchi2, zcoeff, penalty = calc_zchi2_targets(redshifts, targets, template)
    return zchi2[0], zcoeff[0], penalty[0]


def _wrap_zchi2(redshifts, targets, template, qout):
    for t in targets:
        t.sharedmem_unpack()
    try:
        results = calc_zchi2_targets(redshifts, targets, template)
    except Exception as err:
        import traceback, sys
        message = "".join(traceback.format_exception(*sys.exc_info()))
        results = (err, message)

    tids = [t.id for t in targets]
    qout.put((tids, results))


def parallel_calc_zchi2_targets(redshifts, targets, template, verbose=False, \
    ncpu=None):
    '''
    Parallel version of calc_zchi2_targets; see that docstring for details

    ncpu is number of multiprocessing processes to use
    '''

    if len(targets) == 0:
        raise ValueError('Input target list is empty')

    if len(redshifts) == 0:
        raise ValueError('Input redshift array is empty')

    import multiprocessing as mp
    if ncpu is None:
        ncpu = mp.cpu_count()

    if ncpu > len(targets):
        ncpu = len(targets)
        print('WARNING: Using {} cores for {} redshifts'.format(ncpu, ncpu))

    #- Pack targets for passing to multiprocessing function without pickling
    #- large numpy arrays of spectra
    for t in targets:
        t.sharedmem_pack()

    #- launch processes, returning results via Queue.  This is largely for
    #- historical reasons; mp.Pool.map probably would have been fine too.
    qout = mp.Queue()
    target_split = np.array_split(targets, ncpu)
    for i in range(ncpu):
        p = mp.Process(target=_wrap_zchi2, args=(redshifts, target_split[i], template, qout))
        p.start()

    #- Get results, one per process
    results = list()
    for i in range(ncpu):
        results.append(qout.get())

    #- Figure out what order the results arrived in the output queue
    tids = [t.id for t in targets]

    zchi2 = [None]*len(targets)
    zcoeff = [None]*len(targets)
    zchi2penalty = [None]*len(targets)
    for r in results:
        tmpzchi2, tmpzcoeff, tmpzchi2penalty = r[1]
        for i,t in enumerate(r[0]):
            i0 = tids.index(t)
            zchi2[i0] = tmpzchi2[i]
            zcoeff[i0] = tmpzcoeff[i]
            zchi2penalty[i0] = tmpzchi2penalty[i]

    zchi2 = np.array(zchi2)
    zcoeff = np.array(zcoeff)
    zchi2penalty = np.array(zchi2penalty)

    #- restore the state of targets
    for t in targets:
        t.sharedmem_unpack()

    return zchi2, zcoeff, zchi2penalty


def mpi_calc_zchi2_targets(redshifts, targets, template, verbose=False, \
    comm=None):
    '''
    MPI Parallel version of calc_zchi2_targets
    '''
    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    # Option 1: parallelize over redshifts- less rebinning of templates,
    # but more memory use when copying all target data out of shared
    # memory.

    zsplit = np.array_split(redshifts, nproc)
    zchi2, zcoeff, zchi2penalty = calc_zchi2_targets(zsplit[rank], 
        targets, template)
    if comm is not None:
        zchi2        = comm.allgather(zchi2)
        zcoeff       = comm.allgather(zcoeff)
        zchi2penalty = comm.allgather(zchi2penalty)
        zchi2        = np.hstack(zchi2)
        zcoeff       = np.hstack(zcoeff)
        zchi2penalty = np.hstack(zchi2penalty)

    # Option 2: parallelize over targets- more rebinning of templates,
    # but less memory use / copying.

    # tsplit = np.array_split(targets, nproc)
    # zchi2, zcoeff, zchi2penalty = calc_zchi2_targets(redshifts, tsplit[rank], 
    #     template)
    # if comm is not None:
    #     zchi2        = comm.allgather(zchi2)
    #     zcoeff       = comm.allgather(zcoeff)
    #     zchi2penalty = comm.allgather(zchi2penalty)
    #     zchi2        = np.vstack(zchi2)
    #     zcoeff       = np.vstack(zcoeff)
    #     zchi2penalty = np.vstack(zchi2penalty)
    
    return zchi2, zcoeff, zchi2penalty


def calc_zchi2_targets(redshifts, targets, template, verbose=False):
    '''Calculates chi2 vs. redshift for a given PCA template.

    Args:
        redshifts: array of redshifts to evaluate
        targets : list of Target objects
        template: dictionary with keys
            - wave : array of wavelengths [Angstroms]
            - flux[i,wave] : template basis vectors of flux densities

    Returns: zchi2, zcoeff
        zchi2[ntargets, nz] array with one element per target per redshift
        zcoeff[ntargets, nz, ncoeff]

    Notes:
        template.flux is a basis set; spectra will be modeled as
        flux = sum_i a[i] template.flux[i]
        To use an archetype, provide a template with dimensions [1,nwave]
    '''
    nz = len(redshifts)
    ntargets = len(targets)
    nbasis = template.flux.shape[0]
    zchi2 = np.zeros( (ntargets, nz) )
    zchi2penalty = np.zeros( (ntargets, nz) )
    zcoeff = np.zeros( (ntargets, nz, nbasis) )

    #- Regroup fluxes and ivars into 1D arrays per target
    fluxlist = list()
    wfluxlist = list()
    weightslist = list()
    Wlist = list()
    for t in targets:
        weights = np.concatenate( [s.ivar for s in t.spectra] )
        weightslist.append( weights )
        nflux = len(weights)
        Wlist.append( scipy.sparse.dia_matrix((weights, 0), (nflux, nflux)) )
        flux = np.concatenate( [s.flux for s in t.spectra] )
        fluxlist.append(flux)
        wfluxlist.append(weights*flux)

    #- Gather reference spectra for projecting templates to data wavelengths
    refspectra = dict()
    for t in targets:
        for s in t.spectra:
            if s.wavehash not in refspectra:
                refspectra[s.wavehash] = s

    refspectra = list(refspectra.values())

    #- Redshifts near [OII]; used only for galaxy templates
    isOII = (3724 <= template.wave) & (template.wave <= 3733)
    OIItemplate = template.flux[:,isOII].T

    #- Loop over redshifts, solving for template fit coefficients
    nflux = len(fluxlist[0])
    Tb = np.zeros( (nflux, nbasis) )
    for i, z in enumerate(redshifts):
        if verbose and i in istatus:
            print('{} {:d}% done'.format(time.asctime(), round(100.0*i/len(redshifts))))
        #- if all targets have the same number of spectra with the same
        #- wavelength grids, we only need to calculate this once per redshift.
        Tx = rebin_template(template, z, refspectra)

        for j in range(ntargets):
            Tb = list()
            for k, s in enumerate(targets[j].spectra):
                key = s.wavehash
                Tb.append(s.Rcsr.dot(Tx[key]))

            Tb = np.vstack(Tb)

            flux = fluxlist[j]
            wflux = wfluxlist[j]
            weights = weightslist[j]
            W = Wlist[j]
            M = Tb.T.dot(W.dot(Tb))
            y = Tb.T.dot(wflux)
            a = np.linalg.solve(M, y)

            model = Tb.dot(a)
            zchi2[j,i] = np.sum( (flux - model)**2 * weights )
            zcoeff[j,i] = a

            #- Penalize chi2 for negative [OII] flux; ad-hoc
            if template.type == 'GALAXY':
                OIIflux = np.sum( OIItemplate.dot(a) )
                if OIIflux < 0:
                    zchi2penalty[j,i] = -OIIflux

    return zchi2, zcoeff, zchi2penalty


#- DEBUG: duplicated code, but provide a direct way to fit a template to a
#- set of spectra at a given redshift
def template_fit(spectra, z, template):
    Tb = list()
    Tx = rebin_template(template, z, spectra)
    for k, s in enumerate(spectra):
        Tb.append(s.Rcsr.dot(Tx[s.wavehash]))
        ### Tb.append(s.R.dot(Tx[key]))

    Tbx = np.vstack(Tb)

    weights = np.concatenate( [s.ivar for s in spectra] )
    flux = np.concatenate( [s.flux for s in spectra] )
    nflux = len(flux)
    wflux = weights * flux
    W = scipy.sparse.dia_matrix((weights, 0), (nflux, nflux))
    a = np.linalg.solve(Tbx.T.dot(W.dot(Tbx)), Tbx.T.dot(wflux))

    return [T.dot(a) for T in Tb], a


def rebin_template(template, z, spectra):
    '''rebin template to match the wavelengths of the input spectra'''
    nbasis = template.flux.shape[0]  #- number of template basis vectors
    Tx = dict()
    for i, s in enumerate(spectra):
        key = s.wavehash
        if key not in Tx:
            Ti = np.zeros((s.nwave, nbasis))
            for j in range(nbasis):
                t = rebin.trapz_rebin((1+z)*template.wave, template.flux[j], s.wave)
                Ti[:,j] = t

            Tx[key] = Ti

    return Tx

# def _orig_rebin_template(template, z, spectra):
#     '''rebin template to match the wavelengths of the input spectra'''
#     nbasis = template.flux.shape[0]  #- number of template basis vectors
#     Tx = list()
#     nspec = len(spectra)
#     for i, s in enumerate(spectra):
#         Ti = np.zeros((s.nwave, nbasis))
#         for j in range(nbasis):
#             t = rebin.trapz_rebin((1+z)*template.wave, template.flux[j], s.wave)
#             Ti[:,j] = t
#
#         Tx.append(Ti)
#
#     return Tx
