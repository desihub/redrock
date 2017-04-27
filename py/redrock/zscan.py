from __future__ import division, print_function
import time

import os
import numpy as np
import scipy.sparse

from . import rebin, Spectrum, Target
import redrock

def calc_zchi2(redshifts, spectra, template):
    '''
    TODO: document
    '''
    targets = [Target(0, spectra), ]
    zchi2, zcoeff, penalty = calc_zchi2_targets(redshifts, targets, template)
    return zchi2[0], zcoeff[0], penalty[0]

def parallel_calc_zchi2_targets(redshifts, targets, template, verbose=False, \
    ncpu=None, numthreads=-1):
    '''
    Parallel version of calc_zchi2_targets; see that docstring for details

    TODO: document ncpu and numthreads
    '''
    import multiprocessing as mp
    if ncpu is None:
        ncpu = mp.cpu_count()

    if ncpu > len(redshifts):
        ncpu = len(redshifts)
        print('WARNING: Using {} cores for {} redshifts'.format(ncpu, ncpu))

    if numthreads == 0:
        n = max(mp.cpu_count() // ncpu, 1)
        os.environ['OMP_NUM_THREADS'] = str(n)
        os.environ['MKL_NUM_THREADS'] = str(n)
        if verbose:
            print('DEBUG: setting OMP_NUM_THREADS={}'.format(n))
            print('DEBUG: setting MKL_NUM_THREADS={}'.format(n))
    elif numthreads > 0:
        os.environ['OMP_NUM_THREADS'] = str(numthreads)
        os.environ['MKL_NUM_THREADS'] = str(numthreads)

    if verbose:
        print('DEBUG: $OMP_NUM_THREADS={}'.format(os.getenv('OMP_NUM_THREADS')))
        print('DEBUG: $MKL_NUM_THREADS={}'.format(os.getenv('MKL_NUM_THREADS')))

    def foo(i, qin, qout):
        try:
            zz = qin.get()
            if verbose:
                print('Process {}: {} redshifts from {:.4f} to {:.4f}'.format(i, len(zz), zz[0], zz[-1]))
            results = redrock.zscan.calc_zchi2_targets(zz, targets, template, verbose=verbose)
        except Exception as err:
            import traceback, sys
            message = "".join(traceback.format_exception(*sys.exc_info()))
            results = (err, message)

        qout.put((zz[0], results))

    ii = np.linspace(0, len(redshifts), ncpu+1).astype(int)
    qin = mp.Queue()
    qout = mp.Queue()
    for i in range(ncpu):
        tmpz = redshifts[ii[i]:ii[i+1]]
        assert len(tmpz) > 0
        qin.put(tmpz)

    for i in range(ncpu):
        p = mp.Process(target=foo, args=(i, qin, qout))
        p.start()

    results = list()
    for i in range(ncpu):
        results.append(qout.get())

    #- Check for any errors
    mpfail = False
    message = 'ok'
    for i, (z0, r) in enumerate(results):
        if isinstance(r[0], Exception):
            err, message = r
            print("ERROR: result {} starting z={} generated an exception".format(i, z0))
            print(message)
            mpfail = True
    
    if mpfail:
        raise RuntimeError(message)

    #- Figure out what order the results arrived in the output queue
    z0 = [r[0] for r in results]
    izsort = np.argsort(z0)

    zchi2 = list()
    zcoeff = list()
    zchi2penalty = list()
    for i in izsort:
        tmpzchi2, tmpzcoeff, tmpzchi2penalty = results[i][1]
        zchi2.append(tmpzchi2)
        zcoeff.append(tmpzcoeff)
        zchi2penalty.append(tmpzchi2penalty)
        
    zchi2 = np.hstack(zchi2)
    zcoeff = np.hstack(zcoeff)
    zchi2penalty = np.hstack(zchi2penalty)

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
