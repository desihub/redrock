from __future__ import division, print_function

import numpy as np
import scipy.sparse

from . import rebin, Spectrum, Target

def calc_zchi2(redshifts, spectra, template):
    targets = [Target(0, spectra), ]
    zchi2, zcoeff = calc_zchi2_targets(redshifts, targets, template)
    return zchi2[0], zcoeff[0]

def calc_zchi2_targets(redshifts, targets, template):
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
    
    #- NOT GENERAL AT ALL:
    #- assume all targets have same number of spectra with same wavelengths,
    #- so we can just use the first target spectra to get wavelength grids
    #- for all of them
    refspectra = targets[0].spectra
    
    #- Loop over redshifts, solving for template fit coefficients
    nflux = len(fluxlist[0])
    Tb = np.zeros( (nflux, nbasis) )
    for i, z in enumerate(redshifts):
        #- TODO: if all targets have the same number of spectra with the same
        #- wavelength grids, we only need to calculate this once per redshift.
        #- That isn't general; this is an area for optimization.
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
            a = np.linalg.solve(Tb.T.dot(W.dot(Tb)), Tb.T.dot(wflux))

            zchi2[j,i] = np.sum( (flux - Tb.dot(a))**2 * weights )
            zcoeff[j,i] = a

    return zchi2, zcoeff

#- DEBUG: duplicated code, but provide a direct way to fit a template to a
#- set of spectra at a given redshift
def template_fit(spectra, z, template):
    Tb = list()
    Tx = rebin_template(template, z, spectra)
    for k, s in enumerate(spectra):
        key = _wavekey(s.wave)
        Tb.append(s.Rcsr.dot(Tx[key]))
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

def _orig_rebin_template(template, z, spectra):
    '''rebin template to match the wavelengths of the input spectra'''
    nbasis = template.flux.shape[0]  #- number of template basis vectors
    Tx = list()
    nspec = len(spectra)
    for i, s in enumerate(spectra):
        Ti = np.zeros((s.nwave, nbasis))
        for j in range(nbasis):
            t = rebin.trapz_rebin((1+z)*template.wave, template.flux[j], s.wave)
            Ti[:,j] = t

        Tx.append(Ti)

    return Tx
