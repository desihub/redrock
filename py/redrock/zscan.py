from __future__ import division, print_function

import numpy as np
import scipy.sparse
from redrock import rebin

def calc_zchi2(redshifts, spectra, template):
    target = (0, spectra)
    return calc_zchi2_targets(redshifts, [target,], template)[0]

def calc_zchi2_targets(redshifts, targets, template):
    '''Calculates chi2 vs. redshift for a given PCA template.

    Args:
        redshifts: array of redshifts to evaluate
        targets : list of (targetid, spectra), where spectra are a list of
            dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        template: dictionary with keys
            - wave : array of wavelengths [Angstroms]
            - flux[i,wave] : template basis vectors of flux densities

    Returns:
        chi2 array with one element per input redshift

    Notes:
        template['flux'] is a basis set; spectra will be modeled as
        flux = sum_i a[i] template['flux'][i]
        To use an archetype, provide a template with dimensions [1,nwave]
    '''
    nz = len(redshifts)
    ntargets = len(targets)
    zchi2 = np.zeros( (ntargets, nz) )
    
    #- Regroup fluxes and ivars into 1D arrays per target
    fluxlist = list()
    wfluxlist = list()
    weightslist = list()
    Wlist = list()
    for targetid, spectra in targets:
        weights = np.concatenate( [s['ivar'] for s in spectra] )
        weightslist.append( weights )
        nflux = len(weights)
        Wlist.append( scipy.sparse.dia_matrix((weights, 0), (nflux, nflux)) )
        flux = np.concatenate( [s['flux'] for s in spectra] )
        fluxlist.append(flux)
        wfluxlist.append(weights*flux)
    
    #- NOT GENERAL AT ALL:
    #- assume all targets have same number of spectra with same wavelengths,
    #- so we can just use the first target spectra to get wavelength grids
    #- for all of them
    refspectra = targets[0][1]
    
    #- Pre-convert resolution matrices to csr for faster dot products
    for targetid, spectra in targets:
        for s in spectra:
            s['Rcsr'] = s['R'].tocsr()
    
    #- Loop over redshifts, solving for template fit coefficients
    nbasis = template['flux'].shape[0]
    nflux = len(fluxlist[0])
    Tb = np.zeros( (nflux, nbasis) )
    for i, z in enumerate(redshifts):
        Tx = rebin_template(template, z, refspectra)        
        for j in range(ntargets):
            targetid, spectra = targets[j]
            Tb = list()
            for k, s in enumerate(spectra):
                Tb.append(s['Rcsr'].dot(Tx[k]))

            Tb = np.vstack(Tb)

            flux = fluxlist[j]
            wflux = wfluxlist[j]
            weights = weightslist[j]
            W = Wlist[j]
            a = np.linalg.solve(Tb.T.dot(W.dot(Tb)), Tb.T.dot(wflux))

            zchi2[j,i] = np.sum( (flux - Tb.dot(a))**2 * weights )

    return zchi2        

def rebin_template(template, z, spectra):
    '''rebin template to match the wavelengths of the input spectra'''
    nbasis = template['flux'].shape[0]  #- number of template basis vectors
    Tx = list()
    nspec = len(spectra)
    for i, s in enumerate(spectra):
        Ti = np.zeros((len(s['wave']), nbasis))
        for j in range(nbasis):
            t = rebin.trapz_rebin((1+z)*template['wave'], template['flux'][j], s['wave'])
            Ti[:,j] = t

        Tx.append(Ti)
    
    return Tx
