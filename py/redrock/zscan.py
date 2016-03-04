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
    weightslist = list()
    for targetid, spectra in targets:
        fluxlist.append( np.concatenate( [s['flux'] for s in spectra] ) )
        weightslist.append( np.concatenate( [s['ivar'] for s in spectra] ) )
    
    #- NOT GENERAL AT ALL:
    #- assume all targets have same number of spectra with same wavelengths,
    #- so we can just use the first target spectra to get wavelength grids
    #- for all of them
    refspectra = targets[0][1]
        
    #- Loop over redshifts, solving for template fit coefficients
    nbasis = template['flux'].shape[0]
    nflux = len(fluxlist[0])
    Tb = np.zeros( (nflux, nbasis) )
    for i, z in enumerate(redshifts):
        Tx = rebin_template(template, z, refspectra)
        
        if i%100 == 0: print('redshift', z)
        
        for j in range(ntargets):
            targetid, spectra = targets[j]
            flux = fluxlist[j]
            weights = weightslist[j]
            Tb = list()
            for k, s in enumerate(spectra):
                Tb.append(s['R'].dot(Tx[k]))
                
            Tb = np.vstack(Tb)
        
            W = scipy.sparse.dia_matrix((weights, 0), (nflux, nflux))
            a = np.linalg.solve(Tb.T.dot(W.dot(Tb)), Tb.T.dot(W.dot(flux)))
        
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
    
#-------------------------------------------------------------------------
#- Cache templates rebinned onto particular wavelength grids since that is
#- a computationally expensive operation
_template_cache = dict()

def template_fit(z, spectra, template, flux=None, weights=None, npoly=0):
    '''Fit a template to the data at a given redshift
    
    flux = sum_i a[i] template['flux'][i]
    
    Args:
        z : redshift
        spectra: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        template: dictionary with keys
            - wave : array of wavelengths [Angstroms]
            - flux[i,wave] : template basis vectors of flux densities
            
    Optional:
        npoly: number of Legendre polynomial terms to add as nuisance background

    Optional for efficiency, since they may be pre-calculated for all z:
        flux : precalculated np.concatenate( [s['flux'] for s in spectra] )
        weights : precalculated np.concatenate( [s['ivar'] for s in spectra] )
        
    Returns a, T:
        a : coefficients that fit this template to these spectra
        T : list of matrices which sample the template basis vectors to the
            binning and resolution of each spectrum.
            
    Notes:
        T[i].dot(a) is the model for spectra[i]['flux']
    '''
    
    if flux is None:
        flux = np.concatenate( [s['flux'] for s in spectra] )
    if weights is None:
        weights = np.concatenate( [s['ivar'] for s in spectra] )
        
    nflux = len(flux)

    #- Make a list of matrices that bin the template basis for each spectrum
    nbasis = template['flux'].shape[0]  #- number of template basis vectors
    T = list()
    nspec = len(spectra)
    for i, s in enumerate(spectra):
        Ti = np.zeros((len(s['wave']), nbasis+npoly*nspec))
        #- Template basis
        for j in range(nbasis):
            key = (z, id(template), j, len(s['wave']), s['wave'][0], s['wave'][-1])
            if key not in _template_cache:
                t = rebin.trapz_rebin((1+z)*template['wave'], template['flux'][j], s['wave'])
                _template_cache[key] = t
            else:
                t = _template_cache[key]
                
            Ti[:,j] = s['R'].dot(t)

        #- Add legendre background terms
        w = s['wave']
        wx = 2 * (w-w[0]) / (w[-1] - w[0]) - 1.0  #- Map wave -> [-1,1]
        for j in range(npoly):
            c = np.zeros(npoly)
            c[j] = 1.0
            k = nbasis + i*npoly + j
            Ti[:, k] = np.polynomial.legendre.legval(wx, c)
        
        T.append(Ti)
        
    #- Convert to a single matrix for solving `a`
    Tx = np.vstack(T)
    
    #- solve s = Tx * a
    W = scipy.sparse.dia_matrix((weights, 0), (nflux, nflux))
    a = np.linalg.solve(Tx.T.dot(W.dot(Tx)), Tx.T.dot(W.dot(flux)))
    return a, T
    
