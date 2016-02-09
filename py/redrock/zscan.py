from __future__ import division, print_function

import numpy as np
import scipy.sparse
from redrock import rebin

def calc_zchi2(redshifts, spectra, template, rchi2=False, npoly=0):
    '''Calculates chi2 vs. redshift for a given PCA template.

    Args:
        redshifts: array of redshifts to evaluate
        spectra: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        template: dictionary with keys
            - wave : array of wavelengths [Angstroms]
            - flux[i,wave] : template basis vectors of flux densities

    Optional:
        rchi2 : if True, return reduced chi2/dof instead of chi2
        npoly : number if Legendre poly terms to add as nuisance background

    Returns:
        chi2 array with one element per input redshift

    Notes:
        template['flux'] is a basis set; spectra will be modeled as
        flux = sum_i a[i] template['flux'][i]
        To use an archetype, provide a template with dimensions [1,nwave]
    '''
    nz = len(redshifts)
    zchi2 = np.zeros(nz)
    
    #- Regroup fluxes and ivars into 1D arrays
    flux = np.concatenate( [s['flux'] for s in spectra] )
    weights = np.concatenate( [s['ivar'] for s in spectra] )
    nflux = len(flux)
    
    #- Loop over redshifts, solving for template fit coefficients
    for i, z in enumerate(redshifts):
        a, T = template_fit(z, spectra, template, flux=flux, weights=weights, npoly=npoly)
        Tx = np.vstack(T)
        zchi2[i] = np.sum( (flux - Tx.dot(a))**2 * weights )
        if rchi2:
            zchi2[i] /= len(flux) - 1
    
    return zchi2        

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
    for i, s in enumerate(spectra):
        Ti = np.zeros((len(s['wave']), nbasis+npoly))
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
            Ti[:, nbasis+j] = np.polynomial.legendre.legval(wx, c)
        
        T.append(Ti)
        
    #- Convert to a single matrix for solving `a`
    Tx = np.vstack(T)
    
    #- solve s = Tx * a
    W = scipy.sparse.dia_matrix((weights, 0), (nflux, nflux))
    a = np.linalg.solve(Tx.T.dot(W.dot(Tx)), Tx.T.dot(W.dot(flux)))
    return a, T
    
