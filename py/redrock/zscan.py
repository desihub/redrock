from __future__ import division

import numpy as np
from redrock import rebin

def calc_norm(x1, x2, weights):
    '''Calculate weighted normalization for x1 = norm*x2'''
    assert len(x1) == len(x1)
    assert len(x1) == len(weights)
    assert np.all(weights >= 0)
    
    #- catch unnormalizable (but not fatal) case
    blat = np.sum(x2*x2*weights)
    if blat == 0:
        return 1.0
    else:
        return (x1*weights).dot(x2) / blat

def calc_zchi2(redshifts, spectra, templates, rchi2=False):
    '''Calculates chi2 vs. redshift for a set of templates.

    Args:
        redshifts: array of redshifts to evaluate
        spectra: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        templates: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]

    Optional:
        rchi2 : if True, return reduced chi2/dof instead of chi2

    Returns:
        chi2[num_templates, num_redshifts] 2D array with chi2 for each
            template at each redshift

    Notes:
        The spectra are multiple individual spectra for the same object.
        They are assumed to be flux calibrated and consistent with each other.
        The only free parameter is an overall scale factor per template per
        redshift; there is no spectrum-to-spectrum re-normalization.
    '''
    zchi2 = np.zeros([len(templates), len(redshifts)])
    for i in range(len(templates)):
        zchi2[i] = calc_zchi2_template(redshifts, spectra, templates[i], rchi2=rchi2)
        
    return zchi2

def calc_zchi2_template(redshifts, spectra, template, rchi2=False):
    '''Calculates chi2 vs. redshift for a given template.

    Args:
        redshifts: array of redshifts to evaluate
        spectra: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        template: dictionary with keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]

    Optional:
        rchi2 : if True, return reduced chi2/dof instead of chi2

    Returns:
        chi2 array with one element per input redshift
    '''
    
    nz = len(redshifts)
    zchi2 = np.zeros(nz)
    
    #- Regroup spectra fluxes and ivars into two arrays
    sflux = np.concatenate( [s['flux'] for s in spectra] )
    weights = np.concatenate( [s['ivar'] for s in spectra] )
    
    for i, z in enumerate(redshifts):
        #- rebin and convolve the template to the resolution of each spectrum
        tflux = list()
        for s in spectra:
            t = rebin.trapz_rebin((1+z)*template['wave'], template['flux'], s['wave'])
            tflux.append(s['R'].dot(t))

        tflux = np.concatenate(tflux)
        norm = calc_norm(sflux, tflux, weights)

        zchi2[i] = np.sum( (sflux - tflux*norm)**2 * weights )
        if rchi2:
            zchi2[i] /= len(sflux) - 1
    
    return zchi2        

    
    
