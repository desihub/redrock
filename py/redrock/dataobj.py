from __future__ import absolute_import, division, print_function

import numpy as np

class Template(object):
    def __init__(self, template_type, redshifts, wave, flux):
        '''
        Create a spectral Template PCA object
        
        Args:
            template_type : str, type of template, e.g. 'galaxy' or 'qso'
            redshifts : array of redshifts to consider for this template
            wave : 1D array of wavelengths
            flux : 2D array of PCA eigenvectors[nbasis, nwave]
        '''
        wave = np.asarray(wave)
        flux = np.asarray(flux)
        
        assert flux.shape[1] == len(wave)
        
        self.type = template_type
        self.redshifts = np.asarray(redshifts)
        self.wave = wave
        self.flux = flux
        self.nbasis = flux.shape[0]
        self.nwave = flux.shape[1]

class Spectrum(object):
    def __init__(self, wave, flux, ivar, R):
        """
        create a Spectrum object
        
        Args:
            wave : wavelength array
            flux : flux array
            ivar : array of inverse variances of flux
            R : resolution matrix, sparse 2D[nwave, nwave]
        """
        wave = np.asarray(wave)
        flux = np.asarray(flux)
        ivar = np.asarray(ivar)

        nwave = len(wave)
        assert(len(flux) == nwave)
        assert(len(ivar) == nwave)
        assert(R.shape == (nwave, nwave))
        
        self.wave = wave
        self.flux = flux
        self.ivar = ivar
        self.R = R
        self.nwave = nwave
        
        #- Precalculate R as a CSR sparse matrix
        self.Rcsr = R.tocsr()
        
        #- NOT EXACT: hash of wavelengths
        self.wavehash = hash((len(wave), wave[0], wave[1], wave[-2], wave[-1]))

class Target(object):
    def __init__(self, targetid, spectra):
        """
        Create a Target object

        Args:
            targetid : unique targetid (integer or str)
            spectra : list of Spectra objects
        """
        self.id = targetid
        self.spectra = spectra
