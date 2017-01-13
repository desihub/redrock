from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse

from redrock.rebin import trapz_rebin

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
        
    def eval(self, coeff, wave, z):
        '''
        Return template for given coefficients, wavelengths, and redshift
        
        Args:
            coeff : array of coefficients length self.nbasis
            wave : wavelengths at which to evaluate template flux
            z : redshift at which to evaluate template flux
        
        Returns:
            template flux array
        
        Notes:
            A single factor of (1+z)^-1 is applied to the resampled flux
            to conserve integrated flux after redshifting.
        '''
        assert len(coeff) == self.nbasis
        flux = self.flux.T.dot(coeff).T / (1+z)
        return trapz_rebin(self.wave*(1+z), flux, wave)

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

        #- Make a basic coadd
        self.coadd = list()
        for key in set([s.wavehash for s in spectra]):
            wave = None
            unweightedflux = None
            weightedflux = None
            weights = None
            R = None
            nspec = 0
            for s in spectra:
                if s.wavehash != key: continue
                nspec += 1
                if weightedflux is None:
                    wave = s.wave
                    unweightedflux = s.flux
                    weightedflux = s.flux * s.ivar
                    weights = s.ivar.copy()
                    n = len(s.ivar)
                    W = scipy.sparse.dia_matrix((s.ivar, [0,]), (n,n))
                    weightedR = W * s.R
                else:
                    assert len(s.ivar) == n
                    unweightedflux += s.flux
                    weightedflux += s.flux * s.ivar
                    weights += s.ivar
                    W = scipy.sparse.dia_matrix((s.ivar, [0,]), (n,n))
                    weightedR += W * s.R

            isbad = (weights == 0)
            flux = weightedflux / (weights + isbad)
            flux[isbad] = unweightedflux[isbad] / nspec
            Winv = scipy.sparse.dia_matrix((1/(weights+isbad), [0,]), (n,n))
            R = Winv * weightedR
            self.coadd.append(Spectrum(wave, flux, weights, R))
