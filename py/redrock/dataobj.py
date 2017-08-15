from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse

from redrock.rebin import trapz_rebin
from redrock import sharedmem

class Template(object):
    def __init__(self, template_type, redshifts, wave, flux, subtype=''):
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
        self.subtype = subtype
        self.redshifts = np.asarray(redshifts)
        self.wave = wave
        self.flux = flux
        self.nbasis = flux.shape[0]
        self.nwave = flux.shape[1]

    @property
    def fulltype(self):
        '''Return formatted type:subtype string'''
        if self.subtype != '':
            return '{}:{}'.format(self.type, self.subtype)
        else:
            return self.type

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
        self.nwave = len(wave)
        assert(len(flux) == self.nwave)
        assert(len(ivar) == self.nwave)
        assert(R.shape == (self.nwave, self.nwave))
        
        self._shmem = dict()
        self._shmem['wave'] = sharedmem.fromarray(wave)
        self._shmem['flux'] = sharedmem.fromarray(flux)
        self._shmem['ivar'] = sharedmem.fromarray(ivar)
        self._shmem['R.data'] = sharedmem.fromarray(R.data)
        self._shmem['R.offsets'] = R.offsets

        self._shmem['R.shape'] = R.shape

        self.sharedmem_unpack()

        #- NOT EXACT: hash of wavelengths
        self.wavehash = hash((len(wave), wave[0], wave[1], wave[-2], wave[-1]))

    def sharedmem_unpack(self):
        '''Unpack shared memory buffers back into numpy array views of those
        buffers; to be called after self.sharedmem_pack() to restore the object
        back to a working state (as opposed to a state optimized for sending
        to a new process).'''
        self.wave = sharedmem.toarray(self._shmem['wave'])
        self.flux = sharedmem.toarray(self._shmem['flux'])
        self.ivar = sharedmem.toarray(self._shmem['ivar'])
        Rdata = sharedmem.toarray(self._shmem['R.data'])
        Roffsets = self._shmem['R.offsets']
        Rshape = self._shmem['R.shape']
        self.R = scipy.sparse.dia_matrix((Rdata, Roffsets), shape=Rshape)
        self.Rcsr = self.R.tocsr()   #- Precalculate R as a CSR sparse matrix

    def sharedmem_pack(self):
        '''
        Prepare for passing to multiprocessing process function;
        use self.sharedmem_unpack() to restore to the original state.
        '''
        if hasattr(self, 'wave'):
            del self.wave
        if hasattr(self, 'flux'):
            del self.flux
        if hasattr(self, 'ivar'):
            del self.ivar
        if hasattr(self, 'R'):
            del self.R
        if hasattr(self, 'Rcsr'):
            del self.Rcsr

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
                    unweightedflux = s.flux.copy()
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
            R = R.todia()
            self.coadd.append(Spectrum(wave, flux, weights, R))

    def sharedmem_pack(self):
        '''Prepare underlying numpy arrays for sending to a new process;
        call self.sharedmem_unpack() to restore to original state.'''
        for s in self.spectra:
            s.sharedmem_pack()

        if hasattr(self, 'coadd'):
            for s in self.coadd:
                s.sharedmem_pack()

    def sharedmem_unpack(self):
        '''Unpack shared memory arrays into numpy array views of them.
        To be used after self.sharedmem_pack() was called to pack arrays
        before passing them to a new process.'''
        for s in self.spectra:
            s.sharedmem_unpack()

        if hasattr(self, 'coadd'):
            for s in self.coadd:
                s.sharedmem_unpack()
