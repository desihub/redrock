from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse
from collections import OrderedDict

from .rebin import trapz_rebin
from . import sharedmem
from .sharedmem_mpi import MPIShared


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

        
class MultiprocessingSharedSpectrum(object):

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


class SimpleSpectrum(object):

    def __init__(self, wave, flux, ivar, R):
        self.nwave=wave.size
        self.wave=wave
        self.flux=flux
        self.ivar=ivar
        self.R=R
        self.Rcsr = self.R.tocsr()
        self.wavehash = hash((len(wave), wave[0], wave[1], wave[-2], wave[-1]))


class MPISharedTargets(object):

    def __init__(self, targets, comm):
        """
        Place a list of targets into MPI shared memory.

        Args:
            targets (list): the list of targets.  This is ONLY meaningful
                on the rank zero process.  Data on other processes is
                ignored.
            comm (mpi4py.MPI.Comm): the communicator to use (or None).
        """

        self._comm = comm
        self._dtype = np.dtype(np.float64)

        root = False
        if self._comm is None:
            root = True
        elif self._comm.rank == 0:
            root = True

        # We are going to pack the target data into one big memory buffer.
        # First compute the displacements of the data for each target.

        self._target_dictionary = None
        self._bufsize = None

        if root:
            self._target_dictionary, self._bufsize = \
                self._fill_target_dictionary(targets)
        
        if self._comm is not None:
            self._target_dictionary = self._comm.bcast(self._target_dictionary,
                root=0)
            self._bufsize = self._comm.bcast(self._bufsize, root=0)

        # Allocate the shared buffer
        self._shared = MPIShared((self._bufsize,), self._dtype, self._comm)

        # Now set the shared data buffer, one target at a time

        targetnum = 0
        for id, tdict in self._target_dictionary.items():
            for spectra_label in ["spectra", "coadd"]:
                specnum = 0
                for spectra_dict in tdict[spectra_label]:
                    target_begin = spectra_dict["wave"][0]
                    target_end = spectra_dict["rshape"][1]
                    target_buffer = None
                    if root:
                        assert targets[targetnum].id == id
                        target_buffer = np.empty((target_end - target_begin,),
                            dtype=np.float64)
                        spectrum = None
                        if spectra_label == "spectra":
                            spectrum = targets[targetnum].spectra[specnum]
                        elif spectra_label == "coadd":
                            spectrum = targets[targetnum].coadd[specnum]
                        for label, an_array in zip(["wave", "flux", "ivar", 
                            "rdata", "roffsets", "rshape"], [spectrum.wave,
                            spectrum.flux, spectrum.ivar, spectrum.R.data, 
                            spectrum.R.offsets.astype(self._dtype),
                            np.array(spectrum.R.shape).astype(self._dtype)]):
                            begin = spectra_dict[label][0] - target_begin
                            end = spectra_dict[label][1] - target_begin
                            target_buffer[begin:end] = an_array.ravel()
                    self._shared.set(target_buffer, (target_begin,), 
                        fromrank=0)
                    specnum += 1
            targetnum += 1

        # Wrap the target data in a friendlier format.  This will likely
        # duplicate some data from the resolution matrix.

        self.targets = self._get_targets()


    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.close()
        return False

    def close(self):
        return


    def _fill_target_dictionary(self, targets):
        n = 0
        dictionary = OrderedDict() # keep targets as ordered in list
        for target in targets :
            dictionary[target.id] = {}
            for spectra_label, spectra in zip(["spectra", "coadd"], 
                [target.spectra, target.coadd]):
                spectra_list = list()
                for spectrum in spectra :
                    spectrum_dict = {}
                    for label, asize in zip(["wave","flux","ivar","rdata",
                        "roffsets","rshape"], [spectrum.wave.size, 
                        spectrum.flux.size, spectrum.ivar.size, 
                        spectrum.R.data.size, spectrum.R.offsets.size,
                        len(spectrum.R.shape)]):
                        spectrum_dict[label] = [n, n + asize]
                        n += asize
                    spectra_list.append(spectrum_dict)
                dictionary[target.id][spectra_label] = spectra_list
        return dictionary, n


    def _get_targets(self):
        targets = list()
        for id, tdict in self._target_dictionary.items():
            spectra = list()
            coadd = list()
            for spectra_label, spectra_ref in zip(["spectra", "coadd"],
                [spectra, coadd]):
                spectra_dict_list = tdict[spectra_label]
                for spectrum_dict in spectra_dict_list :
                    # we are not doing a copy here (except maybe for R ...)
                    wave = self._shared[spectrum_dict["wave"][0]:spectrum_dict["wave"][1]]
                    flux = self._shared[spectrum_dict["flux"][0]:spectrum_dict["flux"][1]]
                    ivar = self._shared[spectrum_dict["ivar"][0]:spectrum_dict["ivar"][1]]
                    rdata = self._shared[spectrum_dict["rdata"][0]:spectrum_dict["rdata"][1]]
                    roffsets = self._shared[spectrum_dict["roffsets"][0]:spectrum_dict["roffsets"][1]]
                    rshape = tuple(self._shared[spectrum_dict["rshape"][0]:spectrum_dict["rshape"][1]].astype(int))
                    rdata = rdata.reshape((roffsets.size,rdata.shape[0]//roffsets.size))
                    spectra_ref.append( SimpleSpectrum(wave, flux, ivar, 
                        scipy.sparse.dia_matrix((rdata, roffsets), 
                        shape=rshape)) )            
            targets.append(Target(id, spectra, coadd=coadd))

        return targets


class Target(object):
    def __init__(self, targetid, spectra, coadd=None):
        """
        Create a Target object

        Args:
            targetid : unique targetid (integer or str)
            spectra : list of Spectra objects
        """
        self.id = targetid
        self.spectra = spectra

        if coadd is None :
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
                self.coadd.append(MultiprocessingSharedSpectrum(wave, flux, weights, R))
        else :
            self.coadd=coadd
    
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
