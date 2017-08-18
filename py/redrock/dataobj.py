from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.sparse

from redrock.rebin import trapz_rebin
from redrock import sharedmem
from mpi4py import MPI

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
        self.wave=wave
        self.flux=flux
        self.ivar=ivar
        self.R=R
        self.wavehash = hash((len(wave), wave[0], wave[1], wave[-2], wave[-1]))



class MPISharedTargets(object) :
    def __init__(self, targets, comm):
        
        
        if comm is None :
            raise ValueError('I NEED A MPI COMMUNICATOR')
        
        # Global communicator.        
        self._comm = comm
        self._rank = self._comm.rank
        self._procs = self._comm.size
        
        print('rank #%d : MPISharedTargets.__init__ starting'%self._rank)

        self._n = 0
        self._dtype = np.dtype("float64")
        self._target_dictionnary = None
        
        # for the moment, the whole memory allocation is going to the first rank
        if self._rank == 0 :
            dictionnary,ndata = self.fill_target_dictionnary(targets)
            self._target_dictionnary = dictionnary
            self._n = ndata
            
        print('rank #%d : _n = %d'%(self._rank,self._n))
        print('rank #%d : allocating memory'%self._rank)

            
        # We are actually using MPI, so we need to ensure that
        # our specified numpy dtype has a corresponding MPI datatype.
        status = 0
        try:
            # Technically this is an internal variable, but online
            # forum posts from the developers indicate this is stable
            # at least until a public interface is created.
            self._mpitype = MPI._typedict[self._dtype.char]
        except:
            status = 1
        self._checkabort(self._comm, status, "numpy to MPI type conversion")
        
        # Number of bytes in our buffer
        dsize = self._mpitype.Get_size()
        nbytes = self._n * dsize
                    
        # allocating the shared memory
        # all ranks have to go through this even if they don't allocate anything (_n=nbytes=0 for rank>0)
        # otherwise this does not work (don't ask me why)
        status = 0
        try:
            self._win = MPI.Win.Allocate_shared(nbytes, dsize,comm=self._comm)
        except:
            status = 1
        self._checkabort(self._comm, status, "shared memory allocation")
        print('rank #%d : MPISharedTargets.__init__ memory allocated'%self._rank)
        
        # Every process looks up the memory address of rank zero's piece,
        # which is the start of the contiguous shared buffer.
        print('rank #%d : MPISharedTargets.__init__ get memory address'%self._rank)
        status = 0
        try:
            self._buffer, dsize = self._win.Shared_query(0)
        except:
            status = 1
        self._checkabort(self._comm, status, "shared memory query")

        # Create a numpy array which acts as a "view" of the buffer.
        self._dbuf = np.array(self._buffer, dtype="B", copy=False)
        self._data = self._dbuf.view(self._dtype) # here is the data !!
        print('rank #%d : MPISharedTargets.__init__ have the data view'%self._rank)
        
        
        if self._rank == 0 :

            print('rank #%d : MPISharedTargets.__init__ fill the memory with the data'%self._rank)
            
            # Get a write-lock on the shared memory
            self._win.Lock(self._rank, MPI.LOCK_EXCLUSIVE)

            # Copy data ...
            for target in targets :
                tdict=self._target_dictionnary[target.id]
                for spectra_dict_list_label,spectra_list in zip(["spectra","coadd"],[target.spectra,target.coadd]) :
                    spectra_dict_list=tdict[spectra_dict_list_label]
                    for spectrum_dict , spectrum in zip(spectra_dict_list,spectra_list) :
                        for label,an_array in zip(["wave","flux","ivar","rdata","roffsets","rshape"],
                                                  [spectrum.wave,spectrum.flux,spectrum.ivar,spectrum.R.data,spectrum.R.offsets.astype(self._dtype),np.array(spectrum.R.shape).astype(self._dtype)]) :
                            begin=spectrum_dict[label][0]
                            end=spectrum_dict[label][1]                            
                            self._data[begin:end] = an_array.ravel()
                        first=False
            # Release the write-lock
            self._win.Unlock(self._rank)

            print('rank #%d : MPISharedTargets.__init__ done filling the shared memory'%self._rank)
            
        # Explicit barrier here, to ensure that other processes do not try
        # reading data before the writing processes have finished.
        self._comm.barrier()

        print('rank #%d : MPISharedSpectrum.__init__ broadcasting target_dictionnary'%self._rank)
        self._target_dictionnary = comm.bcast(self._target_dictionnary,root=0)
        
        
        print('rank #%d : MPISharedSpectrum.__init__ ending'%self._rank)

        
    def _checkabort(self, comm, status, msg):
        
        if status != 0 :
            print("rank #{} : MPI ERROR : {}".format(comm.rank,msg))
            sys.stdout.flush()
            comm.Abort()  # unfortunately this removes all print messages ...
        
    
    
    def fill_target_dictionnary(self,targets) :
        print("rank #%d : fill_target_dictionnary"%self._rank)
        n=0
        dictionnary={}
        
        for target in targets :
            dictionnary[target.id]={}
            for spectra_label, spectra in zip(["spectra","coadd"],[target.spectra,target.coadd]) :
                spectra_list=list()
                for spectrum in spectra :
                    spectrum_dic={}
                    for label,asize in zip(["wave","flux","ivar","rdata","roffsets","rshape"],[spectrum.wave.size,spectrum.flux.size,spectrum.ivar.size,spectrum.R.data.size,spectrum.R.offsets.size,len(spectrum.R.shape)]) :
                        spectrum_dic[label]=[n,n+asize] ; n+=asize
                    spectra_list.append(spectrum_dic)
                dictionnary[target.id][spectra_label]=spectra_list
        return dictionnary,n

    def get_targets(self) :
        print("rank #%d : get_targets"%self._rank)
        targets = list()
        for targetid in self._target_dictionnary.keys() :
            tdict=self._target_dictionnary[targetid]
            spectra=list()
            coadd=list()
            for spectra_dict_list_label, spectra_ref in zip(["spectra","coadd"],[spectra,coadd]) :
                spectra_dict_list=tdict[spectra_dict_list_label]
                for spectrum_dict in spectra_dict_list :
                    # this is where we have to do the buffer ...
                    # it's not working
                    # wave=np.frombuffer(self._data,dtype=self._dtype, count=(spectrum_dict["wave"][1]-spectrum_dict["wave"][0]), offset=spectrum_dict["wave"][0])
                    # flux=np.frombuffer(self._data,dtype=self._dtype, count=(spectrum_dict["flux"][1]-spectrum_dict["flux"][0]), offset=spectrum_dict["flux"][0])
                    # ivar=np.frombuffer(self._data,dtype=self._dtype, count=(spectrum_dict["ivar"][1]-spectrum_dict["ivar"][0]), offset=spectrum_dict["ivar"][0])
                    # rdata=np.frombuffer(self._data,dtype=self._dtype, count=(spectrum_dict["rdata"][1]-spectrum_dict["rdata"][0]), offset=spectrum_dict["rdata"][0])
                    # roffsets=np.frombuffer(self._data,dtype=self._dtype, count=(spectrum_dict["roffsets"][1]-spectrum_dict["roffsets"][0]), offset=spectrum_dict["roffsets"][0]).astype(int)
                    # rshape=tuple(np.frombuffer(self._data,dtype=self._dtype, count=(spectrum_dict["rshape"][1]-spectrum_dict["rshape"][0]), offset=spectrum_dict["rshape"][0]).astype(int))

                    # we not doing a copy here (except for R)
                    wave=self._data[spectrum_dict["wave"][0]:spectrum_dict["wave"][1]]
                    flux=self._data[spectrum_dict["flux"][0]:spectrum_dict["flux"][1]]
                    ivar=self._data[spectrum_dict["ivar"][0]:spectrum_dict["ivar"][1]]
                    rdata=self._data[spectrum_dict["rdata"][0]:spectrum_dict["rdata"][1]]
                    roffsets=self._data[spectrum_dict["roffsets"][0]:spectrum_dict["roffsets"][1]]
                    rshape=tuple(self._data[spectrum_dict["rshape"][0]:spectrum_dict["rshape"][1]].astype(int))
                    rdata=rdata.reshape((roffsets.size,rdata.shape[0]//roffsets.size))
                    spectra_ref.append(SimpleSpectrum(wave,flux,ivar,scipy.sparse.dia_matrix((rdata, roffsets), shape=rshape)))
            
            targets.append(Target(targetid,spectra,coadd=coadd))
        print("rank #%d : done get_targets"%self._rank)
        return targets
    
        
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.close()
        return False

    def close(self):
        # The shared memory window is automatically freed
        # when the class instance is garbage collected.
        # This function is for any other clean up on destruction.
        return
    
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
