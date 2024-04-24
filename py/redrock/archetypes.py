"""
redrock.archetypes
==================

Classes and functions for archetypes.
"""
import os
from glob import glob
from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
import scipy.special

from .zscan import calc_zchi2_one, calc_zchi2_batch

from .rebin import trapz_rebin

from .igm import transmission_Lyman

from .zscan import per_camera_coeff_with_least_square_batch

from .templates import make_fulltype

class Archetype():
    """Class to store all different archetypes from the same spectype.

    The archetype data are read from a redrock-format archetype file.

    Args:
        filename (str): the path to the archetype file

    """
    def __init__(self, filename):

        # Load the file
        self._filename = filename
        h = fits.open(os.path.expandvars(filename), memmap=False)

        hdr = h['ARCHETYPES'].header
        self.flux = np.asarray(h['ARCHETYPES'].data['ARCHETYPE']).astype('float64') # trapz_rebin only works with 'f8' arrays
        self._narch = self.flux.shape[0]
        self._nwave = self.flux.shape[1]
        self._rrtype = hdr['RRTYPE'].strip()
        self._subtype = np.array(np.char.strip(h['ARCHETYPES'].data['SUBTYPE'].astype(str)))
        self._subtype = np.char.add(np.char.add(self._subtype,'_'),np.arange(self._narch,dtype=int).astype(str))
        self._full_type = np.array([make_fulltype(self._rrtype, subtype) for subtype in self._subtype])
        self._version = hdr['VERSION']

        self.wave = np.asarray(hdr['CRVAL1'] + hdr['CDELT1']*np.arange(self.flux.shape[1]))
        if hdr['LOGLAM']:
            self.wave = 10**self.wave
        self.minwave = self.wave[0]
        self.maxwave = self.wave[-1]

        self._archetype = {}
        self._archetype['INTERP'] = np.array([None]*self._narch)
        for i in range(self._narch):
            self._archetype['INTERP'][i] = interp1d(self.wave,self.flux[i,:],fill_value='extrapolate',kind='linear')

        h.close()

        #It is much more efficient to calculate edges once if rebinning
        #and copy to GPU and store here rather than doing this every time
        self._gpuwave = None
        self._gpuflux = None

        # TODO: Allow Archetype files to specify their IGM model
        self.igm_model = 'Inoue14'

        return

    @property
    def gpuwave(self):
        if (self._gpuwave is None):
            #Copy to GPU once
            import cupy as cp
            self._gpuwave = cp.asarray(self.wave)
        return self._gpuwave

    @property
    def gpuflux(self):
        if (self._gpuflux is None):
            #Copy to GPU once
            import cupy as cp
            self._gpuflux = cp.asarray(self.flux)
        return self._gpuflux

    @property
    def version(self):
        return self._version

    @property
    def filename(self):
        return self._filename

    #- template_type, sub_type, and full_type like Template object methods
    @property
    def template_type(self):
        return self._rrtype

    @property
    def sub_type(self):
        return self._subtype

    @property
    def full_type(self):
        return self._full_type

    def rebin_template(self,index,z,dwave,trapz=True):
        """
        """
        if trapz:
            return {hs:trapz_rebin((1.+z)*self.wave, self.flux[index], wave) for hs, wave in dwave.items()}
        else:
            return {hs:self._archetype['INTERP'][index](wave/(1.+z)) for hs, wave in dwave.items()}

    def rebin_template_batch(self,z,dwave,trapz=True,dedges=None,indx=None,use_gpu=False):
        """Rebin templates to a set of wavelengths.

        The templates are part of this archetype.
        The wavelengths can be passed as centers (dwave) or edges (dedges)
        of the wavelength bins.

        Args:
            z (float): the redshift
            dwave (dict): the keys are the "wavehash" and the values
                are a 1D array containing the wavelength grid.
            trapz (bool): Whether to use the trapz algorithm or interpolation
            dedges (dict): in GPU mode, use pre-computed dict of wavelength
                bin edges, already on GPU
            indx (array): Only rebin these indices in the flux array
                (e.g. rebin flux[indx] rather than flux[:])
            use_gpu (bool): whether or not to use the GPU algorithm

        Returns:
            dict:  The rebinned template
        """
        if (use_gpu):
            import cupy as cp
            xmin = self.minwave
            xmax = self.maxwave
            wave = self.gpuwave
            flux = self.gpuflux
        else:
            wave = self.wave
            flux = self.flux

        if (indx is not None):
            flux = flux[indx]
        minedge = None
        maxedge = None
        result = dict()
        if (trapz and use_gpu and dedges is not None):
            #Use GPU mode with bin edges already calculated
            for hs, edges in dedges.items():
                #Check if edges is a 1-d array or a tuple also containing scalar min/max values
                if type(edges) is tuple:
                    (edges, minedge, maxedge) = edges
                result[hs] = trapz_rebin(wave, flux, edges=edges, use_gpu=use_gpu, myz=cp.array([z]), xmin=xmin, xmax=xmax, edge_min=minedge, edge_max=maxedge)[0,:,:]
            return result
        if trapz:
            #Use batch mode of trapz_rebin
            return {hs:trapz_rebin((1.+z)*wave, flux, w, use_gpu=use_gpu) for hs, w in dwave.items()}
        else:
            for hs, w in dwave.items():
                result[hs] = np.empty((len(w), self._narch))
                for i in range(self._narch):
                    result[hs][:,i] = self._archetype['INTERP'][i](w/(1.+z))
                if (use_gpu):
                    result[hs] = cp.asarray(result[hs])
            #return {hs:self._archetype['INTERP'](wave/(1.+z)) for hs, wave in dwave.items()}
        return result

    def eval(self, subtype, dwave, coeff, wave, z):
        """

        """

        deg_legendre = (coeff!=0.).size-1
        index = np.arange(self._narch)[self._subtype==subtype][0]

        w = np.concatenate([ w for w in dwave.values() ])
        wave_min = w.min()
        wave_max = w.max()
        legendre = np.array([scipy.special.legendre(i)( (wave-wave_min)/(wave_max-wave_min)*2.-1. ) for i in range(deg_legendre)])
        binned = trapz_rebin((1+z)*self.wave, self.flux[index], wave)*transmission_Lyman(z,wave,model=self.igm_model)
        flux = np.append(binned[None,:],legendre, axis=0)
        flux = flux.T.dot(coeff).T / (1+z)

        return flux
    
    def nearest_neighbour_model(self, target,weights,flux,wflux,dwave,z, n_nearest, zzchi2, trans, per_camera, dedges=None, binned=None, use_gpu=False, prior=None, ncam=None):
        
        """Nearest neighbour archetype approach; fitting with a combinating of nearest archetypes in chi2 space
        
        Args:
            target (Target): the target object that contains spectra
            weights (array): concatenated spectral weights (ivar).
            flux (array): concatenated flux values.
            wflux (array): concatenated weighted flux values.
            dwave (dict): dictionary of wavelength grids
            z (float): best redshift
            n_nearest (int): number of nearest neighbours to be used in chi2 space (including best archetype)
            zchi2 (array); chi2 array for all archetypes
            trans (dict); dictionary of transmission Ly-a arrays
            per_camera (bool): If True model will be solved in each camera
            dedges (dict): in GPU mode, use pre-computed dict of wavelength bin edges, already on GPU
            binned (dict): already computed dictionary of rebinned fluxes
            use_gpu (bool): use GPU or not
            prior (2d array): prior matrix on coefficients (1/sig**2)
            ncam (int): Number of camera for given Instrument
        
        Returns:
            chi2 (float): chi2 of best fit model
            zcoeff (array): zcoeff of best fit model
            fulltype (str): fulltype of best archetypes

        """

        #trans and dedges only need to be passed if binned is not as binned already
        #is multiplied by trans in get_best_archetype
        spectra = target.spectra
        nleg = target.nleg
        legendre = target.legendre(nleg=nleg, use_gpu=False) #Get previously calculated legendre

        nleg = legendre[list(legendre.keys())[0]].shape[0]
        iBest = np.argsort(zzchi2)[0:n_nearest]
        tdata = dict()
        if (binned is not None):
            if (use_gpu):
                binned = { hs:binned[hs][:,iBest].get() for hs in binned }
            else:
                binned = { hs:binned[hs][:,iBest] for hs in binned }
        else:
            binned = self.rebin_template_batch(z, dwave, trapz=True, dedges=dedges, indx=iBest, use_gpu=use_gpu)
            for hs, w in dwave.items():
                if (use_gpu):
                    binned[hs] = binned[hs].get()
                if (trans[hs] is not None):
                    #Only multiply if trans[hs] is not None
                    #Both arrays are on CPU so no need to wrap with asarray
                    binned[hs] *= trans[hs][:,None]
        for hs, w in dwave.items():
            tdata[hs] = binned[hs][None,:,:]
            if (nleg > 0):
                tdata[hs] = np.append(tdata[hs], legendre[hs].transpose()[None,:,:], axis=2)
            nbasis = tdata[hs].shape[2]
            
        if per_camera:
            #Use CPU mode since small tdata
            (zzchi2, zzcoeff) = per_camera_coeff_with_least_square_batch(target, tdata, weights, flux, wflux, nleg, 1, method='bvls', n_nbh=n_nearest, prior=prior, use_gpu=False, ncam=ncam)
        else:
            #Use CPU mode for calc_zchi2 since small tdata
            (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, tdata, weights, flux, wflux, 1, nbasis, use_gpu=False)

        sstype = ['%s'%(self._subtype[k]) for k in iBest] # subtypes of best archetypes
        fsstype = '_'.join(sstype)
        #print(sstype)
        #print(z, zzchi2, zzcoeff, fsstype)
        return zzchi2[0], zzcoeff[0], make_fulltype(self._rrtype, fsstype)

    def get_best_archetype(self,target,weights,flux,wflux,dwave,z, per_camera, n_nearest, trans=None, solve_method='bvls', prior=None, use_gpu=False):

        """Get the best archetype for the given redshift and spectype.

        Args:
            spectra (list): list of Spectrum objects.
            weights (array): concatenated spectral weights (ivar).
            flux (array): concatenated flux values.
            wflux (array): concatenated weighted flux values.
            dwave (dict): dictionary of wavelength grids
            z (float): best redshift
            per_camera (bool): True if fitting needs to be done in each camera
            n_nearest (int): number of nearest neighbours to be used in chi2 space (including best archetype)
            trans (dict): pass previously calcualated Lyman transmission instead of recalculating
            solve_method (string): bvls or pca
            use_gpu (bool): use GPU or not
            prior (2d array): prior matrix on coefficients (1/sig**2)
            
        Returns:
            chi2 (float): chi2 of best archetype
            zcoef (array): zcoeff of best archetype
            fulltype (str): fulltype of best archetype

        """
        spectra = target.spectra
        nleg = target.nleg
        legendre = target.legendre(nleg=nleg, use_gpu=use_gpu) #Get previously calculated legendre

        #Select np or cp for operations as arrtype
        if (use_gpu):
            import cupy as cp
            arrtype = cp
            #Get CuPy arrays of weights, flux, wflux
            #These are created on the first call of gpu_spectral_data() for a
            #target and stored.  They are retrieved on subsequent calls.
            (gpuweights, gpuflux, gpuwflux) = target.gpu_spectral_data()
            # Build dictionaries of wavelength bin edges, min/max, and centers
            dedges = { s.wavehash:(s.gpuedges, s.minedge, s.maxedge) for s in spectra }
        else:
            arrtype = np
            dedges = None

        if per_camera:
            ncam=len(list(dwave.keys())) # for e.g., DESI has three cameras namely b, r, z
        else:
            ncam = 1 # entire spectra
        
        wkeys = list(dwave.keys())
        new_keys = [wkeys[0], wkeys[2], wkeys[1]]

        obs_wave = np.concatenate([dwave[key] for key in new_keys])
        
        nleg = legendre[list(legendre.keys())[0]].shape[0]
        zzchi2 = np.zeros(self._narch, dtype=np.float64)
        zzcoeff = np.zeros((self._narch,  1+ncam*(nleg)), dtype=np.float64)
        
        #TODO: return best fit model as well
        #zzmodel = np.zeros((self._narch, obs_wave.size), dtype=np.float64)

        if (trans is None):
            #Calculate Lyman transmission if not passed as dict
            trans = { hs:transmission_Lyman(z,w, use_gpu=False, model=self.igm_model) for hs, w in dwave.items() }
        else:
            #Use previously calculated Lyman transmission
            for hs in trans:
                if (trans[hs] is not None):
                    trans[hs] = trans[hs][0,:]

        #Rebin in batch
        binned = self.rebin_template_batch(z, dwave, trapz=True, dedges=dedges, use_gpu=use_gpu)

        tdata = dict()
        nbasis = 1
        
        ## Prior must be redefined to remove nearest neighbour approach, 
        # because prior was defined based on n_nearest argument..
        # this logic is needed because the first fitting is done with just one archetype 
        #and then nearest neighbour approach is implemented
        if n_nearest is not None:
            nnearest_prior = prior.copy() # prior corresponding to nearest_nbh method
            prior = prior[n_nearest-1:,][:,n_nearest-1:] # removing first rows/columns corresponding to the nearest_archetypes, and keeping just one row for one archetype

        for hs, wave in dwave.items():
            if (trans[hs] is not None):
                #Only multiply if trans[hs] is not None
                binned[hs] *= arrtype.asarray(trans[hs][:,None])
            #Create 3-d tdata with narch x nwave x nbasis where nbasis = 1+nleg
            if nleg > 0:
                tdata[hs] = arrtype.append(binned[hs].transpose()[:,:,None], arrtype.tile(arrtype.asarray(legendre[hs]).transpose()[None,:,:], (self._narch, 1, 1)), axis=2)
            else:
                tdata[hs] = binned[hs].transpose()[:,:,None]
            nbasis = tdata[hs].shape[2]
        if per_camera:
            if (use_gpu):
                (zzchi2, zzcoeff) = per_camera_coeff_with_least_square_batch(target, tdata, gpuweights, gpuflux, gpuwflux, nleg, self._narch, method=solve_method, n_nbh=1, prior=prior, use_gpu=use_gpu, ncam=ncam)
            else:
                (zzchi2, zzcoeff) = per_camera_coeff_with_least_square_batch(target, tdata, weights, flux, wflux, nleg, self._narch, method=solve_method, n_nbh=1, prior=prior, use_gpu=use_gpu, ncam=ncam)
        else:
            if (use_gpu):
                (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, tdata, gpuweights, gpuflux, gpuwflux, self._narch, nbasis, use_gpu=use_gpu)
            else:
                (zzchi2, zzcoeff) = calc_zchi2_batch(spectra, tdata, weights, flux, wflux, self._narch, nbasis, use_gpu=use_gpu)

        if n_nearest is not None:
            best_chi2, best_coeff, best_fulltype = self.nearest_neighbour_model(target,weights,flux,wflux,dwave,z, n_nearest, zzchi2, trans, per_camera, dedges=dedges, binned=binned, use_gpu=use_gpu, prior=nnearest_prior, ncam=ncam)
            #print(best_chi2, best_coeff, best_fulltype)
            return best_chi2, best_coeff, best_fulltype
        else:
            iBest = np.argmin(zzchi2)
            #print(z, zzchi2[iBest], zzcoeff[iBest], self._full_type[iBest])
            return zzchi2[iBest], zzcoeff[iBest], self._full_type[iBest]


class All_archetypes():
    """Class to store all different archetypes of all the different spectype.

    Args:
        lstfilename (lst str): List of file to get the templates from
        archetypes_dir (str): Directory to the archetypes

    """
    def __init__(self, lstfilename=None, archetypes_dir=None, verbose=False):

        # Get list of path to archetype
        if lstfilename is None:
            lstfilename = find_archetypes(archetypes_dir)

        # Load archetype
        self.archetypes = {}
        for f in lstfilename:
            archetype = Archetype(f)
            if verbose:
                print('DEBUG: Found {} archetypes for SPECTYPE {} in file {}'.format(archetype._narch, archetype._rrtype, f) )

            self.archetypes[archetype._rrtype] = archetype

        return

def find_archetypes(archetypes_dir=None):
    """Return list of rrarchetype-\*.fits archetype files

    Search directories in this order, returning results from first one found:
        - archetypes_dir
        - $RR_ARCHETYPE_DIR
        - <redrock_code>/archetypes/

    Args:
        archetypes_dir (str): optional directory containing the archetypes.

    Returns:
        list: a list of archetype files.

    """
    if archetypes_dir is None:
        if 'RR_ARCHETYPE_DIR' in os.environ:
            archetypes_dir = os.environ['RR_ARCHETYPE_DIR']
        else:
            thisdir = os.path.dirname(__file__)
            archdir = os.path.join(os.path.abspath(thisdir), 'archetypes')
            if os.path.exists(archdir):
                archetypes_dir = archdir
            else:
                raise IOError("ERROR: can't find archetypes_dir, $RR_ARCHETYPE_DIR, or {rrcode}/archetypes/")
        lstfilename = sorted(glob(os.path.join(archetypes_dir, 'rrarchetype-*.fits')))
    else:
        if os.path.isfile(archetypes_dir):
            lstfilename = [archetypes_dir]
        else:
            archetypes_dir_expand = os.path.expandvars(archetypes_dir)
            lstfilename = glob(os.path.join(archetypes_dir_expand, 'rrarchetype-*.fits'))
            lstfilename = sorted([ f.replace(archetypes_dir_expand,archetypes_dir) for f in lstfilename])

    return lstfilename
