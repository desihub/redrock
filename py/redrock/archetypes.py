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

from .zscan import calc_zchi2_one

from .rebin import trapz_rebin

from .utils import transmission_Lyman

from .nearest_neighbours import return_N_nearest_archetypes_from_synthetic_spectra
from .nearest_neighbours import return_galaxy_archetype_properties
from .nearest_neighbours import params_for_all_galaxies

## Loading some global data once as it will be used many times
archetype_galaxies = params_for_all_galaxies() ## Dictionary for ELG, LRG, BGS
## Dictionary of galaxy properties of archetypes
rrarchs_params = return_galaxy_archetype_properties('/global/cfs/cdirs/desi/users/abhijeet/new-archetypes/rrarchetype-galaxy.fits') 


class Archetype():
    """Class to store all different archetypes from the same spectype.

    The archetype data are read from a redrock-format archetype file.

    Args:
        filename (str): the path to the archetype file

    """
    def __init__(self, filename):

        # Load the file
        h = fits.open(os.path.expandvars(filename), memmap=False)

        hdr = h['ARCHETYPES'].header
        self.flux = np.asarray(h['ARCHETYPES'].data['ARCHETYPE'])
        self._narch = self.flux.shape[0]
        self._nwave = self.flux.shape[1]
        self._rrtype = hdr['RRTYPE'].strip()
        self._subtype = np.array(np.char.strip(h['ARCHETYPES'].data['SUBTYPE'].astype(str)))
        self._subtype = np.char.add(np.char.add(self._subtype,'_'),np.arange(self._narch,dtype=int).astype(str))
        self._full_type = np.char.add(self._rrtype+':::',self._subtype)
        self._version = hdr['VERSION']

        self.wave = np.asarray(hdr['CRVAL1'] + hdr['CDELT1']*np.arange(self.flux.shape[1]))
        if hdr['LOGLAM']:
            self.wave = 10**self.wave

        self._archetype = {}
        self._archetype['INTERP'] = np.array([None]*self._narch)
        for i in range(self._narch):
            self._archetype['INTERP'][i] = interp1d(self.wave,self.flux[i,:],fill_value='extrapolate',kind='linear')

        h.close()

        return
    def rebin_template(self,index,z,dwave,trapz=True):
        """
        """
        if trapz:
            return {hs:trapz_rebin((1.+z)*self.wave, self.flux[index], wave) for hs, wave in dwave.items()}
        else:
            return {hs:self._archetype['INTERP'][index](wave/(1.+z)) for hs, wave in dwave.items()}

    def eval(self, subtype, dwave, coeff, wave, z):
        """

        """

        deg_legendre = (coeff!=0.).size-1
        index = np.arange(self._narch)[self._subtype==subtype][0]

        w = np.concatenate([ w for w in dwave.values() ])
        wave_min = w.min()
        wave_max = w.max()
        legendre = np.array([scipy.special.legendre(i)( (wave-wave_min)/(wave_max-wave_min)*2.-1. ) for i in range(deg_legendre)])
        binned = trapz_rebin((1+z)*self.wave, self.flux[index], wave)*transmission_Lyman(z,wave)
        flux = np.append(binned[None,:],legendre, axis=0)
        flux = flux.T.dot(coeff).T / (1+z)

        return flux


    def get_best_archetype(self,spectra,weights,flux,wflux,dwave,z,legendre, nearest_nbh, n_nbh):
        """Get the best archetype for the given redshift and spectype.

        Args:
            spectra (list): list of Spectrum objects.
            weights (array): concatenated spectral weights (ivar).
            flux (array): concatenated flux values.
            wflux (array): concatenated weighted flux values.
            dwave (dic): dictionary of wavelength grids
            z (float): best redshift
            legendre (dic): legendre polynomial

        Returns:
            chi2 (float): chi2 of best archetype
            zcoef (array): zcoef of best archetype
            fulltype (str): fulltype of best archetype

        """

        nleg = legendre[list(legendre.keys())[0]].shape[0]
        zzchi2 = np.zeros(self._narch, dtype=np.float64)
        zzcoeff = np.zeros((self._narch, nleg+1), dtype=np.float64)
        trans = { hs:transmission_Lyman(z,w) for hs, w in dwave.items() }

        for i in range(self._narch):
            binned = self.rebin_template(i, z, dwave,trapz=False)
            binned = { hs:trans[hs]*binned[hs] for hs, w in dwave.items() }
            tdata = { hs:np.append(binned[hs][:,None],legendre[hs].transpose(), axis=1 ) for hs, wave in dwave.items() }
            zzchi2[i], zzcoeff[i] = calc_zchi2_one(spectra, weights, flux, wflux, tdata)
        iBest = np.argmin(zzchi2)
        if nearest_nbh:
            ### Applying nearest neighbour method based on artifical galaxies with known physical properties
            zzchi2 = np.zeros(n_nbh, dtype=np.float64)
            zzcoeff = np.zeros((n_nbh, nleg+1), dtype=np.float64)
            spectype = self._full_type[iBest].split(':::')[0] #redrock best spectype
            subtype = self._full_type[iBest].split(':::')[1].split('_') #redrock best subtype
            if spectype=='GALAXY':
                new_arch, gal_inds = return_N_nearest_archetypes_from_synthetic_spectra(arch_id=iBest, archetype_data=rrarchs_params, gal_data=archetype_galaxies[subtype], n_nbh=n_nbh, ret_wave=False)
                for i in range(n_nbh):
                    binned = {hs:trapz_rebin(self.wave*(1.+z), new_arch[i], wave) for hs, wave in dwave.items()}
                    binned = { hs:trans[hs]*binned[hs] for hs, w in dwave.items() }
                    tdata = {hs:np.append(binned[hs][:,None], legendre[hs].transpose(), axis=1 ) for hs in dwave.keys()}
                    zzchi2[i], zzcoeff[i] = calc_zchi2_one(spectra, weights, flux, wflux, tdata)
                i_new_Best = np.argmin(zzchi2) # new archetype
                gal_id = gal_inds[i_new_Best]
                new_fulltype = 'GALAXY:::%s_%d'%(subtype, gal_id) #same as Redrock format
                chi2, coeff, fin_fulltype = zzchi2[i_new_Best], zzcoeff[i_new_Best], new_fulltype
            else:
                # For QSOs and Stars
                binned = self.rebin_template(iBest, z, dwave,trapz=True)
                binned = { hs:trans[hs]*binned[hs] for hs, w in dwave.items() }
                tdata = { hs:np.append(binned[hs][:,None],legendre[hs].transpose(), axis=1 ) for hs, wave in dwave.items() }
                zzchi2, zzcoeff = calc_zchi2_one(spectra, weights, flux, wflux, tdata)
                chi2, coeff, fin_fulltype =  zzchi2, zzcoeff, self._full_type[iBest]

        else:
            #Without New archetypes
            binned = self.rebin_template(iBest, z, dwave,trapz=True)
            binned = { hs:trans[hs]*binned[hs] for hs, w in dwave.items() }
            tdata = { hs:np.append(binned[hs][:,None],legendre[hs].transpose(), axis=1 ) for hs, wave in dwave.items() }
            zzchi2, zzcoeff = calc_zchi2_one(spectra, weights, flux, wflux, tdata)
            chi2, coeff, fin_fulltype =  zzchi2, zzcoeff, self._full_type[iBest]
        
        return chi2, coeff, fin_fulltype


class All_archetypes():
    """Class to store all different archetypes of all the different spectype.

    Args:
        lstfilename (lst str): List of file to get the templates from
        archetypes_dir (str): Directory to the archetypes

    """
    def __init__(self, lstfilename=None, archetypes_dir=None):

        # Get list of path to archetype
        if lstfilename is None:
            lstfilename = find_archetypes(archetypes_dir)

        # Load archetype
        self.archetypes = {}
        for f in lstfilename:
            archetype = Archetype(f)
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
        archetypes_dir_expand = os.path.expandvars(archetypes_dir)
        lstfilename = glob(os.path.join(archetypes_dir_expand, 'rrarchetype-*.fits'))
        lstfilename = sorted([ f.replace(archetypes_dir_expand,archetypes_dir) for f in lstfilename])

    return lstfilename
