"""
Classes and functions for archetypes.
"""

import sys
import os
from glob import glob
from astropy.io import fits
import scipy as sp
from scipy.interpolate import interp1d
from scipy import special

from .zscan import spectral_data, calc_zchi2_one

from ._zscan import _zchi2_one

from .fitz import fitz, get_dv

from .zwarning import ZWarningMask as ZW

from . import constants

class Archetype():
    """Class to store all different archetypes from the same spectype.

    The archetype data are read from a redrock-format archetype file.

    Args:
        filename (str): the path to the archetype file

    """
    def __init__(self, filename):

        # Load the file
        h = fits.open(filename, memmap=False)

        hdr = h['ARCHETYPES'].header
        self.flux = sp.array(h['ARCHETYPES'].data['ARCHETYPE'])
        self._rrtype = hdr['RRTYPE'].strip()
        self._subtype = sp.array(sp.char.strip(h['ARCHETYPES'].data['SUBTYPE'].astype(str)))

        self.wave = sp.asarray(hdr['CRVAL1'] + hdr['CDELT1']*sp.arange(self.flux.shape[1]))
        if 'LOGLAM' in hdr and hdr['LOGLAM'] != 0:
            self.wave = 10**self.wave

        h.close()

        self._narch = self.flux.shape[0]
        self._nwave = self.flux.shape[1]
        self._full_type = sp.char.add(self._rrtype+':::',self._subtype)
        self._full_type[self._subtype==''] = self._rrtype

        # Dic of templates
        self._archetype = {}
        self._archetype['INTERP'] = sp.array([None]*self._narch)
        for i in range(self._narch):
            self._archetype['INTERP'][i] = interp1d(self.wave,self.flux[i,:],fill_value='extrapolate',kind='linear')

        return

    def get_best_archetype(self,spectra,weights,flux,wflux,dwave,z,legendre):
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
            subtype (str): subtype of best archetype

        """
        wave = sp.concatenate([ spec.wave for spec in spectra ])
        waveRF = wave/(1.+z)

        leg = sp.concatenate([ v for v in legendre.values() ]).transpose()
        Tb = sp.append( sp.zeros((flux.size,1)),leg, axis=1 )
        nbasis = 1+leg.shape[1]

        zzchi2 = sp.zeros(self._narch, dtype=sp.float64)
        zzcoeff = sp.zeros((self._narch, nbasis), dtype=sp.float64)

        for i, arch in enumerate(self._archetype['INTERP']):
            # TODO: use rebin_template and calc_zchi2_one to use
            #   the resolution matrix and the different spectrograph
            #binned = rebin_template(template, z, dwave)
            #zzchi2[i], zzcoeff[i] = calc_zchi2_one(spectra, weights, flux, wflux, binned)
            Tb[:,0] = arch(waveRF)
            zcoeff = sp.zeros(nbasis, dtype=sp.float64)
            zzchi2[i] = _zchi2_one(Tb, weights, flux, wflux, zcoeff)
            zzcoeff[i] = zcoeff

        iBest = sp.argmin(zzchi2)
        # TODO: should we look at the value of zzcoeff[0] and if negative
        #   set the chi2 to very big?

        return zzchi2[iBest], zzcoeff[iBest], self._subtype[iBest]


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
            self.archetypes[archetype._rrtype] = archetype

        return
    def get_best_archetype(self,spectra,tzfit):
        """Rearange tzfit according to chi2 from archetype

        Args:
            spectra (list): list of Spectrum objects.
            tzfit (astropy.table): attributes of all the different minima

        Returns:
            tzfit (astropy.table): attributes of all the different minima

        """

        # TODO: set this as a parameter
        deg_legendre = 3

        # Build dictionary of wavelength grids
        wave = sp.concatenate([ spec.wave for spec in spectra ])
        dwave = {}
        legendre = {}
        for s in spectra:
            if s.wavehash not in dwave:
                dwave[s.wavehash] = s.wave
                x = (s.wave-wave.min())/(wave.max()-wave.min())*2.-1.
                legendre[s.wavehash] = sp.array( [special.legendre(i)(x) for i in range(deg_legendre) ] )

        (weights, flux, wflux) = spectral_data(spectra)

        # Fit each archetype
        for res in tzfit:
            # TODO keep coeff archetype?
            res['chi2'], coeff, res['subtype'] = self.archetypes[res['spectype']].get_best_archetype(spectra,
                weights, flux, wflux, dwave, res['z'], legendre)

        tzfit.sort('chi2')
        tzfit['znum'] = sp.arange(len(tzfit))
        tzfit['deltachi2'] = sp.ediff1d(tzfit['chi2'], to_end=0.0)

        #- set ZW.SMALL_DELTA_CHI2 flag
        for i in range(len(tzfit)-1):
            noti = sp.arange(len(tzfit))!=i
            alldeltachi2 = sp.absolute(tzfit['chi2'][noti]-tzfit['chi2'][i])
            alldv = sp.absolute(get_dv(z=tzfit['z'][noti],zref=tzfit['z'][i]))
            zwarn = sp.any( (alldeltachi2<constants.min_deltachi2) & (alldv>=constants.max_velo_diff) )
            if zwarn:
                tzfit['zwarn'][i] |= ZW.SMALL_DELTA_CHI2
            elif tzfit['zwarn'][i]&ZW.SMALL_DELTA_CHI2:
                tzfit['zwarn'][i] &= ~ZW.SMALL_DELTA_CHI2

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

    return sorted(glob(os.path.join(archetypes_dir, 'rrarchetype-*.fits')))
