"""
Classes and functions for archetypes.
"""

import os
from glob import glob
from astropy.io import fits
import numpy as np

from .zscan import calc_zchi2_one

from ._zscan import _zchi2_one

from .rebin import trapz_rebin


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
        self.flux = np.asarray(h['ARCHETYPES'].data['ARCHETYPE'])
        self._rrtype = hdr['RRTYPE'].strip()
        self._subtype = np.array(np.char.strip(h['ARCHETYPES'].data['SUBTYPE'].astype(str)))

        self.wave = np.asarray(hdr['CRVAL1'] + hdr['CDELT1']*np.arange(self.flux.shape[1]))
        if 'LOGLAM' in hdr and hdr['LOGLAM'] != 0:
            self.wave = 10**self.wave

        h.close()

        self._narch = self.flux.shape[0]
        self._nwave = self.flux.shape[1]
        self._full_type = np.char.add(self._rrtype+':::',self._subtype)
        self._full_type[self._subtype==''] = self._rrtype

        return
    def rebin_template(self,index,z,dwave):
        """
        """
        return {hs:trapz_rebin((1.+z)*self.wave, self.flux[index], wave) for hs, wave in dwave.items()}

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

        nleg = legendre[list(legendre.keys())[0]].shape[0]
        leg = np.array([np.concatenate( [legendre[k][i] for k in legendre.keys()] ) for i in range(nleg)])
        Tb = np.append( np.zeros((flux.size,1)),leg.transpose(), axis=1 )

        zzchi2 = np.zeros(self._narch, dtype=np.float64)
        zzcoeff = np.zeros((self._narch, Tb.shape[1]), dtype=np.float64)
        zcoeff = np.zeros(Tb.shape[1], dtype=np.float64)

        for i in range(self._narch):
            print(i,self._narch)
            # TODO: use rebin_template and calc_zchi2_one to use
            #   the resolution matrix and the different spectrograph
            binned = self.rebin_template(i, z, dwave)
            #zzchi2[i], zzcoeff[i] = calc_zchi2_one(spectra, weights, flux, wflux, binned)
            Tb[:,0] = np.concatenate([ spec for spec in binned.values()])
            zzchi2[i] = _zchi2_one(Tb, weights, flux, wflux, zcoeff)
            zzcoeff[i] = zcoeff

        iBest = np.argmin(zzchi2)
        # TODO: should we look at the value of zzcoeff[0] and if negative
        #   set the chi2 to very big?

        import matplotlib.pyplot as plt
        wave = np.concatenate([ w for w in dwave.values() ])
        binned = self.rebin_template(iBest, z, dwave)
        Tb[:,0] = np.concatenate([ spec for spec in binned.values()])
        plt.plot(wave*(1.+z),flux)
        plt.plot(wave*(1.+z),Tb.dot(zzcoeff[iBest]))
        plt.grid()
        plt.show()

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
