"""
Classes and functions for archetypes.
"""

import sys
import os
from glob import glob
import fitsio
import scipy as sp
from scipy.interpolate import interp1d

class Archetype():
    """

    """
    def __init__(self, filename):

        ### Load the file
        h = fitsio.FITS(filename)

        hdr = h['ARCHETYPES'].read_header()
        self.flux = sp.array(h['ARCHETYPES']['ARCHETYPE'][:])
        self._rrtype = hdr['RRTYPE'].strip()
        self._subtype = sp.array(sp.char.strip(h['ARCHETYPES']['SUBTYPE'][:].astype(str)))

        self.wave = sp.asarray(hdr['CRVAL1'] + hdr['CDELT1']*sp.arange(self.flux.shape[1]))
        if 'LOGLAM' in hdr and hdr['LOGLAM'] != 0:
            self.wave = 10**self.wave

        h.close()

        self._narch = self.flux.shape[0]
        self._nwave = self.flux.shape[1]
        self._full_type = sp.char.add(self._rrtype+':::',self._subtype)
        self._full_type[self._subtype==''] = self._rrtype

        ### Dic of templates
        self._archetype = {}
        self._archetype['INTERP'] = sp.array([None]*self._narch)
        for i in range(self._narch):
            self._archetype['INTERP'][i] = interp1d(self.wave,self.flux[i,:],fill_value='extrapolate',kind='linear')

        return
class All_archetypes():
    """

    """
    def __init__(self, lstfilename=None, archetypes_dir=None):

        ### Get list of path to archetype
        if lstfilename is None:
            lstfilename = find_archetypes(archetypes_dir)

        ### Load archetype
        self.lst_archetypes = {}
        self.archetypes = {}
        for f in lstfilename:
            archetype = Archetype(f)
            self.lst_archetypes[archetype._rrtype] = archetype
            self.archetypes[archetype._rrtype] = {}
            self.archetypes[archetype._rrtype]['SUBTYPE'] = archetype._subtype
            self.archetypes[archetype._rrtype]['INTERP'] = archetype._archetype

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

