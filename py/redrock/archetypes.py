"""
Classes and functions for archetypes.
"""

import sys
import os
from glob import glob
import fitsio
import scipy as sp
from scipy.interpolate import interp1d
from scipy import special
import numba

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

    def get_best_archetype(self,wave,flux,ivar,z,legendre):

        chi2Best = None
        iBest = None
        paramBest = None

        waveRF = wave/(1.+z)
        Tb = sp.append( sp.zeros((flux.size,1)),legendre, axis=1 )

        for i, arch in enumerate(self._archetype['INTERP']):

            Tb[:,0] = arch(waveRF)

            M = Tb.T.dot(sp.multiply(ivar[:,None], Tb))
            y = Tb.T.dot(flux*ivar)
            zcoeff = sp.linalg.solve(M, y)
            m = Tb.dot(zcoeff)
            fval = sp.sum( (flux-m)**2*ivar )

            if chi2Best is None or chi2Best>fval:
                chi2Best = fval
                paramBest = zcoeff
                iBest = i

        return chi2Best, paramBest, self._subtype[iBest]
class All_archetypes():
    """

    """
    def __init__(self, lstfilename=None, archetypes_dir=None):

        ### Get list of path to archetype
        if lstfilename is None:
            lstfilename = find_archetypes(archetypes_dir)

        ### Load archetype
        self.archetypes = {}
        for f in lstfilename:
            archetype = Archetype(f)
            self.archetypes[archetype._rrtype] = archetype

        return
    def get_best_archetype(self,spectra,tzfit):
        """

        """

        wave = sp.concatenate([ spec.wave for spec in spectra ])
        flux = sp.concatenate([ spec.flux for spec in spectra ])
        ivar = sp.concatenate([ spec.ivar for spec in spectra ])

        ### TODO: set this as a parameter
        deg_legendre = 3
        waveLegendre = (wave-wave[0])/(wave[-1]-wave[0])*2.-1.
        legendre = sp.array([ special.legendre(i)(waveLegendre) for i in range(deg_legendre) ]).transpose()

        tzfit_arch = {}
        for res in tzfit:
            znum = res['znum']
            tzfit_arch[znum] = {}
            tzfit_arch[znum]['zwarn'] = res['zwarn']
            tzfit_arch[znum]['spectype'] = res['spectype']
            tzfit_arch[znum]['chi2'], tzfit_arch[znum]['coeff'], tzfit_arch[znum]['subtype'] = \
                self.archetypes[res['spectype']].get_best_archetype(wave,flux,ivar,res['z'],legendre)

        '''
        chi2 = sp.array([ tzfit_arch[znum]['chi2'] for znum in tzfit_arch ])
        new_min_idx = sp.array([ znum for znum in tzfit_arch ])[chi2==chi2.min()][0]
        tz = tzfit_arch[new_min_idx]['z']
        tchi2 = tzfit_arch[new_min_idx]['chi2']

        noti = sp.arange(chi2.size)!=new_min_idx
        dv = sp.absolute( sp.array([ get_dv(tzfit_arch[znum]['z'],tz) for znum in list(tzfit_arch.keys()) if znum!=new_min_idx ]) )
        deltachi2 = sp.array([ tzfit_arch[znum]['chi2']-tchi2 for znum in list(tzfit_arch.keys()) if znum!=new_min_idx ])

        dic_cat['Z'][i] = tz
        dic_cat['SPECTYPE'][i] = tzfit_arch[new_min_idx]['SPECTYPE']
        if sp.any( (deltachi2<9.) & (dv>=1000.) ):
            dic_cat['ZWARN'][i] |= 2**2
        elif dic_cat['ZWARN'][i] == 2**2:
            dic_cat['ZWARN'][i] = 0
        '''
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

