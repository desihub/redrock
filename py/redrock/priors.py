"""
Classes and functions for priors.
"""

from astropy.io import fits
import os
import numpy as np

class Priors():
    """Class to store all different redshift priors.

    Args:
        filename (str): the path to the redshift prior file.
        The file should have at least one HDU with
        EXTNAME='PRIORS', composed of:
            TARGETID: the id of the object
            Z: the mean of the prior
            SIGMA: the sigma (in dz) for the prior

    """
    def __init__(self, filename):

        print('DEBUG: Using priors')
        print('DEBUG: read {}'.format(filename))

        h = fits.open(os.path.expandvars(filename), memmap=False)

        targetid = h['PRIORS'].data['TARGETID']
        z = h['PRIORS'].data['Z']
        sigma = h['PRIORS'].data['SIGMA']
        self._param = { targetid[i]:{'Z':z[i], 'SIGMA':sigma[i]} for i in range(z.size) }

        h.close()

        self._type = 'gaussian'
        self._func = getattr(self,self._type)

        return
    def eval(self, targetid, z):
        """Return prior contribution to the chi2 for the given TARGETID and redshift grid
        Args:
            targetid : TARGETID of the object
            z : redshift at which to evaluate template flux
        Returns:
            prior values on the redshift grid
        """
        try:
            z0 = self._param[targetid]['Z']
            s0 = self._param[targetid]['SIGMA']
            return self._func(z,z0,s0)
        except KeyError:
            print('DEBUG: targetid {} not in priors'.format(targetid))
            return 0.

    @staticmethod
    def gaussian(z,z0,s0):
        """Return a Gaussian prior of mean z0 and sigma s0 on the grid z
        Args:
            z : redshift grid
            z0 : mean
            s0 : sigma
        Returns:
            prior values on the redshift grid
        """
        return ((z-z0)/s0)**2
    @staticmethod
    def lorentzien(z,z0,s0):
        """Return a Lorentzien prior of mean z0 and sigma s0 on the grid z
        Args:
            z : redshift grid
            z0 : mean
            s0 : sigma
        Returns:
            prior values on the redshift grid
        """
        return -np.log(1.+((z-z0)/s0)**2)
