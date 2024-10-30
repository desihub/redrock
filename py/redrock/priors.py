"""
redrock.priors
==============

Classes and functions for priors.
"""

from astropy.io import fits
import os
import numpy as np

class Priors():
    """Class to store all different redshift priors.

    Args:
        filename (str): the path to the redshift prior file.

    Note:
        The file should have at least one HDU with EXTNAME='PRIORS', composed of:

        * TARGETID: the id of the object
        * Z: the mean of the prior
        * SIGMA: the sigma (in dz) for the prior
    """
    def __init__(self, filename):

        print('Using priors')
        print(f'read {filename}')

        h = fits.open(os.path.expandvars(filename), memmap=False)

        targetid = h['PRIORS'].data['TARGETID']
        z = h['PRIORS'].data['Z']
        sigma = h['PRIORS'].data['SIGMA']
        self._param = { targetid[i]:{'Z':z[i], 'SIGMA':sigma[i]} for i in range(z.size) }

        h.close()

        self._type = h['PRIORS'].data['FUNCTION'][0]
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
            print(f'targetid {targetid} not in priors')
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
        return -2*np.log(1/(1.+((z-z0)/s0)**2))


    @staticmethod
    def tophat(z, z0, s0):
        """Return a tophat prior of mean z0 and width s0 on the grid z.

        Args:
            z : redshift grid
            z0 : mean
            s0 : width

        Returns:
            prior values on the redshift grid.

        Warning:
            * np.NaN <= np.NaN -> False
            * np.NaN <=/>= 0.0  -> False
            * np.inf >= 0.0 -> True
            * np.inf >= np.inf -> True

        Todo:
            * We need to use np.Nan value and not 1e10 value outside the prior to avoid
              the case where they are only one or two minima in the tophat
              since otherwise the second/third minima will be selected outside the prior
            * Cannot use np.inf value for that since np.inf <= np.inf is True ...
            * Need to had np.inf value in the left and right of the tophat in the case where
              the minima is the first or the last point !
        """

        prior = np.where(np.abs(z - z0) < s0/2, 0., np.NaN)

        if np.all(np.isnan(prior)):
            return prior

        index_left, index_right = np.argwhere(prior>=0.0)[0], np.argwhere(prior>=0.0)[-1]

        if index_left == 0:
            prior[index_left] = np.inf
        else:
            prior[index_left - 1] = np.inf
        if index_right == (prior.size -1):
            prior[index_right] = np.inf
        else:
            prior[index_right + 1] = np.inf

        return prior
