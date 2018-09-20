"""
Classes and functions for priors.
"""

from astropy.io import fits
import os


class Priors():
    """Class to store all different redshift priors.

    Args:
        filename (str): the path to the redshift prior file

    """
    def __init__(self, filename):

        print('DEBUG: Using priors')
        print('DEBUG: read {}'.format(filename))

        h = fits.open(os.path.expandvars(filename), memmap=False)

        targetid = h['PRIORS'].data['TARGETID']
        z = h['PRIORS'].data['Z']
        sigma = h['PRIORS'].data['SIGMA']
        self._priors = { targetid[i]:{'Z':z[i], 'SIGMA':sigma[i]} for i in range(z.size) }

        h.close()

        return
    def eval(self, targetid, z):
        try:
            z0 = self._priors[targetid]['Z']
            s0 = self._priors[targetid]['SIGMA']
            return ((z-z0)/s0)**2
        except KeyError:
            print('DEBUG: targetid {} not in priors'.format(targetid))
            return 0.
