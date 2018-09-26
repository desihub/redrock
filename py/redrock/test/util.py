#- Utility functions for testing

from __future__ import division, print_function

import numpy as np
import scipy.sparse

from ..targets import Spectrum, Target, DistTargetsCopy
from ..templates import Template


def get_template(wavemin=100, wavemax=9000, wavestep=5, spectype='GALAXY', subtype='', redshifts=None):
    """Returns fake template PCA eigenvectors to use for testing
    """
    if redshifts is None:
        redshifts = np.linspace(0, 1, 20)
    wave = np.arange(wavemin, wavemax + wavestep/2.0, wavestep)
    flux = np.zeros((3, len(wave)))
    wavemid = (wavemin + wavemax) / 2.0
    wx = np.pi*(2*(wave - wavemin) / (wavemax-wavemin) - 1)
    flux[0] = 1.0
    flux[1] = np.arange(len(wave)) / len(wave)
    flux[2] = np.exp(-(wave-wavemid)**2/(2*20**2)) + 0.1*np.sin(wx)

    return Template(spectype=spectype, redshifts=redshifts, wave=wave,
        flux=flux, subtype=subtype)


def get_target(z=0.5, wavestep=5):
    """Returns a fake target at redshift z to use for testing
    """
    template = get_template()
    c = np.random.normal(size=template.nbasis)
    c = [1,2,3]
    spectra = list()
    for wave in [np.arange(4000, 6700, wavestep), np.arange(6502, 8001, wavestep)]:
        flux = template.eval(c, wave, z)
        sigma = np.random.normal(loc=1, scale=0.1, size=len(wave)).clip(0.5, 1.5)
        ivar = 1/sigma**2
        R = _getR(len(wave), 2.0)
        assert isinstance(R, scipy.sparse.dia_matrix)
        assert hasattr(R, 'offsets')
        for _ in range(2):
            noisyflux = flux + np.random.normal(scale=sigma)
            spectra.append(Spectrum(wave, noisyflux, ivar, R, R.tocsr()))

    return Target(123, spectra)


# Return a couple of fake targets
def fake_targets():
    t1 = get_target(0.2)
    t1.id = 111
    t2 = get_target(0.5)
    t2.id = 222
    # Make a distributed targets object that just copies existing
    # targets.
    dtarg = DistTargetsCopy([t1, t2])
    return dtarg


#- Return a normalized sampled Gaussian (no integration, just sampling)
def _norm_gauss(x, sigma):
    y = np.exp(-x**2/(2.0*sigma))
    return y / np.sum(y)

#- Utility function to create a resolution matrix with a given sigma in pixel units
def _getR(n, sigma):
    '''Returns a (n,n) sized sparse Resolution matrix with constant sigma'''
    x = np.arange(-5, 6)
    y = _norm_gauss(x, sigma)
    data = np.zeros((11, n))
    for i in range(n):
        data[:,i] = y
    return scipy.sparse.dia_matrix((data, x), shape=(n,n))
