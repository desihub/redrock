"""
redrock.utils
=============

Redrock utility functions.
"""

import sys
import os

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from . import constants

class Inoue14(object):
    def __init__(self, scale_tau=1.):
        """
        IGM absorption from Inoue et al. (2014)
        
        Parameters
        ----------
        scale_tau : float
            Parameter multiplied to the IGM :math:`\tau` values (exponential 
            in the linear absorption fraction).  
            I.e., :math:`f_\mathrm{igm} = e^{-\mathrm{scale\_tau} \tau}`.

        Copyright (c) 2016-2022 Gabriel Brammer

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        Code updated in 2023 by Craig Warner to add GPU support.
        """
        super(Inoue14, self).__init__()
        
        self._load_data()
        self.scale_tau = scale_tau
        self.has_gpu_data = False

    @staticmethod
    def _pow(a, b):
        """C-like power, a**b
        """
        return a**b
    
    def _load_data(self):
        from importlib import resources
        LAF_file = resources.files('redrock').joinpath('data/LAFcoeff.txt')
        DLA_file = resources.files('redrock').joinpath('data/DLAcoeff.txt')
    
        data = np.loadtxt(LAF_file, unpack=True)
        ix, lam, ALAF1, ALAF2, ALAF3 = data
        self.lam = lam[:,np.newaxis]
        self.ALAF1 = ALAF1[:,np.newaxis]
        self.ALAF2 = ALAF2[:,np.newaxis]
        self.ALAF3 = ALAF3[:,np.newaxis]
        
        data = np.loadtxt(DLA_file, unpack=True)
        ix, lam, ADLA1, ADLA2 = data
        self.ADLA1 = ADLA1[:,np.newaxis]
        self.ADLA2 = ADLA2[:,np.newaxis]
                
        return True

    def copy_data_to_gpu(self):
        if (self.has_gpu_data):
            return
        import cupy as cp
        self.gpu_ALAF1 = cp.asarray(self.ALAF1[:,:,None])
        self.gpu_ALAF2 = cp.asarray(self.ALAF2[:,:,None])
        self.gpu_ALAF3 = cp.asarray(self.ALAF3[:,:,None])
        self.gpu_ADLA1 = cp.asarray(self.ADLA1[:,:,None])
        self.gpu_ADLA2 = cp.asarray(self.ADLA2[:,:,None])
        self.gpu_lam = cp.asarray(self.lam[:,:,None])
        self.has_gpu_data = True
        return

    @property
    def NA(self):
        """
        Number of Lyman-series lines
        """
        return self.lam.shape[0]

    def tLSLAF(self, zS, lobs, use_gpu=False):
        """
        Lyman series, Lyman-alpha forest
        """
        if (use_gpu):
            import cupy as cp
            ALAF1 = self.gpu_ALAF1
            ALAF2 = self.gpu_ALAF2
            ALAF3 = self.gpu_ALAF3
            l2 = self.gpu_lam
            zS = cp.asarray(zS)
        else:
            ALAF1 = self.ALAF1[:,:,None]
            ALAF2 = self.ALAF2[:,:,None]
            ALAF3 = self.ALAF3[:,:,None]
            l2 = self.lam[:,:,None]

        z1LAF = 1.2
        z2LAF = 4.7

        tLSLAF_value = np.zeros_like(lobs*l2).T

        x0 = (lobs / (1+zS)[:,None]) < l2
        x1 = x0 & (lobs < l2*(1+z1LAF))
        x2 = x0 & ((lobs >= l2*(1+z1LAF)) & (lobs < l2*(1+z2LAF)))
        x3 = x0 & (lobs >= l2*(1+z2LAF))

        tLSLAF_value = np.zeros_like(lobs*l2)
        tLSLAF_value[x1] += ((ALAF1/l2**1.2)*lobs**1.2)[x1]
        tLSLAF_value[x2] += ((ALAF2/l2**3.7)*lobs**3.7)[x2]
        tLSLAF_value[x3] += ((ALAF3/l2**5.5)*lobs**5.5)[x3]

        return tLSLAF_value.sum(axis=0)


    def tLSDLA(self, zS, lobs, use_gpu=False):
        """
        Lyman Series, DLA
        """
        if (use_gpu):
            import cupy as cp
            ADLA1 = self.gpu_ADLA1
            ADLA2 = self.gpu_ADLA2
            l2 = self.gpu_lam
            zS = cp.asarray(zS)
        else:
            ADLA1 = self.ADLA1[:,:,None]
            ADLA2 = self.ADLA2[:,:,None]
            l2 = self.lam[:,:,None]
        z1DLA = 2.0
        
        tLSDLA_value = np.zeros_like(lobs*l2)
        
        x0 = ((lobs / (1+zS)[:,None]) < l2) & (lobs < l2*(1.+z1DLA))
        x1 = ((lobs / (1+zS)[:,None]) < l2) & ~(lobs < l2*(1.+z1DLA))
        
        tLSDLA_value[x0] += ((ADLA1/l2**2)*lobs**2)[x0]
        tLSDLA_value[x1] += ((ADLA2/l2**3)*lobs**3)[x1]
                
        return tLSDLA_value.sum(axis=0)


    def tLCDLA(self, zS, lobs, use_gpu=False):
        """
        Lyman continuum, DLA
        """
        if (use_gpu):
            import cupy as cp
            power = cp.power
            zS = cp.asarray(zS)
        else:
            power = np.power
        z1DLA = 2.0
        lamL = 911.8
        
        tLCDLA_value = np.zeros_like(lobs)
        
        x0 = lobs < lamL*(1.+zS)[:,None]

        y0 = x0 & (zS[:,None] < z1DLA)
        tLCDLA_value[y0] = 0.2113 * (power(1.0+zS[:,None], 2)*y0)[y0] - 0.07661 * (power(1.0+zS[:,None], 2.3)*y0)[y0] * power(lobs[y0]/lamL, (-3e-1)) - 0.1347 * power(lobs[y0]/lamL, 2)

        y0 = x0 & (zS[:,None] >= z1DLA)
        x1 = lobs >= lamL*(1.+z1DLA)
        tLCDLA_value[y0 & x1] = 0.04696 * (power(1.0+zS[:,None], 3)*y0)[y0 & x1] - 0.01779 * (power(1.0+zS[:,None], 3.3)*y0)[y0 & x1] * power(lobs[y0 & x1]/lamL, (-3e-1)) - 0.02916 * power(lobs[y0 & x1]/lamL, 3)
        tLCDLA_value[y0 & ~x1] =0.6340 + 0.04696 * (power(1.0+zS[:,None], 3)*y0)[y0 & ~x1] - 0.01779 * (power(1.0+zS[:,None], 3.3)*y0)[y0 & ~x1] * power(lobs[y0 & ~x1]/lamL, (-3e-1)) - 0.1347 * power(lobs[y0 & ~x1]/lamL, 2) - 0.2905 * power(lobs[y0 & ~x1]/lamL, (-3e-1))
        
        return tLCDLA_value


    def tLCLAF(self, zS, lobs, use_gpu=False):
        """
        Lyman continuum, LAF
        """
        if (use_gpu):
            import cupy as cp
            power = cp.power
            zS = cp.asarray(zS)
        else:
            power = np.power
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = np.zeros_like(lobs)
        
        x0 = lobs < lamL*(1.+zS)[:,None]
        y0 = x0 & (zS[:,None] < z1LAF) 
        tLCLAF_value[y0] = 0.3248 * (power(lobs[y0]/lamL, 1.2) - (power(1.0+zS[:,None], -9e-1)*y0)[y0] * power(lobs[y0]/lamL, 2.1))

        y0 = x0 & (zS[:,None] < z2LAF)
        x1 = lobs >= lamL*(1+z1LAF)
        tLCLAF_value[y0 & x1] = 2.545e-2 * ((power(1.0+zS[:,None], 1.6)*y0)[y0 & x1] * power(lobs[y0 & x1]/lamL, 2.1) - power(lobs[y0 & x1]/lamL, 3.7))
        tLCLAF_value[y0 & ~x1] = 2.545e-2 * (power(1.0+zS[:,None], 1.6)*y0)[y0 & ~x1] * power(lobs[y0 & ~x1]/lamL, 2.1) + 0.3248 * power(lobs[y0 & ~x1]/lamL, 1.2) - 0.2496 * power(lobs[y0 & ~x1]/lamL, 2.1)

        y0 = x0 & (zS[:,None] >= z2LAF)
        x1 = lobs > lamL*(1.+z2LAF)
        x2 = (lobs >= lamL*(1.+z1LAF)) & (lobs < lamL*(1.+z2LAF))
        x3 = lobs < lamL*(1.+z1LAF)

        tLCLAF_value[y0 & x1] = 5.221e-4 * ((power(1.0+zS[:,None], 3.4)*y0)[y0 & x1] * power(lobs[y0 & x1]/lamL, 2.1) - power(lobs[y0 & x1]/lamL, 5.5))
        tLCLAF_value[y0 & x2] = 5.221e-4 * (power(1.0+zS[:,None], 3.4)*y0)[y0 & x2] * power(lobs[y0 & x2]/lamL, 2.1) + 0.2182 * power(lobs[y0 & x2]/lamL, 2.1) - 2.545e-2 * power(lobs[y0 & x2]/lamL, 3.7)
        tLCLAF_value[y0 & x3] = 5.221e-4 * (power(1.0+zS[:,None], 3.4)*y0)[y0 & x3] * power(lobs[y0 & x3]/lamL, 2.1) + 0.3248 * power(lobs[y0 & x3]/lamL, 1.2) - 3.140e-2 * power(lobs[y0 & x3]/lamL, 2.1)
            
        return tLCLAF_value


    def full_IGM(self, z, lobs, use_gpu=False):
        """Get full Inoue IGM absorption

        Parameters
        ----------
        z : float array
            Redshift to evaluate IGM absorption

        lobs : array
            Observed-frame wavelength(s) in Angstroms.

        Returns
        -------
        abs : array
            IGM absorption

        """

        if (use_gpu):
            import cupy as cp
            arrexp = cp.exp
            self.copy_data_to_gpu()
        else:
            arrexp = np.exp

        tau_LS = self.tLSLAF(z, lobs, use_gpu=use_gpu) + self.tLSDLA(z, lobs, use_gpu=use_gpu)
        tau_LC = self.tLCLAF(z, lobs, use_gpu=use_gpu) + self.tLCDLA(z, lobs, use_gpu=use_gpu)

        ### Upturn at short wavelengths, low-z
        #k = 1./100
        #l0 = 600-6/k
        #clip = lobs/(1+z) < 600.
        #tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip = 0.

        return arrexp(-self.scale_tau*(tau_LC + tau_LS + tau_clip))

    def build_grid(self, zgrid, lrest):
        """Build a spline interpolation object for fast IGM models
        
        Returns: self.interpolate
        """
        
        from scipy.interpolate import CubicSpline
        igm_grid = np.zeros((len(zgrid), len(lrest)))
        for iz in range(len(zgrid)):
            igm_grid[iz,:] = self.full_IGM(zgrid[iz], lrest*(1+zgrid[iz]))
        
        self.interpolate = CubicSpline(zgrid, igm_grid)


IGM = None
def transmission_IGM_Inoue14(zObj, lObs, use_gpu=False):
    """Calculate the transmitted flux fraction from the Lyman series
    and due to the IGM.  This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)

    Args:
        zObj (array of float): Redshifts of objects
        lObs (array of float): wavelength grid
        use_gpu (boolean): whether to use CUPY

    Returns:
        array of float: transmitted flux fraction T[nz, nlambda]
    """
    global IGM
    if IGM is None:
        IGM = Inoue14()

    if use_gpu:
        xp = cp
    else:
        xp = np

    #- tile observer frame wavelengths lObs to match lRest dimensions
    lObs = xp.tile(lObs, (zObj.size, 1))

    # Transmission array to fill in
    T = xp.ones(lObs.shape)

    # Only process redshifts with restframe wavelenths at or shorter than LyA.
    min_restwave_per_z = lObs.min()/(1.+zObj)
    ii = min_restwave_per_z < constants.LyA_wavelength
    T[ii, :] = IGM.full_IGM(zObj[ii], lObs[ii,:], use_gpu=use_gpu)

    return T


def transmission_Lyman_CaluraKamble(zObj,lObs, use_gpu=False,
                                    model='Calura12'):
    """Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)

    Args:
        zObj (array of float): Redshifts of objects
        lObs (array of float): wavelength grid
        use_gpu (boolean): whether to use CUPY
        model (str): Calura12 or Kamble20, IGM model constants to use

    Returns:
        array of float: transmitted flux fraction T[nz, nlambda]
    """
    if use_gpu:
        xp = cp
    else:
        xp = np

    Lyman_series = constants.Lyman_series[model]

    #- lRest[num_redshifts, num_wavelengths]
    #- restframe wavelengths for lObs for each zObj
    lRest = xp.outer(1.0/(1.0+zObj), lObs)
    min_wave = lRest.min()

    #- tile observer frame wavelengths lObs to match lRest dimensions
    lObs = xp.tile(lObs, (zObj.size, 1))

    #- Transmission array to fill in
    T = xp.ones(lRest.shape)

    for l in list(Lyman_series.keys()):
        if (min_wave > Lyman_series[l]['line']):
            continue

        w      = lRest<Lyman_series[l]['line']
        zpix   = lObs[w]/Lyman_series[l]['line']-1.
        tauEff = Lyman_series[l]['A']*(1.+zpix)**Lyman_series[l]['B']
        T[w]  *= np.exp(-tauEff)

    return T


igm_models = ('Calura12', 'Kamble20', 'Inoue14', 'None', None)

def transmission_Lyman(zObj, lObs, use_gpu=False, model='Calura12', always_return_array=True):
    """Calculate the transmitted flux fraction from the Lyman series
    This returns the transmitted flux fraction:
    1 -> everything is transmitted (medium is transparent)
    0 -> nothing is transmitted (medium is opaque)

    Args:
        zObj (float or array of float): Redshift(s) of object
        lObs (float or array of float): wavelength grid
        use_gpu (boolean): whether to use CUPY
        model: which IGM model to use; Calura12, Kamble20, or Inoue14
        always_return_array: if True (default), always return array even if all ones

    Returns:
        float or array of float: transmitted flux fraction
        if both zObj and lObs are scalar, this returns a float.
        If one is scalar and other is array, this returns array
        If both are arrays, this returns 2D array[nz, nwave]

    if always_return_array is False, returns None if there is no IGM absoption
    for this redshift (faster).
    """
    if model not in igm_models:
        raise ValueError(f'Unrecognized model {model}; should be one of {igm_models}')

    #- Handle no-op cases quickly
    if (model == 'None' or model is None) and not always_return_array:
        return None

    #- lowest observed wavelength at any redshift
    min_wave = np.min(lObs) / (1. + np.max(zObj))
    if min_wave > constants.LyA_wavelength and not always_return_array:
        return None

    #- Proceed with slower calculations
    if use_gpu:
        xp = cp
    else:
        xp = np

    scalar_zObj = xp.isscalar(zObj)
    scalar_lObs = xp.isscalar(lObs)

    zObj = xp.atleast_1d(zObj)
    lObs = xp.atleast_1d(lObs)

    nz = len(zObj)
    nwave = len(lObs)

    if always_return_array and (min_wave > constants.LyA_wavelength or model is None or model == 'None'):
        T = xp.ones( (nz, nwave) )
    elif model in ('Calura12', 'Kamble20'):
        T = transmission_Lyman_CaluraKamble(zObj, lObs, use_gpu=use_gpu, model=model)
    elif model == 'Inoue14':
        T = transmission_IGM_Inoue14(zObj, lObs, use_gpu=use_gpu)
    else:
        # should have been caught by first check, but complain here just in case
        raise ValueError(f'Unrecognized model {model}; should be one of {igm_models}')

    #- convert transmission T back to dimensionality of input
    if scalar_zObj and scalar_lObs:
        assert T.shape == (1,1)
        T = T[0,0]
    elif scalar_zObj:
        assert T.shape[0] == 1
        T = T[0]
    elif scalar_lObs:
        assert T.shape[1] == 1
        T = T[:,0]

    return T

