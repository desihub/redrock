import unittest
import numpy as np

cp_available = False
try:
    import cupy as cp
    d = cp.cuda.Device()
    cp_available = cp.is_available() 
except Exception:
    cp_available = False

from .. import igm
from ..constants import Lyman_series, LyA_wavelength

class TestUtils(unittest.TestCase):
    """Test redrock utils.
    """

    @classmethod
    def setUpClass(cls):
        cls.obswave = np.arange(3600, 9801, 50)
        cls.redshifts = np.arange(0, 5.1, 0.5)

    def test_transmission_Lyman_scalars(self, use_gpu=False):
        # scalar inputs -> scalar output
        for model in igm.igm_models:
            for z in self.redshifts:
                for w in (self.obswave[0], self.obswave[-1]):
                    print(f'{model=} {z=} {w=}')
                    t = igm.transmission_Lyman(z, w, model=model, always_return_array=False)                        
                    if w/(1+z) > LyA_wavelength or model == 'None':
                        self.assertEqual(t, None)
                    else:
                        self.assertTrue(np.isscalar(t))
                        self.assertLess(t, 1.0)

                    t = igm.transmission_Lyman(z, w, model=model, always_return_array=True)
                    self.assertTrue(np.isscalar(t))
                    if w/(1+z) > LyA_wavelength or model == 'None':
                        self.assertEqual(t, 1.0)
                    else:
                        self.assertLess(t, 1.0)


    def test_transmission_Lyman_scalar_redshift(self, use_gpu=False):
        for model in igm.igm_models:
            for z in self.redshifts:
                print(f'{model=} {z=}')
                t = igm.transmission_Lyman(z, self.obswave, model=model,
                                           use_gpu=use_gpu, always_return_array=False)                
                if self.obswave[0]/(1+z) > LyA_wavelength or model == 'None':
                    self.assertEqual(t, None)
                else:
                    self.assertEqual(t.shape, (len(self.obswave),) )
                    self.assertTrue(np.any(t < 1.0))

                t = igm.transmission_Lyman(z, self.obswave, model=model,
                                           use_gpu=use_gpu, always_return_array=True)
                self.assertEqual(t.shape, (len(self.obswave),) )
                if self.obswave[0]/(1+z) > LyA_wavelength or model == 'None':
                    self.assertTrue(np.all(t == 1.0))
                else:
                    self.assertTrue(np.any(t < 1.0))

    def test_transmission_Lyman_scalar_wavelength(self, use_gpu=False):
        for model in igm.igm_models:
            for w in self.obswave:
                print(f'{model=} {w=}')
                t = igm.transmission_Lyman(self.redshifts, w, model=model,
                                           use_gpu=use_gpu, always_return_array=False)
                if w/(1+self.redshifts[-1]) > LyA_wavelength or model == 'None':
                    self.assertEqual(t, None)
                else:
                    self.assertEqual(t.shape, (len(self.redshifts),) )
                    self.assertTrue(np.any(t < 1.0))

                t = igm.transmission_Lyman(self.redshifts, w, model=model,
                                           use_gpu=use_gpu, always_return_array=True)
                self.assertEqual(t.shape, (len(self.redshifts),) )
                if w/(1+self.redshifts[-1]) > LyA_wavelength or model == 'None':
                    self.assertTrue(np.all(t == 1.0))
                else:
                    self.assertTrue(np.any(t < 1.0))

    def test_transmission_Lyman(self, use_gpu=False):
        restwave = np.outer(1/(1+self.redshifts), self.obswave)
        for model in igm.igm_models:
            print(f'{model=}')
            t = igm.transmission_Lyman(self.redshifts, self.obswave, model=model,
                                       use_gpu=use_gpu, always_return_array=False)
            if model == 'None':
                self.assertEqual(t, None)
            else:
                self.assertEqual(t.shape, (len(self.redshifts), len(self.obswave)) )
                self.assertTrue(np.any(t < 1.0))

                ii = restwave > LyA_wavelength
                self.assertTrue(np.all(t[ii] == 1.0))

            t = igm.transmission_Lyman(self.redshifts, self.obswave, model=model,
                                       use_gpu=use_gpu, always_return_array=True)
            if model == 'None':
                self.assertTrue(np.all(t == 1.0))
            else:
                self.assertEqual(t.shape, (len(self.redshifts), len(self.obswave)) )
                self.assertTrue(np.any(t < 1.0))

                ii = restwave > LyA_wavelength
                self.assertTrue(np.all(t[ii] == 1.0))


    @unittest.skipUnless(cp_available, "Skipping test_Lyman_transmission_GPU because no GPU found.")
    def test_Lyman_transmission_GPU(self):
        '''Test the GPU version of Lyman transmission'''
        self.test_transmission_Lyman_scalars(use_gpu=True)
        self.test_transmission_Lyman_scalar_redshift(use_gpu=True)
        self.test_transmission_Lyman_scalar_wavelength(use_gpu=True)
        self.test_transmission_Lyman(use_gpu=True)
