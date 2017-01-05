import unittest
import numpy as np
import scipy.sparse

import redrock

#- Return a normalized sampled Gaussian (no integration, just sampling)
def norm_gauss(x, sigma):
    y = np.exp(-x**2/(2.0*sigma))
    return y / np.sum(y)

#- Utility function to create a resolution matrix with a given sigma in pixel units
def getR(n, sigma):
    '''Returns a (n,n) sized sparse Resolution matrix with constant sigma'''
    x = np.arange(-5, 6)
    y = norm_gauss(x, sigma)
    data = np.zeros((11, n))
    for i in range(n):
        data[:,i] = y
    return scipy.sparse.dia_matrix((data, x), shape=(n,n))


class TestZScan(unittest.TestCase):
    
    def setUp(self):
        pass
            
    def test_zscan(self):
        #- Create 2 component template
        wave = np.arange(100,900)
        flux = np.zeros((2, len(wave)))
        flux[0, 100] = 1
        flux[1, 120] = 1
        redshifts = np.linspace(0.3, 0.7, 200)
        template = redrock.Template('BLAT', redshifts, wave, flux)

        z = 0.5       #- redshift to use

        #- Create some noisy spectra on different wavelength grids
        spectra = list()
        sn2 = 0.0;
        a = [200, 200]
        for i in range(5):
            wave = np.arange(210+10*i, 400+10*i, 2.1)
            nwave = len(wave)
            noise = 10.0
            flux = np.random.normal(scale=noise, size=nwave)
            R = getR(nwave, sigma=1.0+0.5*i)  #- different resolutions per spectra
            Tx = np.zeros((template.flux.shape[0], nwave))
            for j in range(Tx.shape[0]):
                Tx[j] = redrock.rebin.trapz_rebin((1+z)*template.wave, template.flux[j], wave)

            signal = R.dot(Tx.T.dot(a))
            flux += np.random.poisson(signal)
            ivar = np.ones(nwave) / (noise**2 + signal) 
            sn2 += np.sum(signal**2 * ivar)
            spectra.append( redrock.Spectrum(wave, flux, ivar, R) )

        zchi2, zcoeff = redrock.zscan.calc_zchi2(redshifts, spectra, template)

        zmin = redshifts[np.argmin(zchi2)]
        self.assertAlmostEqual(zmin, z, delta=0.005)
        
        #- Test dimensions of template_fit return
        #- Too low S/N to reproducibly check afit values
        # afit, Tfit = redrock.zscan.template_fit(zmin, spectra, template)
        # self.assertEqual(len(afit), 2)
        # self.assertEqual(len(Tfit), 5)
        # for i in range(5):
        #     self.assertEqual(Tfit[i].shape, (len(spectra[i]['flux']), 2))
            
        #- Also test fitz since we are here
        zbest, zerr, zwarn, chi2min, deltachi2 = redrock.fitz.fitz(zchi2, redshifts, spectra, template)
        self.assertAlmostEqual(zbest, z, delta=0.01)
        self.assertLess(zerr, 0.01)
        self.assertEqual(zwarn, 0)
        self.assertGreater(deltachi2, 0)
        
        #- Test zwarning
        zchi2[0] = 0
        zbest, zerr, zwarn, chi2min, deltachi2 = redrock.fitz.fitz(zchi2, redshifts, spectra, template)
        self.assertNotEqual(zwarn, 0)
        
                
if __name__ == '__main__':
    unittest.main()
