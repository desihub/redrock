import unittest
import numpy as np
import scipy.sparse

import redrock
import redrock.test.util

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
        z1 = 0.2
        z2 = 0.5
        t1 = redrock.test.util.get_target(z1); t1.id = 111
        t2 = redrock.test.util.get_target(z2); t2.id = 222
        template = redrock.test.util.get_template()
        zscan, zfit = redrock.zfind([t1,t2], [template,], ncpu=1)
        
        self.assertAlmostEqual(zfit['z'][0], z1, delta=0.005)
        
        #- Test dimensions of template_fit return
        #- Too low S/N to reproducibly check afit values
        # afit, Tfit = redrock.zscan.template_fit(zmin, spectra, template)
        # self.assertEqual(len(afit), 2)
        # self.assertEqual(len(Tfit), 5)
        # for i in range(5):
        #     self.assertEqual(Tfit[i].shape, (len(spectra[i]['flux']), 2))
            
        #- Also test fitz since we are here
        # results = redrock.fitz.fitz(zchi2, redshifts, spectra, template)
        # self.assertGreater(len(results), 1)
        # self.assertAlmostEqual(results[0]['z'], z, delta=0.01)
        # self.assertLess(results[0]['zerr'], 0.01)
        # self.assertEqual(results[0]['zwarn'], 0)
        #
        # #- Test zwarning
        # zchi2[0] = 0
        # results = redrock.fitz.fitz(zchi2, redshifts, spectra, template)
        # self.assertNotEqual(results[0]['zwarn'], 0)
        
                
if __name__ == '__main__':
    unittest.main()
