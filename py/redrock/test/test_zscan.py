import unittest
import numpy as np
import scipy.sparse

from .. import zfind as rrzfind
from .. import zscan as rrzscan
from . import util

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
        z2 = 0.25
        seed = np.random.randint(2**31)
        print('TEST: Using random seed {}'.format(seed))
        np.random.seed(seed)
        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        template = util.get_template()
        template.redshifts = np.linspace(0.15, 0.3, 50)
        zscan, zfit = rrzfind([t1,t2], [template,], ncpu=1)

        zx1 = zfit[zfit['targetid'] == 111][0]
        zx2 = zfit[zfit['targetid'] == 222][0]
        self.assertLess(np.abs(zx1['z'] - z1)/zx1['zerr'], 5)
        self.assertLess(np.abs(zx2['z'] - z2)/zx2['zerr'], 5)
        self.assertLess(zx1['zerr'], 0.002)
        self.assertLess(zx2['zerr'], 0.002)
        
        #- Test dimensions of template_fit return
        zmin = zx1['z']
        fitflux, fitcoeff = rrzscan.template_fit(t1.spectra, zmin, template)

        self.assertEqual(len(fitcoeff), template.nbasis)
        self.assertEqual(len(fitflux), len(t1.spectra))
        for i in range(len(fitflux)):
            self.assertEqual(len(fitflux[i]), len(t1.spectra[i].flux))
    
    def test_parallel_zscan(self):
        z1 = 0.2
        z2 = 0.25
        seed = np.random.randint(2**31)
        print('TEST: Using random seed {}'.format(seed))
        np.random.seed(seed)
        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        template = util.get_template()
        redshifts = np.linspace(0.15, 0.3, 25)
        zchi2a, zcoeffa, penaltya = rrzscan.calc_zchi2_targets(redshifts, [t1,t2], template)
        zchi2b, zcoeffb, penaltyb = rrzscan.parallel_calc_zchi2_targets(redshifts, [t1,t2], template, ncpu=2)
        
        self.assertEqual(zchi2a.shape, zchi2b.shape)
        self.assertEqual(zcoeffa.shape, zcoeffb.shape)
        self.assertTrue(np.all(zchi2a == zchi2b))
        self.assertTrue(np.all(zcoeffa == zcoeffb))

    def test_subtype(self):
        z1 = 0.0
        z2 = 1e-4
        seed = np.random.randint(2**31)
        print('TEST: Using random seed {}'.format(seed))
        np.random.seed(seed)
        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        Fstar = util.get_template(spectype='STAR', subtype='F')
        Mstar = util.get_template(spectype='STAR', subtype='M')
        Fstar.redshifts = np.linspace(-1e-3, 1e-3, 25)
        Fstar.redshifts = np.linspace(-1e-3, 1e-3, 25)
        nminima = 3
        zscan, zfit = rrzfind([t1,t2], [Fstar, Mstar,], ncpu=1, 
            nminima=nminima)
        self.assertEqual(len(zfit), 2*nminima)
        self.assertTrue(np.all(zfit['spectype'] == 'STAR'))

if __name__ == '__main__':
    unittest.main()
