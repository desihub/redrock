import os
import unittest
import numpy as np
cp_available = False
try:
    import cupy as cp
    cp_available = True
except Exception:
    cp_available = False

from .. import rebin

class TestRebin(unittest.TestCase):

    def setUp(self):
        #- Supposed to turn off numba.jit of _trapz_rebin, but coverage
        #- still doesn't see the function.  Leaving this here anyway.
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        try:
            d = cp.cuda.Device()
            cp_available = True
        except Exception:
            cp_available = False

    def tearDown(self):
        del os.environ['NUMBA_DISABLE_JIT']

    def test_centers2edges(self):
        c2e = rebin.centers2edges  #- shorthand
        self.assertTrue(np.allclose(c2e([1,2,3]), [0.5, 1.5, 2.5, 3.5]))
        self.assertTrue(np.allclose(c2e([1,3,5]), [0, 2, 4, 6]))
        self.assertTrue(np.allclose(c2e([1,3,4]), [0, 2, 3.5, 4.5]))

    def test_trapzrebin(self):
        '''Test constant flux density at various binnings'''
        nx = 10
        x = np.arange(nx)*1.1
        y = np.ones(nx)

        #- test various binnings
        for nedge in range(3,10):
            edges = np.linspace(min(x), max(x), nedge)
            yy = rebin.trapz_rebin(x, y, edges=edges)
            self.assertTrue(np.all(yy == 1.0), msg=str(yy))

        #- edges starting/stopping in the interior
        summ = rebin.trapz_rebin(x, y, edges=[0.5, 8.3])[0]
        for nedge in range(3, 3*nx):
            edges = np.linspace(0.5, 8.3, nedge)
            yy = rebin.trapz_rebin(x, y, edges=edges)
            self.assertTrue(np.allclose(yy, 1.0), msg=str(yy))

    def test_centers(self):
        '''Test with centers instead of edges'''
        nx = 10
        x = np.arange(nx)
        y = np.ones(nx)
        xx = np.linspace(0.5, nx-1.5)
        yy = rebin.trapz_rebin(x, y, xx)
        self.assertTrue(np.allclose(yy, 1.0), msg=str(yy))

    def test_error(self):
        '''Test that using edges outside of x range raises ValueError'''
        nx = 10
        x = np.arange(nx)
        y = np.ones(nx)
        with self.assertRaises(ValueError):
            yy = rebin.trapz_rebin(x, y, edges=np.arange(-1, nx-1))
        with self.assertRaises(ValueError):
            yy = rebin.trapz_rebin(x, y, edges=np.arange(1, nx+1))

    def test_nonuniform(self):
        '''test rebinning a non-uniform density'''
        for nx in range(5,12):
            x = np.linspace(0, 2*np.pi, nx)
            y = np.sin(x)
            edges = [0, 2*np.pi]
            yy = rebin.trapz_rebin(x, y, edges=edges)
            self.assertTrue(np.allclose(yy, 0.0))

        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        edges = [0, 0.5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]
        yy = rebin.trapz_rebin(x, y, edges=edges)
        self.assertTrue(np.allclose(yy[0:2], 2/np.pi, atol=5e-4))
        self.assertTrue(np.allclose(yy[2:4], -2/np.pi, atol=5e-4))

    def test_gpu_trapzrebin(self):
        '''Test that GPU version matches CPU for constant flux density at various binnings'''
        if (not cp_available):
            self.assertTrue(True)
            return
        nx = 10
        x = np.arange(nx)*1.1
        y = np.ones(nx)

        #- test various binnings
        for nedge in range(3,10):
            edges = np.linspace(min(x), max(x), nedge)
            yy = rebin.trapz_rebin(x, y, edges=edges, use_gpu=True)
            self.assertTrue(np.all(yy == 1.0), msg=str(yy))
            self.assertTrue(type(yy) == cp.ndarray)

        #- edges starting/stopping in the interior
        summ = rebin.trapz_rebin(x, y, edges=[0.5, 8.3], use_gpu=True)[0]
        self.assertTrue(type(summ) == cp.ndarray)
        for nedge in range(3, 3*nx):
            edges = np.linspace(0.5, 8.3, nedge)
            yy = rebin.trapz_rebin(x, y, edges=edges, use_gpu=True)
            self.assertTrue(np.allclose(yy, 1.0), msg=str(yy))
            self.assertTrue(type(yy) == cp.ndarray)

    def test_gpu_trapzrebin_uneven(self):
        '''test rebinning unevenly spaced x for GPU vs CPU'''
        if (not cp_available):
            self.assertTrue(True)
            return
        x = np.linspace(0, 10, 100)**2
        y = np.sqrt(x)
        edges = np.linspace(0,100,21)
        c = rebin.trapz_rebin(x, y, edges=edges)
        g = rebin.trapz_rebin(x, y, edges=edges, use_gpu=True)
        self.assertTrue(np.allclose(c,g))
        self.assertTrue(type(g) == cp.ndarray)
        self.assertTrue(type(c) == np.ndarray)

    def test_gpu_trapzrebin_multidimensional(self):
        '''Test that batch CPU and GPU matches CPU 1d version'''
        if (not cp_available):
            self.assertTrue(True)
            return
        #First test multiple bases, no redshifts
        x = np.arange(10)
        y = np.ones((2,10)) #nbasis = 2
        y[1,:] = np.arange(10)
        g = rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], use_gpu=True) #nbasis=2, GPU mode
        cbatch = rebin.trapz_rebin(x, y, edges=[0,2,4,6,8]) #nbasis = 2, CPU batch mode
        self.assertTrue(type(g) == cp.ndarray)
        self.assertTrue(type(cbatch) == np.ndarray)
        self.assertTrue(np.allclose(g, cbatch))
        self.assertTrue(cbatch.shape == (4,2))
        for j in range(2):
            c = rebin.trapz_rebin(x, y[j,:], edges=[0,2,4,6,8]) #CPU mode
            self.assertTrue(np.allclose(g[:,j], c))
            self.assertTrue(type(c) == np.ndarray)
            self.assertTrue(c.shape == (4,))

        #Now test 3d mode with multiple redshifts
        myz = np.linspace(0, 1, 11)
        g = rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], myz=myz,use_gpu=True) #nbasis=2, multiple redshifts, GPU mode
        cbatch = rebin.trapz_rebin(x, y, edges=[0,2,4,6,8], myz=myz) #nbasis = 2, multiple redshifts, CPU batch mode
        self.assertTrue(type(g) == cp.ndarray)
        self.assertTrue(type(cbatch) == np.ndarray)
        self.assertTrue(np.allclose(g, cbatch))
        self.assertTrue(cbatch.shape == (11,4,2))
        for z in range(len(myz)):
            for j in range(2):
                c = rebin.trapz_rebin(x*(1+myz[z]), y[j,:], edges=[0,2,4,6,8]) #CPU mode
                self.assertTrue(np.allclose(g[z,:,j], c))
                self.assertTrue(type(c) == np.ndarray)
                self.assertTrue(c.shape == (4,))

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
