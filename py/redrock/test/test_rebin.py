import os
import unittest
import numpy as np

from redrock import rebin

class TestBlat(unittest.TestCase):
    
    def setUp(self):
        #- Supposed to turn off numba.jit of _trapz_rebin, but coverage
        #- still doesn't see the function.  Leaving this here anyway.
        os.environ['NUMBA_DISABLE_JIT'] = '1'
        pass

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
        sum = rebin.trapz_rebin(x, y, edges=[0.5, 8.3])[0]
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
        
        
if __name__ == '__main__':
    import os
    unittest.main()
