import unittest
from io import BytesIO

import numpy as np
from astropy.table import Table

from ..zfind import zfind, sort_zfit, calc_deltachi2
from .. zwarning import ZWarningMask as ZW

class TestZFind(unittest.TestCase):
    """Test redrock.zfind
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sort_zfit(self):

        #- rank by chi2 since zwarn=0 for everyone
        zfit = Table.read(BytesIO(b"""
                id zwarn chi2 rank
                1  0     30   2
                2  0     20   1
                3  0     10   0
                """), format='ascii')
        sort_zfit(zfit)
        self.assertEqual(zfit.colnames, ['id', 'zwarn', 'chi2', 'rank'])
        self.assertEqual(list(zfit['rank']), list(range(len(zfit))))

        #- zwarn Z_FITLIMIT=32 moves id=3 to the end, even with good chi2
        zfit = Table.read(BytesIO(b"""
                id zwarn chi2 rank
                1  0     30   1
                2  0     20   0
                3  32    10   2
                """), format='ascii')
        sort_zfit(zfit)
        self.assertEqual(list(zfit['rank']), list(range(len(zfit))))

        #- also test other specific bad bits
        for bad in (ZW.NEGATIVE_MODEL,
                    ZW.MANY_OUTLIERS,
                    ZW.Z_FITLIMIT,
                    ZW.NEGATIVE_EMISSION,
                    ZW.BAD_MINFIT):
            zfit.sort('id')
            zfit['zwarn'][2] = bad
            sort_zfit(zfit)
            self.assertEqual(list(zfit['rank']), list(range(len(zfit))))

        #- zwarn BAD_MINFIT=1024 moves id=3 to the end, even with good chi2
        zfit = Table.read(BytesIO(b"""
                id zwarn chi2 rank
                1  0     30   1
                2  0     20   0
                3  1024  10   2
                """), format='ascii')
        sort_zfit(zfit)
        self.assertEqual(list(zfit['rank']), list(range(len(zfit))))

        #- zwarn BAD_TARGET=256 is ok (not about this single fit)
        zfit = Table.read(BytesIO(b"""
                id zwarn chi2 rank
                1  0     30   2
                2  0     20   1
                3  256   10   0
                """), format='ascii')
        sort_zfit(zfit)
        self.assertEqual(list(zfit['rank']), list(range(len(zfit))))

        #- test mix of zwarn bits
        zfit = Table.read(BytesIO(b"""
                id zwarn chi2 rank
                1  0     30   1
                2  256   20   0
                3  288   10   2
                """), format='ascii')
        sort_zfit(zfit)
        self.assertEqual(list(zfit['rank']), list(range(len(zfit))))

    def test_calc_deltachi2(self):

        #- Well separated redshifts, but some close chi2
        chi2  = np.array([1.0, 2.0, 20.0, 40.0])
        z     = np.array([3.0, 3.1,  3.2,  3.3])
        zwarn = np.array([0,   0,    0,    0])
        dchi2, setzwarn = calc_deltachi2(chi2, z, zwarn)
        self.assertTrue(np.all(dchi2 == np.array([1, 18, 20, 0.0])), dchi2)
        #- Note: setzwarn[1] is True even though deltachi2[1]>9 because it
        #- is close to the next *better* solution, not just the next worse
        self.assertTrue(np.all(setzwarn == np.array([True,True,False,False])))

        #- Fits at similar redshifts count as the same solution
        chi2  = np.array([1.0, 2.0, 20.0, 40.0])
        z     = np.array([3.0, 3.0001, 4.0, 4.0001])
        zwarn = np.array([0,   0,    0,    0])
        dchi2, setzwarn = calc_deltachi2(chi2, z, zwarn)
        self.assertTrue(np.all(dchi2 == np.array([19, 18, 0.0, 0.0])), dchi2)
        self.assertTrue(np.all(setzwarn == np.array([False,False,False,False])))

        #- deltachi2 and warning based on next good fit; ignore zwarn=32 entry
        chi2  = np.array([1.0, 2.0, 20.0, 40.0])
        z     = np.array([3.0, 3.1,  3.2,  3.3])
        zwarn = np.array([0,   32,   0,    0])
        dchi2, setzwarn = calc_deltachi2(chi2, z, zwarn)
        self.assertTrue(np.all(dchi2 == np.array([19, 18, 20, 0.0])), dchi2)
        self.assertTrue(np.all(setzwarn == np.array([False,True,False,False])))

def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m redrock.test.test_zfind
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
