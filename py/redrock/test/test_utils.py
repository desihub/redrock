import unittest
import numpy as np
cp_available = False
try:
    import cupy as cp
    cp_available = cp.is_available() 
except Exception:
    cp_available = False

from .. import utils
from .. import igm

class TestUtils(unittest.TestCase):
    """Test redrock utils.
    """

    def setUp(self):
        try:
            d = cp.cuda.Device()
            cp_available = True
        except Exception:
            cp_available = False

    def tearDown(self):
        pass

    def test_distribute_work(self):
        nproc = 2
        ids = [0, 1, 2, 3]
        dist = utils.distribute_work(nproc, ids)
        self.assertEqual(set(map(len, dist)), {2, 2})

        weights = [3, 1, 1, 1]
        dist = utils.distribute_work(nproc, ids, weights=weights)
        self.assertEqual(set(map(len, dist)), {1, 3})

        capacities = [1, 3]
        dist = utils.distribute_work(nproc, ids, capacities=capacities)
        self.assertEqual(list(map(len, dist)), [1, 3])

    def test_Lyman_transmission_batch(self):
        '''Test the 2D version of Lyman transmission in batch versus
        the legacy 1D mode'''
        x = np.arange(300)+800.
        myz = np.arange(11)*0.1

        c = igm.transmission_Lyman(myz,x)
        self.assertTrue(type(c) == np.ndarray)
        for i in range(len(myz)):
            a = igm.transmission_Lyman(myz[i], x)
            self.assertTrue(np.allclose(c[i], a))
            self.assertTrue(type(a) == np.ndarray)

    @unittest.skipUnless(cp_available, "Skipping test_Lyman_transmission_GPU because no GPU found.")
    def test_Lyman_transmission_GPU(self):
        '''Test the GPU version of Lyman transmission in batch versus
        both the 2D mode and the legacy 1D mode'''
        x = np.arange(300)+800.
        myz = np.arange(11)*0.1

        g = igm.transmission_Lyman(myz,x,use_gpu=True)
        self.assertTrue(type(g) == cp.ndarray)
        c = igm.transmission_Lyman(myz,x)
        self.assertTrue(type(c) == np.ndarray)
        self.assertTrue(np.allclose(c, g))
        for i in range(len(myz)):
            a = igm.transmission_Lyman(myz[i], x)
            self.assertTrue(np.allclose(g[i], a))
            self.assertTrue(type(a) == np.ndarray)


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
