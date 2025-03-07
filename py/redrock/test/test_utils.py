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

    def test_reduced_wavelength(self):
        x = utils.reduced_wavelength(np.arange(10))
        self.assertEqual(x[0], -1.0)
        self.assertEqual(x[-1], 1.0)
        x = utils.reduced_wavelength(np.linspace(3600, 5800, 10))
        self.assertEqual(x[0], -1.0)
        self.assertEqual(x[-1], 1.0)

        #- even out-of-order non-linear ok
        x = utils.reduced_wavelength(np.random.uniform(-5, 20, size=100))
        self.assertEqual(np.min(x), -1.0)
        self.assertEqual(np.max(x), 1.0)

        #- also works on cupy if installed,
        #- and answer remains on GPU as a cupy array
        if cp_available:
            x = utils.reduced_wavelength(cp.linspace(3600, 5800, 10))
            self.assertEqual(x[0], -1.0)
            self.assertEqual(x[-1], 1.0)
            self.assertTrue(isinstance(x, cp.ndarray))

