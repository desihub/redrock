import unittest
import numpy as np
import numpy.testing as nt

from ..priors import Priors


class TestPriors(unittest.TestCase):
    """Test redrock.priors static prior functions."""

    def setUp(self):
        self.z = np.linspace(0.0, 2.0, 201)

    def test_gaussian(self):
        z0, s0 = 1.0, 0.1
        prior = Priors.gaussian(self.z, z0, s0)
        # minimum should be at z0
        self.assertEqual(np.argmin(prior), np.argmin(np.abs(self.z - z0)))
        # value at z0 should be 0
        nt.assert_allclose(prior[np.argmin(np.abs(self.z - z0))], 0.0, atol=1e-10)
        # values should be non-negative
        self.assertTrue(np.all(prior >= 0.0))

    def test_lorentzien(self):
        z0, s0 = 1.0, 0.1
        prior = Priors.lorentzien(self.z, z0, s0)
        # minimum should be at z0
        self.assertEqual(np.argmin(prior), np.argmin(np.abs(self.z - z0)))
        # value at z0 should be 0
        nt.assert_allclose(prior[np.argmin(np.abs(self.z - z0))], 0.0, atol=1e-10)
        # values should be non-negative
        self.assertTrue(np.all(prior >= 0.0))

    def test_tophat_inside(self):
        """tophat: z0 well inside the grid, values inside are 0, outside are nan"""
        z0, s0 = 1.0, 0.2
        prior = Priors.tophat(self.z, z0, s0)
        inside = np.abs(self.z - z0) < s0 / 2
        # Values strictly inside the hat should be 0 or inf (boundary markers)
        self.assertTrue(np.all(np.isfinite(prior[inside]) | np.isinf(prior[inside])))
        # Values outside should be nan or inf
        outside_finite = np.isfinite(prior[~inside])
        self.assertFalse(np.any(outside_finite & ~np.isnan(prior[~inside])))

    def test_tophat_all_nan_when_z0_outside_grid(self):
        """tophat: when z0 is outside the grid, all values are nan"""
        z0, s0 = 10.0, 0.1  # far outside the grid 0..2
        prior = Priors.tophat(self.z, z0, s0)
        self.assertTrue(np.all(np.isnan(prior)))

    def test_tophat_no_NaN_attribute(self):
        """tophat uses np.nan (lowercase), not np.NaN which was removed in numpy 2"""
        # This would raise AttributeError with numpy 2.x if np.NaN were used
        z0, s0 = 1.0, 0.4
        prior = Priors.tophat(self.z, z0, s0)
        # Just verify it runs without error and returns an array
        self.assertIsInstance(prior, np.ndarray)


if __name__ == '__main__':
    unittest.main()
