"""
Test redrock.archetypes

TODO: expand tests
"""

import unittest
import os
import importlib
import numpy as np
from scipy.interpolate import interp1d
from . import util
from ..archetypes import Archetype
from ..zscan import spectral_data
from ..fitz import prior_on_coeffs

class TestArchetypes(unittest.TestCase):

    def setUp(self):
        self.filename = importlib.resources.files('redrock.test').joinpath('data/rrarchetype-galaxy-test.fits')
        # self.filename = os.path.abspath(
        #     os.path.join(os.path.dirname(__file__), '..', 'data', 'rrarchetype-galaxy.fits')
        # )
        self.archetypes = Archetype(self.filename)  # moved here

    def test_archetype_load(self):
        self.assertIsNotNone(self.archetypes)  # now safe to access here


    def test_split_archetype_coeff(self):
        """test archetypes.split_archetype_coeff"""
        from redrock.archetypes import split_archetype_coeff

        # for archetypes, the coeff array is N archetype coefficients,
        # followed by nbands * nleg Legendre coefficients

        #- 1 archetype, 3 bands, 1 legendre per band
        ac, lc = split_archetype_coeff('ELG_12', [1,2,3,4], nbands=3, nleg=1)
        self.assertEqual(ac, [1,])
        self.assertEqual(lc, [[2,], [3,], [4,]])

        #- 1 archetype, 3 bands, 2 legendre per band
        ac, lc = split_archetype_coeff('ELG_12', [1,2,3,4,5,6,7], nbands=3, nleg=2)
        self.assertEqual(ac, [1,])
        self.assertEqual(lc, [[2,3], [4,5], [6,7]])

        #- 1 archetype, 3 bands, 2 legendre per band, with trailing zeros
        ac, lc = split_archetype_coeff('ELG_12', [1,2,3,4,5,6,7,0,0,0], nbands=3, nleg=2)
        self.assertEqual(ac, [1,])
        self.assertEqual(lc, [[2,3], [4,5], [6,7]])

        #- 2 archetypes, 3 bands, 2 legendre per band, with trailing zeros
        ac, lc = split_archetype_coeff('ELG_12;LRG_5', [1,0,2,3,4,5,6,7,0,0,0], nbands=3, nleg=2)
        self.assertEqual(ac, [1,0])
        self.assertEqual(lc, [[2,3], [4,5], [6,7]])

        #- no legendre terms
        ac, lc = split_archetype_coeff('ELG_12;LRG_5', [1,0,], nbands=3, nleg=0)
        self.assertEqual(ac, [1,0])
        self.assertEqual(lc, [list(), list(), list()])

    def test_archetype_with_legendre(self):
        """Test archetype method with legendre and prior terms"""
        self.target = util.get_target()
        spectra = self.target.spectra
        self.target.nleg = 2
        # Build dictionary of wavelength grids
        dwave = { s.wavehash:s.wave for s in spectra }
        self.target.bands = ['b', 'r']
        (weights, flux, wflux) = spectral_data(self.target.spectra)
        prior = prior_on_coeffs(1, self.target.nleg, 0.1, len(dwave.keys()))

        # per camera mode
        chi2, coeff, fulltype = self.archetypes.get_best_archetype(self.target, weights, flux, wflux, dwave, z=0.5, per_camera=True, n_nearest=None, trans=None, solve_method='bvls', prior=prior, use_gpu=False)
        assert len(coeff) == 1 + self.target.nleg * len(self.target.bands)

        # without per camera mode
        chi2, coeff, fulltype = self.archetypes.get_best_archetype(self.target, weights, flux, wflux, dwave, z=0.5, per_camera=False, n_nearest=None, trans=None, solve_method='bvls', prior=prior, use_gpu=False)
        assert len(coeff) == 1 + self.target.nleg

    def test_archetype_without_legendre(self):
        """Test archetype method without legendre and prior terms"""
        self.target = util.get_target()
        spectra = self.target.spectra
        self.target.nleg = 0
        # Build dictionary of wavelength grids
        dwave = { s.wavehash:s.wave for s in spectra }
        self.target.bands = ['b', 'r']
        (weights, flux, wflux) = spectral_data(self.target.spectra)
        prior = None
        chi2, coeff, fulltype = self.archetypes.get_best_archetype(self.target, weights, flux, wflux, dwave, z=0.5, per_camera=False, n_nearest=None, trans=None, solve_method='bvls', prior=prior, use_gpu=False)
        assert len(coeff) == 1 # single archetype






