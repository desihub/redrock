"""
Test redrock.archetypes

TODO: expand tests
"""

import unittest

class TestArchetypes(unittest.TestCase):

    def test_split_archetype_coeff(self):
        """test archetypes.split_archetype_coeff"""
        from redrock.archetypes import split_archetype_coeff

        # for archetypes, the coeff array is N archetype coefficients,
        # followed by nbands * nleg Legendre coefficients

        #- 1 archetype, 3 bands, 1 legendre per band
        ac, lc = split_archetype_coeff('ELG_12', [1,2,3,4], nbands=3)
        self.assertEqual(ac, [1,])
        self.assertEqual(lc, [[2,], [3,], [4,]])

        #- 1 archetype, 3 bands, 2 legendre per band
        ac, lc = split_archetype_coeff('ELG_12', [1,2,3,4,5,6,7], nbands=3)
        self.assertEqual(ac, [1,])
        self.assertEqual(lc, [[2,3], [4,5], [6,7]])

        #- 1 archetype, 3 bands, 2 legendre per band, with trailing zeros
        ac, lc = split_archetype_coeff('ELG_12', [1,2,3,4,5,6,7,0,0,0], nbands=3)
        self.assertEqual(ac, [1,])
        self.assertEqual(lc, [[2,3], [4,5], [6,7]])

        #- 2 archetypes, 3 bands, 2 legendre per band, with trailing zeros
        ac, lc = split_archetype_coeff('ELG_12;LRG_5', [1,0,2,3,4,5,6,7,0,0,0], nbands=3)
        self.assertEqual(ac, [1,0])
        self.assertEqual(lc, [[2,3], [4,5], [6,7]])

        #- no legendre terms
        ac, lc = split_archetype_coeff('ELG_12;LRG_5', [1,0,], nbands=3)
        self.assertEqual(ac, [1,0])
        self.assertEqual(lc, [list(), list(), list()])


