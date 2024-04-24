from __future__ import division, print_function

import os, tempfile
from shutil import rmtree
import unittest
from uuid import uuid1
import numpy as np

from .. import utils as rrutils
from ..results import read_zscan, write_zscan
from ..zfind import zfind
from ..templates import DistTemplate

from . import util

class TestIO(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testDir = tempfile.mkdtemp()        
        cls.testfile = os.path.join(cls.testDir, 'test-{uuid}.h5'.format(uuid=uuid1()))

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.testfile):
            os.remove(cls.testfile)

    def setUp(self):
        #- remove testfile if leftover from a previous test
        if os.path.exists(self.testfile):
            os.remove(self.testfile)

    def test_endian(self):
        x1 = np.arange(5, dtype='>f')
        x2 = np.arange(5, dtype='<f')
        self.assertTrue(rrutils.native_endian(x1).dtype.isnative)
        self.assertTrue(rrutils.native_endian(x2).dtype.isnative)
        if x1.dtype.isnative:
            self.assertTrue(x1 is rrutils.native_endian(x1))
        else:
            self.assertTrue(x2 is rrutils.native_endian(x2))

    def test_zscan_io(self):
        dtarg = util.fake_targets()

        # Get the dictionary of wavelength grids
        dwave = dtarg.wavegrids()

        # Construct the distributed template.
        template = util.get_template(subtype='BLAT')
        dtemp = DistTemplate(template, dwave)

        zscan1, zfit1 = zfind(dtarg, [ dtemp ])

        write_zscan(self.testfile, zscan1, zfit1)
        write_zscan(self.testfile, zscan1, zfit1, clobber=True)
        zscan2, zfit2 = read_zscan(self.testfile)

        self.assertEqual(zfit1.colnames, zfit2.colnames)
        for cn in zfit1.colnames:
            np.testing.assert_equal(zfit1[cn], zfit2[cn])

        for targetid in zscan1:
            for spectype in zscan1[targetid]:
                for key in zscan1[targetid][spectype]:
                    d1 = zscan1[targetid][spectype][key]
                    d2 = zscan2[targetid][spectype][key]
                    self.assertTrue(np.all(d1==d2), 'data mismatch {}/{}/{}'.format(targetid, spectype, key))

    def test_read_zscan(self):
        """Additional read_zscan testing with pre-generated data"""
        import importlib
        zscanfile = importlib.resources.files('redrock.test').joinpath('data/rrdetails-test.h5')
        zscan, zfit = read_zscan(zscanfile)

        for targetid in np.unique(zfit['targetid']):
            for fulltype in zscan[targetid].keys():
                zz = zscan[targetid][fulltype]
                self.assertEqual(len(zz['redshifts']), len(zz['zchi2']))
                self.assertGreater(len(zz['zfit']), 0, f'Empty zfit Table for {targetid=} {fulltype=}')



