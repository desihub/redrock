from __future__ import division, print_function

import os, tempfile
from shutil import rmtree
import unittest
from uuid import uuid1
import numpy as np

from .. import utils as rrutils
from ..results import read_zscan, write_zscan, read_zfit
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

        zfit1.remove_column('fitmethod')

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

        #- targetid subsets
        targetids = np.unique(zfit['targetid'])

        zscan, zfit = read_zscan(zscanfile, select_targetids=targetids[1])
        self.assertEqual(len(np.unique(zfit['targetid'])), 1)
        self.assertEqual(np.unique(zfit['targetid'])[0], targetids[1])

        zscan, zfit = read_zscan(zscanfile, select_targetids=targetids[0:2])
        self.assertEqual(len(np.unique(zfit['targetid'])), 2)
        self.assertTrue(np.all(np.unique(zfit['targetid']) == targetids[0:2]))

        #- UPPERCASE column names
        zscan, zfit = read_zscan(zscanfile, upper=True)
        self.assertIn('TARGETID', zfit.colnames)

        #- only zscan; not (zscan,zfit)
        results = read_zscan(zscanfile, nozfit=True)
        self.assertTrue(isinstance(results, dict))  #- not tuple
        self.assertIn(targetids[0], results.keys())

    def test_read_zfit(self):
        """Test read_zfit with pre-generated data"""
        import importlib
        filename = importlib.resources.files('redrock.test').joinpath('data/rrdetails-test.h5')
        zfit = read_zfit(filename)
        targetids = np.unique(zfit['targetid'])
        self.assertEqual(len(targetids), 3)

        #- Select two targetids
        zfit2 = read_zfit(filename, select_targetids=targetids[0:2])
        targetids2 = np.unique(zfit2['targetid'])
        self.assertEqual(len(targetids2), 2)
        self.assertTrue(np.all(targetids2 == targetids[0:2]))

        #- Select a single targetid as integer
        zfit1 = read_zfit(filename, select_targetids=targetids[1])
        targetids1 = np.unique(zfit1['targetid'])
        self.assertEqual(len(targetids1), 1)
        self.assertEqual(targetids1[0], targetids[1])

        #- force column names to UPPERCASE
        zfit = read_zfit(filename, upper=True)
        self.assertIn('TARGETID', zfit.colnames)
        self.assertIn('Z', zfit.colnames)
        self.assertNotIn('z', zfit.colnames)


