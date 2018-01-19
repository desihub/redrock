from __future__ import division, print_function

import os
import unittest
from uuid import uuid1
import numpy as np

from .. import utils as rrutils
from ..results import read_zscan, write_zscan
from ..targets import DistTargetsCopy
from ..templates import DistTemplate, find_templates, load_dist_templates
from ..zfind import zfind

from . import util


class TestIO(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testfile = 'test-{uuid}.h5'.format(uuid=uuid1())

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.testfile):
            os.remove(cls.testfile)

    def setUp(self):
        pass

    def test_endian(self):
        x1 = np.arange(5, dtype='>f')
        x2 = np.arange(5, dtype='<f')
        self.assertTrue(rrutils.native_endian(x1).dtype.isnative)
        self.assertTrue(rrutils.native_endian(x2).dtype.isnative)
        if x1.dtype.isnative:
            self.assertTrue(x1 is rrutils.native_endian(x1))
        else:
            self.assertTrue(x2 is rrutils.native_endian(x2))

    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_find_templates(self):
        templates = find_templates()
        self.assertTrue(len(templates) > 0)
        template_dir = os.path.dirname(templates[0])
        templates = find_templates(template_dir = template_dir)
        self.assertTrue(len(templates) > 0)

    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_read_templates(self):
        dtarg = util.fake_targets()
        dwave = dtarg.wavegrids()
        for dtp in load_dist_templates(dwave):
            self.assertIn('wave', dtp.template.__dict__)
            self.assertIn('flux', dtp.template.__dict__)
            wave = dtp.template.wave
            flux = dtp.template.flux
            self.assertEqual(wave.shape[0], flux.shape[1])
            self.assertEqual(wave.ndim, 1)
            self.assertEqual(flux.ndim, 2)

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


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
