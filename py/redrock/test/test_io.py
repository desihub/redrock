from __future__ import division, print_function

import os
import unittest
from uuid import uuid1
import numpy as np

from .. import io as rrio
from .. import zfind as rrzfind
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
        self.assertTrue(rrio.native_endian(x1).dtype.isnative)
        self.assertTrue(rrio.native_endian(x2).dtype.isnative)
        if x1.dtype.isnative:
            self.assertTrue(x1 is rrio.native_endian(x1))
        else:
            self.assertTrue(x2 is rrio.native_endian(x2))
                
    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_find_templates(self):
        templates = rrio.find_templates()
        self.assertTrue(len(templates) > 0)
        template_dir = os.path.dirname(templates[0])
        templates = rrio.find_templates(template_dir = template_dir)
        self.assertTrue(len(templates) > 0)
        
    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_read_templates(self):
        for template in rrio.read_templates().values():
            self.assertIn('wave', template.__dict__)
            self.assertIn('flux', template.__dict__)
            self.assertIn('type', template.__dict__)
            wave = template.wave
            flux = template.flux
            self.assertEqual(wave.shape[0], flux.shape[1])
            self.assertEqual(wave.ndim, 1)
            self.assertEqual(flux.ndim, 2)
        
    def test_zscan_io(self):
        t1 = util.get_target(0.2)
        t1.id = 111
        t2 = util.get_target(0.5)
        t2.id = 222
        template = util.get_template(subtype='BLAT')
        zscan1, zfit1 = rrzfind([t1,t2], {template.fulltype:template}, ncpu=1)

        rrio.write_zscan(self.testfile, zscan1, zfit1)
        rrio.write_zscan(self.testfile, zscan1, zfit1, clobber=True)
        zscan2, zfit2 = rrio.read_zscan(self.testfile)

        self.assertEqual(zfit1.colnames, zfit2.colnames)
        for cn in zfit1.colnames:
            np.testing.assert_equal(zfit1[cn], zfit2[cn])

        for targetid in zscan1:
            for spectype in zscan1[targetid]:
                for key in zscan1[targetid][spectype]:
                    d1 = zscan1[targetid][spectype][key]
                    d2 = zscan2[targetid][spectype][key]
                    self.assertTrue(np.all(d1==d2), 'data mismatch {}/{}/{}'.format(targetid, spectype, key))
                
if __name__ == '__main__':
    unittest.main()
