from __future__ import division, print_function

import os
import unittest
from uuid import uuid1
import numpy as np
from redrock import io

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
        self.assertTrue(io.native_endian(x1).dtype.isnative)
        self.assertTrue(io.native_endian(x2).dtype.isnative)
        if x1.dtype.isnative:
            self.assertTrue(x1 is io.native_endian(x1))
        else:
            self.assertTrue(x2 is io.native_endian(x2))
                
    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_find_templates(self):
        templates = io.find_templates()
        self.assertTrue(len(templates) > 0)
        template_dir = os.path.dirname(templates[0])
        templates = io.find_templates(template_dir = template_dir)
        self.assertTrue(len(templates) > 0)
        
    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_read_templates(self):
        for template in io.read_templates():
            self.assertIn('wave', template.__dict__)
            self.assertIn('flux', template.__dict__)
            self.assertIn('type', template.__dict__)
            wave = template.wave
            flux = template.flux
            self.assertEqual(wave.shape[0], flux.shape[1])
            self.assertEqual(wave.ndim, 1)
            self.assertEqual(flux.ndim, 2)
        
    def test_zscan_io(self):
        #- setup fake zscan results dictionary
        z1 = {123: dict(), 456: dict()}
        z1[123]['GALAXY'] = _random_zscan()
        z1[123]['STAR'] = _random_zscan()
        z1[456]['GALAXY'] = _random_zscan()
        z1[456]['STAR'] = _random_zscan()

        io.write_zscan(self.testfile, z1)
        io.write_zscan(self.testfile, z1, clobber=True)
        zbest, z2 = io.read_zscan(self.testfile)
        
        for key in ('Z', 'ZERR', 'ZWARN', 'TARGETID'):
            self.assertIn(key, zbest.dtype.names)
        
        for targetid in z1:
            for templatetype in z1[targetid]:
                for key in z1[targetid][templatetype]:
                    x1 = z1[targetid][templatetype][key]
                    x2 = z2[targetid][templatetype][key]
                    self.assertTrue(np.all(x1 == x2))
                
            
def _random_zscan(n=10):
    '''Returns random zscan result dictionary'''
    z = np.arange(n, dtype=float) + np.random.uniform()
    zchi2 = np.random.uniform(size=n)
    zbest = np.random.uniform()
    minchi2 = np.random.uniform()
    zerr = np.random.uniform()
    zwarn = np.random.randint(100)
    deltachi2 = np.random.uniform(1000)
    return dict(z=z, zchi2=zchi2, zbest=zbest, minchi2=minchi2, zerr=zerr, zwarn=zwarn, deltachi2=deltachi2)
                
if __name__ == '__main__':
    unittest.main()
