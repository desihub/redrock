from __future__ import division, print_function

import os, tempfile
from shutil import rmtree
import unittest
from uuid import uuid1
import numpy as np

from .. import utils as rrutils
from ..results import read_zscan, write_zscan
from ..templates import DistTemplate, find_templates, load_templates, load_dist_templates
from ..zfind import zfind

from . import util


class TestIO(unittest.TestCase):

    #- Create unique test filename in a subdirectory
    @classmethod
    def setUpClass(cls):
        cls.testDir = tempfile.mkdtemp()        
        cls.testfile = os.path.join(cls.testDir, 'test-{uuid}.h5'.format(uuid=uuid1()))

        # create dummy template files in a separate dir
        cls.testTemplateDir = tempfile.mkdtemp()
        cls.default_templates = (
                'rrtemplate-galaxy-v2.fits',
                'rrtemplate-qso-v1.fits',
                )
        cls.alternate_templates = (
                'rrtemplate-galaxy-v1.fits',
                'rrtemplate-blat-v1.fits',
                )

        for filename in cls.default_templates + cls.alternate_templates:
            with open(f'{cls.testTemplateDir}/{filename}', 'w'):
                pass

        with open(cls.testTemplateDir + '/default_templates.txt', 'w') as fx:
            fx.write('# Header comment\n')
            fx.write('\n')
            for filename in cls.default_templates:
                fx.write(f'{filename}\n')

        with open(cls.testTemplateDir + '/alternate_templates.txt', 'w') as fx:
            fx.write('# Header comment\n')
            fx.write('\n')
            for filename in cls.alternate_templates:
                fx.write(f'{filename}\n')

    #- Cleanup test files if they exist
    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.testfile):
            os.remove(cls.testfile)

        if os.path.isdir(cls.testDir):
            rmtree(cls.testDir)

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

    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_find_templates(self):
        templates = find_templates()
        self.assertTrue(len(templates) > 0)
        template_dir = os.path.dirname(templates[0])
        templates = find_templates(template_dir)
        self.assertTrue(len(templates) > 0)

        template_dir = os.path.dirname(templates[0])
        templates = find_templates(template_dir)
        self.assertTrue(len(templates) > 0)

        templates = find_templates(self.testTemplateDir)
        self.assertEqual(len(templates), len(self.default_templates))
        for filename in templates:
            self.assertIn(os.path.basename(filename), self.default_templates)

        templates = find_templates(self.testTemplateDir+'/alternate_templates.txt')
        self.assertEqual(len(templates), len(self.alternate_templates))
        for filename in templates:
            self.assertIn(os.path.basename(filename), self.alternate_templates)

        with self.assertRaises(ValueError):
            templates = find_templates(self.testTemplateDir+'/blat.txt')

        with self.assertRaises(ValueError):
            templates = find_templates('blat.txt')

        with self.assertRaises(ValueError):
            templates = find_templates('blat.foo')


    def test_load_templates(self):
        templates = load_templates()
        self.assertTrue(len(templates) > 0)

        template_files = find_templates()
        templates = load_templates(template_files[0:2])
        self.assertEqual(len(templates), 2)


    ### @unittest.skipIf('RR_TEMPLATE_DIR' not in os.environ, '$RR_TEMPLATE_DIR not set')
    def test_load_dist_templates(self):
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


