from __future__ import division, print_function

import os, tempfile
from shutil import rmtree
import unittest
from uuid import uuid1
import numpy as np

from .. import utils as rrutils
from ..templates import DistTemplate, find_templates, load_templates, load_dist_templates

from . import util as testutil

class TestTemplates(unittest.TestCase):

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

        with open(cls.testTemplateDir + '/templates-default.txt', 'w') as fx:
            fx.write('# Header comment\n')
            fx.write('\n')
            for filename in cls.default_templates:
                fx.write(f'{filename}\n')

        with open(cls.testTemplateDir + '/templates-alternate.txt', 'w') as fx:
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

        templates = find_templates(self.testTemplateDir+'/templates-alternate.txt')
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
        dtarg = testutil.fake_targets()
        dwave = dtarg.wavegrids()
        for dtp in load_dist_templates(dwave):
            self.assertIn('wave', dtp.template.__dict__)
            self.assertIn('flux', dtp.template.__dict__)
            wave = dtp.template.wave
            flux = dtp.template.flux
            self.assertEqual(wave.shape[0], flux.shape[1])
            self.assertEqual(wave.ndim, 1)
            self.assertEqual(flux.ndim, 2)

    def test_header2templatefiles(self):
        """Test templates.header2templatefiles"""
        from ..templates import header2templatefiles

        #- Derive filenames from TEMNAM/TEMVER, with whitespace
        #- STAR version unknown defaults to 0.1
        hdr = dict(
                TEMNAM00='GALAXY  ',  TEMVER00='2.6     ',
                TEMNAM01='QSO:::HIZ', TEMVER01='1.0     ',
                TEMNAM02='STAR:::M',  TEMVER02='unknown ',
            )
        filenames = header2templatefiles(hdr)
        self.assertTrue(filenames[0].endswith('rrtemplate-GALAXY-None-v2.6.fits'), f'Wrong {filenames[0]}')
        self.assertTrue(filenames[1].endswith('rrtemplate-QSO-HIZ-v1.0.fits'),     f'Wrong {filenames[1]}')
        self.assertTrue(filenames[2].endswith('rrtemplate-STAR-M-v0.1.fits'),      f'Wrong {filenames[2]}')

        #- Derive filenames from TEMNAM/TEMVER, no whitespace
        #- star version used if not "unknown"
        hdr = dict(
                TEMNAM00='GALAXY',    TEMVER00='2.6',
                TEMNAM01='QSO:::HIZ', TEMVER01='1.0',
                TEMNAM02='STAR:::M',  TEMVER02='3.14',
            )
        filenames = header2templatefiles(hdr)
        self.assertTrue(filenames[0].endswith('rrtemplate-GALAXY-None-v2.6.fits'), f'Wrong {filenames[0]}')
        self.assertTrue(filenames[1].endswith('rrtemplate-QSO-HIZ-v1.0.fits'),     f'Wrong {filenames[1]}')
        self.assertTrue(filenames[2].endswith('rrtemplate-STAR-M-v3.14.fits'),     f'Wrong {filenames[2]}')

        #- TEMFIL overrides TEMNAM/TEMVER if it exists
        hdr = dict(
                TEMNAM00='GALAXY',    TEMVER00='2.6',  TEMFIL00='blat.fits',
                TEMNAM01='QSO:::HIZ', TEMVER01='1.0',  TEMFIL01='foo.fits',
                TEMNAM02='STAR:::M',  TEMVER02='3.14', TEMFIL02='bar.fits'
            )
        filenames = header2templatefiles(hdr)
        self.assertTrue(filenames[0].endswith('blat.fits'), f'Wrong {filenames[0]}')
        self.assertTrue(filenames[1].endswith('foo.fits'),  f'Wrong {filenames[1]}')
        self.assertTrue(filenames[2].endswith('bar.fits'),  f'Wrong {filenames[2]}')

        #- Testing future possibilities for GALAXY suptypes
        #- Also test dict entries "out of order"
        hdr = dict(
                TEMNAM00='GALAXY:::BGS',    TEMVER00='3.0',
                TEMNAM02='GALAXY:::LRG',    TEMVER02='3.0',
                TEMNAM01='GALAXY:::ELG',    TEMVER01='3.0',
            )
        filenames = header2templatefiles(hdr)
        self.assertTrue(filenames[0].endswith('rrtemplate-GALAXY-BGS-v3.0.fits'), f'Wrong {filenames[0]}')
        self.assertTrue(filenames[1].endswith('rrtemplate-GALAXY-ELG-v3.0.fits'), f'Wrong {filenames[1]}')
        self.assertTrue(filenames[2].endswith('rrtemplate-GALAXY-LRG-v3.0.fits'), f'Wrong {filenames[2]}')


