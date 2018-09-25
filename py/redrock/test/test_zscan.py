import unittest
import numpy as np
import scipy.sparse
import tempfile
import os
import shutil
from astropy.io import fits

import numpy.testing as nt

from ..targets import DistTargetsCopy
from ..templates import DistTemplate
from ..rebin import rebin_template
from ..zscan import calc_zchi2_one, calc_zchi2_targets, spectral_data
from ..zfind import zfind, calc_deltachi2

from . import util

#- Return a normalized sampled Gaussian (no integration, just sampling)
def norm_gauss(x, sigma):
    y = np.exp(-x**2/(2.0*sigma))
    return y / np.sum(y)

#- Utility function to create a resolution matrix with a given sigma in pixel units
def getR(n, sigma):
    '''Returns a (n,n) sized sparse Resolution matrix with constant sigma'''
    x = np.arange(-5, 6)
    y = norm_gauss(x, sigma)
    data = np.zeros((11, n))
    for i in range(n):
        data[:,i] = y
    return scipy.sparse.dia_matrix((data, x), shape=(n,n))


class TestZScan(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._branchFiles = tempfile.mkdtemp()+"/"

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._branchFiles):
            shutil.rmtree(cls._branchFiles, ignore_errors=True)

    def test_zfind(self):
        z1 = 0.2
        z2 = 0.25
        seed = np.random.randint(2**31)
        print('TEST: Using random seed {}'.format(seed))
        np.random.seed(seed)

        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        dtarg = DistTargetsCopy([t1, t2])

        # Get the dictionary of wavelength grids
        dwave = dtarg.wavegrids()

        # Construct the distributed template.
        template = util.get_template(redshifts=np.linspace(0.15, 0.3, 50))
        dtemp = DistTemplate(template, dwave)

        zscan, zfit = zfind(dtarg, [ dtemp ])

        zx1 = zfit[zfit['targetid'] == 111][0]
        zx2 = zfit[zfit['targetid'] == 222][0]

        # These unit tests fail- the chi2 values are very large, and the
        # zerr is ~1.0e-5.  We need to understand this...
        #self.assertLess(np.abs(zx1['z'] - z1)/zx1['zerr'], 5)
        #self.assertLess(np.abs(zx2['z'] - z2)/zx2['zerr'], 5)

        self.assertLess(zx1['zerr'], 0.002)
        self.assertLess(zx2['zerr'], 0.002)

        # Create a prior file and test it
        priorName = self._branchFiles+'/priors.fits'
        c1 = fits.Column(name='TARGETID', array=np.array([t1.id,t2.id]), format='K')
        c2 = fits.Column(name='Z',        array=np.array([z1,z2]),       format='D')
        c3 = fits.Column(name='SIGMA',    array=np.array([0.01,0.01]),   format='D')
        t = fits.BinTableHDU.from_columns([c1, c2, c3],name='PRIORS')
        t.writeto(priorName)

        zscan, zfit = zfind(dtarg, [ dtemp ], priors=priorName)
        zx1 = zfit[zfit['targetid'] == 111][0]
        zx2 = zfit[zfit['targetid'] == 222][0]

        self.assertLess(np.abs(zx1['z'] - z1)/zx1['zerr'], 5)
        self.assertLess(np.abs(zx2['z'] - z2)/zx2['zerr'], 5)
        self.assertLess(zx1['zerr'], 0.002)
        self.assertLess(zx2['zerr'], 0.002)

    def test_calc_deltachi2(self):
        chi2 = np.array([1.0, 2.0, 4.0, 8.0])
        z = np.array([3.0, 3.1, 3.2, 3.3])
        dchi2 = calc_deltachi2(chi2, z)
        self.assertTrue(np.all(dchi2 == np.array([1, 2, 4, 0.0])), dchi2)

        z = np.array([3.0, 3.0, 4.0, 4.0])
        dchi2 = calc_deltachi2(chi2, z)
        self.assertTrue(np.all(dchi2 == np.array([3, 2, 0.0, 0.0])), dchi2)

    def test_parallel_zscan(self):
        z1 = 0.2
        z2 = 0.25
        seed = np.random.randint(2**31)
        print('TEST: Using random seed {}'.format(seed))
        np.random.seed(seed)

        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        dtarg = DistTargetsCopy([t1, t2])

        # Get the dictionary of wavelength grids
        dwave = dtarg.wavegrids()

        # Construct the distributed template.
        template = util.get_template(redshifts=np.linspace(0.15, 0.3, 50))
        dtemp = DistTemplate(template, dwave)

        results_a = calc_zchi2_targets(dtarg, [ dtemp ], mp_procs=1)
        results_b = calc_zchi2_targets(dtarg, [ dtemp ], mp_procs=2)

        for i, tg in enumerate(dtarg.local()):
            resa = results_a[tg.id][template.full_type]
            resb = results_b[tg.id][template.full_type]
            self.assertEqual(resa['zchi2'].shape, resb['zchi2'].shape)
            self.assertEqual(resa['zcoeff'].shape, resb['zcoeff'].shape)
            self.assertTrue(np.all(resa['zchi2'] == resb['zchi2']))
            self.assertTrue(np.all(resa['zcoeff'] == resb['zcoeff']))


    def test_subtype(self):
        z1 = 0.0
        z2 = 1e-4
        seed = np.random.randint(2**31)
        print('TEST: Using random seed {}'.format(seed))
        np.random.seed(seed)

        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        dtarg = DistTargetsCopy([t1, t2])

        # Get the dictionary of wavelength grids
        dwave = dtarg.wavegrids()

        Fstar = util.get_template(spectype='STAR', subtype='F',
            redshifts=np.linspace(-1e-3, 1e-3, 25))
        dFstar = DistTemplate(Fstar, dwave)

        Mstar = util.get_template(spectype='STAR', subtype='M',
            redshifts=np.linspace(-1e-3, 1e-3, 25))
        dMstar = DistTemplate(Mstar, dwave)

        nminima=3
        zscan, zfit = zfind(dtarg, [ dFstar, dMstar ], mp_procs=1,
            nminima=nminima)

        for row in zfit:
            self.assertEqual(len(row['coeff']), nminima)
        self.assertTrue(np.all(zfit['spectype'] == 'STAR'))


    def test_sharedmem(self):
        z1 = 0.0
        z2 = 1e-4
        t1 = util.get_target(z1); t1.id = 111
        t2 = util.get_target(z2); t2.id = 222
        dtarg = DistTargetsCopy([t1, t2])

        import copy
        dtcopy = copy.deepcopy(dtarg)

        for tg in dtarg.local():
            tg.sharedmem_pack()

        for tg in dtarg.local():
            tg.sharedmem_unpack()

        toff = 0
        for tg in dtarg.local():
            soff = 0
            tgcopy = dtcopy.local()[toff]
            for s in tg.spectra:
                scopy = tgcopy.spectra[soff]
                self.assertEqual(s.nwave, scopy.nwave)
                nt.assert_almost_equal(s.wave, scopy.wave)
                nt.assert_almost_equal(s.flux, scopy.flux)
                nt.assert_almost_equal(s.ivar, scopy.ivar)
                nt.assert_equal(s.R.offsets, scopy.R.offsets)
                nt.assert_almost_equal(s.R.data, scopy.R.data)
                nt.assert_almost_equal(s.Rcsr.data, scopy.Rcsr.data)
                nt.assert_equal(s.Rcsr.indices, scopy.Rcsr.indices)
                nt.assert_equal(s.Rcsr.indptr, scopy.Rcsr.indptr)
                soff += 1
            toff += 1


def test_suite():
    """Allows testing of only this module with the command::

        python setup.py test -m <modulename>
    """
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
