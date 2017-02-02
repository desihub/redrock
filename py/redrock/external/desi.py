import sys
import warnings
if sys.version_info[0] > 2:
    basestring = str

import numpy as np
import desispec.io
from desispec.resolution import Resolution

from .. import Target
from .. import Spectrum

def write_zbest(outfile, zbest):
    '''
    Write zbest Table to outfile
    
    Adds blank BRICKNAME and SUBTYPE columns if needed
    Adds zbest.meta['EXTNAME'] = 'ZBEST'
    '''
    ntargets = len(zbest)
    if 'BRICKNAME' not in zbest.colnames:
        zbest['BRICKNAME'] = np.zeros(ntargets, dtype='S8')

    if 'SUBTYPE' not in zbest.colnames:
        zbest['SUBTYPE'] = np.zeros(ntargets, dtype='S8')

    zbest.meta['EXTNAME'] = 'ZBEST'
    zbest.write(outfile, overwrite=True)

def read_bricks(brickfiles, trueflux=False, targetids=None):
    '''
    Read targets from a list of brickfiles
    
    Args:
        brickfiles : list of input brick files, or string glob to match
        
    Returns list of Target objects
    '''
    if isinstance(brickfiles, basestring):
        import glob
        brickfiles = glob.glob(brickfiles)

    assert len(brickfiles) > 0

    bricks = list()
    brick_targetids = set()

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for infile in brickfiles:
            b = desispec.io.Brick(infile)
            bricks.append(b)
            brick_targetids.update(b.get_target_ids())

    if targetids is None:
        targetids = brick_targetids

    targets = list()
    for targetid in targetids:
        spectra = list()
        for brick in bricks:
            wave = brick.get_wavelength_grid()
            flux, ivar, Rdata, info = brick.get_target(targetid)

            #- work around desispec.io.Brick returning 32-bit non-native endian
            flux = flux.astype(float)
            ivar = ivar.astype(float)
            Rdata = Rdata.astype(float)

            for i in range(flux.shape[0]):
                if np.all(flux[i] == 0):
                    # print('WARNING: Skipping spectrum {} of target {} on brick {} with flux=0'.format(i, targetid, brick.brickname))
                    continue

                if np.all(ivar[i] == 0):
                    # print('WARNING: Skipping spectrum {} of target {} on brick {} with ivar=0'.format(i, targetid, brick.brickname))
                    continue

                R = Resolution(Rdata[i])
                spectra.append(Spectrum(wave, flux[i], ivar[i], R))

        if len(spectra) > 0:
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on brick {} has no good spectra'.format(targetid, brick.brickname))
    
    for brick in bricks:
        brick.close()

    return targets
