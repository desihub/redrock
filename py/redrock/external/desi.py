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

def read_bricks(brickfiles, trueflux=False):
    '''
    Read targets from a list of brickfiles
    
    Args:
        brickfiles : list of input brick files
        
    Returns list of Target objects
    '''
    bricks = list()
    targetids = set()
    for infile in brickfiles:
        b = desispec.io.Brick(infile)
        bricks.append(b)
        targetids.update(b.get_target_ids())

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
                R = Resolution(Rdata[i])
                spectra.append(Spectrum(wave, flux[i], ivar[i], R))
                
        targets.append(Target(targetid, spectra))
    
    return targets
