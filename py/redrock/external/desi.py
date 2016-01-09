import desispec.io
from desispec.resolution import Resolution

def read_simbricks(brickfiles):
    '''
    Read targets from a list of simulated brickfiles, replacing the flux
    with the _TRUEFLUX HDU, and the ivar with ones.
    '''
    targets = dict()
    for filename in brickfiles:
        fx = open(filename)
        fibermap = fx['FIBERMAP']        
        wave = fx['WAVELENGTH'].astype(float)
        flux = fx['_TRUEFLUX'].astype(float)
        ivar = np.ones_like(flux)
        Rdata = fx['RESOLUTION'].astype(float)

        for i in range(flux.shape[0]):
            targetid = fibermap['TARGETID'][i]
            if targetid not in targets:
                targets[targetid] = list()

            R = Resolution(Rdata[i])
            spectrum = dict(wave=wave, flux=flux[i], ivar=ivar[i], R=R)
            targets[targetid].append(spectrum)
            
    return targets.keys(), targets.values()
        
    
def read_bricks(brickfiles, trueflux=False):
    '''
    Read targets from a list of brickfiles
    
    Args:
        brickfiles : list of input brick files
        
    Returns list of (targetid, spectra) where spectra are a list of
        dictionaries with keys 'wave', 'flux', 'ivar', 'R'
        
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
                spectra.append(dict(wave=wave, flux=flux[i], ivar=ivar[i], R=R))
                
        targets.append(spectra)
    
    return zip(targetids, targets)
