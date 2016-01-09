from __future__ import absolute_import, division, print_function

import numpy as np
import redrock
from redrock.zwarning import ZWarningMask as ZW

def pickz(zchi2, redshifts, spectra, template):
    '''Refines redshift measurement
    '''
    #- Scan at a finer resolution around the initial minimum
    zmin = redshifts[np.argmin(zchi2)]
    dz = 0.003
    zz = np.linspace(zmin-dz, zmin+dz, 31)
    zzchi2 = redrock.zscan.calc_zchi2(zz, spectra, template)
    chi2min = np.min(zzchi2)

    #- Fit a parabola to that finder resolution chi2 vs. z scan
    zmin = zz[np.argmin(zzchi2)]
    dchi2 = 50
    while True:
        jj = (zzchi2 < np.min(zzchi2)+dchi2)
        if np.count_nonzero(jj) >= 3:
            break
        else:
            dchi2 *= 2

    p = np.polyfit(zz[jj]-zmin, zzchi2[jj], 2)

    #- For zwarning mask bits
    zwarn = 0

    #- Get minimum and error from the parabola fit parameters
    a, b, c = p
    zbest = -b/(2*a) + zmin
    chi2min = c - b**2/(4*a)
    blat = b**2 - 4*a*(c-1-chi2min)
    if blat >= 0:
        zp = (-b + np.sqrt(blat)) / (2*a)
        zerr = zp-(zbest-zmin)
    else:
        zerr = -1
        zwarn |= ZW.BAD_MINFIT

    #- Other minima are too similar
    #- TODO: tune exclusion of close values
    if np.any((np.abs(redshifts-zbest)>5*zerr) & (zchi2 < chi2min + 9)):
        zwarn |= ZW.SMALL_DELTA_CHI2

    #- Initial minimum or best fit too close to edge of redshift range
    if zbest < redshifts[1] or zbest > redshifts[-2]:
        zwarn |= ZW.Z_FITLIMIT
    if zmin < redshifts[1] or zmin > redshifts[-2]:
        zwarn |= ZW.Z_FITLIMIT
        
    #- parabola minimum outside fit range; replace with min of scan
    if zbest < zz[0] or zbest > zz[-1]:
        zwarn |= ZW.BAD_MINFIT
        imin = np.where(zbest == np.min(zbest))[0][0]
        zbest = zz[zbest]
       
    #- Parabola fit considerably different than minimum of scan 
    #- (somewhat arbitrary cutoff)
    if abs(zbest - zmin) > 0.01:
        zwarn |= ZW.BAD_MINFIT
    
    #- chi2 at zbest
    chi2min = redrock.zscan.calc_zchi2([zbest,], spectra, template)[0]
    
    return zbest, zerr, zwarn, chi2min

