from __future__ import absolute_import, division, print_function

import numpy as np
import redrock
from redrock.zwarning import ZWarningMask as ZW

def pickz(zchi2, redshifts, spectra, template):
    '''Refines redshift measurement
    '''
    #- Scan at a finer resolution around the initial minimum
    zmin = redshifts[np.argmin(zchi2)]
    dz = 0.001
    zz = np.linspace(zmin-dz, zmin+dz, 21)
    zzchi2 = redrock.zscan.calc_zchi2(zz, spectra, template)
    chi2min = np.min(zzchi2)

    #- Fit a parabola to that finder resolution chi2 vs. z scan
    zmin = zz[np.argmin(zzchi2)]
    jj = (zzchi2 < np.min(zzchi2)+50)
    p = np.polyfit(zz[jj], zzchi2[jj], 2)

    #- For zwarning mask bits
    zwarn = 0

    #- Get minimum and error from the parabola fit parameters
    a, b, c = p
    zbest = -b/(2*a)
    chi2min = c - b**2/(4*a)
    blat = b**2 - 4*a*(c-1-chi2min)
    if blat >= 0:
        zp = (-b + np.sqrt(blat)) / (2*a)
        zerr = zp-zbest
    else:
        zerr = 1e6
        zwarn |= ZW.BAD_MINFIT

    #- Initial minimum or best fit too close to edge of redshift range
    if zbest < redshifts[1] or zbest > redshifts[-2]:
        zwarn |= ZW.Z_FITLIMIT
    if zmin < redshifts[1] or zmin > redshifts[-2]:
        zwarn |= ZW.Z_FITLIMIT
        
    #- parabola minimum outside fit range
    if zbest < zz[0] or zbest > zz[-1]:
        zwarn |= ZW.BAD_MINFIT
       
    #- Parabola fit considerably different than minimum of scan 
    #- (somewhat arbitrary cutoff)
    if abs(zbest - zmin) > 0.01:
        zwarn |= ZW.BAD_MINFIT
        
    return zbest, zerr, zwarn

