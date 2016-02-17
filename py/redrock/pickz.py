from __future__ import absolute_import, division, print_function

import numpy as np
import redrock
from redrock.zwarning import ZWarningMask as ZW

def minfit(x, y):
    '''
    Fit a parabola to y(x) assuming that it contains an interior minimum

    Returns coeff, xmin, ymin, sigma

    coeff = np.polyfit(x, y, 2)
    '''
    imin = np.argmin(y)        
    # assert imin != 0
    # assert imin != len(y)-1
    assert len(x) == len(y)

    c = np.polyfit(x, y, 2)
    d = np.polyder(c)
    xmin = -d[1] / d[0]
    ymin = np.polyval(c, xmin)

    cx = c - [0, 0, ymin+1]
    y1, y2 = np.roots(cx)
    sigma = abs(y2-y1)/2.0

    return c, xmin, ymin, sigma

def pickz(zchi2, redshifts, spectra, template):
    '''Refines redshift measurement
    '''
    assert len(zchi2) == len(redshifts)
    
    #- Scan at a finer resolution around the initial minimum
    imin = np.argmin(zchi2)
    ilo = max(0, imin-2)
    ihi = min(imin+3, len(zchi2))

    c, zmin, chi2min, sigma = minfit(redshifts[ilo:ihi], zchi2[ilo:ihi])
    zz = np.linspace(zmin-2*sigma, zmin+2*sigma, 15)
    zzchi2 = redrock.zscan.calc_zchi2(zz, spectra, template)

    c, zbest, chi2min, zerr = minfit(redshifts[ilo:ihi], zchi2[ilo:ihi])

    #- For zwarning mask bits
    zwarn = 0

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
    if abs(zbest - zmin) > sigma:
        zwarn |= ZW.BAD_MINFIT
        
    return zbest, zerr, zwarn, chi2min

