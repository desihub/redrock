from __future__ import absolute_import, division, print_function

import numpy as np
import redrock
from redrock.zwarning import ZWarningMask as ZW

def find_minima(x):
    '''
    return indices of local minima of x, including edges, sorted small to large
    
    Note: this is somewhat conservative in the case of repeated values:
    
    find_minima([1,1,1,2,2,2]) -> [0,1,2,4,5]
    '''
    x = np.asarray(x)
    ii = np.where(np.r_[True, x[1:]<=x[:-1]] & np.r_[x[:-1]<=x[1:], True])[0]
    
    jj = np.argsort(x[ii])
    
    return ii[jj]

def minfit(x,y):
    '''
    Fits y = y0 + ((x-x0)/xerr)**2
    
    Returns (x0, xerr, y0, zwarn) where zwarn=0 is good fit
    
    See redrock.zwarning.ZWarningMask.BAD_MINFIT for zwarn failure flags
    '''
    try:
        #- y = a x^2 + b x + c
        a,b,c = np.polyfit(x,y,2)
    except np.linalg.LinAlgError:
        return (-1,-1,-1,ZW.BAD_MINFIT)
    
    if a == 0.0:
        return (-1,-1,-1,ZW.BAD_MINFIT)

    #- recast as y = y0 + ((x-x0)/xerr)^2
    x0 = -b / (2*a)
    y0 = -(b**2) / (4*a) + c

    zwarn = 0
    if (x0 <= np.min(x)) or (np.max(x) <= x0):
        zwarn |= ZW.BAD_MINFIT

    if a > 0.0:
        xerr = 1 / np.sqrt(a)
    else:
        xerr = 1 / np.sqrt(-a)
        zwarn |= ZW.BAD_MINFIT

    return (x0, xerr, y0, zwarn)


def fitz(zchi2, redshifts, spectra, template, nminima=3):
    '''Refines redshift measurement around up to nminima minima
    
    TODO: document return values
    TODO: fix deltachi2
    TODO: if there are fewer than nminima minima, consider padding
    '''
    assert len(zchi2) == len(redshifts)

    results = list()

    for imin in find_minima(zchi2):
        if len(results) == nminima:
            break

        #- Skip this minimum if it is within 1000 km/s of a previous one
        zprev = np.array([tmp['z'] for tmp in results])
        dv = 3e5 * (redshifts[imin] - zprev) / (1+redshifts[imin])
        if np.any(np.abs(dv) < 1000):
            continue

        #- Sample more finely around the minimum
        ilo = max(0, imin-1)
        ihi = min(imin+1, len(zchi2)-1)
        zz = np.linspace(redshifts[ilo], redshifts[ihi], 15)
        zzchi2, zzcoeff, zzpenalty = redrock.zscan.calc_zchi2(zz, spectra, template)

        #- fit parabola to 3 points around minimum
        i = min(max(np.argmin(zzchi2),1), len(zz)-1)
        zmin, sigma, chi2min, zwarn = minfit(zz[i-1:i+2], zzchi2[i-1:i+2])
        try:
            coeff = redrock.zscan.calc_zchi2([zmin,], spectra, template)[1][0]
        except ValueError as err:
            if zmin<redshifts[0] or redshifts[-1]<zmin:
                #- beyond redshift range can be invalid for template
                coeff = np.zeros(template.nbasis)
                zwarn |= ZW.Z_FITLIMIT
                zwarn |= ZW.BAD_MINFIT
            else:
                #- Unknown problem; re-raise error
                raise err

        zbest = zmin
        zerr = sigma

        #- Other minima are too similar (within 3 sigma significance)
        #- TODO: tune exclusion of close values
        ii = find_minima(zchi2)
        if np.any((np.abs(redshifts-zbest)>5*zerr)[ii] & (zchi2 < chi2min + 9)[ii]):
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
            zbest = zz[imin]
            chi2min = zzchi2[imin]

        results.append(dict(z=zbest, zerr=zerr, zwarn=zwarn,
            chi2=chi2min, zz=zz, zzchi2=zzchi2,
            coeff=coeff))

    #- Sort results by chi2min; detailed fits may have changed order
    ii = np.argsort([tmp['chi2'] for tmp in results])
    results = [results[i] for i in ii]

    #- Convert list of dicts -> Table
    from astropy.table import Table
    results = Table(results)
    
    assert len(results) > 0
    
    return results
