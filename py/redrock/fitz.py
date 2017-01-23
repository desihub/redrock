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


def parallel_fitz_targets(zchi2, redshifts, targets, template, nminima=3, ncpu=None, verbose=False):
    '''
    TODO: document
    '''
    assert zchi2.shape == (len(targets), len(redshifts))
    
    import multiprocessing as mp
    if ncpu is None:
        ncpu = mp.cpu_count() // 2
    
    #- Wrapper function for fitz.  This can use targets, template, etc. without
    #- copying them.  multiprocessing.Queue is used for I/O to know which
    #- targets to process.
    def wrap_fitz(i, qin, qout):
        '''
        i: process number
        qin, qout: input and output multiprocessing.Queue

        qin sends (start_index, number_of_targets_to_process)
        qout receives (target_index, fitz_results)
        '''
        try:
            start, n = qin.get()
            if verbose:
                print('Process {} targets[]{}:{}]'.format(i, start, start+n))
            for j in range(start, start+n):
                results = fitz(zchi2[j], redshifts, targets[j].spectra, template, nminima=nminima)
                #- return index with results so that they can be sorted
                qout.put( (j, results) )
        except Exception as err:
            import traceback, sys
            message = "".join(traceback.format_exception(*sys.exc_info()))
            qout.put( (i, err, message) )

    #- Load Queue with start,n indices of targets to process
    ii = np.linspace(0, len(targets), ncpu+1).astype(int)
    qin = mp.Queue()
    qout = mp.Queue()
    for i in range(ncpu):
        start = ii[i]
        n = ii[i+1] - ii[i]
        qin.put( (start, n) )

    #- Start processes to run wrap_fitz
    for i in range(ncpu):
        p = mp.Process(target=wrap_fitz, args=(i, qin, qout))
        p.start()

    #- Pull results from queue
    results = list()
    for i in range(len(targets)):
        results.append(qout.get())

    #- Check for any errors
    mpfail = False
    message = 'ok'
    for r in results:
        if isinstance(r[1], Exception):
            i, err, message = r
            print("ERROR: result {} generated an exception".format(i))
            print(message)
            mpfail = True
    
    if mpfail:
        print("ERROR: Raising the last of the exceptions")
        raise RuntimeError(message)
    
    #- Sort results into original order of targets
    isort = np.argsort([r[0] for r in results])
    results = [results[i][1] for i in isort]
    
    return results

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
