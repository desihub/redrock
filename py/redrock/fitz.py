from __future__ import absolute_import, division, print_function

import sys
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
    if len(x) < 3:
        return (-1,-1,-1,ZW.BAD_MINFIT)

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

def _wrap_fitz(target_indices, qout, zchi2, redshifts, targets, template, nminima):
    #- unpack shared memory buffers into numpy arrays
    zchi2 = redrock.sharedmem.toarray(zchi2)
    try:
        for j in target_indices:
            targets[j].sharedmem_unpack()
            results = fitz(zchi2[j], redshifts, targets[j].spectra, template, nminima=nminima)
            #- return index with results so that they can be sorted
            qout.put( (j, results) )
    except Exception as err:
        import traceback, sys
        message = "".join(traceback.format_exception(*sys.exc_info()))
        qout.put( (target_indices[0], err, message) )

    qout.close()

def parallel_fitz_targets(zchi2, redshifts, targets, template, nminima=3, ncpu=None):
    '''
    run `fitz` in parallel using `ncpu` CPU cores.

    See fitz for inputs and outputs.
    '''
    assert zchi2.shape == (len(targets), len(redshifts))

    import multiprocessing as mp
    if ncpu is None:
        ncpu = max(1, mp.cpu_count() // 2)
    else:
        ncpu = max(1, ncpu)

    ncpu = min(len(targets), ncpu)

    #- Pack arrays into shared memory for sending to processes
    zchi2 = redrock.sharedmem.fromarray(zchi2)
    for t in targets:
        t.sharedmem_pack()

    #- Start processes to run _wrap_fitz, returning results in a Queue
    #- NOTE: this is somewhat for historical reasons; mp.Pool.map probably
    #- would have been fine too.
    qout = mp.Queue()
    target_indices = np.array_split(range(len(targets)), ncpu)
    for i in range(ncpu):
        ### print('fitz process {}/{} doing {} targets'.format(i+1, ncpu, len(target_indices[i])))
        p = mp.Process(target=_wrap_fitz, args=(target_indices[i], qout, 
                zchi2, redshifts, targets, template, nminima))
        p.start()

    #- restore targets to unpacked numpy array state
    for t in targets:
        t.sharedmem_unpack()

    #- Pull results from queue
    #- NOTE: one per target, not one per process
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


def mpi_fitz_targets(zchi2, redshifts, targets, template, nminima=3, comm=None):

    if comm is None:
        raise ValueError("I NEED A COMMUNICATOR")
    
    assert zchi2.shape == (len(targets), len(redshifts))
    
    target_indices = np.array_split(range(len(targets)), comm.size)

    print("rank #%d : redrock.fitz for %s targets %d:%d"%(comm.rank,template.fulltype,target_indices[comm.rank][0],target_indices[comm.rank][-1]))
    #sys.stdout.flush() #  this would helps seeing something, but why is python saying it does know sys ?????
    result=[]
    try:
        for j in target_indices[comm.rank]:
            res = fitz(zchi2[j], redshifts, targets[j].spectra, template, nminima=nminima)
            result.append( (j,res) ) # same result format as _wrap_fitz
    except Exception as err:
        import traceback, sys
        message = "".join(traceback.format_exception(*sys.exc_info()))
        result.append( (target_indices[comm.rank][0], err, message) )
    
    print("rank #%d : done redrock.fitz"%comm.rank)
    #- all the results gather to rank #0
    results = comm.gather(result,root=0)
    print("rank #%d : done gathering results to rank 0"%comm.rank)

    if comm.rank == 0 :
        
        # rearrange results ( list of lists -> list )
        tmp=list()
        for result in results :
            for res in result :
                tmp.append(res)
        results=tmp
        print("rank #{} : results = {}".format(comm.rank,results))
        
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
    else : # not rank 0
        results = None
    
    # bcast results? no, rank 0 keeps the results
            
    return results



def fitz(zchi2, redshifts, spectra, template, nminima=3):
    '''Refines redshift measurement around up to nminima minima

    TODO: document return values    
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
        i = min(max(np.argmin(zzchi2),1), len(zz)-2)
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
