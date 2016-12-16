from __future__ import division, print_function
import time

import numpy as np

import redrock.zscan
import redrock.pickz
from redrock.zwarning import ZWarningMask as ZW

import multiprocessing as mp

def _wrap_calc_zchi2(args):
    try:
        return redrock.zscan.calc_zchi2_targets(*args)
    except Exception as oops:
        print('-'*60)
        print('ERROR: calc_zchi2_targets raised exception; original traceback:')
        import traceback
        traceback.print_exc()
        print('...propagating exception upwards')
        print('-'*60)
        raise oops

def zfind(targets, templates, ncpu=None):
    '''
    Given a list of targets and a list of templates, find redshifts
    
    Args:
        targets : list of Target objects
        templates: list of Template objects
        ncpu: number of CPU cores to use for multiprocessing

    Returns nested dictionary results[targetid][templatetype] with keys
        - z: array of redshifts scanned
        - zchi2: array of chi2 fit at each z
        - zbest: best fit redshift (finer resolution fit around zchi2 minimum)
        - minchi2: chi2 at zbest
        - zerr: uncertainty on zbest
        - zwarn: 0=good, non-0 is a warning flag    
    '''
    # redshifts = dict(
    #     GALAXY  = 10**np.arange(np.log10(0.1), np.log10(2.0), 4e-4),
    #     STAR = np.arange(-0.001, 0.00101, 0.0001),
    #     QSO  = 10**np.arange(np.log10(0.5), np.log10(4.0), 5e-4),
    # )

    results = dict()
    for target in targets:
        results[target.id] = dict()

    if ncpu is None:
        ncpu = max(mp.cpu_count() // 2, 1)

    if ncpu > 1:
        print("INFO: using multiprocessing with {} cores".format(ncpu))
    else:
        print("INFO: not using multiprocessing")

    for t in templates:
        print('zchi2 scan for '+t.type)
        
        ntargets = len(targets)
        chunksize = max(1, ntargets // ncpu)
        args = list()
        for i in range(0, ntargets, chunksize):
            verbose = (i==0)
            args.append( [t.redshifts, targets[i:i+chunksize], t, verbose] )
   
        print('{} starting calc_zchi2 map'.format(time.asctime()))
        if ncpu > 1:
            pool = mp.Pool(ncpu)
            zchi2_results = pool.map(_wrap_calc_zchi2, args)
            pool.close()
            pool.join()
        else:
            zchi2_results = [_wrap_calc_zchi2(x) for x in args]

        zchi2 = np.vstack([x[0] for x in zchi2_results])
        zcoeff = np.vstack([x[1] for x in zchi2_results])

        print('pickz')
        for i in range(len(targets)):
            try:
                zbest, zerr, zwarn, minchi2, deltachi2 = redrock.pickz.pickz(
                    zchi2[i], t.redshifts, targets[i].spectra, t)
            except ValueError:
                print('ERROR: pickz failed for target {} id {}'.format(i, targets[i].id))
                zbest = zerr = minchi2 = deltachi2 = -1.0
                zwarn = ZW.BAD_MINFIT

            results[targets[i].id][t.type] = dict(
                z=t.redshifts, zchi2=zchi2[i], zbest=zbest, zerr=zerr, zwarn=zwarn,
                minchi2=minchi2, zcoeff=zcoeff[i], deltachi2=deltachi2,
            )
                
    return results
    
