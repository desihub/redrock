from __future__ import division, print_function

import numpy as np

import redrock.zscan
import redrock.fitz
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

    zscan = dict()
    for target in targets:
        zscan[target.id] = dict()
        for t in templates:
            zscan[target.id][t.type] = dict()
            
    if ncpu is None:
        ncpu = max(mp.cpu_count() // 2, 1)

    if ncpu > 1:
        print("INFO: using multiprocessing with {} cores".format(ncpu))
    else:
        print("INFO: not using multiprocessing")

    for t in templates:
        print('zchi2 scan for '+t.type)
        
        ntargets = len(targets)
        chunksize = min(20, max(1, ntargets // ncpu))

        args = list()
        for i in range(0, ntargets, chunksize):
            verbose = (i==0)
            args.append( [t.redshifts, targets[i:i+chunksize], t, verbose] )
        
        if ncpu > 1:
            print("DEBUG: multiprocessing with {} chunks of {} targets each".format(len(args), chunksize))
            pool = mp.Pool(ncpu)
            zchi2_results = pool.map(_wrap_calc_zchi2, args)
            pool.close()
            pool.join()
        else:
            zchi2_results = [_wrap_calc_zchi2(x) for x in args]

        zchi2 = np.vstack([x[0] for x in zchi2_results])
        zcoeff = np.vstack([x[1] for x in zchi2_results])

        print('fitz')
        for i in range(len(targets)):
            zscan[targets[i].id][t.type]['redshifts'] = t.redshifts
            zscan[targets[i].id][t.type]['zchi2'] = zchi2[i]
            zscan[targets[i].id][t.type]['zcoeff'] = zcoeff[i]
            zscan[targets[i].id][t.type]['zfit'] = \
                redrock.fitz.fitz(zchi2[i], t.redshifts, targets[i].spectra, t)

    #- Convert individual zfit results into a zall array
    import astropy.table
    zfit = list() 
    for target in targets:
        tzfit = list()
        for spectype in zscan[target.id]:
            tmp = zscan[target.id][spectype]['zfit']
            tmp['spectype'] = spectype
            tzfit.append(tmp)

        maxncoeff = max([tmp['coeff'].shape[1] for tmp in tzfit])
        for tmp in tzfit:
            if tmp['coeff'].shape[1] < maxncoeff:
                n = maxncoeff - tmp['coeff'].shape[1]
                c = np.append(tmp['coeff'], np.zeros((len(tmp), n)), axis=1)
                tmp.replace_column('coeff', c)

        tzfit = astropy.table.vstack(tzfit)
        ii = np.argsort(tzfit['chi2'])
        tzfit = tzfit[ii]
        tzfit['targetid'] = target.id
        tzfit['znum'] = np.arange(len(tzfit))
        tzfit['deltachi2'] = np.ediff1d(tzfit['chi2'], to_end=0.0)
        ii = np.where(tzfit['deltachi2'] < 9)[0]
        tzfit['zwarn'][ii] |= ZW.SMALL_DELTA_CHI2
        zfit.append(tzfit)

    zfit = astropy.table.vstack(zfit)
    return zscan, zfit
    
