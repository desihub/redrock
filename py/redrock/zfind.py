from __future__ import division, print_function

import numpy as np

import redrock.zscan
import redrock.pickz

import multiprocessing as mp

def _wrap_calc_zchi2(args):
    try:
        return redrock.zscan.calc_zchi2_targets(*args)
    except Exception, oops:
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
        targets : list of (targetid, spectra), where spectra are a list of
            dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux : array of flux densities [10e-17 erg/s/cm^2/Angstrom]
            - ivar : inverse variances of flux
            - R : spectro-perfectionism resolution matrix
        templates: list of dictionaries, each of which has keys
            - wave : array of wavelengths [Angstroms]
            - flux[i,wave] : template basis vectors of flux densities
        ncpu: number of CPU cores to use for multiprocessing

    Returns nested dictionary results[targetid][templatetype] with keys
        - z: array of redshifts scanned
        - zchi2: array of chi2 fit at each z
        - zbest: best fit redshift (finer resolution fit around zchi2 minimum)
        - minchi2: chi2 at zbest
        - zerr: uncertainty on zbest
        - zwarn: 0=good, non-0 is a warning flag    
    '''
    redshifts = dict(
        GALAXY  = 10**np.arange(np.log10(0.1), np.log10(2.0), 4e-4),
        STAR = np.arange(-0.001, 0.00101, 0.0001),
        QSO  = 10**np.arange(np.log10(0.5), np.log10(4.0), 5e-4),
    )

    results = dict()
    for targetid, spectra in targets:
        results[targetid] = dict()

    if ncpu is None:
        ncpu = max(mp.cpu_count() // 2, 1)
    
    for t in templates:
        zz = redshifts[t['type']]
        print('zchi2 scan for '+t['type'])
        
        ntargets = len(targets)
        chunksize = max(1, ntargets // ncpu)
        args = list()
        for i in range(0, ntargets, chunksize):
            args.append( [zz, targets[i:i+chunksize], t] )
        
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
            targetid, spectra = targets[i]
            zbest, zerr, zwarn, minchi2 = redrock.pickz.pickz(
                zchi2[i], zz, spectra, t)
            results[targetid][t['type']] = dict(
                z=zz, zchi2=zchi2[i], zbest=zbest, zerr=zerr, zwarn=zwarn,
                minchi2=minchi2, zcoeff=zcoeff[i],
            )
                
    return results
    
