from __future__ import division, print_function
import time
import os, sys

import time
import numpy as np

from . import zscan as rrzscan
from . import fitz
from . import constants
from .zwarning import ZWarningMask as ZW

import multiprocessing as mp


def _wrap_calc_zchi2(args):
    try:
        return rrzscan.calc_zchi2_targets(*args)
    except Exception as oops:
        print('-'*60)
        print('ERROR: calc_zchi2_targets raised exception; original traceback:')
        import traceback
        traceback.print_exc()
        print('...propagating exception upwards')
        print('-'*60)
        raise oops


def zfind(targets, templates, ncpu=None, comm=None, nminima=3):
    '''
    Given a list of targets and a list of templates, find redshifts
    
    Args:
        targets : list of Target objects
        templates: dictionary of Template objects (template_name: template)

    Options:
        ncpu: number of CPU cores to use for multiprocessing
        nminima: number of minima per spectype to return

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

    pid = os.getpid()
    zscan = dict()
    for target in targets:
        zscan[target.id] = dict()
        for t in templates.values():
            zscan[target.id][t.fulltype] = dict()
            
    if ncpu is None :
        if comm is None :
            # use multiprocessing by default
            ncpu = max(mp.cpu_count() // 2, 1)
        else :
            ncpu = 1

    if comm is not None :
        if comm.rank == 0 :
            print("INFO: using MPI with {} processes".format(comm.size))
            sys.stdout.flush()
    elif ncpu > 1:
        print("INFO: using multiprocessing with {} cores".format(ncpu))
    else:
        print("INFO: not using multiprocessing")
        
    for t in templates.values():

        if comm is not None and (comm.rank == 0) :
            print('INFO: starting zchi2 scan for {}'.format(comm.rank,
                t.fulltype))
            sys.stdout.flush() #  this helps seeing something
        
        t0 = time.time()
        if ncpu > 1:
            zchi2, zcoeff, penalty = rrzscan.parallel_calc_zchi2_targets(t.redshifts, targets, t, ncpu=ncpu)
        elif comm is not None:
            zchi2, zcoeff, penalty = rrzscan.mpi_calc_zchi2_targets(t.redshifts, targets, t, comm=comm)
        else:
            zchi2, zcoeff, penalty = rrzscan.calc_zchi2_targets(t.redshifts, targets, t)
        dt = time.time() - t0

        if comm is None or comm.rank==0 :
            print('DEBUG: PID {} {} zscan in {:.1f} seconds'.format(pid, t.fulltype, dt))

        t0 = time.time()
        if comm is None : # multiprocessing version
            zfits = fitz.parallel_fitz_targets(
                zchi2+penalty, t.redshifts, targets, t,
                ncpu=ncpu, nminima=nminima)
        else : # mpi version
            zfits = fitz.mpi_fitz_targets(
                zchi2+penalty, t.redshifts, targets, t,
                comm=comm, nminima=nminima)
        dt = time.time() - t0

        if comm is None or comm.rank==0 :
            print('DEBUG: PID {} {} fitz in {:.1f} seconds'.format(pid, t.fulltype, dt))
  
        if comm is None or comm.rank == 0 :
            
            for i,zfit in enumerate(zfits) :
                zscan[targets[i].id][t.fulltype]['zfit'] = zfit
        
            for i in range(len(targets)):
                zscan[targets[i].id][t.fulltype]['redshifts'] = t.redshifts
                zscan[targets[i].id][t.fulltype]['zchi2'] = zchi2[i]
                zscan[targets[i].id][t.fulltype]['penalty'] = penalty[i]
                zscan[targets[i].id][t.fulltype]['zcoeff'] = zcoeff[i]

    sys.stdout.flush()
    
    if comm is None or comm.rank == 0 :
        #- Convert individual zfit results into a zall array
        t0 = time.time()
        ### print('DEBUG: PID {} Making zall'.format(pid))
        sys.stdout.flush()
        import astropy.table
        zfit = list() 
        for target in targets:
            tzfit = list()
            for fulltype in zscan[target.id]:
                tmp = zscan[target.id][fulltype]['zfit']
                #- TODO: reconsider fragile parsing of fulltype
                if fulltype.count(':') > 0:
                    spectype, subtype = fulltype.split(':')
                else:
                    spectype, subtype = (fulltype, '')
                tmp['spectype'] = spectype
                tmp['subtype'] = subtype
                tzfit.append(tmp)
                del zscan[target.id][fulltype]['zfit']

            maxncoeff = max([tmp['coeff'].shape[1] for tmp in tzfit])
            for tmp in tzfit:
                if tmp['coeff'].shape[1] < maxncoeff:
                    n = maxncoeff - tmp['coeff'].shape[1]
                    c = np.append(tmp['coeff'], np.zeros((len(tmp), n)), axis=1)
                    tmp.replace_column('coeff', c)

            tzfit = astropy.table.vstack(tzfit)
            tzfit.sort('chi2')
            tzfit['targetid'] = target.id
            tzfit['znum'] = np.arange(len(tzfit))
            tzfit['deltachi2'] = np.ediff1d(tzfit['chi2'], to_end=0.0)
            ii = np.where(tzfit['deltachi2'] < 9)[0]
            tzfit['zwarn'][ii] |= ZW.SMALL_DELTA_CHI2

            #- Trim down cases of multiple subtypes for a single type (e.g. STARs)
            #- tzfit is already sorted by chi2, so keep first nminima of each type
            iikeep = list()
            for spectype in np.unique(tzfit['spectype']):
                ii = np.where(tzfit['spectype'] == spectype)[0]
                iikeep.extend(ii[0:nminima])
            if len(iikeep) < len(tzfit):
                tzfit = tzfit[iikeep]
                #- grouping by spectype could get chi2 out of order; resort
                tzfit.sort('chi2')

            #- Unflag zwarn |= ZW.SMALL_DELTA_CHI2 when znum0 and znum1 have same spectype,
            #- and different subtype
            for i in range(len(tzfit)-1):
                cond = (tzfit['zwarn'][i] & ZW.SMALL_DELTA_CHI2)!=0
                cond &= (tzfit['zwarn'][i+1] & ZW.SMALL_DELTA_CHI2)==0
                cond &= (tzfit['spectype'][i]==tzfit['spectype'][i+1])
                cond &= (tzfit['subtype'][i]!=tzfit['subtype'][i+1])
                dv = (constants.speed_light/1000.) * (tzfit['z'][i]-tzfit['z'][i+1]) / (1.+(tzfit['z'][i]+tzfit['z'][i+1])/2. )
                cond &= (np.absolute(dv) < constants.max_velo_diff)
                if cond: tzfit['zwarn'][i] -= ZW.SMALL_DELTA_CHI2

            zfit.append(tzfit)

        zfit = astropy.table.vstack(zfit)
        dt = time.time() - t0
        #print('DEBUG: PID {} zall in {:.1f} seconds'.format(pid, dt))
    else :
       zscan = None
       zfit = None
    
    # no need to broadcast at the end of this routine
    # because only rank 0 will write the results
    
    return zscan, zfit
    
