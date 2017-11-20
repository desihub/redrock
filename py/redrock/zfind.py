"""
redrock.zfind
=============

Redshift finding algorithms.
"""

from __future__ import division, print_function

import os
import sys
import traceback

import time

import numpy as np

import astropy.table

from . import constants

from .utils import elapsed

from .targets import Spectrum, Target, DistTargets, distribute_targets

from .templates import Template, DistTemplate

from .zscan import calc_zchi2_targets

from .fitz import fitz, get_dv

from .zwarning import ZWarningMask as ZW


def _mp_fitz(chi2, target_data, t, nminima, qout):
    """Wrapper for multiprocessing version of fitz.
    """
    try:
        # Unpack targets from shared memory
        for tg in target_data:
            tg.sharedmem_unpack()
        results = list()
        for i, tg in enumerate(target_data):
            zfit = fitz(chi2[i], t.template.redshifts, tg.spectra,
                t.template, nminima=nminima)
            npix = 0
            for spc in tg.spectra:
                npix += (spc.ivar > 0.).sum()
            results.append( (tg.id, zfit, npix) )
        qout.put(results)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "MP calc_zchi2: {}".format(x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()


def zfind(targets, templates, mp_procs=1, nminima=3):
    """Compute all redshift fits for the local set of targets and collect.

    Given targets and templates distributed across a set of MPI processes,
    compute the redshift fits for all redshifts and our local set of targets.
    Each process computes the fits for a slice of redshift range and then
    cycles through redshift slices by passing the interpolated templates along
    to the next process in order.

    Note:
        If using MPI, only the rank 0 process will return results- all other
        processes with return a tuple of (None, None).

    Args:
        targets (DistTargets): distributed targets.
        templates (list): list of DistTemplate objects.
        mp_procs (int): if not using MPI, this is the number of multiprocessing
            processes to use.
        nminima (int): number of chi^2 minima to consider.  Passed to fitz().

    Returns:
        tuple: (allresults, allzfit), where "allresults" is a dictionary of the
            full chi^2 fit information, suitable for writing to a redrock scan
            file.  "allzfit" is an astropy Table of only the best fit parameters
            for a limited set of minima.

    """

    # Find most likely candidate redshifts by scanning over the
    # pre-interpolated templates on a coarse redshift spacing.

    # See if we are the process that should be printing stuff...
    am_root = False
    if targets.comm is None:
        am_root = True
    elif targets.comm.rank == 0:
        am_root = True

    # If we are not using MPI, our DistTargets object will have all the targets
    # on the main process.  In that case, we would like to distribute our
    # targets across multiprocesses.  Here we compute that distribution, if
    # needed.

    mpdist = None
    if targets.comm is None:
        mpdist = distribute_targets(targets.local(), mp_procs)

    # Compute the coarse-binned chi2 for all local targets.

    results = calc_zchi2_targets(targets, templates, mp_procs=mp_procs)

    # For each of our local targets, refine the redshift fit close to the
    # minima in the coarse fit.

    for t in templates:
        ft = t.template.full_type

        if am_root:
            print("  Finding best fits for template {}"\
                .format(t.template.full_type))
            sys.stdout.flush()

        start = elapsed(None, "", comm=t.comm)

        # Here we have another parallelization choice between MPI and
        # multiprocessing.

        if targets.comm is not None:
            # MPI case.  Every process just works with its local targets.
            for tg in targets.local():
                zfit = fitz(results[tg.id][ft]['zchi2'] \
                    + results[tg.id][ft]['penalty'],
                    t.template.redshifts, tg.spectra,
                    t.template, nminima=nminima)
                results[tg.id][ft]['zfit'] = zfit
                results[tg.id][ft]['zfit']['npixels'] = 0
                for spectrum in tg.spectra:
                    results[tg.id][ft]['zfit']['npixels'] += \
                        (spectrum.ivar>0.).sum()

        else:
            # Multiprocessing case.
            import multiprocessing as mp

            # Ensure that all targets are packed into shared memory
            for tg in targets.local():
                tg.sharedmem_pack()

            qout = mp.Queue()

            procs = list()
            for i in range(mp_procs):
                if len(mpdist[i]) == 0:
                    continue
                target_data = [ x for x in targets.local() if x.id in mpdist[i] ]
                eff_chi2 = np.zeros((len(target_data),
                    len(t.template.redshifts)), dtype=np.float64)

                for i, tg in enumerate(target_data):
                    eff_chi2[i,:] = results[tg.id][ft]['zchi2'] \
                        + results[tg.id][ft]['penalty']
                p = mp.Process(target=_mp_fitz, args=(eff_chi2,
                    target_data, t, nminima, qout))
                procs.append(p)
                p.start()

            # Extract the output
            for i in range(mp_procs):
                if len(mpdist[i]) == 0:
                    continue
                res = qout.get()
                for rs in res:
                    results[rs[0]][ft]['zfit'] = rs[1]
                    results[rs[0]][ft]['zfit']['npixels'] = rs[2]

        stop = elapsed(start, "    Finished in", comm=t.comm)

    # Gather our results to the root process and split off the zfit data.
    # Only process zero returns data- other ranks return None.

    allresults = None
    allzfit = None

    if targets.comm is not None:
        results = targets.comm.gather(results, root=0)
    else:
        results = [ results ]

    if am_root:
        allresults = dict()
        for p in results:
            allresults.update(p)
        del results

        allzfit = list()
        for tid in targets.all_target_ids:
            tzfit = list()
            for fulltype in allresults[tid]:
                tmp = allresults[tid][fulltype]['zfit']
                #- TODO: reconsider fragile parsing of fulltype
                if fulltype.count(':') > 0:
                    spectype, subtype = fulltype.split(':')
                else:
                    spectype, subtype = (fulltype, '')
                tmp['spectype'] = spectype
                tmp['subtype'] = subtype
                tmp['ncoeff'] = tmp['coeff'].shape[1]
                tzfit.append(tmp)
                del allresults[tid][fulltype]['zfit']

            maxncoeff = max([tmp['coeff'].shape[1] for tmp in tzfit])
            for tmp in tzfit:
                if tmp['coeff'].shape[1] < maxncoeff:
                    n = maxncoeff - tmp['coeff'].shape[1]
                    c = np.append(tmp['coeff'], np.zeros((len(tmp), n)), axis=1)
                    tmp.replace_column('coeff', c)

            tzfit = astropy.table.vstack(tzfit)
            tzfit.sort('chi2')
            tzfit['targetid'] = tid
            tzfit['znum'] = np.arange(len(tzfit))
            tzfit['deltachi2'] = np.ediff1d(tzfit['chi2'], to_end=0.0)
            tzfit['zwarn'][ (tzfit['npixels']<10*tzfit['ncoeff']) ] |= \
                ZW.LITTLE_COVERAGE

            #- set ZW.SMALL_DELTA_CHI2 flag
            for i in range(len(tzfit)-1):
                noti = (np.arange(len(tzfit))!=i)
                alldeltachi2 = np.absolute(tzfit['chi2'][noti]-tzfit['chi2'][i])
                alldv = np.absolute(get_dv(z=tzfit['z'][noti],
                    zref=tzfit['z'][i]))
                zwarn = np.any( (alldeltachi2<9.) & \
                    (alldv>=constants.max_velo_diff) )
                if zwarn:
                    tzfit['zwarn'][i] |= ZW.SMALL_DELTA_CHI2

            # Trim down cases of multiple subtypes for a single type (e.g.
            # STARs) tzfit is already sorted by chi2, so keep first nminima of
            # each type.
            iikeep = list()
            for spectype in np.unique(tzfit['spectype']):
                ii = np.where(tzfit['spectype'] == spectype)[0]
                iikeep.extend(ii[0:nminima])
            if len(iikeep) < len(tzfit):
                tzfit = tzfit[iikeep]
                #- grouping by spectype could get chi2 out of order; resort
                tzfit.sort('chi2')

            allzfit.append(tzfit)

        allzfit = astropy.table.vstack(allzfit)

    return allresults, allzfit
