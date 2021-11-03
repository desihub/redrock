"""
redrock.zfind
=============

Redshift finding algorithms.
"""

from __future__ import division, print_function

import re
import sys
import traceback

import numpy as np

import astropy.table

from . import constants

from .utils import elapsed

from .targets import distribute_targets

from .archetypes import All_archetypes

from .priors import Priors

from .results import read_zscan_redrock

from .zscan import calc_zchi2_targets

from .fitz import fitz, get_dv

from .zwarning import ZWarningMask as ZW


def _mp_fitz(chi2, target_data, t, nminima, qout, archetype):
    """Wrapper for multiprocessing version of fitz.
    """
    try:
        # Unpack targets from shared memory
        for tg in target_data:
            tg.sharedmem_unpack()
        results = list()
        for i, tg in enumerate(target_data):
            zfit = fitz(chi2[i], t.template.redshifts, tg.spectra,
                t.template, nminima=nminima, archetype=archetype)
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

def calc_deltachi2(chi2, z, dvlimit=None):
    '''
    Calculate chi2 differences, excluding candidates with close z

    Args:
        chi2 : array of chi2 values
        z : array of redshifts

    Options:
        dvlimit: exclude candidates that are closer than dvlimit [km/s]

    Note: The final target always has deltachi2=0.0 because we don't know
        what the next chi2 would have been.  This can also occur for the
        last N targets if all N of them are within dvlimit of each other.
    '''
    if dvlimit is None:
        dvlimit = constants.max_velo_diff

    deltachi2 = np.zeros(len(chi2))
    for i in range(len(chi2)-1):
        dv = get_dv(z[i+1:], z[i])
        ii = np.abs(dv)>dvlimit
        if np.any(ii):
            dchi2 = chi2[i+1:] - chi2[i]
            deltachi2[i] = np.min(dchi2[ii])

    return deltachi2

def zfind(targets, templates, mp_procs=1, nminima=3, archetypes=None, priors=None, chi2_scan=None):
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
        mp_procs (int, optional): if not using MPI, this is the number of
            multiprocessing processes to use.
        nminima (int, optional): number of chi^2 minima to consider.
            Passed to fitz().
        archetypes (str, optional): file or directory containing archetypes
            to use for final fitz choice of best chi2 vs. z minimum.
        priors (str, optional): file containing redshift priors
        chi2_scan (str, optional): file containing already computed chi2 scan

    Returns:
        tuple: (allresults, allzfit), where "allresults" is a dictionary of the
            full chi^2 fit information, suitable for writing to a redrock scan
            file.  "allzfit" is an astropy Table of only the best fit parameters
            for a limited set of minima.

    """

    if archetypes:
        archetypes = All_archetypes(archetypes_dir=archetypes).archetypes

    if not priors is None:
        priors = Priors(priors)

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

    #- 75% on runtime
    # Compute the coarse-binned chi2 for all local targets.
    if chi2_scan is None:
        results = calc_zchi2_targets(targets, templates, mp_procs=mp_procs)
    else:
        results = read_zscan_redrock(chi2_scan)

    # Apply redshift prior
    if not priors is None:
        for tg in results.keys():
            for ft in results[tg].keys():
                results[tg][ft]['zchi2'] += priors.eval(tg, results[tg][ft]['redshifts'])

    # For each of our local targets, refine the redshift fit close to the
    # minima in the coarse fit.

    sort = np.array([ t.template.full_type for t in templates]).argsort()
    for t in np.array(list(templates))[sort]:
        ft = t.template.full_type
        if archetypes:
            archetype = archetypes[t.template._rrtype]
        else:
            archetype = None

        if am_root:
            print("  Finding best fits for template {}"\
                .format(t.template.full_type))
            sys.stdout.flush()

        start = elapsed(None, "", comm=t.comm)

        # Here we have another parallelization choice between MPI and
        # multiprocessing.

        #- 25% of runtime
        if targets.comm is not None:
            # MPI case.  Every process just works with its local targets.
            for tg in targets.local():
                zfit = fitz(results[tg.id][ft]['zchi2'] \
                    + results[tg.id][ft]['penalty'],
                    t.template.redshifts, tg.spectra,
                    t.template, nminima=nminima,archetype=archetype)
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
                    target_data, t, nminima, qout, archetype))
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

        elapsed(start, "    Finished in", comm=t.comm)

    # Add the target metadata to the results

    for tg in targets.local():
        results[tg.id]['meta'] = tg.meta

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

        # for t in templates:
        #     ft = t.template.full_type

        allzfit = list()
        for tid in targets.all_target_ids:
            tzfit = list()
            for fulltype in allresults[tid]:
                if fulltype == 'meta':
                    continue
                tmp = allresults[tid][fulltype]['zfit']

                #- TODO: reconsider fragile parsing of fulltype
                if archetypes is None:
                    if fulltype.count(':::') > 0:
                        spectype, subtype = fulltype.split(':::')
                    else:
                        spectype, subtype = (fulltype, '')
                else:
                    spectype = [ el.split(':::')[0] for el in tmp['fulltype'] ]
                    subtype = [ el.split(':::')[1] for el in tmp['fulltype'] ]
                    tmp.remove_column('fulltype')
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
            tzfit['zwarn'][ tzfit['npixels']==0 ] |= ZW.NODATA
            tzfit['zwarn'][ (tzfit['npixels']<10*tzfit['ncoeff']) ] |= \
                ZW.LITTLE_COVERAGE
            if archetypes:
                tzfit['zwarn'][ tzfit['coeff'][:,0]<=0. ] |= ZW.NEGATIVE_MODEL

            #- set ZW.SMALL_DELTA_CHI2 flag
            tzfit['deltachi2'] = calc_deltachi2(tzfit['chi2'], tzfit['z'])
            ii = (tzfit['deltachi2'] < constants.min_deltachi2)
            tzfit['zwarn'][ii] |= ZW.SMALL_DELTA_CHI2

            for i in range(len(tzfit)-1):
                noti = (np.arange(len(tzfit))!=i)
                alldeltachi2 = np.absolute(tzfit['chi2'][noti]-tzfit['chi2'][i])
                alldv = np.absolute(get_dv(z=tzfit['z'][noti],
                    zref=tzfit['z'][i]))
                zwarn = np.any( (alldeltachi2<constants.min_deltachi2) & \
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

            tzfit['znum'] = np.arange(len(tzfit))

            # Store
            allzfit.append(tzfit)

        allzfit = astropy.table.vstack(allzfit)

        # Cosmetic: move TARGETID to be first column as primary key
        try:
            allzfit.columns.move_to_end('targetid', last=False)
        except:
            # Must be using python2, don't mess with the order.
            pass

        # Now we have the final table of best fit results.  We want to add any
        # extra columns from the target metadata.  We assume that the meta keys
        # for the first target are the same keys for all targets...

        zfitids = list(allzfit['targetid'])
        allmetakeys = list(sorted(allresults[zfitids[0]]['meta'].keys()))

        # Parse any type information for the metadata.

        typepat = re.compile(r'(.*)_datatype')
        metakeys = list()
        metatypes = dict()
        for mk in allmetakeys:
            mat = typepat.match(mk)
            if mat is None:
                # this is a real key
                metakeys.append(mk)
            else:
                # get the data type
                metatypes[mat.group(1)] = allresults[zfitids[0]]['meta'][mk]
        for mk in metakeys:
            if mk not in metatypes:
                metatypes[mk] = None

        # Append the columns

        for mk in metakeys:
            if metatypes[mk] is not None:
                col = astropy.table.Column(np.array([ \
                    allresults[x]['meta'][mk] for x in zfitids ],
                    dtype=metatypes[mk]), name=mk)
            else:
                col = astropy.table.Column([ allresults[x]['meta'][mk] \
                    for x in zfitids ], name=mk)
            allzfit.add_column(col)

        # Remove the meta data from the dictionary, so that it is not later
        # interpreted as a template type.

        for tid in targets.all_target_ids:
            del allresults[tid]['meta']

        # Standardize column sizes
        if allzfit['subtype'].dtype != '<U20':
            allzfit.replace_column('subtype', allzfit['subtype'].astype('<U20'))

        if allzfit['spectype'].dtype != '<U6':
            allzfit.replace_column('spectype',allzfit['spectype'].astype('<U6'))

        maxcoeff = np.max([t.template.nbasis for t in templates])
        ntarg, ncoeff = allzfit['coeff'].shape
        if ncoeff != maxcoeff:
            coeff = np.zeros((ntarg, maxcoeff), dtype=allzfit['coeff'].dtype)
            coeff[:,0:ncoeff] = allzfit['coeff']
            allzfit.replace_column('coeff', coeff)

    return allresults, allzfit
