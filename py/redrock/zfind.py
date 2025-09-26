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

from .utils import elapsed, extra_columns_in_archetype_mode

from .targets import distribute_targets

from .templates import parse_fulltype

from .archetypes import All_archetypes

from .priors import Priors

from .results import read_zscan_redrock

from .zscan import calc_zchi2_targets

from .fitz import fitz, get_dv

from .zwarning import ZWarningMask as ZW
from .zwarning import badfit_mask

def sort_dict_by_col(d, colname):
    """Sort a dict of np.ndarrays by one key.
    Replacement for astropy.Table.sort
    """
    if (not colname in d):
        raise KeyError('Key '+str(colname)+' is not in dictionary')
    for k in d.keys():
        if (type(d[k]) is not np.ndarray):
            raise ValueError('Column '+str(k)+' is not an np.array')
    idx = d[colname].argsort(0).flatten()
    for k in d.keys():
        d[k] = d[k][idx]
    return

def sort_dict_by_cols(d, colnames, sort_first_column_first = True):
    """Sort a dict of np.ndarrays by multiple keys
    Replacement for astropy.Table.sort

    Args:
        d : a dictionary
        colnames : a tuple or list of column names
        sort_first_column_first : boolean - np.lexsort((a,b)) will sort by b
            first and then sort by a.  This is the opposite of
            astropy.Table.sort behavior.  Setting this to true will ensure
            that sort_dict_by_cols(d, ('a','b')) will result in sorting by a
            first and then b.
    """
    for c in colnames:
        if (not c in d):
            raise KeyError('Key '+str(c)+' is not in dictionary')
    for k in d.keys():
        if (type(d[k]) is not np.ndarray):
            raise ValueError('Column '+str(k)+' is not an np.array')
    valsToSort = ()
    for c in colnames:
        valsToSort += (d[c],)
    if (sort_first_column_first):
        #Reverse order of valsToSort
        valsToSort = valsToSort[::-1]
    idx = np.lexsort(valsToSort, axis=0).flatten()
    for k in d.keys():
        d[k] = d[k][idx]
    return

def _mp_fitz(chi2, target_data, t, nminima, qout, archetype, use_gpu, deg_legendre, zminfit_npoints, per_camera, n_nearest, prior_sigma):
    """Wrapper for multiprocessing version of fitz.
    """
    try:
        # Unpack targets from shared memory
        for tg in target_data:
            tg.sharedmem_unpack()
        results = list()
        for i, tg in enumerate(target_data):
            zfit = fitz(chi2[i], t.template.redshifts, tg,
                t.template, nminima=nminima, archetype=archetype, use_gpu=use_gpu, zminfit_npoints=zminfit_npoints, per_camera=per_camera, deg_legendre=deg_legendre, n_nearest=n_nearest, prior_sigma=prior_sigma)
            results.append( (tg.id, zfit) )
        qout.put(results)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        lines = [ "MP calc_zchi2: {}".format(x) for x in lines ]
        print("".join(lines))
        sys.stdout.flush()

def calc_deltachi2(chi2, z, zwarn, dvlimit=constants.max_velo_diff):
    '''
    Calculate chi2 differences, excluding candidates with close z or bad fits

    Args:
        chi2 : array of chi2 values
        z : array of redshifts
        zwarn : array of zwarn values

    Options:
        dvlimit: exclude candidates that are closer than dvlimit [km/s],
                uses minumum value of the pair

    Returns (deltachi2, setzwarn) where `deltachi2` is array of chi2 differences
        to next best good fit, and `setzwarn` is boolean array of whether
        a SMALL_DELTACHI2 zwarn bit should be set.

    Note: The final target always has deltachi2=0.0 because we don't know
        what the next chi2 would have been.  This can also occur for the
        last N targets if all N of them are within dvlimit of each other.
    '''
    nz = len(chi2)
    deltachi2 = np.zeros(nz)
    okfit = (zwarn & badfit_mask) == 0

    # if no dvlimit passed, use constant threshold
    if np.isscalar(dvlimit):
        dvlimit = np.full(nz, dvlimit)

    for i in range(len(chi2)-1):
        dv = get_dv(z[i+1:], z[i])
        ii = (np.abs(dv)>np.minimum(dvlimit[i],dvlimit[i+1:])) & okfit[i+1:]
        if np.any(ii):
            dchi2 = chi2[i+1:] - chi2[i]
            deltachi2[i] = np.min(dchi2[ii])

    #- zwarn SMALL_DELTA_CHI2 is based upon small difference to any good fit,
    #- including a slightly better one
    noti = np.ones(nz, dtype=bool)
    setzwarn = np.zeros(nz, dtype=bool)
    for i in range(nz):
        noti[:] = True
        noti[i] = False
        alldeltachi2 = np.absolute(chi2[noti] - chi2[i])
        alldv = np.absolute(get_dv(z=z[noti], zref=z[i]))
        alldvlimit = np.minimum(dvlimit[i], dvlimit[noti])
        zwarn = np.any( okfit[noti] &
                    (alldeltachi2 < constants.min_deltachi2) &
                    (alldv >= alldvlimit) )
        if zwarn:
            setzwarn[i] = True

    return deltachi2, setzwarn

def _rebalance_after_scan(targets, results):
    """Helper for rebalancing targets and results after lopsided zscan
    """

    # gather lopsided results and targets on rank 0 for rebalancing
    results = targets.comm.gather(results, root=0)
    lopsided_targets = targets.comm.gather(targets.local(), root=0)

    if targets.comm.rank == 0:
        # Flatten lopsided distributed targets (list of list of targets)
        flattened_targets = [t for sl in lopsided_targets for t in sl]
        # Split targets into approximately equal lengths sublists
        ix = np.array_split(np.arange(len(flattened_targets)), len(lopsided_targets))
        #dist_targets = [flattened_targets[i[0]:i[0] + len(i)] for i in ix]
        #Less elegantly create a list of lists, appending empty lists where
        #len(targets) == 0 in the case of MPI ranks > ntargets
        dist_targets = []
        for i in ix:
            if len(i) == 0:
                dist_targets.append([])
            else:
                dist_targets.append(flattened_targets[i[0]:i[0] + len(i)])
        # Merge list of result dictionaries
        results = {k: v for d in results for k, v in d.items()}
        # Split results using rebalanced target lists
        dist_results = [{t.id: results[t.id] for t in s} for s in dist_targets]
    else:
        dist_targets = None
        dist_results = None

    # distribute rebalance targets and results
    local_targets = targets.comm.scatter(dist_targets, root=0)
    results = targets.comm.scatter(dist_results, root=0)

    return local_targets, results

def sort_zfit(zfit):
    """
    Sorts zfit table by goodness of fit, using 'zwarn' and 'chi2' columns

    Args:
        zfit: astropy Table with columns 'zwarn' and 'chi2'

    Modifies zfit in-place by sorting it
    """
    zfit['__badfit__'] = (zfit['zwarn'] & badfit_mask) != 0
    zfit.sort( ('__badfit__', 'chi2') )
    zfit.remove_column('__badfit__')

def sort_zfit_dict(zfit):
    """
    Sorts zfit dict by goodness of fit, using 'zwarn' and 'chi2' columns

    Args:
        zfit: dict of numpy arrays with columns 'zwarn' and 'chi2'

    Modifies zfit in-place by sorting it
    """

    zfit['__badfit__'] = (zfit['zwarn'] & badfit_mask) != 0
    sort_dict_by_cols(zfit, ('__badfit__', 'chi2'), sort_first_column_first=True)
    zfit.pop('__badfit__')

def zfind(targets, templates, mp_procs=1, nminima=3, archetypes=None, priors=None, chi2_scan=None, use_gpu=False,
          zminfit_npoints=15, per_camera=None, deg_legendre=None, n_nearest=None, prior_sigma=None, ncamera=None):
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
        archetypes (list optional): list of Archetype objects
            to use for final fitz choice of best chi2 vs. z minimum.
        priors (str, optional): file containing redshift priors
        chi2_scan (str, optional): file containing already computed chi2 scan
        use_gpu (bool, optional): use gpu for calc_zchi2
        deg_legendre (int): in archetype mode polynomials upto deg_legendre-1 will be used
        zminfit_npoints (int): number of finer redshift pixels to search for final redshift
        per_camera: (bool): True if fitting needs to be done in each camera for archetype mode
        n_nearest (int): number of nearest neighbours to be used in chi2 space (including best archetype)
        prior_sigma (float): prior to add in the final solution matrix: added as 1/(prior_sigma**2) only for per-camera mode
        zminfit_npoints (int): number of minimum redshift to be fit for final redshift estimation

    Returns:
        tuple: (allresults, allzfit), where "allresults" is a dictionary of the
            full chi^2 fit information, suitable for writing to a redrock scan
            file.  "allzfit" is an astropy Table of only the best fit parameters
            for a limited set of minima.

    """

    if archetypes:
        archetype_spectype = list(archetypes.keys()) # to account for the case if only one archetype is provided
        pca_map, zero_like_keys = extra_columns_in_archetype_mode()

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

    # Compute the coarse-binned chi2 for all local targets.
    start_zscan = elapsed(None, "", comm=targets.comm)
    if chi2_scan is None:
        results = calc_zchi2_targets(targets, templates, mp_procs=mp_procs, use_gpu=use_gpu)
    else:
        results = read_zscan_redrock(chi2_scan)
    elapsed(start_zscan, "Scanning redshifts", comm=targets.comm)

    if (use_gpu):
        #If using GPU, copy template flux and wave arrays to cupy objects
        #on the GPU once here so it is not copied every iteration of
        #rebinning below.  Use gpuwave and gpuflux so as to not overwrite
        #numpy arrays.
        import cupy as cp
        for t in templates:
            t.template.gpuwave = cp.asarray(t.template.wave)
            t.template.gpuflux = cp.asarray(t.template.flux)

    # Note: rebalancing no longer needs to be done now that following steps
    # have been GPU-ized - CW 12/22
    # Note: GPU zscan accommodates lopsided distribution of targets but this
    # is not great for the following steps that have not been GPU-ified yet.
    # Rebalance targets and results before proceeded.
    #if hasattr(targets, 'is_lopsided') and targets.is_lopsided and targets.comm is not None:
    #    start_rebalance = elapsed(None, "", comm=targets.comm)
    #    local_targets, results = _rebalance_after_scan(targets, results)
    #    elapsed(start_rebalance, "Rebalancing targets", comm=targets.comm)
    #else:
    #    local_targets = targets.local()
    local_targets = targets.local()

    # Apply redshift prior
    if not priors is None:
        for tg in results.keys():
            for ft in results[tg].keys():
                results[tg][ft]['zchi2'] += priors.eval(tg, results[tg][ft]['redshifts'])

    # For each of our local targets, refine the redshift fit close to the
    # minima in the coarse fit.

    start_findbest = elapsed(None, "", comm=targets.comm)
    sort = np.array([ t.template.full_type for t in templates]).argsort()

    #creating list of template spectype to make use in case archetypes would be used
    all_spectype = []
    for spec in templates:
        if spec.template.template_type not in all_spectype:
            all_spectype.append(spec.template.template_type)

    for t in np.array(list(templates))[sort]:
        ft = t.template.full_type
        if archetypes:
            if t.template._rrtype in archetypes.keys():
                archetype = archetypes[t.template._rrtype]
            else:
                archetype = None
        else:
            archetype = None

        if am_root:
            print("  Finding best fits for template {}"\
                .format(t.template.full_type))
            sys.stdout.flush()

        start = elapsed(None, "", comm=targets.comm)

        # Here we have another parallelization choice between MPI and
        # multiprocessing.

        if targets.comm is not None or mp_procs == 1:
            # MPI case.  Every process just works with its local targets.
            for tg in local_targets:
                zfit = fitz(results[tg.id][ft]['zchi2'] \
                    + results[tg.id][ft]['penalty'],
                    t.template.redshifts, tg,
                    t.template, nminima=nminima,archetype=archetype, use_gpu=use_gpu, deg_legendre=deg_legendre, zminfit_npoints=zminfit_npoints, per_camera=per_camera, n_nearest=n_nearest, prior_sigma=prior_sigma)
                results[tg.id][ft]['zfit'] = zfit
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
                    target_data, t, nminima, qout, archetype, use_gpu, deg_legendre, zminfit_npoints, per_camera, n_nearest, prior_sigma))
                procs.append(p)
                p.start()

            # Extract the output
            for i in range(mp_procs):
                if len(mpdist[i]) == 0:
                    continue
                res = qout.get()
                for rs in res:
                    results[rs[0]][ft]['zfit'] = rs[1]

        elapsed(start, "    Finished in", comm=targets.comm)
    elapsed(start_findbest, "Finding best redshift", comm=targets.comm)

    # Add the target metadata to the results

    start_finalize = elapsed(None, "", comm=targets.comm)
    for tg in local_targets:
        results[tg.id]['meta'] = tg.meta

    # Gather our results to the root process and split off the zfit data.
    # Only process zero returns data- other ranks return None.

    allresults = None
    allzfit = None

    if targets.comm is not None:
        # gather failes with results>2GB, so use point-to-point instead
        ### results = targets.comm.gather(results, root=0)

        # Rank 0 receives results in batches from each of the other ranks;
        # this preserves the same order as comm.gather
        if targets.comm.rank == 0:
            results = [results,]
            for other_rank in range(1,targets.comm.size):

                # first get the number of batches expected
                num_batches = targets.comm.recv(source=other_rank)

                # then collect those batches
                other_results = dict()
                for _ in range(num_batches):
                    other_results.update( targets.comm.recv(source=other_rank) )

                results.append(other_results)

        # Ranks>0 send results to rank 0
        # first send num_batches (int), and then the batches of results (dict)
        else:
            max_targets_per_send = 5000
            if len(results) < max_targets_per_send:
                targets.comm.send(1, dest=0)
                targets.comm.send(results, dest=0)
            else:
                # first send the number of batches that will be sent
                num_batches = (len(results)-1) // max_targets_per_send + 1
                targets.comm.send(num_batches, dest=0)

                # split into (targetid, values) items for dict "slicing"
                result_items = list(results.items())
                for i in range(0, len(results), max_targets_per_send):
                    subresults = dict(result_items[i:i+max_targets_per_send])
                    targets.comm.send(subresults, dest=0)

        targets.comm.barrier()
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

                if archetypes is None:
                    spectype, subtype = parse_fulltype(fulltype)
                else:
                    if 'fulltype' in tmp.keys():#to take care of case when archetype is applied only for one template
                        spectype = list()
                        subtype = list()
                        # el is a list with one element (corresponding to each minimum)
                        for el in tmp['fulltype']:
                            this_spectype, this_subtype = parse_fulltype(el[0])
                            spectype.append(this_spectype)
                            subtype.append(this_subtype)
                        del tmp['fulltype'] #it's a dictionary
                    else:
                        spectype, subtype = parse_fulltype(fulltype)

                #Have to create arrays of correct length since using dict of
                #np arrays instead of astropy Table
                nmin = len(tmp['chi2'])

                if nmin == 0:
                    print(f'WARNING: no {fulltype} chi2 vs. z minima for targetid {tid}')
                    continue

                if np.isscalar(spectype):
                    tmp['spectype'] = np.full((nmin, 1), spectype)
                    tmp['subtype'] = np.full((nmin, 1), subtype)
                else:
                    assert len(spectype)==nmin
                    tmp['spectype'] = np.array([spectype]).reshape((nmin, 1))
                    tmp['subtype'] = np.array([subtype]).reshape((nmin, 1))

                tmp['ncoeff'] = np.array([tmp['coeff'].shape[1]]*nmin).reshape((nmin, 1))

                # set max_velo_diff differently for STARs, but watch out
                # for archtypes which have spectype as list instead of scalar
                if (np.isscalar(spectype) and spectype.upper() == 'STAR') or spectype[0].upper() == 'STAR':
                    max_velo_diff = constants.max_velo_diff_star
                else:
                    max_velo_diff = constants.max_velo_diff
                tmp['max_velo_diff'] = np.full((nmin,1), max_velo_diff)

                tzfit.append(tmp)
                del allresults[tid][fulltype]['zfit']

            maxncoeff = max([tmp['coeff'].shape[1] for tmp in tzfit])

            for tmp in tzfit:
                if tmp['coeff'].shape[1] < maxncoeff:
                    n = maxncoeff - tmp['coeff'].shape[1]
                    nx = tmp['coeff'].shape[0]
                    c = np.append(tmp['coeff'], np.zeros((nx, n)), axis=1)
                    tmp['coeff'] = c
            #tzfit = astropy.table.vstack(tzfit)
            ## Equivalent of astropy.table.vstack(tzfit) - vstack each array

            tzfit2 = dict()
            for k in tzfit[0].keys():
                tzfit2[k] = list()
                ref = tzfit[0][k]  # reference array
                for i in range(len(tzfit)):
                    if k in tzfit[i].keys():
                        val = tzfit[i][k]
                        if archetypes and k in pca_map:
                            # this is to make sure in case of archetypes for multiple spectype we have uniform size for arrays
                            val = np.pad(val, ((0,0),(0, max(0, ref.shape[1] - val.shape[1]))), constant_values=0)[:, :ref.shape[1]]
                    else:
                        row = tzfit[i]
                        if archetypes and k in pca_map and pca_map[k] in row:
                            val = row[pca_map[k]]
                            val = np.pad(val, ((0,0),(0, max(0, ref.shape[1] - val.shape[1]))), constant_values=0)[:, :ref.shape[1]]
                        elif k in zero_like_keys:
                            val = np.zeros_like(ref)
                    tzfit2[k].append(val)

                tzfit2[k] = np.vstack(tzfit2[k])
                if (tzfit2[k].shape[1] == 1):
                    tzfit2[k] = tzfit2[k].flatten()
            tzfit = tzfit2

            #Have to create arrays of correct length since using dict of
            #np arrays instead of astropy Table
            l = len(tzfit['chi2'])
            tzfit['targetid'] = np.array([tid]*l)

            if (archetypes):
                if (len(archetype_spectype)==len(all_spectype)):
                    if n_nearest is None:
                        if tzfit["coeff"].ndim==1: ## this makes sure that code doesn't fail if --archetypes-no-legendre is provided
                            tzfit["coeff"] = tzfit["coeff"].reshape(-1, 1)
                        ibad = tzfit['coeff'][:,0]<=0. # means that best archetype has negative model
                    else:
                        ibad = np.any(tzfit['coeff'][:,:n_nearest]<0, axis=1) # any archetype has negative coeff
                else:
                    ## need to check if this bitmask is only applied to objects for which archetypes are used
                    ibad = np.isin(tzfit['spectype'], np.array(archetype_spectype))
                    index_check = np.where(ibad)[0]
                    for k in index_check:
                        if n_nearest is None:
                            if tzfit['coeff'][:,0][k]>=0.: # don't reject physical model
                                ibad[k]=False
                        else:
                            if np.all(tzfit['coeff'][:,:n_nearest][k]>=0):
                                ibad[k]=False

                tzfit['zwarn'][ibad] |= ZW.NEGATIVE_MODEL

            tzfit['zwarn'][ tzfit['npixels']==0 ] |= ZW.NODATA
            tzfit['zwarn'][ (tzfit['npixels']<10*tzfit['ncoeff']) ] |= \
                ZW.LITTLE_COVERAGE

            #- Sort by badfit zwarn bits and chi2
            sort_zfit_dict(tzfit)

            # Trim down cases of multiple subtypes for a single type (e.g.
            # STARs) tzfit is already sorted by chi2, so keep first nminima of
            # each type.
            iikeep = list()
            for spectype in np.unique(tzfit['spectype']):
                ii = np.where(tzfit['spectype'] == spectype)[0]
                iikeep.extend(ii[0:nminima])


            if (len(iikeep) < l):
                for k in tzfit.keys():
                    tzfit[k] = tzfit[k][iikeep]
                #- grouping by spectype could get chi2 out of order; resort
                sort_zfit_dict(tzfit)

            #Length may have changed
            l = len(tzfit['chi2'])
            tzfit['znum'] = np.arange(l)

            #- calc deltachi2 and set ZW.SMALL_DELTA_CHI2 flag
            deltachi2, setzwarn = calc_deltachi2(tzfit['chi2'], tzfit['z'], tzfit['zwarn'],
                                                 dvlimit=tzfit['max_velo_diff'])
            tzfit['deltachi2'] = deltachi2
            tzfit['zwarn'][setzwarn] |= ZW.SMALL_DELTA_CHI2

            # remove max_velo_diff column
            del tzfit['max_velo_diff']

            # Store
            # Here convert to astropy table
            allzfit.append(astropy.table.Table(tzfit))

        allzfit = astropy.table.vstack(allzfit)
        #print(allzfit['targetid', 'z', 'zwarn', 'chi2', 'deltachi2','spectype','subtype'])
        # Cosmetic: move TARGETID to be first column as primary key
        try:
            allzfit.columns.move_to_end('targetid', last=False)
            for k in allzfit.colnames:
                if 'pca' in k.lower():
                    allzfit.columns.move_to_end(k)
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
            max_length = max(len(s) for s in allzfit['subtype']) # this is particularly important if ever use archetype nearest neighbour approach
            dtype = f'<U{max_length}'
            allzfit.replace_column('subtype', allzfit['subtype'].astype(dtype))

        if allzfit['spectype'].dtype != '<U6':
            allzfit.replace_column('spectype',allzfit['spectype'].astype('<U6'))

        if archetypes is None or not per_camera:
            maxcoeff = np.max([t.template.nbasis for t in templates])
        else:
            if n_nearest is not None:
                maxcoeff = max(np.max([t.template.nbasis for t in templates]), ncamera*(deg_legendre)+n_nearest)
            else:
                maxcoeff = max(np.max([t.template.nbasis for t in templates]), ncamera*(deg_legendre)+1)

        if allzfit['coeff'].ndim == 1:
            ntarg = allzfit['coeff'].shape
            ncoeff = 1
        else:
            ntarg, ncoeff = allzfit['coeff'].shape

        if ncoeff != maxcoeff:
            coeff = np.zeros((ntarg, maxcoeff), dtype=allzfit['coeff'].dtype)
            coeff[:,0:ncoeff] = allzfit['coeff']
            allzfit.replace_column('coeff', coeff)

        # this is just to save space in COEFF array in case multiple archetypes are used
        if len(np.unique(allzfit['ncoeff']))==1:
            maxcoeff = int(np.unique(allzfit['ncoeff']))
            coeff = np.zeros((ntarg, maxcoeff), dtype=allzfit['coeff'].dtype)
            coeff = allzfit['coeff'][:,0:maxcoeff]
            allzfit.replace_column('coeff', coeff)

    elapsed(start_finalize, "Finalizing results", comm=targets.comm)

    return allresults, allzfit
