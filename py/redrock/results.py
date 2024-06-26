"""
redrock.results
===============

Functions for reading and writing full redrock results to HDF5.
"""

from __future__ import absolute_import, division, print_function

import os
import os.path
import numpy as np
from astropy.table import Table

from .utils import encode_column
from .templates import parse_fulltype

def write_zscan(filename, zscan, zfit, clobber=False):
    """Writes redrock.zfind results to a file.

    The nested dictionary structure of results is mapped into a nested
    group structure of the HDF5 file:

    /targetids[nt]
    /zscan/{spectype}/redshifts[nz]
    /zscan/{spectype}/zchi2[nt, nz]
    /zscan/{spectype}/penalty[nt, nz]
    /zscan/{spectype}/zcoeff[nt, nz, nc] or zcoeff[nt, nc, nz] ?
    /zfit/{targetid}/zfit table...

    Args:
        filename (str): the output file path.
        zscan (dict): the full set of fit results.
        zfit (Table): the best fit redshift results.
        clobber (bool): if True, delete the file if it exists.

    """
    import h5py
    filename = os.path.expandvars(filename)
    if clobber and os.path.exists(filename):
        os.remove(filename)

    outdir = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    zfit = zfit.copy()

    #- convert unicode to byte strings
    for colname in ('spectype', 'subtype', 'fitmethod'):
        if colname in zfit.columns:
            if zfit[colname].dtype.kind == 'U':
                zfit.replace_column(colname, np.char.encode(zfit[colname], 'ascii'))

    zbest = zfit[zfit['znum'] == 0]
    zbest.remove_column('znum')

    zbest.write(filename, path='zbest', format='hdf5')

    targetids = np.asarray(zbest['targetid'])
    spectypes = list(zscan[targetids[0]].keys())

    tempfile = filename + '.tmp'
    fx = h5py.File(tempfile, mode='w')
    fx['targetids'] = targetids

    for spectype in spectypes:
        zchi2 = np.vstack([zscan[t][spectype]['zchi2'] for t in targetids])
        penalty = np.vstack([zscan[t][spectype]['penalty'] for t in targetids])
        zcoeff = list()
        for t in targetids:
            tmp = zscan[t][spectype]['zcoeff']
            tmp = tmp.reshape((1,)+tmp.shape)
            zcoeff.append(tmp)
        zcoeff = np.vstack(zcoeff)
        fx['zscan/{}/zchi2'.format(spectype)] = zchi2
        fx['zscan/{}/penalty'.format(spectype)] = penalty
        fx['zscan/{}/zcoeff'.format(spectype)] = zcoeff
        fx['zscan/{}/redshifts'.format(spectype)] = \
            zscan[targetids[0]][spectype]['redshifts']

    for targetid in targetids:
        ii = np.where(zfit['targetid'] == targetid)[0]
        fx['zfit/{}/zfit'.format(targetid)] = zfit[ii].as_array()
        #- TODO: fx['zfit/{}/model']

    fx.close()
    os.rename(tempfile, filename)


def read_zscan(filename, select_targetids=None):
    """Read redrock.zfind results from a file.

    Args:
        filename (str): redrock details (.h5) filename
        select_targetids (list): array of TARGETIDs to read

    Returns:
        tuple: ``(zscan, zfit)`` where
        ``zfit`` is a Table with the N best fits for each target
        per spectype and subtype; and
        ``zscan`` is a nested dictionary ``zscan[targetid][templatetype]``
        with keys:

                - redshifts: array of redshifts scanned
                - zchi2: array of chi2 of fit vs. redshifts
                - penalty: array of chi2 penalties for unphysical fits at each z
                - zcoeff: array of coefficients fit per redshifts
                - zfit: table of N best-fit redshifts for just this target and templatetype
    """
    import h5py
    # zbest = Table.read(filename, format='hdf5', path='zbest')
    with h5py.File(os.path.expandvars(filename), mode='r') as fx:
        targetids = fx['targetids'][()]  # .value
        fulltypes = list(fx['zscan'].keys())

        if select_targetids is not None:
            indx = np.where(np.isin(targetids, select_targetids))[0]
        else:
            indx = np.arange(len(targetids))

        zscan = dict()
        for targetid in targetids[indx]:
            zscan[targetid] = dict()
            for fulltype in fulltypes:
                zscan[targetid][fulltype] = dict()

        for fulltype in fulltypes:
            spectype, subtype = parse_fulltype(fulltype)
            # blat[()] is obtuse syntax for what used to be clear blat.value
            zchi2 = fx['/zscan/{}/zchi2'.format(fulltype)][()]
            penalty = fx['/zscan/{}/penalty'.format(fulltype)][()]
            zcoeff = fx['/zscan/{}/zcoeff'.format(fulltype)][()]
            redshifts = fx['/zscan/{}/redshifts'.format(fulltype)][()]
            for i, targetid in zip(indx, targetids[indx]):
                zscan[targetid][fulltype]['redshifts'] = redshifts
                zscan[targetid][fulltype]['zchi2'] = zchi2[i]
                zscan[targetid][fulltype]['penalty'] = penalty[i]
                zscan[targetid][fulltype]['zcoeff'] = zcoeff[i]
                thiszfit = fx['/zfit/{}/zfit'.format(targetid)][()]
                ii = (thiszfit['spectype'].astype('U') == spectype)
                ii &= (thiszfit['subtype'].astype('U') == subtype)
                thiszfit = Table(thiszfit[ii])
                thiszfit.remove_columns(['targetid', 'znum', 'deltachi2'])
                thiszfit.replace_column('spectype',
                    encode_column(thiszfit['spectype']))
                thiszfit.replace_column('subtype',
                    encode_column(thiszfit['subtype']))
                zscan[targetid][fulltype]['zfit'] = thiszfit

        zfit = [fx['zfit/{}/zfit'.format(tid)][()] for tid in targetids[indx]]
        zfit = Table(np.hstack(zfit))
        zfit.replace_column('spectype', encode_column(zfit['spectype']))
        zfit.replace_column('subtype', encode_column(zfit['subtype']))

    return zscan, zfit


def read_zscan_redrock(filename):
    """Read redrock.zfind results from a file to be reused by redrock itself.

    Returns:
        dict: dictionary of results for each local target ID.
            dic.keys() are TARGETID
            dic[tg].keys() are TEMPLATE
            dic[tg][ft].keys() are ['penalty', 'zcoeff', 'zchi2', 'redshifts']

    """
    import h5py

    with h5py.File(os.path.expandvars(filename), mode='r') as fx:
        targetids = fx['targetids'][()]
        spectypes = list(fx['zscan'].keys())

        tmp_results = { ft:
            {'redshifts':fx[f'/zscan/{ft}/redshifts'][()],
            'zchi2':fx[f'/zscan/{ft}/zchi2'][()],
            'penalty':fx[f'/zscan/{ft}/penalty'][()],
            'zcoeff':fx[f'/zscan/{ft}/zcoeff'][()]}
            for ft in spectypes}
        results = { tg:{ ft:
            {'redshifts':tmp_results[ft]['redshifts'],
            'zchi2':tmp_results[ft]['zchi2'][i],
            'penalty':tmp_results[ft]['penalty'][i],
            'zcoeff':tmp_results[ft]['zcoeff'][i]}
            for ft in spectypes } for i, tg in enumerate(targetids) }

    return results
