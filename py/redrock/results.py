"""
Functions for reading and writing full redrock results to HDF5.
"""

from __future__ import absolute_import, division, print_function

import os
import os.path
import numpy as np
from astropy.table import Table

from .utils import encode_column


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
    zfit.replace_column('spectype', np.char.encode(zfit['spectype'], 'ascii'))
    zfit.replace_column('subtype', np.char.encode(zfit['subtype'], 'ascii'))

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
        # "fulltype" is a keyname in Archetypes files
        #HDF5 conflicts with 'U23' string type, so defining a list and save them as dictionary in the final hdf5 file
        #Therefore to read the content use: data = f["fulltype"]["TARGETID"]["fulltype"][()] and it should work
        if "fulltype" in zfit[ii].dtype.names:
            arch_dict = {} # dictionary that will save archetypes "fulltype" information
            arch_dict['fulltype'] = zfit[ii]['fulltype'].tolist()
            fx['fulltype/{}/fulltype'.format(targetid)] = np.string_(arch_dict)
            zfit[ii].remove_columns(['fulltype']) #removing this column from the table
        fx['zfit/{}/zfit'.format(targetid)] = zfit[ii].as_array()

    fx.close()
    os.rename(tempfile, filename)


def read_zscan(filename):
    """Read redrock.zfind results from a file.

    Returns:
        tuple: (zbest, results) where zbest is a Table with keys TARGETID, Z,
            ZERR, ZWARN and results is a nested dictionary
            results[targetid][templatetype] with keys:

                - z: array of redshifts scanned
                - zchi2: array of chi2 fit at each z
                - penalty: array of chi2 penalties for unphysical fits at each z
                - zbest: best fit redshift (finer resolution fit around zchi2
                    min)
                - minchi2: chi2 at zbest
                - zerr: uncertainty on zbest
                - zwarn: 0=good, non-0 is a warning flag

    """
    import h5py
    # zbest = Table.read(filename, format='hdf5', path='zbest')
    with h5py.File(os.path.expandvars(filename), mode='r') as fx:
        targetids = fx['targetids'][()]  # .value
        spectypes = list(fx['zscan'].keys())

        zscan = dict()
        for targetid in targetids:
            zscan[targetid] = dict()
            for spectype in spectypes:
                zscan[targetid][spectype] = dict()

        for spectype in spectypes:
            # blat[()] is obtuse syntax for what used to be clear blat.value
            zchi2 = fx['/zscan/{}/zchi2'.format(spectype)][()]
            penalty = fx['/zscan/{}/penalty'.format(spectype)][()]
            zcoeff = fx['/zscan/{}/zcoeff'.format(spectype)][()]
            redshifts = fx['/zscan/{}/redshifts'.format(spectype)][()]
            for i, targetid in enumerate(targetids):
                zscan[targetid][spectype]['redshifts'] = redshifts
                zscan[targetid][spectype]['zchi2'] = zchi2[i]
                zscan[targetid][spectype]['penalty'] = penalty[i]
                zscan[targetid][spectype]['zcoeff'] = zcoeff[i]
                thiszfit = fx['/zfit/{}/zfit'.format(targetid)][()]
                ii = (thiszfit['spectype'].astype('U') == spectype)
                thiszfit = Table(thiszfit[ii])
                thiszfit.remove_columns(['targetid', 'znum', 'deltachi2'])
                thiszfit.replace_column('spectype',
                    encode_column(thiszfit['spectype']))
                thiszfit.replace_column('subtype',
                    encode_column(thiszfit['subtype']))
                zscan[targetid][spectype]['zfit'] = thiszfit

        zfit = [fx['zfit/{}/zfit'.format(tid)][()] for tid in targetids]
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
