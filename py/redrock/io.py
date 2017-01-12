from __future__ import absolute_import, division, print_function

import sys
from glob import glob
import os.path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import Template

#- for python 3 compatibility
if sys.version_info.major > 2:
    basestring = str

#- From https://github.com/desihub/desispec io.util.native_endian
def native_endian(data):
    """Convert numpy array data to native endianness if needed.

    Returns new array if endianness is swapped, otherwise returns input data

    Context:
    By default, FITS data from astropy.io.fits.getdata() are not Intel
    native endianness and scipy 0.14 sparse matrices have a bug with
    non-native endian data.
    """
    if data.dtype.isnative:
        return data
    else:
        return data.byteswap().newbyteorder()

def read_template(filename):
    '''
    Read template from filename
    
    Returns a Template object
    '''
    if os.path.exists(filename):
        fx = fits.open(filename, memmap=False)
    else:
        xfilename = os.path.join(os.getenv('RR_TEMPLATE_DIR'), filename)
        if os.path.exists(xfilename):
            fx = fits.open(xfilename, memmap=False)
        else:
            raise IOError('unable to find '+filename)

    hdr = fx['BASIS_VECTORS'].header
    wave = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(hdr['NAXIS1'])
    if 'LOGLAM' in hdr and hdr['LOGLAM'] != 0:
        wave = 10**wave

    flux = native_endian(fx['BASIS_VECTORS'].data)
    fx.close()

    rrtype = hdr['RRTYPE'].strip().upper()
    if rrtype == 'GALAXY':
        ### redshifts = 10**np.arange(np.log10(1+0.005), np.log10(1+2.0), 1.5e-4) - 1
        redshifts = 10**np.arange(np.log10(1+0.005), np.log10(1+2.0), 3e-4) - 1
    elif rrtype == 'STAR':
        redshifts = np.arange(-0.001, 0.00101, 0.0001)
    elif rrtype == 'QSO':
        redshifts = 10**np.arange(np.log10(1+0.5), np.log10(1+4.0), 5e-4) - 1
        redshifts = 10**np.arange(np.log10(1+0.5), np.log10(1+4.0), 5e-4) - 1
    else:
        raise ValueError('Unknown redshift range to use for template type {}'.format(rrtype))

    return Template(rrtype, redshifts, wave, flux)

def find_templates(template_dir=None):
    '''
    Return list of redrock-*.fits template files
    
    Search directories in this order, returning results from first one found:
        * template_dir
        * $RR_TEMPLATE_DIR
        * {redrock_code}/templates/
    '''
    if template_dir is None:
        if 'RR_TEMPLATE_DIR' in os.environ:
            template_dir = os.environ['RR_TEMPLATE_DIR']
        else:
            thisdir = os.path.dirname(__file__)
            tempdir = os.path.join(os.path.abspath(thisdir), 'templates')
            if os.path.exists(tempdir):
                template_dir = tempdir
        
    if template_dir is None:
        raise IOError("ERROR: can't find template_dir, $RR_TEMPLATE_DIR, or {rrcode}/templates/")

    return glob(os.path.join(template_dir, 'rrtemplate-*.fits'))

def read_templates(template_list=None, template_dir=None):
    '''
    Return a list of templates from the files in template_list
    
    If template_list is None, use list from find_templates(template_dir)
    If template_list is a filename, return 1-element list with that template
    '''
    if template_list is None:
        template_list = find_templates(template_dir)

    templates = list()
    if isinstance(template_list, basestring):
        templates.append(read_template(template_list))
    else:
        for tfile in template_list:
            templates.append(read_template(tfile))
    
    if len(templates) == 0:
        raise IOError('No templates found')
    
    return templates

def write_zscan(filename, zscan, zfit, clobber=False):
    '''
    Writes redrock.zfind results to filename
    
    The nested dictionary structure of results is mapped into a nested
    group structure of the HDF5 file:

    TODO: document structure
    
    /targetids[nt]
    /zscan/{spectype}/redshifts[nz]
    /zscan/{spectype}/zchi2[nt, nz]
    /zscan/{spectype}/zcoeff[nt, nz, nc] or zcoeff[nt, nc, nz] ?
    /zfit/{targetid}/zfit table...
    
    if clobber=True, replace pre-existing file
    '''
    import h5py
    if clobber and os.path.exists(filename):
        os.remove(filename)

    zfit = zfit.copy()
    zfit.replace_column('spectype', np.char.encode(zfit['spectype'], 'ascii'))

    zbest = zfit[zfit['znum'] == 0]
    zbest.remove_column('znum')
        
    zbest.write(filename, path='zbest', format='hdf5')

    targetids = np.asarray(zbest['targetid'])
    spectypes = list(zscan[targetids[0]].keys())

    fx = h5py.File(filename)
    fx['targetids'] = targetids

    for spectype in spectypes:
        zchi2 = np.vstack([zscan[t][spectype]['zchi2'] for t in targetids])
        zcoeff = list()
        for t in targetids:
            tmp = zscan[t][spectype]['zcoeff']
            tmp = tmp.reshape((1,)+tmp.shape)
            zcoeff.append(tmp)
        zcoeff = np.vstack(zcoeff)
        fx['zscan/{}/zchi2'.format(spectype)] = zchi2
        fx['zscan/{}/zcoeff'.format(spectype)] = zcoeff
        fx['zscan/{}/redshifts'.format(spectype)] = zscan[targetids[0]][spectype]['redshifts']

    for targetid in targetids:
        ii = np.where(zfit['targetid'] == targetid)[0]
        fx['zfit/{}/zfit'.format(targetid)] = zfit[ii].as_array()
        #- TODO: fx['zfit/{}/model']

    fx.close()
    
def read_zscan(filename):
    '''Return redrock.zfind results stored in hdf5 file as written
    by write_zscan
    
    returns (zbest, results) tuple:
        zbest is a Table with keys TARGETID, Z, ZERR, ZWARN
        results is a nested dictionary results[targetid][templatetype] with keys
            - z: array of redshifts scanned
            - zchi2: array of chi2 fit at each z
            - zbest: best fit redshift (finer resolution fit around zchi2 min)
            - minchi2: chi2 at zbest
            - zerr: uncertainty on zbest
            - zwarn: 0=good, non-0 is a warning flag    
    '''
    import h5py
    # zbest = Table.read(filename, format='hdf5', path='zbest')
    with h5py.File(filename, mode='r') as fx:
        targetids = fx['targetids'].value
        spectypes = list(fx['zscan'].keys())
    
        zscan = dict()
        for targetid in targetids:
            zscan[targetid] = dict()
            for spectype in spectypes:
                zscan[targetid][spectype] = dict()

        for spectype in spectypes:
            zchi2 = fx['/zscan/{}/zchi2'.format(spectype)].value
            zcoeff = fx['/zscan/{}/zcoeff'.format(spectype)].value
            redshifts = fx['/zscan/{}/redshifts'.format(spectype)].value
            for i, targetid in enumerate(targetids):
                zscan[targetid][spectype]['redshifts'] = redshifts
                zscan[targetid][spectype]['zchi2'] = zchi2[i]
                zscan[targetid][spectype]['zcoeff'] = zcoeff[i]
                thiszfit = fx['/zfit/{}/zfit'.format(targetid)].value
                ii = (thiszfit['spectype'].astype('U') == spectype)
                thiszfit = Table(thiszfit[ii])
                thiszfit.remove_columns(['targetid', 'znum', 'deltachi2'])
                thiszfit.replace_column('spectype', _encode_column(thiszfit['spectype']))
                zscan[targetid][spectype]['zfit'] = thiszfit
    
        zfit = [fx['zfit/{}/zfit'.format(tid)].value for tid in targetids]
        zfit = Table(np.hstack(zfit))
        zfit.replace_column('spectype', _encode_column(zfit['spectype']))

    return zscan, zfit

def _encode_column(c):
    '''Returns a bytes column encoded into a string column'''
    return c.astype((str, c.dtype.itemsize))

            
    