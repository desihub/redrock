from __future__ import absolute_import, division, print_function

from glob import glob
import os.path

import numpy as np
from astropy.io import fits

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
    
    Returns dictionary with keys:
        wave : restframe wavelength array [Angstroms]
        flux : 2D basis set template flux[i, jwave]
        archetype_coeff : 2D[narchetypes, ncoeff] coefficients describing a
            set of archetypes
        type : string descripting template type
        subtype : string describing template subtype
    '''
    fx = fits.open(filename, memmap=False)

    hdr = fx['BASIS_VECTORS'].header
    wave = hdr['CRVAL1'] + hdr['CDELT1']*np.arange(hdr['NAXIS1'])
    if 'LOGLAM' in hdr and hdr['LOGLAM'] != 0:
        wave = 10**wave

    template = {
        'wave':  wave,
        'flux': native_endian(fx['BASIS_VECTORS'].data),
        'archetype_coeff': native_endian(fx['ARCHETYPE_COEFF'].data),
        'type': hdr['RRTYPE'].strip(),
        'subtype': hdr['RRSUBTYP'].strip(),
        }

    return template

def find_templates(template_dir=None):
    '''
    Return list of redrock-*.fits template files found in either
    template_dir or $RR_TEMPLATE_DIR
    '''
    if template_dir is None:
        template_dir = os.getenv('RR_TEMPLATE_DIR')
        
    if template_dir is None:
        raise IOError('ERROR: must provide template_dir or $RR_TEMPLATE_DIR')

    return glob(os.path.join(template_dir, 'redrock-*.fits'))

def read_templates(template_list=None, template_dir=None):
    '''
    Return a list of templates from the files in template_list
    
    If template_list is None, use list from find_templates(template_dir)
    '''
    if template_list is None:
        template_list = find_templates(template_dir)

    templates = list()
    for tfile in template_list:
        templates.append(read_template(tfile))
    
    if len(templates) == 0:
        raise IOError('No templates found')
    
    return templates
    
    