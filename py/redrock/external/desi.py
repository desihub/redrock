'''
redrock wrapper tools for DESI
'''

from __future__ import absolute_import, division, print_function

import os, sys
import warnings
if sys.version_info[0] > 2:
    basestring = str

import numpy as np
import desispec.io
from desispec.resolution import Resolution

from .. import Target
from .. import Spectrum

def write_zbest(outfile, zbest):
    '''
    Write zbest Table to outfile

    Adds blank BRICKNAME and SUBTYPE columns if needed
    Adds zbest.meta['EXTNAME'] = 'ZBEST'
    '''
    ntargets = len(zbest)
    if 'BRICKNAME' not in zbest.colnames:
        zbest['BRICKNAME'] = np.zeros(ntargets, dtype='S8')

    if 'SUBTYPE' not in zbest.colnames:
        zbest['SUBTYPE'] = np.zeros(ntargets, dtype='S8')

    zbest.meta['EXTNAME'] = 'ZBEST'
    zbest.write(outfile, overwrite=True)

def read_spectra(spectrafiles, targetids=None):
    '''
    Read targets from a list of spectra files

    Args:
        spectrafiles : list of input spectra files, or string glob to match

    Returns list of Target objects
    '''
    if isinstance(spectrafiles, basestring):
        import glob
        spectrafiles = glob.glob(spectrafiles)

    assert len(spectrafiles) > 0

    input_spectra = list()
    input_targetids = set()

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    for infile in spectrafiles:
        sp = desispec.io.read_spectra(infile)
        input_spectra.append(sp)
        input_targetids.update(sp.fmap['TARGETID'])

    if targetids is None:
        targetids = input_targetids

    targets = list()
    for targetid in targetids:
        spectra = list()
        for sp in input_spectra:
            for x in sp.bands:          #- typically 'b', 'r', 'z'
                wave = sp.wave[x]
                ii = (sp.fmap['TARGETID'] == targetid)
                flux = sp.flux[x][ii]
                ivar = sp.ivar[x][ii]
                Rdata = sp.resolution_data[x][ii]

                #- work around desispec.io.Brick returning 32-bit non-native endian
                # flux = flux.astype(float)
                # ivar = ivar.astype(float)
                # Rdata = Rdata.astype(float)

                for i in range(flux.shape[0]):
                    if np.all(flux[i] == 0):
                        # print('WARNING: Skipping spectrum {} of target {} on brick {} with flux=0'.format(i, targetid, brick.brickname))
                        continue

                    if np.all(ivar[i] == 0):
                        # print('WARNING: Skipping spectrum {} of target {} on brick {} with ivar=0'.format(i, targetid, brick.brickname))
                        continue

                    R = Resolution(Rdata[i])
                    spectra.append(Spectrum(wave, flux[i], ivar[i], R))

        if len(spectra) > 0:
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(targetid, os.path.basename(brickfiles[0])))

    return targets

def read_bricks(brickfiles, trueflux=False, targetids=None):
    '''
    Read targets from a list of brickfiles

    Args:
        brickfiles : list of input brick files, or string glob to match

    Returns list of Target objects

    Note: these don't actually have to be bricks anymore; they are read via
        desispec.io.read_frame()
    '''
    if isinstance(brickfiles, basestring):
        import glob
        brickfiles = glob.glob(brickfiles)

    assert len(brickfiles) > 0

    bricks = list()
    brick_targetids = set()

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    for infile in brickfiles:
        b = desispec.io.read_frame(infile)
        bricks.append(b)
        brick_targetids.update(b.fibermap['TARGETID'])

    if targetids is None:
        targetids = brick_targetids

    targets = list()
    for targetid in targetids:
        spectra = list()
        for brick in bricks:
            wave = brick.wave
            ii = (brick.fibermap['TARGETID'] == targetid)
            flux = brick.flux[ii]
            ivar = brick.ivar[ii]
            Rdata = brick.resolution_data[ii]

            #- work around desispec.io.Brick returning 32-bit non-native endian
            # flux = flux.astype(float)
            # ivar = ivar.astype(float)
            # Rdata = Rdata.astype(float)

            for i in range(flux.shape[0]):
                if np.all(flux[i] == 0):
                    # print('WARNING: Skipping spectrum {} of target {} on brick {} with flux=0'.format(i, targetid, brick.brickname))
                    continue

                if np.all(ivar[i] == 0):
                    # print('WARNING: Skipping spectrum {} of target {} on brick {} with ivar=0'.format(i, targetid, brick.brickname))
                    continue

                R = Resolution(Rdata[i])
                spectra.append(Spectrum(wave, flux[i], ivar[i], R))

        if len(spectra) > 0:
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(targetid, os.path.basename(brickfiles[0])))

    return targets

def rrdesi(options=None):
    import redrock
    import optparse
    from astropy.io import fits

    parser = optparse.OptionParser(usage = "%prog [options] spectra1 spectra2...")
    parser.add_option("-t", "--templates", type="string",  help="template file or directory")
    parser.add_option("-o", "--output", type="string",  help="output file")
    parser.add_option("--zbest", type="string",  help="output zbest fits file")
    parser.add_option("-n", "--ntargets", type=int,  help="number of targets to process")
    parser.add_option("--mintarget", type=int,  help="first target to include", default=0)
    parser.add_option("--ncpu", type=int,  help="number of cpu cores for multiprocessing", default=None)
    parser.add_option("--debug", help="debug with ipython", action="store_true")
    ### parser.add_option("--coadd", help="use coadd instead of individual spectra", action="store_true")
    parser.add_option("--allspec", help="use individual spectra instead of coadd", action="store_true")
    parser.add_option("--bricks", help="inputs are old brick format instead of spectra format", action="store_true")

    if options is None:
        opts, infiles = parser.parse_args()
    else:
        opts, infiles = parser.parse_args(options)

    if (opts.output is None) and (opts.zbest is None):
        print('ERROR: --output or --zbest required')
        sys.exit(1)

    if len(infiles) == 0:
        print('ERROR: must provide input spectra/brick files')
        sys.exit(1)

    if opts.bricks:
        targets = read_bricks(infiles)
    else:
        targets = read_spectra(infiles)

    if not opts.allspec:
        for t in targets:
            t._all_spectra = t.spectra
            t.spectra = t.coadd

    if opts.ntargets is not None:
        targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]

    print('INFO: fitting {} targets'.format(len(targets)))

    templates = redrock.io.read_templates(opts.templates)
    zscan, zfit = redrock.zfind(targets, templates, ncpu=opts.ncpu)

    if opts.output:
        print('INFO: writing {}'.format(opts.output))
        redrock.io.write_zscan(opts.output, zscan, zfit, clobber=True)

    if opts.zbest:
        zbest = zfit[zfit['znum'] == 0]

        #- Remove extra columns not needed for zbest
        zbest.remove_columns(['zz', 'zzchi2', 'znum'])

        #- Change to upper case like DESI
        for colname in zbest.colnames:
            if colname.islower():
                zbest.rename_column(colname, colname.upper())

        #- Get the BRICKNAME from the first brick file
        hdr = fits.getheader(brickfiles[0])
        if 'BRICKNAM' in hdr:
            zbest['BRICKNAME'] = hdr['BRICKNAM']

        print('INFO: writing {}'.format(opts.zbest))
        write_zbest(opts.zbest, zbest)

    if opts.debug:
        import IPython
        IPython.embed()
