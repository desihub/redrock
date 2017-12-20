'''
redrock wrapper tools for DESI
'''
from __future__ import absolute_import, division, print_function

import os, sys
import warnings
if sys.version_info[0] > 2:
    basestring = str

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import desispec.io
from desispec.resolution import Resolution

from ..dataobj import (Target, MultiprocessingSharedSpectrum, 
    SimpleSpectrum, MPISharedTargets)

from .. import io
from .. import zfind


def write_zbest(outfile, zbest, fibermap):
    '''
    Write zbest and fibermap Tables to outfile

    Adds blank BRICKNAME and SUBTYPE columns if needed
    Adds zbest.meta['EXTNAME'] = 'ZBEST'
    Adds fibermap.meta['EXTNAME'] = 'FIBERMAP'
    '''
    ntargets = len(zbest)
    if 'BRICKNAME' not in zbest.colnames:
        zbest['BRICKNAME'] = np.zeros(ntargets, dtype='S8')

    if 'SUBTYPE' not in zbest.colnames:
        zbest['SUBTYPE'] = np.zeros(ntargets, dtype='S8')

    zbest.meta['EXTNAME'] = 'ZBEST'
    fibermap.meta['EXTNAME'] = 'FIBERMAP'

    hx = fits.HDUList()
    hx.append(fits.PrimaryHDU())
    hx.append(fits.convenience.table_to_hdu(zbest))
    hx.append(fits.convenience.table_to_hdu(fibermap))
    hx.writeto(outfile, overwrite=True)

def read_spectra(spectrafiles, targetids=None, spectrum_class=SimpleSpectrum):
    '''
    Read targets from a list of spectra files.

    Args:
        spectrafiles : list of input spectra files, or string glob to match

    Options:
        targetids : list of target ids. If set, only those target spectra will be read.
        spectrum_class :  The spectrum_class argument is needed to use the same read_spectra
        routine for the two parallelism approaches for redrock. The multiprocessing version
        uses a shared memory at the spectrum class initialization, see
        the redrock.dataobj.MultiprocessingSharedSpectrum class, whereas the MPI version
        implements the shared memory after all spectra have been read by the root process,
        and so the MPI version used another more simple spectrum class (see redrock.dataobj.SimpleSpectrum).
    
    Returns tuple of (targets, fibermap) where
        targets is a list of Target objects and
        fibermap is the Table from the input spectra files

    Note that due to multiple exposures of the same targets,
    len(targets) != len(fibermap)
    '''
    if isinstance(spectrafiles, basestring):
        import glob
        spectrafiles = glob.glob(spectrafiles)

    assert len(spectrafiles) > 0

    input_spectra = list()
    input_targetids = set()
    input_fibermaps = list()

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    for infile in spectrafiles:
        sp = desispec.io.read_spectra(infile)
        if hasattr(sp, 'fmap'):
            sp.fibermap = sp.fmap   #- for future compatibility
        input_spectra.append(sp)
        input_targetids.update(sp.fibermap['TARGETID'])
        input_fibermaps.append(sp.fibermap)

    if targetids is None:
        targetids = input_targetids

    targets = list()
    for targetid in targetids:
        spectra = list()
        for sp in input_spectra:
            ii = np.where(sp.fibermap['TARGETID'] == targetid)[0]
            if np.count_nonzero(ii) == 0:
                continue
            for x in sp.bands:          #- typically 'b', 'r', 'z'
                wave = sp.wave[x]                
                flux = sp.flux[x][ii]
                ivar = sp.ivar[x][ii]*(sp.mask[x][ii]==0)
                Rdata = sp.resolution_data[x][ii]

                for i in range(flux.shape[0]):
                    ifiber = ii[i]      #- index into sp.fibermap
                    if np.all(flux[i] == 0):
                        continue

                    if np.all(ivar[i] == 0):
                        continue

                    R = Resolution(Rdata[i])

                    meta = dict()
                    meta['NIGHT'] = sp.fibermap['NIGHT'][ifiber]
                    meta['EXPID'] = sp.fibermap['EXPID'][ifiber]

                    #- for backwards compatibility, allow missing TILEID
                    if 'TILEID' in sp.fibermap.dtype.names:
                        meta['TILEID'] = sp.fibermap['TILEID'][ifiber]
                    else:
                        meta['TILEID'] = -1

                    if 'HPXPIXEL' in sp.fibermap.dtype.names:
                        meta['HPXPIXEL'] = sp.fibermap['HPXPIXEL'][ifiber]
                        meta['HPXNSIDE'] = sp.meta['HPXNSIDE']
                    else:
                        import desimodel.footprint
                        meta['HPXNSIDE'] = 64
                        ra = sp.fibermap['RA_TARGET'][ifiber]
                        dec = sp.fibermap['DEC_TARGET'][ifiber]
                        meta['HPXPIXEL'] = desimodel.footprint.radec2pix(64, ra, dec)

                    spectra.append(spectrum_class(wave, flux[i], ivar[i], R, meta=meta))

        #- end of for targetid in targetids loop

        if len(spectra) > 0:
            numexp = len(set([sp.meta['EXPID'] for sp in spectra]))
            if spectra[0].meta['TILEID'] >= 0:
                numtile = len(set([sp.meta['TILEID'] for sp in spectra]))
            else:
                numtile = -1

            meta = dict(NUMEXP=numexp, NUMTILE=numtile)
            targets.append(Target(targetid, spectra, meta=meta))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(targetid, os.path.basename(brickfiles[0])))

    fibermap = vstack(input_fibermaps)

    return targets, fibermap


def rrdesi(options=None, comm=None):
    import optparse
    from astropy.io import fits
    import time

    # Note, in the pure multiprocessing case (comm == None), "rank"
    # will always be set to zero, which is fine since we are outside
    # any areas using multiprocessing and this is just used to control
    # program flow.
    rank = 0
    nproc = 1
    if comm is not None:
        rank = comm.rank
        nproc = comm.size

    start_time = time.time()
    pid = os.getpid()

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

    templates = None
    targets   = None
    meta      = None
    if rank == 0:
        t0 = time.time()
        print('INFO: reading targets')
        sys.stdout.flush()
        
        spectrum_class = SimpleSpectrum
        if opts.ncpu is None or opts.ncpu > 1:
            spectrum_class = MultiprocessingSharedSpectrum

        targets, fibermap = read_spectra(infiles,spectrum_class=spectrum_class)
            
        if not opts.allspec:
            for t in targets:
                t._all_spectra = t.spectra
                t.spectra = t.coadd

        if opts.ntargets is not None:
            targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
            targetids = [t.id for t in targets]
            ii = np.in1d(fibermap['TARGETID'], targetids)
            fibermap = fibermap[ii]
        
        print('INFO: reading templates')
        sys.stdout.flush()
        
        if not opts.templates is None and os.path.isdir(opts.templates):
            templates = io.read_templates(template_list=None, template_dir=opts.templates)
        else:
            templates = io.read_templates(template_list=opts.templates, template_dir=None)

        dt = time.time() - t0
        # print('DEBUG: PID {} read targets and templates in {:.1f} seconds'.format(pid,dt))
        sys.stdout.flush()
        
    # all processes get a copy of the templates from rank 0
    if comm is not None:
        templates = comm.bcast(templates, root=0)

    # Call zfind differently depending on our type of parallelism.

    if comm is not None:
        # Use MPI
        with MPISharedTargets(targets, comm) as shared_targets:
            zscan, zfit = zfind(shared_targets.targets, templates, 
                ncpu=None, comm=comm)
    else:
        # Use pure multiprocessing
        zscan, zfit = zfind(targets, templates, ncpu=opts.ncpu)
    
    if rank == 0:
        if opts.output:
            print('INFO: writing {}'.format(opts.output))
            io.write_zscan(opts.output, zscan, zfit, clobber=True)

        if opts.zbest:
            zbest = zfit[zfit['znum'] == 0]

            #- Remove extra columns not needed for zbest
            zbest.remove_columns(['zz', 'zzchi2', 'znum'])

            #- Change to upper case like DESI
            for colname in zbest.colnames:
                if colname.islower():
                    zbest.rename_column(colname, colname.upper())

            #- DEPRECATED, but keep for now: add brickname column
            for i in range(len(zbest)):
                ii = np.where(fibermap['TARGETID'] == zbest['TARGETID'][i])[0]
                zbest['BRICKNAME'] = fibermap['BRICKNAME'][ii[0]]

            #- Cosmetic: move TARGETID to be first column as primary key
            zbest.columns.move_to_end('TARGETID', last=False)

            ntargets = len(targets)
            zbest['NUMEXP'] = np.zeros(ntargets, dtype=np.int16)
            zbest['NUMTILE'] = np.zeros(ntargets, dtype=np.int16)
            for i in range(ntargets):
                assert targets[i].id == zbest['TARGETID'][i]
                zbest['NUMEXP'][i] = targets[i].meta['NUMEXP']
                zbest['NUMTILE'][i] = targets[i].meta['NUMTILE']

            #- Cosmetic: move TARGETID to be first column as primary key
            zbest.columns.move_to_end('TARGETID', last=False)

            ntargets = len(targets)
            zbest['NUMEXP'] = np.zeros(ntargets, dtype=np.int16)
            zbest['NUMTILE'] = np.zeros(ntargets, dtype=np.int16)
            for i in range(ntargets):
                assert targets[i].id == zbest['TARGETID'][i]
                zbest['NUMEXP'][i] = targets[i].meta['NUMEXP']
                zbest['NUMTILE'][i] = targets[i].meta['NUMTILE']

            print('INFO: writing {}'.format(opts.zbest))
            write_zbest(opts.zbest, zbest, fibermap)

    run_time = time.time() - start_time
    
    if comm is None or comm.rank == 0:
        print('INFO: finished {} in {:.1f} seconds'.format(os.path.basename(infiles[0]), run_time))
    
    if opts.debug:
        if comm is not None:
            print('INFO: ignoring ipython debugging when using MPI')
        else:
            import IPython
            IPython.embed()

    return



