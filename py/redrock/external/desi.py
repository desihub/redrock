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
from .. import MultiprocessingSharedSpectrum, SimpleSpectrum,  MPISharedTargets

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

def read_spectra(spectrafiles, targetids=None, spectrum_class=None):
    '''
    Read targets from a list of spectra files

    Args:
        spectrafiles : list of input spectra files, or string glob to match

    Returns tuple of (targets, meta) where
        targets is a list of Target objects and
        meta is a Table of metadata (currently only BRICKNAME)
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
        if hasattr(sp, 'fmap'):
            sp.fibermap = sp.fmap   #- for future compatibility
        input_spectra.append(sp)
        input_targetids.update(sp.fibermap['TARGETID'])

    if targetids is None:
        targetids = input_targetids

    targets = list()
    bricknames = list()
    for targetid in targetids:
        spectra = list()
        for sp in input_spectra:
            ii = (sp.fibermap['TARGETID'] == targetid)
            if np.count_nonzero(ii) == 0:
                continue
            if 'BRICKNAME' in sp.fibermap.dtype.names:
                brickname = sp.fibermap['BRICKNAME'][ii][0]
            else:
                brickname = 'unknown'
            for x in sp.bands:          #- typically 'b', 'r', 'z'
                wave = sp.wave[x]                
                flux = sp.flux[x][ii]
                ivar = sp.ivar[x][ii]
                Rdata = sp.resolution_data[x][ii]

                for i in range(flux.shape[0]):
                    if np.all(flux[i] == 0):
                        # print('WARNING: Skipping spectrum {} of target {} on brick {} with flux=0'.format(i, targetid, brick.brickname))
                        continue

                    if np.all(ivar[i] == 0):
                        # print('WARNING: Skipping spectrum {} of target {} on brick {} with ivar=0'.format(i, targetid, brick.brickname))
                        continue

                    R = Resolution(Rdata[i])
                    spectra.append(spectrum_class(wave, flux[i], ivar[i], R))

        bricknames.append(brickname)
        #- end of for targetid in targetids loop

        if len(spectra) > 0:
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(targetid, os.path.basename(brickfiles[0])))

    #- Create a metadata table in case we might want to add other columns
    #- in the future
    assert len(bricknames) == len(targets)
    dtype = [('BRICKNAME', 'S8'),]
    meta = np.zeros(len(bricknames), dtype=dtype)
    meta['BRICKNAME'] = bricknames

    return targets, meta

def read_bricks(brickfiles, trueflux=False, targetids=None, spectrum_class=None):
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
    bricknames = list()
    for targetid in targetids:
        spectra = list()
        for brick in bricks:
            wave = brick.wave
            ii = (brick.fibermap['TARGETID'] == targetid)
            if np.count_nonzero(ii) == 0:
                continue

            if 'BRICKNAME' in brick.fibermap.dtype.names:
                brickname = brick.fibermap['BRICKNAME'][ii][0]
            else:
                brickname = 'unknown'
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
                spectra.append(spectrum_class(wave, flux[i], ivar[i], R))

        #- end of for targetid in targetids loop

        if len(spectra) > 0:
            bricknames.append(brickname)
            targets.append(Target(targetid, spectra))
        else:
            print('ERROR: Target {} on {} has no good spectra'.format(targetid, os.path.basename(brickfiles[0])))

    #- Create a metadata table in case we might want to add other columns
    #- in the future
    assert len(bricknames) == len(targets)
    dtype = [('BRICKNAME', 'S8'),]
    meta = np.zeros(len(bricknames), dtype=dtype)
    meta['BRICKNAME'] = bricknames

    return targets, meta

def rrdesi_multiprocessing(options=None):
    import redrock
    import optparse
    from astropy.io import fits
    import time
    start_time = time.time()

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

    try:
        targets, meta = read_spectra(infiles,spectrum_class=MultiprocessingSharedSpectrum)
    except RuntimeError:
        targets, meta = read_bricks(infiles,spectrum_class=MultiprocessingSharedSpectrum)

    if not opts.allspec:
        for t in targets:
            t._all_spectra = t.spectra
            t.spectra = t.coadd

    if opts.ntargets is not None:
        targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
        meta = meta[opts.mintarget:opts.mintarget+opts.ntargets]

    print('INFO: fitting {} targets'.format(len(targets)))

    templates = redrock.io.read_templates(opts.templates)

    print('DEBUG: only test IO for now')
    if False :
        
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

            #- Add brickname column
            zbest['BRICKNAME'] = meta['BRICKNAME']

            print('INFO: writing {}'.format(opts.zbest))
            write_zbest(opts.zbest, zbest)

    run_time = time.time() - start_time
    print('INFO: finished {} in {:.1f} seconds'.format(os.path.basename(infiles[0]), run_time))

    if opts.debug:
        import IPython
        IPython.embed()

        
def rrdesi_mpi(options=None):

    
    from mpi4py import MPI
    
    import redrock
    import optparse
    from astropy.io import fits
    import time
    start_time = time.time()
    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    procs = comm.size

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

    
    if comm.rank==0 : # MPI: only one process reads the data and the templates
        print('rank #%d : reading targets'%comm.rank)
        try:
            targets, meta = read_spectra(infiles,spectrum_class=SimpleSpectrum)
        except RuntimeError:
            targets, meta = read_bricks(infiles,spectrum_class=SimpleSpectrum)
            
        if not opts.allspec:
            for t in targets:
                t._all_spectra = t.spectra
                t.spectra = t.coadd

        if opts.ntargets is not None:
            targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
            meta = meta[opts.mintarget:opts.mintarget+opts.ntargets]
        
        print('rank #%d : reading templates'%comm.rank)

        templates = redrock.io.read_templates(opts.templates)

        print('rank #%d : done reading'%comm.rank)
        
    else : # MPI : the others don't do anything
        print('rank #%d : waiting'%comm.rank)
        templates = None
        targets   = None
        meta      = None
        
    # all processes get a copy of the templates  
    templates = comm.bcast(templates,root=0)
    print('rank #%d : I have %d templates from bcast'%(comm.rank,len(templates)))
    meta = comm.bcast(meta,root=0)
    print('rank #%d : I have the targets meta data from bcast'%(comm.rank))
    
    with MPISharedTargets(targets, comm) as shared_targets :
        
        print('rank #%d : shared_targets._data.size=%d'%(comm.rank,shared_targets._data.size))
        print('rank #%d : shared_targets._data=%s'%(comm.rank,shared_targets._data[:12]))
        
        targets = shared_targets.get_targets()
        print('rank #%d : number of targets = %d'%(comm.rank,len(targets)))
    
        
    print('DEBUG: only test IO for now')
    
    if False :
        print('INFO: fitting {} targets'.format(len(targets)))

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

            #- Add brickname column
            zbest['BRICKNAME'] = meta['BRICKNAME']

            print('INFO: writing {}'.format(opts.zbest))
            write_zbest(opts.zbest, zbest)

    run_time = time.time() - start_time
    print('INFO: finished {} in {:.1f} seconds'.format(os.path.basename(infiles[0]), run_time))

    if opts.debug:
        import IPython
        IPython.embed()
