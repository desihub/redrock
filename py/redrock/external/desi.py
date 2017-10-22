'''
redrock wrapper tools for DESI
'''
from __future__ import absolute_import, division, print_function

import os, sys
import re
import warnings
from collections import OrderedDict

if sys.version_info[0] > 2:
    basestring = str

import numpy as np

from astropy.io import fits
from astropy.table import Table

from desiutil.io import encode_table

import desispec.io
from desispec.resolution import Resolution

from ..dataobj import (Target, MultiprocessingSharedSpectrum, 
    SimpleSpectrum, MPISharedTargets)

from .. import io
from .. import zfind


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


def desi_target_offsets(spectrafiles, targetids=None):
    '''
    Read target offsets from a list of spectra files.

    This reads metadata information from a set of spectra files and computes
    a dictionary of shared memory buffer offsets.  See 
    redrock.dataobj.target_offsets() for more details on the format.
   
    Args:
        spectrafiles : list of input spectra files, or string glob to match
        targetids : list of target ids. If set, only those target spectra will
            be read.
    
    Returns tuple of (dict, int):
        the first element is the dictionary of offsets, and the second is
        the number of double precision elements required for the shared
        buffer.
    '''
    if isinstance(spectrafiles, basestring):
        import glob
        spectrafiles = glob.glob(spectrafiles)

    assert len(spectrafiles) > 0

    input_bands = {}
    input_fmap = {}
    input_targetids = set()

    for sfile in spectrafiles:
        input_bands[sfile] = {}
        hdus = fits.open(sfile, memmap=True)
        nhdu = len(hdus)
        for h in range(1, nhdu):
            name = hdus[h].header["EXTNAME"]
            if name == "FIBERMAP":
                input_fmap[sfile] = encode_table(Table(hdus[h].data, 
                    copy=True).as_array())
                input_targetids.update(input_fmap[sfile]['TARGETID'])
            else:
                # Find the band based on the name
                mat = re.match(r"(.*)_(.*)", name)
                if mat is None:
                    raise RuntimeError("FITS extension name {} does not contain the band".format(name))
                band = mat.group(1).lower()
                type = mat.group(2)
                if band not in input_bands[sfile]:
                    input_bands[sfile][band] = {}
                if type == "WAVELENGTH":
                    input_bands[sfile][band]["nwave"] = \
                        int(hdus[h].header["NAXIS1"])
                elif type == "RESOLUTION":
                    input_bands[sfile][band]["ndiag"] = \
                        int(hdus[h].header["NAXIS2"])
        hdus.close()

    # Construct the target dictionary

    if targetids is None:
        targetids = input_targetids

    nelem = 0
    dictionary = OrderedDict()

    for targetid in targetids:
        dictionary[targetid] = {}
        spectra_list = list()
        for sfile in spectrafiles:
            ii = (input_fmap[sfile]['TARGETID'] == targetid)
            tspec = np.count_nonzero(ii)
            if tspec == 0:
                continue
            for spec in range(tspec):
                for band in input_bands[sfile]:
                    spectrum_dict = {}
                    # Here we compute the sizes of the data, offsets, and
                    # shape of sparse resolution matrix.  We want to do this
                    # without actually constructing the matrix, since that
                    # would require reading the data.
                    Rdata_size = input_bands[sfile][band]["nwave"] * \
                        input_bands[sfile][band]["ndiag"]
                    halfdiag = input_bands[sfile][band]["ndiag"] // 2
                    Roffset_size = 2 * halfdiag + 1
                    Rshape_size = 2
                    for label, asize in zip(["wave","flux","ivar","rdata",
                        "roffsets","rshape"], 
                        [input_bands[sfile][band]["nwave"], 
                        input_bands[sfile][band]["nwave"],
                        input_bands[sfile][band]["nwave"],
                        Rdata_size, Roffset_size, Rshape_size]):
                        spectrum_dict[label] = [nelem, nelem + asize]
                        nelem += asize
                    spectra_list.append(spectrum_dict)
                dictionary[targetid]["spectra"] = spectra_list
        
        # Now compute the size of the coadd- assume same bands
        # between files and a resolution matrix of the same size
        spectra_list = list()
        for band in input_bands[spectrafiles[0]]:
            spectrum_dict = {}
            Rdata_size = input_bands[spectrafiles[0]][band]["nwave"] * \
                input_bands[spectrafiles[0]][band]["ndiag"]
            halfdiag = input_bands[spectrafiles[0]][band]["ndiag"] // 2
            Roffset_size = 2 * halfdiag + 1
            Rshape_size = 2
            for label, asize in zip(["wave","flux","ivar","rdata",
                "roffsets","rshape"], 
                [input_bands[spectrafiles[0]][band]["nwave"], 
                input_bands[spectrafiles[0]][band]["nwave"],
                input_bands[spectrafiles[0]][band]["nwave"],
                Rdata_size, Roffset_size, Rshape_size]):
                spectrum_dict[label] = [nelem, nelem + asize]
                nelem += asize
            spectra_list.append(spectrum_dict)
        dictionary[targetid]["coadd"] = spectra_list

    return dictionary, nelem


def read_spectra_shared(shm, spectrafiles):
    """
    """
    if isinstance(spectrafiles, basestring):
        import glob
        spectrafiles = glob.glob(spectrafiles)

    assert len(spectrafiles) > 0

    # We are directly manipulating internal buffers of the shared memory 
    # object, so it seems fine to use internal variables.

    rank = 0
    if shm._comm is not None:
        rank = shm._comm.rank

    offsets = shm.target_offsets

    for sfile in spectrafiles:
        hdus = None
        nhdu = None
        if rank == 0:
            hdus = fits.open(sfile, memmap=True)
            nhdu = len(hdus)
            fmap = encode_table(Table(hdus["FIBERMAP"].data, 
                copy=True).as_array())
        if shm._comm is not None:
            nhdu = shm._comm.bcast(nhdu, root=0)

        for h in range(1, nhdu):
            name = None
            if rank == 0:
                name = hdus[h].header["EXTNAME"]
            if shm._comm is not None:
                name = shm._comm.bcast(name, root=0)
            # Find the band based on the name
            mat = re.match(r"(.*)_(.*)", name)
            if mat is None:
                continue
            band = mat.group(1).lower()
            type = mat.group(2)

            if type == "WAVELENGTH":
                # This is a wavelength HDU, read it and update the
                # data for all targets.  FIXME:  in the case of multiple
                # spectra files, do we allow different wavelength ranges
                # for the same band in different files?
                wdata = None
                if shm._root:
                    wdata = hdus[h].data
                for id in offsets.keys():
                    tdict = offsets[id]
                    for spc in tdict["spectra"]:
                        begin = spc["wave"][0]
                        shm.set(wdata, (begin,), fromrank=0)
                del wdata
            else:
                # this is some other type of data, update the individual
                # spectra.
                nspec = None
                if rank == 0:
                    if type == "RESOLUTION":
                        nspec = int(hdus[h].header["NAXIS3"])
                    else:
                        nspec = int(hdus[h].header["NAXIS2"])
                if shm._comm is not None:
                    nspec = shm._comm.bcast(nspec, root=0)

                #need the global spec offset for this target, not just
                # the offset in THIS FILE!!

                for s in range(nspec):
                    buffer = None
                    if rank == 0:
                        id = fmap[s]["TARGETID"]
                    if shm._comm is not None:
                        id = shm._comm.bcast(id, root=0)
                    if id in offsets:
                        tdict = offsets[id]
                        for spc in tdict["spectra"]:
                            if type == "FLUX":
                                begin = spc["flux"][s][0]
                                if shm._root:
                                    buffer = hdus[h].data[s].ravel()
                                shm.set(buffer, (begin,), fromrank=0)
                            elif type == "IVAR":
                                begin = spc["ivar"][s][0]
                                if shm._root:
                                    buffer = hdus[h].data[s].ravel()
                                shm.set(buffer, (begin,), fromrank=0)
                            elif type == "RESOLUTION":
                                res = None
                                if shm._root:
                                    res = Resolution(hdus[h].data[s])
                                begin = spc["rdata"][s][0]
                                if shm._root:
                                    buffer = res.data
                                shm.set(buffer, (begin,), fromrank=0)
                                begin = spc["roffsets"][s][0]
                                if shm._root:
                                    buffer = res.offsets.as_type(np.float64)
                                shm.set(buffer, (begin,), fromrank=0)
                                begin = spc["rshape"][s][0]
                                if shm._root:
                                    buffer = res.shape.as_type(np.float64)
                                shm.set(buffer, (begin,), fromrank=0)
        hdus.close()


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
                ivar = sp.ivar[x][ii]*(sp.mask[x][ii]==0)
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


def read_bricks(brickfiles, trueflux=False, targetids=None, spectrum_class=SimpleSpectrum):
    '''
    Read targets from a list of brickfiles

    Args:
        brickfiles : list of input brick files, or string glob to match
    
    Options:
        targetids : list of target ids. If set, only those target spectra will be read.
        spectrum_class :  The spectrum_class argument is needed to use the same read_spectra
        routine for the two parallelism approaches for redrock. The multiprocessing version
        uses a shared memory at the spectrum class initialization, see
        the redrock.dataobj.MultiprocessingSharedSpectrum class, whereas the MPI version
        implements the shared memory after all spectra have been read by the root process,
        and so the MPI version used another more simple spectrum class (see redrock.dataobj.SimpleSpectrum).
    
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
            ivar = brick.ivar[ii] * (brick.mask[ii]==0)
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
        offsets, nelem = desi_target_offsets(infiles)
        shared_targets = MPISharedTargets(nelem, offsets, comm)
        read_spectra_shared(shared_targets, infiles)

        zscan, zfit = zfind(shared_targets.targets, templates, 
            ncpu=None, comm=comm)
    else:
        # Use pure multiprocessing
        try:
            targets, meta = read_spectra(infiles, 
                spectrum_class=spectrum_class)
        except RuntimeError:
            targets, meta = read_bricks(infiles, 
                spectrum_class=spectrum_class)
            
        if not opts.allspec:
            for t in targets:
                t._all_spectra = t.spectra
                t.spectra = t.coadd

        if opts.ntargets is not None:
            targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
            meta = meta[opts.mintarget:opts.mintarget+opts.ntargets]

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

            #- Add brickname column
            zbest['BRICKNAME'] = meta['BRICKNAME']

            print('INFO: writing {}'.format(opts.zbest))
            write_zbest(opts.zbest, zbest)

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



