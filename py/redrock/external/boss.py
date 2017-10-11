'''
redrock wrapper tools for BOSS
'''

from __future__ import absolute_import, division, print_function

import os, sys, fitsio
import warnings
if sys.version_info[0] > 2:
    basestring = str

import numpy as np
from scipy import sparse
import desispec.resolution
from desispec.resolution import Resolution
import os.path

from .. import Target
from .. import io
from .. import zfind
from ..dataobj import (Target, MultiprocessingSharedSpectrum, 
    SimpleSpectrum, MPISharedTargets)

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

def read_spectra(spplate_name, spall, targetids=None,spectrum_class=SimpleSpectrum,use_frames=False):
    '''
    Read targets from a list of spectra files

    Args:
        spplate_name: input spPlate file
        spall: spAll file. Required to get THING_IDs from (PLATE,FIBER)

    Returns tuple of (targets, meta) where
        targets is a list of Target objects and
        meta is a Table of metadata (currently only BRICKNAME)
    '''
    spall = fitsio.FITS(spall)
    plates = spall[1]["PLATE"][:]
    fibers = spall[1]["FIBERID"][:]
    thingid = spall[1]["THING_ID"][:]
    pf2thid = {(p,f):t for p,f,t in zip(plates,fibers,thingid)}
    spall.close()

    ## read spplate
    spplate = fitsio.FITS(spplate_name)
    plate = spplate[0].read_header()["PLATEID"]
    mjd = spplate[0].read_header()["MJD"]
    if not use_frames:
        infiles = [spplate_name]
    if use_frames:
        path = os.path.dirname(spplate_name)
        cameras = ['b1','r1','b2','r2']

        infiles = []
        nexp_tot=0
        for c in cameras:
            try:
                nexp = spplate[0].read_header()["NEXP_{}".format(c.upper())]
            except ValueError:
                print("DEBUG: spplate {} has no exposures in camera {} ".format(spplate_name,c))
                continue
            for i in range(1,nexp+1):
                nexp_tot += 1
                expid = str(nexp_tot)
                if nexp_tot<10:
                    expid = '0'+expid
                exp = path+"/spCFrame-"+spplate[0].read_header()["EXPID"+expid][:11]+".fits"
                infiles.append(exp)
    
    for f in infiles:
        print(f)
    spplate.close()
    bricknames=[]
    dic_spectra = {}

    for infile in infiles:
        h = fitsio.FITS(infile)
        assert plate == h[0].read_header()["PLATEID"]
        fs = h[5]["FIBERID"][:]

        fl = h[0].read()
        iv = h[1].read()*(h[2].read()==0)
        wd = h[4].read()

        ## crop to lmin, lmax
        lmin = 3500.
        lmax = 10000.
        if use_frames:
            la = 10**h[3].read()
            if h[0].read_header()["CAMERAS"][0]=="b":
                lmin = 3500.
                lmax = 6000.
            else:
                lmin = 5500.
                lmax = 10000.
        else:
            coeff0 = h[0].read_header()["COEFF0"]
            coeff1 = h[0].read_header()["COEFF1"]
            la = 10**(coeff0 + coeff1*np.arange(fl.shape[1]))
            la = np.broadcast_to(la,fl.shape)
        
        imin = abs(la-lmin).min(axis=0).argmin()
        imax = abs(la-lmax).min(axis=0).argmin()

        print("DEBUG: imin {} imax {}".format(imin,imax))

        la = la[:,imin:imax]
        fl = fl[:,imin:imax]
        iv = iv[:,imin:imax]
        wd = wd[:,imin:imax]

        w = wd<1e-5
        wd[w]=2.
        ii = np.arange(la.shape[1])
        di = ii-ii[:,None]
        di2 = di**2
        ndiag = int(4*np.ceil(wd.max())+1)
        nbins = wd.shape[1]

        for i,f in enumerate(fs):
            if np.all(iv[i]==0):
                print("DEBUG: skipping plate,fid = {},{} (no data)".format(plate,f))
                continue
            t = pf2thid[(plate,f)]
            if t==-1:
                sf = str(f)
                if f<1000:
                    sf = '0'+sf
                if f<100:
                    sf = '0'+sf
                if f<10:
                    sf = '0'+sf
                t = int(str(plate)+str(mjd)+sf)
                print("DEBUG: changing negative thing id to PLATEMJDFIBERID {}".format(t))
            if t not in dic_spectra:
                dic_spectra[t]=[]
                brickname = '{}-{}'.format(plate,mjd)
                bricknames.append(brickname)

            ## build resolution from wdisp
            reso = np.zeros([ndiag,nbins])
            for idiag in range(ndiag):
                offset = ndiag//2-idiag
                d = np.diagonal(di2,offset=offset)
                if offset<0:
                    reso[idiag,:len(d)] = np.exp(-d/2/wd[i,:len(d)]**2)
                else:
                    reso[idiag,nbins-len(d):nbins]=np.exp(-d/2/wd[i,nbins-len(d):nbins]**2)

            R = Resolution(reso)
            ccd = sparse.spdiags(1./R.sum(axis=1).T, 0, *R.shape)
            R = (ccd*R).todia()
            dic_spectra[t].append(spectrum_class(la[i],fl[i],iv[i],R))

        h.close()
        print("DEBUG: read {} ".format(infile))

    if targetids == None:
        targetids = dic_spectra.keys()

    targets = []
    for targetid in targetids:
        spectra = dic_spectra[targetid]
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

def rrboss(options=None, comm=None):
    import redrock
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
    parser.add_option("--spplate", type="string",  help="input plate directory")
    parser.add_option("--spall", type="string",  help="spAll file")
    parser.add_option("-t", "--templates", type="string",  help="template file or directory")
    parser.add_option("-o", "--output", type="string",  help="output file")
    parser.add_option("--zbest", type="string",  help="output zbest fits file")
    parser.add_option("-n", "--ntargets", type=int,  help="number of targets to process")
    parser.add_option("--mintarget", type=int,  help="first target to include", default=0)
    parser.add_option("--ncpu", type=int,  help="number of cpu cores for multiprocessing", default=None)
    parser.add_option("--debug", help="debug with ipython", action="store_true")
    parser.add_option("--use-frames", help="use individual spcframes instead of spplate (the spCFrame files are expected to be in the same directory as the spPlate", action="store_true")

    if options is None:
        opts, infiles = parser.parse_args()
    else:
        opts, infiles = parser.parse_args(options)

    if (opts.output is None) and (opts.zbest is None):
        print('ERROR: --output or --zbest required')
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

        targets, meta = read_spectra(opts.spplate,opts.spall,spectrum_class=spectrum_class,use_frames=opts.use_frames)
        print("DEBUG: read {} targets".format(len(targets)))

        if not opts.use_frames:
            for t in targets:
                t._all_spectra = t.spectra
                t.spectra = t.coadd

        if opts.ntargets is not None:
            targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
            meta = meta[opts.mintarget:opts.mintarget+opts.ntargets]

        print('INFO: fitting {} targets'.format(len(targets)))
        sys.stdout.flush()

        templates = io.read_templates(opts.templates)

        dt = time.time() - t0
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

            #- Add brickname column
            zbest['BRICKNAME'] = meta['BRICKNAME']

            print('INFO: writing {}'.format(opts.zbest))
            write_zbest(opts.zbest, zbest)

    run_time = time.time() - start_time

    if comm is None or comm.rank == 0:
        print('INFO: finished in {:.1f} seconds'.format(run_time))

    if opts.debug:
        if comm is not None:
            print('INFO: ignoring ipython debugging when using MPI')
        else:
            import IPython
            IPython.embed()

    return
