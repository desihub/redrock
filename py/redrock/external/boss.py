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

def read_spectra(indir, spall,targetids=None):
    '''
    Read targets from a list of spectra files

    Args:
        indir : input plate directory
        spall: name of a fits file with a BinaryTable in HDU[1] containing the columns PLATE,MJD,FIBERID,THING_ID (typically spAll).
                These columns are used to obtain the TARGETID (= THING_ID) from the PLATE,MJD,FIBERID

    Returns tuple of (targets, meta) where
        targets is a list of Target objects and
        meta is a Table of metadata (currently only BRICKNAME)
    '''
    import glob
    spcframefiles = glob.glob(indir+"/spCFrame-*.fits")

    assert len(spcframefiles) > 0

    spall = fitsio.FITS(spall)
    pla = spall[1]["PLATE"][:]
    mjd = spall[1]["MJD"][:]
    fib = spall[1]["FIBERID"][:]
    tid = spall[1]["THING_ID"][:]

    pmf2tid = {(p,m,f):t for p,m,f,t in zip(pla,mjd,fib,tid)}

    ## get plate and mjd from the first file
    h = fitsio.FITS(spcframefiles[0])
    p = h[0].read_header()["PLATEID"]
    m = h[0].read_header()["MJD"]
    dic_spectra = {}

    #- Ignore warnings about zdc2 bricks lacking bricknames in header
    bricknames=[]
    for infile in spcframefiles:
        h = fitsio.FITS(infile)
        fs = h[5]["FIBERID"][:]

        la = 10**h[3].read()
        fl = h[0].read()
        iv = h[1].read()*(h[2].read()==0)
        wd = h[4].read()

        ## crop to lmin, lmax in blue and red
        if h[0].read_header()["CAMERAS"][0]=="b":
            lmin = 3500.
            lmax = 6000.
        else:
            lmin = 5500.
            lmax = 10000.
        imin = abs(la-lmin).min(axis=0).argmin()
        imax = abs(la-lmax).min(axis=0).argmin()

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
            t = pmf2tid[(p,m,f)]
            if t==-1 or np.all(iv[i]==0):
                ### print("DEBUG: skipping thing_id {} (no thing_id or no data)".format(t))
                continue
            if t not in dic_spectra:
                dic_spectra[t]=[]
                brickname = '{}-{}'.format(p,m)
                bricknames.append(brickname)

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
            dic_spectra[t].append(Spectrum(la[i],fl[i],iv[i],R))

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

def rrboss(options=None):
    import redrock
    import optparse
    from astropy.io import fits
    import time
    start_time = time.time()

    parser = optparse.OptionParser(usage = "%prog [options] spectra1 spectra2...")
    parser.add_option("--indir", type="string",  help="input plate directory")
    parser.add_option("-t", "--templates", type="string",  help="template file or directory")
    parser.add_option("-o", "--output", type="string",  help="output file")
    parser.add_option("--zbest", type="string",  help="output zbest fits file")
    parser.add_option("-n", "--ntargets", type=int,  help="number of targets to process")
    parser.add_option("--mintarget", type=int,  help="first target to include", default=0)
    parser.add_option("--ncpu", type=int,  help="number of cpu cores for multiprocessing", default=None)
    parser.add_option("--debug", help="debug with ipython", action="store_true")
    parser.add_option("--allspec", help="use individual spectra instead of coadd", action="store_true")
    parser.add_option("--spall", type = "string", help="spAll file")

    if options is None:
        opts, infiles = parser.parse_args()
    else:
        opts, infiles = parser.parse_args(options)

    if (opts.output is None) and (opts.zbest is None):
        print('ERROR: --output or --zbest required')
        sys.exit(1)

    if opts.allspec == False:
        print('ERROR: coaddition not yet implemented, please set --allspec')
        sys.exit(1)

    targets, meta = read_spectra(opts.indir,opts.spall)
    print("DEBUG: read {} targets".format(len(targets)))

    if not opts.allspec:
        for t in targets:
            t.do_coadd()

    if opts.ntargets is not None:
        targets = targets[opts.mintarget:opts.mintarget+opts.ntargets]
        meta = meta[opts.mintarget:opts.mintarget+opts.ntargets]

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

        #- Add brickname column
        zbest['BRICKNAME'] = meta['BRICKNAME']

        print('INFO: writing {}'.format(opts.zbest))
        write_zbest(opts.zbest, zbest)

    run_time = time.time() - start_time
    print('INFO: finished in {:.1f} seconds'.format(run_time))

    if opts.debug:
        import IPython
        IPython.embed()
